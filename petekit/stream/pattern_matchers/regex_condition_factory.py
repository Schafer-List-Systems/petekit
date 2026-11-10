"""Regex-based condition factory for stream buffer routing tables."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

_logger = logging.getLogger("regex_condition_factory")

from peteos import AgenticObject
from ...function.function_manager import FunctionManager
from ...text.buffer_manager import Buffer, BufferManager, HookPoint, _sanitize_buffer_name
from ...text.option_buffer_manager import _origin_name_from_option_buffer_name
from ..stream_router import RoutingConditionError, CONDITION_FN


_CONDITION_LIST_BUFFER = "system:status:regex_condition"
_REGEX_DOC_BUFFER = "doc:regex_conditions"
_REGEX_PATTERN_BUFFER = "system:config:regex_condition"
_CONDITION_BUFFER_PATTERN = r"^system:config:regex_condition:[^[]+$"
_REGEX_CONDITION_SUFFIX = "[regex_condition]"
_REGEX_CONDITION_OPTION_BUFFER_PATTERN = rf".+\[{_REGEX_CONDITION_SUFFIX[1:-1]}\]$"
_READ_BUFFER_FN = Callable[[str], dict[str, Any]]



_REGEX_DOC = f"""\
## Regex Conditions

A regex condition matches incoming text against one or more regular expression
patterns. The text must fullmatch (the entire line including the trailing newline
character '\\n') at least one pattern (^...\\n$) for the condition to return True.

### Create a condition (two buffers)

**Step 1 — Origin buffer** (holds the patterns)

Create an <origin_buffer> buffer (e.g., `example-patterns.json`):

```json
{{
    "<pattern_name>": {{
        "regex": "<regular expression including \\n>",
        "sample_line": "<example text that the regex must match>\\n"
    }}
}}
```

**Step 2 — Option buffer** (registers the condition)

Create a buffer named `<origin_buffer>[regex_condition]`:

```json
{{
    "condition_name": "<condition_name>"
}}
```

The origin buffer must exist before the option buffer is created.
The `condition_name` is immutable after creation — renaming is not allowed.

### Update a condition

Edit the **origin buffer** directly. Changes are validated against `sample_line`
before being accepted.

### Remove a condition

**Option 1** — Drop the option buffer `<origin_name>[regex_condition]`. The
condition is unregistered and its catalog entry is removed. The origin buffer
is left intact.

**Option 2** — Drop the origin buffer. The condition is unregistered and
the catalog entry is removed.

A condition cannot be dropped while it is in use.

### Pattern requirements

- `regex` — a valid regular expression including the trailing '\\n'
- `sample_line` — a line of text the regex must fullmatch exactly (including '\\n').
  Used for validation only, not stored after creation.

### Conditions catalog

`system:status:regex_condition` — a managed catalog of all conditions (do not edit manually):

```json
{{
    "conditions": {{
        "<condition_name>": {{
            "name": "<condition_name>",
            "pattern_count": N,
            "origin": "<origin_buffer_name>"
        }}
    }}
}}
```

Read this buffer to discover available conditions. The `origin` field shows
which buffer holds the patterns.
"""


# -------------------------------------------------------------------------------------------------------------------- #
# Utils                                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #


def _get_option_config(option_buffer: Buffer) -> dict[str, Any] | str:
    """Read and parse the JSON config from an option buffer.

    Returns the parsed config dict on success, or an error string if the buffer is
    absent, empty, or contains invalid JSON.
    """
    # Read the raw text from the option buffer.
    buffer_text = option_buffer.get_text()
    if not buffer_text:
        return f"[RegexConditionFactory] option buffer '{option_buffer.name}' is empty"

    # Parse the JSON config from the buffer text.
    try:
        config = json.loads(buffer_text)
    except json.JSONDecodeError as e:
        return f"[RegexConditionFactory] option buffer '{option_buffer.name}' has invalid JSON: {e}"

    return config


async def _remove_condition_from_catalog(
    factory: RegexConditionFactory,
    condition_name: str,
) -> bool:
    """Remove a condition entry from the conditions catalog.

    Returns True if the entry was found and removed, False otherwise.
    """
    # Get the conditions catalog buffer.
    conditions_buffer = factory.get_buffer(_CONDITION_LIST_BUFFER)
    if conditions_buffer is None:
        return False

    # Parse the conditions catalog JSON.
    try:
        conditions_data = json.loads(conditions_buffer.get_text())
    except json.JSONDecodeError:
        return False

    # Return early when the condition is not in the catalog.
    conditions = conditions_data.get("conditions", {})
    if condition_name not in conditions:
        return False

    # Remove the condition entry from the catalog.
    del conditions["conditions"][condition_name]

    # Write the updated catalog back to the buffer.
    await factory.create_buffer(
        _CONDITION_LIST_BUFFER,
        text=json.dumps(conditions_data, indent=2) + "\n",
        overwrite=True,
    )
    return True


def _make_condition(
    pattern_buffer: str,
    read_buffer_fn: _READ_BUFFER_FN,
    condition_name: str,
) -> CONDITION_FN:
    """Return a condition function that fullmatches text against all patterns in the pattern buffer."""
    # Return a closure that reads the pattern buffer and fullmatches the incoming text
    # against every registered pattern. The routing table calls this with (stream, text, metadata).
    def condition(function_manager: FunctionManager, stream_buffer_name: str, appended_text: str, metadata: dict) -> bool:

        # Read the pattern buffer content — the caller provides the read function.
        read_result = read_buffer_fn(pattern_buffer)
        if not read_result.get("ok"):
            raise RoutingConditionError(
                condition=condition_name,
                reason=f"pattern buffer could not be read: {read_result.get('error')}",
            )

        # Parse the pattern buffer JSON — it must be a valid JSON object.
        try:
            pattern_data = json.loads(read_result["content"])
        except json.JSONDecodeError as e:
            raise RoutingConditionError(
                condition=condition_name,
                reason=f"pattern buffer is not valid JSON: {e.msg} — expected a JSON object in '{pattern_buffer}'",
            )

        # Evaluate each pattern: compile and test against the text with fullmatch.
        for pat in pattern_data.values():
            compiled = re.compile(pat.get("regex", ""))
            if compiled.fullmatch(appended_text):
                return True

        # No pattern matched — return False and let the routing table try the next condition.
        return False

    return condition


def _guard_pattern_value(op_type: str, value) -> re.Pattern | str:
    """Validate a single pattern entry. Return the compiled pattern or an error string."""
    # Reject non-dict values and values missing the required 'regex' key.
    if not isinstance(value, dict) or "regex" not in value:
        return f"regex {op_type} must supply a dict with 'regex' key, got: {type(value).__name__} — see {_REGEX_DOC_BUFFER}"

    # Reject entries missing 'sample_line' — validation requires a sample to test against.
    if "sample_line" not in value:
        return f"regex {op_type} must supply 'sample_line' — the regex cannot be validated without a sample — see {_REGEX_DOC_BUFFER}"

    # Validate the regex against its sample_line; propagate validation errors upward.
    validation = _validate_pattern_against_sample(value["regex"], value["sample_line"])
    if isinstance(validation, str):
        return f"{validation} — see {_REGEX_DOC_BUFFER}"

    return validation


def _validate_pattern_against_sample(pattern: str, sample_line: str) -> re.Pattern | str:
    """Validate a regex pattern against a sample line using fullmatch semantics.

    Returns the compiled pattern on success, or an error string on failure.
    """
    # Compile the pattern first to catch SyntaxErrors early.
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return f"invalid regex in pattern: {e.msg} at position {e.pos}"

    # Reject the pattern if it does not fullmatch the entire sample_line.
    m = compiled.fullmatch(sample_line)
    if m is None:
        return (
            f"sample_line does not match pattern '{pattern}'. "
            f"Match failed — pattern did not match the entire line.\n"
            f"  sample_line: {sample_line}\n"
            f"  pattern: {pattern}\n"
            f"Hint: pattern requires a full match (^...$). Adjust the pattern."
        )

    return compiled


# -------------------------------------------------------------------------------------------------------------------- #
# Hooks                                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #


def _before_drop_conditions_list(
    buffer_manager: BufferManager,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Block direct drops of the conditions list buffer."""
    return "This buffer is protected by the RegexConditionManager."


async def _before_update_condition(
    factory: RegexConditionFactory,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> bool | str:
    """Validate pattern buffer content changes.

    Allows the update if all touched patterns validate against their sample_lines.
    Blocks the update if validation fails.
    """
    # Parse the old buffer text as JSON; on initial creation use empty state.
    try:
        if old_buffer is not None:
            old_json = json.loads(old_buffer.get_text())
        else:
            old_json = {}
    except json.JSONDecodeError as e:
        return f"old text is not valid JSON: {e.msg}"

    # Parse the new buffer text as JSON.
    try:
        new_json = json.loads(new_buffer.get_text())
    except json.JSONDecodeError as e:
        return f"new text is not valid JSON: {e.msg}"

    # Find which patterns were added or changed — only those need validation.
    added = set(new_json.keys()) - set(old_json.keys())
    changed = {k for k in set(old_json.keys()) & set(new_json.keys()) if old_json[k] != new_json[k]}
    touched_patterns = added | changed

    # Validate each touched pattern against its sample_line.
    for name in sorted(touched_patterns):
        validation = _guard_pattern_value("replace", new_json.get(name))
        if isinstance(validation, str):
            return validation

    return True


async def _after_update_condition(
    factory: RegexConditionFactory,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> bool | str:
    """Synchronize the conditions list pattern_count after a pattern buffer update.

    This is a side effect -- the update already succeeded. We only keep
    the conditions list view current here. Handles both the old system:config:
    flow and the new [regex_condition] option buffer flow.
    """
    # Parse the updated buffer text as JSON to count patterns.
    try:
        new_json = json.loads(new_buffer.get_text())
    except json.JSONDecodeError:
        return True  # Best-effort bookkeeping; do not block.

    # Determine the condition name: use the option buffer config if available (new flow),
    # otherwise derive from the buffer name (old flow).
    option_name = new_buffer.name + _REGEX_CONDITION_SUFFIX
    option_buffer = factory.get_buffer(option_name)
    if option_buffer is not None:
        config = _get_option_config(option_buffer)
        if not isinstance(config, str):
            condition_name = config.get("condition_name")
        else:
            condition_name = new_buffer.name.removeprefix(f"{_REGEX_PATTERN_BUFFER}:")
    else:
        condition_name = new_buffer.name.removeprefix(f"{_REGEX_PATTERN_BUFFER}:")

    # Read the conditions list and update the pattern count.
    read_result = factory._read_buffer(_CONDITION_LIST_BUFFER)
    if not read_result.get("ok"):
        return True

    try:
        conditions_data = json.loads(read_result["content"])
    except json.JSONDecodeError:
        return True

    # Get the conditions dict from conditions_data.
    conditions = conditions_data.get("conditions", {})

    # Replace the catalog entry with updated data -- preserve origin if already set (new flow).
    existing = conditions.get(condition_name, {})
    conditions[condition_name] = {
        "name": condition_name,
        "pattern_count": len(new_json),
        "origin": existing.get("origin", new_buffer.name),
    }
    conditions_data["conditions"] = conditions

    # Write the updated conditions list back -- non-critical bookkeeping.
    try:
        await factory.create_buffer(
            _CONDITION_LIST_BUFFER,
            text=json.dumps(conditions_data, indent=2) + "\n",
            overwrite=True,
        )
    except Exception as e:
        _logger.error(f"[_after_update_condition] failed to write conditions catalog: {e}")

    return True


async def _after_create_condition(factory: RegexConditionFactory, buffer: Buffer, args: dict[str, Any]) -> bool | str:
    """Handle AFTER_CREATE_BUFFER on a pattern buffer: create function and register hooks.

    On initial creation: pattern buffer is new, create function and register hooks.
    On restart: function already exists in factory._functions, skip re-registration.
    """
    # Derive the condition name and buffer name from the pattern buffer.
    condition_name = buffer.name.removeprefix(f"{_REGEX_PATTERN_BUFFER}:")
    pattern_buffer_name = buffer.name

    # Skip on replay — the function is already registered from the initial add.
    if factory.has_function(condition_name):
        return True

    # Register the pattern management update hook on the new config buffer.
    hook_result = buffer.register_hook(
        HookPoint.BEFORE_UPDATE_BUFFER,
        "factory",
        _before_update_condition,
    )
    if not hook_result.get("ok"):
        return (
            f"condition '{condition_name}' could not be added: "
            f"pattern hook could not be registered on '{pattern_buffer_name}': "
            f"{hook_result.get('error')}"
        )

    # Create and register the condition closure so the routing table can evaluate it.
    condition_fn = _make_condition(pattern_buffer_name, factory._read_buffer, condition_name)
    func_result = await factory.create_function(
        name=condition_name,
        callable=condition_fn,
        short_description=f"matches lines against patterns in '{pattern_buffer_name}'",
        long_description=(
            f"Evaluates the incoming text against all regex patterns registered in '{pattern_buffer_name}'. "
            f"All patterns must fullmatch for this condition to return True."
        ),
    )
    if not func_result.get("ok"):
        return f"condition '{condition_name}' could not be added: {func_result.get('error')}"

    # Sync the conditions list now that the condition is created.
    return await _after_update_condition(factory, None, buffer, args)


async def _after_drop_condition(factory: RegexConditionFactory, buffer: Buffer, args: dict[str, Any]) -> bool | str:
    """Handle AFTER_DROP_BUFFER on a pattern buffer: update conditions list."""
    # Derive the condition name from the pattern buffer name.
    condition_name = buffer.name.removeprefix(f"{_REGEX_PATTERN_BUFFER}:")

    # Remove the entry from the conditions list — the pattern buffer itself was already
    # dropped by the system, and the function was already removed.
    conditions_buffer = factory.get_buffer(_CONDITION_LIST_BUFFER)

    # Guard: return early if the conditions list buffer is not available.
    if conditions_buffer is None:
        _logger.error(f"[_after_drop_condition] conditions list buffer not found")
        return True

    # Parse the conditions list JSON.
    try:
        conditions = json.loads(conditions_buffer.get_text())
    except json.JSONDecodeError:
        _logger.error(f"[_after_drop_condition] conditions list is not valid JSON")
        return True

    # Guard: only remove if the condition exists in the catalog.
    if condition_name in conditions.get("conditions", {}):
        del conditions["conditions"][condition_name]
        text = json.dumps(conditions, indent=2) + "\n"
        write_result = await factory.write_buffer(_CONDITION_LIST_BUFFER, text, pos="end")
        if not write_result.get("ok"):
            _logger.error(f"[_after_drop_condition] failed to update conditions list: {write_result.get('error')}")

    return True


# -------------------------------------------------------------------------------------------------------------------- #
# Option Buffer Flow -- [regex_condition] suffix                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #


async def _before_create_condition_option(
    factory: RegexConditionFactory,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """BEFORE_CREATE_BUFFER hook for [regex_condition] option buffers -- validate before creation.

    Checks that the config contains a condition_name, that it is not already registered,
    that the origin buffer exists, and that all patterns in the origin validate.
    Persists the condition_name in the config before allowing creation.
    """
    # Parse the config from the option buffer -- must be a dict.
    config = _get_option_config(buffer)
    if isinstance(config, str):
        return config

    # The config must supply a condition_name -- this is the source of truth for the function name.
    condition_name = config.get("condition_name")
    if not condition_name:
        return f"[_before_create_condition_option] config must supply 'condition_name'"

    # Reject if a function with this name is already registered.
    if factory.has_function(condition_name):
        return f"condition '{condition_name}' is already registered"

    # Derive the origin buffer name from the option buffer name.
    origin_name = _origin_name_from_option_buffer_name(buffer.name)
    if not origin_name:
        return f"[_before_create_condition_option] cannot derive origin from '{buffer.name}'"

    # The origin buffer must exist at creation time -- patterns live there.
    origin_buffer = factory.get_buffer(origin_name)
    if origin_buffer is None:
        return f"[_before_create_condition_option] origin buffer '{origin_name}' not found"

    # Validate the regex patterns in the origin buffer.
    try:
        origin_json = json.loads(origin_buffer.get_text())
    except json.JSONDecodeError as e:
        return f"[_before_create_condition_option] origin '{origin_name}' is not valid JSON: {e}"

    # Validate each pattern against its sample_line -- reject if any fails.
    for pat_name, pat_value in origin_json.items():
        validation = _guard_pattern_value("create", pat_value)
        if isinstance(validation, str):
            return f"[_before_create_condition_option] pattern '{pat_name}' in '{origin_name}': {validation}"

    # Persist the condition_name in the config so it survives restarts.
    config["condition_name"] = condition_name
    buffer.set_text(json.dumps(config, indent=2) + "\n")

    return None


async def _before_update_condition_option(
    factory: RegexConditionFactory,
    old_buffer: Buffer,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """BEFORE_UPDATE_BUFFER hook for [regex_condition] option buffers -- reject rename of condition_name.

    The condition_name stored in the config is immutable -- renaming would change the registered
    function and break callers. Allow the update only when the condition_name is unchanged.
    """
    # Parse the old and new configs.
    old_config = _get_option_config(old_buffer)
    if isinstance(old_config, str):
        return f"[_before_update_condition_option] could not read old config: {old_config}"

    # Get the new config from the new buffer.
    new_config = _get_option_config(new_buffer)
    if isinstance(new_config, str):
        return f"[_before_update_condition_option] could not read new config: {new_config}"

    # Retrieve the condition_name from both configs.
    old_name = old_config.get("condition_name")
    new_name = new_config.get("condition_name")

    # Reject the update when the condition_name differs from the persisted one -- no renaming allowed.
    if old_name is not None and new_name != old_name:
        return f"[_before_update_condition_option] renaming conditions is not allowed ('{old_name}' -> '{new_name}')"

    return None


async def _after_create_condition_option(
    factory: RegexConditionFactory,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """AFTER_CREATE_BUFFER hook for [regex_condition] option buffers -- register and wire hooks.

    Creates the condition function, registers it in the function manager for ref_count support,
    and registers update and drop hooks on the origin buffer.
    """
    # Parse the config from the option buffer to retrieve the condition_name.
    config = _get_option_config(buffer)
    if isinstance(config, str):
        _logger.error(f"[_after_create_condition_option] {config}")
        return None

    # Get the condition_name from the config.
    condition_name = config.get("condition_name")
    if not condition_name:
        _logger.error(f"[_after_create_condition_option] config is missing 'condition_name'")
        return None

    # Derive the origin buffer name from the option buffer name.
    origin_name = _origin_name_from_option_buffer_name(buffer.name)
    if not origin_name:
        _logger.error(f"[_after_create_condition_option] cannot derive origin from '{buffer.name}'")
        return None

    # Get the origin buffer to register hooks on it.
    origin_buffer = factory.get_buffer(origin_name)
    if origin_buffer is None:
        _logger.error(f"[_after_create_condition_option] origin buffer '{origin_name}' not found")
        return None

    # Create and register the condition function via register_function (gives ref_count support).
    condition_fn = _make_condition(origin_name, factory._read_buffer, condition_name)
    register_result = await factory.register_function(condition_fn, condition_name)
    if not register_result.get("ok"):
        _logger.error(f"[_after_create_condition_option] register failed for '{condition_name}': {register_result.get('error')}")
        return None

    # Register BEFORE_UPDATE on the origin buffer -- validates pattern changes against sample_lines.
    before_update_result = origin_buffer.register_hook(
        HookPoint.BEFORE_UPDATE_BUFFER,
        "condition_factory:origin_before_update",
        _before_update_condition,
    )
    if not before_update_result.get("ok"):
        _logger.error(f"[_after_create_condition_option] failed to register before_update on '{origin_name}': {before_update_result.get('error')}")
        return None

    # Register AFTER_UPDATE on the origin buffer -- syncs pattern_count in the conditions catalog.
    after_update_result = origin_buffer.register_hook(
        HookPoint.AFTER_UPDATE_BUFFER,
        "condition_factory:origin_after_update",
        _after_update_condition,
    )
    if not after_update_result.get("ok"):
        _logger.error(f"[_after_create_condition_option] failed to register after_update on '{origin_name}': {after_update_result.get('error')}")
        return None

    # Register BEFORE_DROP on the origin buffer -- blocks if refs are held or option buffer still exists.
    before_drop_result = origin_buffer.register_hook(
        HookPoint.BEFORE_DROP_BUFFER,
        "condition_factory:origin_before_drop",
        _before_drop_from_origin,
    )
    if not before_drop_result.get("ok"):
        _logger.error(f"[_after_create_condition_option] failed to register before_drop on '{origin_name}': {before_drop_result.get('error')}")
        return None

    # Register AFTER_DROP on the origin buffer -- unregisters the condition and removes all hooks.
    after_drop_result = origin_buffer.register_hook(
        HookPoint.AFTER_DROP_BUFFER,
        "condition_factory:origin_after_drop",
        _after_drop_from_origin,
    )
    if not after_drop_result.get("ok"):
        _logger.error(f"[_after_create_condition_option] failed to register after_drop on '{origin_name}': {after_drop_result.get('error')}")
        return None

    # Confirm successful registration.
    _logger.info(f"[>>] condition '{condition_name}' registered, hooks wired on '{origin_name}'")

    # Sync the conditions catalog -- record origin so _after_drop_from_origin can find the entry.
    conditions_buffer = factory.get_buffer(_CONDITION_LIST_BUFFER)
    if conditions_buffer is None:
        return None

    # Parse the conditions catalog JSON.
    try:
        conditions_data = json.loads(conditions_buffer.get_text())
    except json.JSONDecodeError as e:
        _logger.error(f"[_after_create_condition_option] conditions catalog is not valid JSON: {e}")
        return None

    # Extract the conditions subdirectory from the catalog.
    conditions = conditions_data.get("conditions", {})

    # Parse the pattern count from the origin buffer's JSON content.
    try:
        origin_json = json.loads(origin_buffer.get_text())
        pattern_count = len(origin_json)
    except Exception as e:
        _logger.error(f"[_after_create_condition_option] failed to parse origin buffer '{origin_name}': {e}")
        pattern_count = 0

    # Update the conditions catalog entry with the new pattern count and origin.
    conditions[condition_name] = {
        "name": condition_name,
        "pattern_count": pattern_count,
        "origin": origin_name,
    }
    conditions_data["conditions"] = conditions

    # Write the updated conditions catalog back to the buffer via create_buffer
    # so that BEFORE_UPDATE / AFTER_UPDATE hooks fire for dependent listeners.
    try:
        await factory.create_buffer(
            _CONDITION_LIST_BUFFER,
            text=json.dumps(conditions_data, indent=2) + "\n",
            overwrite=True,
        )
    except Exception as e:
        _logger.error(f"[_after_create_condition_option] failed to write conditions catalog: {e}")


async def _before_drop_condition_option(
    factory: RegexConditionFactory,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """BEFORE_DROP_BUFFER hook for [regex_condition] option buffers -- reject when refs are held.

    Blocks the drop of the option buffer while the condition function has active references.
    """
    # Parse the config from the option buffer.
    config = _get_option_config(buffer)
    if isinstance(config, str):
        return None

    # Retrieve the condition_name from the config.
    condition_name = config.get("condition_name")
    if not condition_name:
        return None

    # Reject when the function is not registered or has active callable refs.
    if not factory.has_function(condition_name):
        return None
    wrapper = factory._callables.get(condition_name)
    if wrapper is not None and wrapper._reference_count > 0:
        return f"[_before_drop_condition_option] condition '{condition_name}' is still in use (ref_count={wrapper._reference_count})"

    return None


async def _after_drop_condition_option(
    factory: RegexConditionFactory,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """AFTER_DROP_BUFFER hook for [regex_condition] option buffers -- remove hooks from origin, unregister.

    Cleans up the hooks registered on the origin buffer and unregisters the condition
    from the function manager. The origin buffer itself is left intact.
    """
    # Parse the config from the option buffer.
    config = _get_option_config(buffer)
    if isinstance(config, str):
        _logger.error(f"[_after_drop_condition_option] {config}")
        return None

    # Retrieve the condition_name from the config.
    condition_name = config.get("condition_name")
    if not condition_name:
        _logger.error(f"[_after_drop_condition_option] config is missing 'condition_name'")
        return None

    # Derive the origin buffer name from the option buffer name.
    origin_name = _origin_name_from_option_buffer_name(buffer.name)
    if not origin_name:
        _logger.error(f"[_after_drop_condition_option] cannot derive origin from '{buffer.name}'")
        return None

    # Get the origin buffer to remove the hooks from it.
    origin_buffer = factory.get_buffer(origin_name)
    if origin_buffer is not None:
        # Remove the hooks that were registered on the origin buffer at create time.
        origin_buffer.unregister_hook(HookPoint.BEFORE_UPDATE_BUFFER, "condition_factory:origin_before_update")
        origin_buffer.unregister_hook(HookPoint.AFTER_UPDATE_BUFFER, "condition_factory:origin_after_update")
        origin_buffer.unregister_hook(HookPoint.BEFORE_DROP_BUFFER, "condition_factory:origin_before_drop")
        origin_buffer.unregister_hook(HookPoint.AFTER_DROP_BUFFER, "condition_factory:origin_after_drop")
        _logger.debug(f"[_after_drop_condition_option] hooks removed from '{origin_name}'")

    # Unregister the condition from the function manager.
    unregister_result = await factory.unregister_function(condition_name)
    if not unregister_result.get("ok"):
        _logger.error(f"[_after_drop_condition_option] unregister failed for '{condition_name}': {unregister_result.get('error')}")
        return None

    # Remove the entry from the conditions catalog.
    removed = await _remove_condition_from_catalog(factory, condition_name)
    if not removed:
        _logger.warning(f"[_after_drop_condition_option] condition '{condition_name}' was not found in the catalog")

    _logger.info(f"[>>] condition '{condition_name}' unregistered after option buffer drop")


async def _before_drop_from_origin(
    factory: RegexConditionFactory,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """BEFORE_DROP_BUFFER hook on origin buffer -- reject when option buffer exists or refs are held.

    The origin buffer must not be dropped while a [regex_condition] option buffer still points to it,
    and it must not be dropped while the condition function has active references.
    """
    # Derive the option buffer name from the origin buffer name.
    option_name = buffer.name + _REGEX_CONDITION_SUFFIX

    # Check new-flow option buffer if one exists.
    option_buffer = factory.get_buffer(option_name)
    if option_buffer is None:
        _logger.error(f"[_before_drop_from_origin] option buffer not found: {option_name}")
        return None

    # Get the option buffer config.
    config = _get_option_config(option_buffer)
    if isinstance(config, str):
        _logger.error(f"[_before_drop_from_origin] option buffer config invalid: {option_name}")
        return None

    # The option buffer is valid -- check ref_count for the named condition.
    condition_name = config.get("condition_name")
    if not condition_name:
        _logger.error(f"[_before_drop_from_origin] option buffer '{option_name}' is missing 'condition_name' -- inconsistent state")
        return None

    # Block the drop when the condition has active refs.
    wrapper = factory._callables.get(condition_name)
    if wrapper is not None and wrapper._reference_count > 0:
        return f"[_before_drop_from_origin] condition '{condition_name}' is still in use (ref_count={wrapper._reference_count})"

    # Check if any old-flow condition points to this origin.
    conditions_buffer = factory.get_buffer(_CONDITION_LIST_BUFFER)
    if conditions_buffer is None:
        return None

    # Parse the conditions catalog JSON.
    try:
        conditions_data = json.loads(conditions_buffer.get_text())
    except json.JSONDecodeError as e:
        _logger.error(f"[_before_drop_from_origin] conditions catalog is not valid JSON: {e}")
        return None

    # Extract the conditions subdirectory from the catalog.
    conditions = conditions_data.get("conditions", {})

    # Scan for any old-flow condition that points to this origin and check ref_count.
    for cond_name, cond_meta in conditions.items():
        if cond_meta.get("origin") != buffer.name:
            continue
        wrapper = factory._callables.get(cond_name)
        if wrapper is not None and wrapper._reference_count > 0:
            return f"[_before_drop_from_origin] condition '{cond_name}' is still in use (ref_count={wrapper._reference_count})"

    return None


async def _after_drop_from_origin(
    factory: RegexConditionFactory,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """AFTER_DROP_BUFFER hook on origin buffer -- unregister condition and clean up.

    The origin buffer has been dropped. All hooks are already gone. Unregister the condition
    from the function manager and remove the entry from the conditions catalog.
    """
    # Derive the option buffer name from the origin buffer name.
    option_name = buffer.name + _REGEX_CONDITION_SUFFIX
    condition_name: str | None = None

    # Get the option buffer that may hold the condition_name.
    option_buffer = factory.get_buffer(option_name)
    if option_buffer is None:
        _logger.error(f"[_after_drop_from_origin] option buffer not found: {option_name}")
        return None

    # Get the option config from the option buffer.
    config = _get_option_config(option_buffer)
    if isinstance(config, str):
        _logger.error(f"[_after_drop_from_origin] option buffer config invalid: {option_name}")
        return None

    # Get the condition_name from the config.
    condition_name = config.get("condition_name")
    if not condition_name:
        _logger.error(f"[_after_drop_from_origin] condition_name not found in config")
        return None

    # Unregister the condition from the function manager.
    unregister_result = await factory.unregister_function(condition_name)
    if not unregister_result.get("ok"):
        _logger.error(f"[_after_drop_from_origin] unregister failed for '{condition_name}': {unregister_result.get('error')}")
        return None

    # Remove the entry from the conditions catalog.
    removed = await _remove_condition_from_catalog(factory, condition_name)
    if not removed:
        _logger.warning(f"[_after_drop_from_origin] condition '{condition_name}' was not found in the catalog")

    _logger.info(f"[>>] condition '{condition_name}' unregistered after origin buffer drop")
    return None


# -------------------------------------------------------------------------------------------------------------------- #
# RegexConditionFactory                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #


class RegexConditionFactory(FunctionManager, AgenticObject):
    f"""Regex-based pattern conditions. See {_REGEX_DOC_BUFFER} for details."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Seed the conditions list buffer — a reactive view, updated by AFTER_CREATE/DROP on pattern buffers.
        condition_result = self._create_buffer(
            _CONDITION_LIST_BUFFER,
            text=json.dumps({"conditions": {}}) + "\n",
        )
        if not condition_result.get("ok"):
            raise RuntimeError(f"failed to create condition list buffer: {condition_result.get('error')}")

        # Guard the conditions list against direct drops — it is a system buffer.
        conditions_buffer = self.get_buffer(_CONDITION_LIST_BUFFER)
        hook_result = conditions_buffer.register_hook(
            HookPoint.BEFORE_DROP_BUFFER,
            "before_drop_conditions_list",
            _before_drop_conditions_list,
        )
        if not hook_result.get("ok"):
            raise RuntimeError(f"failed to register drop_guard on conditions list: {hook_result.get('error')}")

        # Register hooks on pattern buffers: BEFORE_CREATE validates, AFTER_CREATE creates function,
        # BEFORE_UPDATE validates pattern changes, AFTER_UPDATE syncs pattern_count in conditions list,
        # AFTER_DROP removes from conditions list.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "before_create_condition",
            _CONDITION_BUFFER_PATTERN,
            lambda bm, buffer, args: _before_update_condition(bm, None, buffer, args),
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "after_create_condition",
            _CONDITION_BUFFER_PATTERN,
            _after_create_condition,
        )
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_UPDATE_BUFFER,
            "before_update_condition",
            _CONDITION_BUFFER_PATTERN,
            _before_update_condition,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_UPDATE_BUFFER,
            "after_update_condition",
            _CONDITION_BUFFER_PATTERN,
            _after_update_condition,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            "after_drop_condition",
            _CONDITION_BUFFER_PATTERN,
            _after_drop_condition,
        )

        # Seed the schema documentation buffer.
        self._create_buffer(_REGEX_DOC_BUFFER, text=_REGEX_DOC)

        # Validate the origin buffer and config before creating the option buffer.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "condition_option:before_create",
            _REGEX_CONDITION_OPTION_BUFFER_PATTERN,
            _before_create_condition_option,
            apply_to_existing=True,
        )

        # Reject renames of the condition_name in the option buffer config.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_UPDATE_BUFFER,
            "condition_option:before_update",
            _REGEX_CONDITION_OPTION_BUFFER_PATTERN,
            _before_update_condition_option,
            apply_to_existing=True,
        )

        # Block the option buffer drop when the condition has active refs.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_DROP_BUFFER,
            "condition_option:before_drop",
            _REGEX_CONDITION_OPTION_BUFFER_PATTERN,
            _before_drop_condition_option,
            apply_to_existing=True,
        )

        # Clean up hooks from the origin buffer and unregister the condition after the option buffer is dropped.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            "condition_option:after_drop",
            _REGEX_CONDITION_OPTION_BUFFER_PATTERN,
            _after_drop_condition_option,
            apply_to_existing=True,
        )

        # Register the condition function and wire hooks on the origin buffer after the option buffer is created.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "condition_option:after_create",
            _REGEX_CONDITION_OPTION_BUFFER_PATTERN,
            _after_create_condition_option,
            apply_to_existing=True,
        )
