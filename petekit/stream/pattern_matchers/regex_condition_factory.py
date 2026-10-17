"""Regex-based condition factory for stream buffer routing tables."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

_logger = logging.getLogger("regex_condition_factory")

from peteos import AgenticObject
from ...function.function_manager import FunctionManager, _FUNCTION_STATUS_NAME_BASE
from ...text.buffer_manager import Buffer, BufferManager, HookPoint, _sanitize_buffer_name
from ..stream_router import RoutingConditionError, CONDITION_FN


_CONDITION_LIST_BUFFER = "system:config:regex_condition"
_REGEX_DOC_BUFFER = "doc:regex_conditions"
_REGEX_PATTERN_BUFFER = "system:config:regex_condition"
_CONDITION_BUFFER_PATTERN = r"^system:config:regex_condition:[^[]+$"
_READ_BUFFER_FN = Callable[[str], dict[str, Any]]


def _pattern_buffer_name(condition_name: str) -> str:
    """Derive the canonical buffer name for a single pattern buffer."""
    # Derive the canonical buffer name by combining the base pattern buffer prefix
    # with a sanitized version of the condition name.
    return f"{_REGEX_PATTERN_BUFFER}:{_sanitize_buffer_name(condition_name)}"



_REGEX_DOC = f"""\
## Regex Conditions

A regex condition matches incoming text against one or more regular expression
patterns. The text must fullmatch at least one pattern (^...$) for the
condition to return True.

### Create a condition

Create a pattern buffer with the name `system:config:regex_condition:<condition_name>`
and provide at least one pattern entry:

```json
{{
    "<pattern_name>": {{
        "regex": "<regular expression>",
        "sample_line": "<example text that the regex must match>"
    }}
}}
```

The `regex` and `sample_line` are required. The regex is validated against
`sample_line` immediately — if it does not fullmatch, the creation is rejected.

To add more patterns, write additional entries to the same buffer.

### Remove a condition

Drop the pattern buffer `system:config:regex_condition:<condition_name>`. The
condition is deregistered and its entry is removed from the conditions catalog.

### Requirements per pattern

- `regex` — a valid regular expression
- `sample_line` — a line of text that the regex must fullmatch exactly

The `sample_line` is used for validation only. It is not stored after creation.

### What happens automatically

- The condition is registered as a callable in the function manager
- The `pattern_count` in the conditions catalog is updated

### Conditions catalog

`system:config:regex_condition` — a catalog of all conditions:

```json
{{
    "conditions": {{
        "<condition_name>": {{
            "name": "<condition_name>",
            "pattern_count": N
        }}
    }}
}}
```

Read this buffer to discover available conditions and their pattern counts.
"""


def _before_drop_conditions_list(
    buffer_manager: BufferManager,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Block direct drops of the conditions list buffer."""
    return "This buffer is protected by the RegexConditionManager."


def _before_drop_function_meta(
    buffer_manager: BufferManager,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Block direct drops of function meta buffers."""
    return (
        f"Buffer '{buffer.name}' is managed by RegexConditionManager. "
        f"Remove entries via {_CONDITION_LIST_BUFFER} instead."
    )


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
    old_buffer: Buffer,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> bool | str:
    """Synchronize the conditions list pattern_count after a pattern buffer update.

    This is a side effect — the update already succeeded. We only keep
    the conditions list view current here.
    """
    # Parse the updated buffer text as JSON to count patterns.
    try:
        new_json = json.loads(new_buffer.get_text())
    except json.JSONDecodeError:
        return True  # Best-effort bookkeeping; do not block.

    # Read the conditions list and update the pattern count.
    read_result = factory._read_buffer(_CONDITION_LIST_BUFFER)
    if not read_result.get("ok"):
        return True

    try:
        conditions_data = json.loads(read_result["content"])
    except json.JSONDecodeError:
        return True

    # Get the conditions dict from conditions_data and derive the condition name.
    conditions = conditions_data.get("conditions", {})
    derived_name = old_buffer.name.removeprefix(f"{_REGEX_PATTERN_BUFFER}:")

    # Guard: the condition must exist in the catalog.
    if derived_name not in conditions:
        return True  # Condition not in catalog — nothing to sync.

    # Update the pattern count and write back to conditions_data.
    conditions[derived_name]["pattern_count"] = len(new_json)
    conditions_data["conditions"] = conditions

    # Write the updated conditions list back — non-critical bookkeeping.
    try:
        await factory.create_buffer(
            _CONDITION_LIST_BUFFER,
            text=json.dumps(conditions_data, indent=2) + "\n",
            overwrite=True,
        )
    except Exception:
        pass

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
    # factory._functions preserves state across restart.
    if condition_name in factory._functions:
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

    # Guard the meta buffer against direct drops — remove via the conditions list.
    meta_name = f"{_FUNCTION_STATUS_NAME_BASE}{condition_name}"

    meta_buffer = factory.get_buffer(meta_name)
    reg_result = meta_buffer.register_hook(
        HookPoint.BEFORE_DROP_BUFFER,
        "before_drop_function_meta",
        _before_drop_function_meta,
    )
    if not reg_result.get("ok"):
        return (
            f"condition '{condition_name}' could not be added: "
            f"before_drop_function_meta could not be registered on '{meta_name}': "
            f"{reg_result.get('error')}"
        )

    return True


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
