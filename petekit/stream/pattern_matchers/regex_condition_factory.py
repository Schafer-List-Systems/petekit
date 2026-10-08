"""Regex-based condition factory for stream buffer routing tables."""

from __future__ import annotations

import re
from typing import Any

from peteos import AgenticObject
from ...function.function_manager import FunctionManager
from ...utils.text_codecs import make_json_codec


_CONDITION_LIST_BUFFER = "system:list:regex_conditions"
_REGEX_DOC_BUFFER = "doc:regex_conditions"


def _diff_keys(old: dict, new: dict) -> tuple[set[str], set[str], set[str]]:
    # Compute which keys were added, removed, or changed between two dictionaries.
    removed = set(old.keys()) - set(new.keys())
    added = set(new.keys()) - set(old.keys())
    changed = {k for k in set(old.keys()) & set(new.keys()) if old[k] != new[k]}
    return (added, removed, changed)

_REGEX_DOC = """\
## Schema

system:list:regex_conditions  — catalog of all condition descriptors (JSON)
    {
        "conditions": {
            "<condition_name>": {
                "name": "<condition_name>",
                "pattern_count": N
            }
        }
    }

function:<condition_name>:regex  — per-condition pattern dictionary (JSON)
    {
        "<pattern_name>": {
            "regex": "<regex>",
            "sample_line": "..."
        }
    }

## Usage

A condition evaluates to True when the incoming text matches ANY of its patterns
using fullmatch semantics — the entire line must match (^...$).

To add a condition, add an entry to the conditions dictionary in the conditions list buffer:
    "conditions": {"<condition_name>": {"name": "<condition_name>"}}
- The pattern buffer function:<condition_name>:regex is created automatically.

To remove a condition, remove its entry from the conditions list buffer.

To add or update a pattern in the pattern buffer function:<condition_name>:regex, write an entry:
    "<pattern_name>": {"regex": "<regex>", "sample_line": "<example>"}
    The regex is validated against sample_line immediately — a mismatch is rejected.

To remove a pattern, remove its entry from the pattern buffer.
"""


def _pattern_buffer_name(condition_name: str) -> str:
    # Derive the canonical buffer name for a single pattern buffer.
    return f"function:{condition_name}:regex"


def _drop_guard(buf, old_text: str | None, start: int, end: int, new_text: str | None) -> str | None:
    """Block direct buffer drops. Remove entries via system:list:regex_conditions instead."""
    if new_text is None:
        return f"This buffer cannot be dropped. Remove entries via {_CONDITION_LIST_BUFFER} instead."
    return None


def _condition_closure(pattern_buffer: str, read_buffer_fn: callable) -> callable:
    # Build a closure that reads the named config buffer and fullmatches text against all patterns.
    def condition(stream: str, text: str, metadata: dict) -> bool:
        import json
        read_result = read_buffer_fn(pattern_buffer)
        if not read_result.get("ok"):
            return False
        try:
            data = json.loads(read_result["content"])
        except json.JSONDecodeError:
            return False

        # Iterate over pattern entries — data is the patterns dict directly.
        for pat in data.values():
            compiled = re.compile(pat.get("regex", ""))
            if compiled.fullmatch(text):
                return True

        return False

    return condition


def _guard_pattern_value(op_type: str, value) -> re.Pattern | str:
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
    # Compile the pattern first to catch SyntaxErrors early.
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return f"invalid regex in pattern: {e.msg} at position {e.pos}"

    # Reject the pattern if it does not fullmatch the entire sample_line.
    m = compiled.fullmatch(sample_line)
    if m is None:
        # Return a clear error so the agent knows what went wrong.
        return (
            f"sample_line does not match pattern '{pattern}'. "
            f"Match failed — pattern did not match the entire line.\n"
            f"  sample_line: {sample_line}\n"
            f"  pattern: {pattern}\n"
            f"Hint: pattern requires a full match (^...$). Adjust the pattern."
        )

    return compiled


async def _pattern_update_hook(factory, buf, old_json: dict, new_json: dict, patch) -> bool | str:
    # Guard: new_json is None when drop_buffer fires hooks before deleting the buffer.
    # Block direct drops — the factory must remove conditions via the conditions list buffer.
    if new_json is None:
        return (
            "Cannot delete this buffer directly. "
            "Remove the condition from system:list:regex_conditions instead."
        )

    # Compute the diff between old and new patterns to find which are added, removed, or changed.
    added, removed, changed = _diff_keys(old_json, new_json)

    # Collect all touched pattern names — skipping validation for removals.
    touched_patterns = added | changed

    # Validate each touched pattern against its sample_line in the new state.
    for name in sorted(touched_patterns):
        # Reject the patch if the regex does not match its sample_line.
        validation = _guard_pattern_value("replace", new_json.get(name))
        if isinstance(validation, str):
            return validation

    # Refresh the conditions listing buffer so that pattern counts stay current.
    # To that, we read the buffer as the source of truth.
    import json
    read_result = factory._read_buffer(_CONDITION_LIST_BUFFER)
    if not read_result.get("ok"):
        return f"conditions buffer could not be read: {read_result.get('error')} — the conditions buffer '{_CONDITION_LIST_BUFFER}' may have been deleted or is inaccessible"
    try:
        data = json.loads(read_result["content"])
    except json.JSONDecodeError as e:
        return f"conditions buffer is not valid JSON: {e.msg} — expected a JSON object with a 'conditions' key in '{_CONDITION_LIST_BUFFER}'"

    # Derive the condition name from the buffer name and update its pattern count.
    conditions = data.get("conditions", {})
    derived_name = buf.name.removeprefix("function:").removesuffix(":regex")
    if derived_name not in conditions:
        raise RuntimeError(f"condition '{derived_name}' not found in conditions list — internal state corruption")
    conditions[derived_name]["pattern_count"] = len(new_json)

    # Write the updated conditions list back.
    data["conditions"] = conditions
    try:
        await factory.create_buffer(
            _CONDITION_LIST_BUFFER,
            text=json.dumps(data, indent=2) + "\n",
            overwrite=True,
        )
    except Exception:
        pass  # best-effort: non-critical bookkeeping update

    return True


async def _condition_list_update_hook(factory, buf, old_json: dict, new_json: dict, patch) -> bool | str:
    # Compute the diff between old and new conditions to find which are added, removed, or changed.
    added, removed, changed = _diff_keys(
        old_json.get("conditions", {}),
        new_json.get("conditions", {}),
    )

    # Collect all touched condition names and track their intent.
    touched_conditions: set[str] = added | removed | changed

    # Determine whether each touched condition is an add or a remove by consulting the new state.
    # Dispatch add or remove accordingly so the final state drives the decision.
    conditions = new_json.get("conditions", {})
    for name in sorted(touched_conditions):
        # Add if present in new state
        if name in added:
            # Validate the entry shape.
            entry = conditions[name]
            if not isinstance(entry, dict):
                return f"condition '{name}' must be a dict, got: {type(entry).__name__}"

            # Derive the pattern buffer name from the condition name — always deterministic.
            pattern_buffer = _pattern_buffer_name(name)

            # Overwrite is safe — replaces content in place and preserves existing hooks.
            import json
            init_content = json.dumps({"patterns": {}}, indent=2)
            create_result = await factory.create_buffer(pattern_buffer, text=init_content, overwrite=True)
            if not create_result.get("ok"):
                return (
                    f"condition '{name}' could not be created due to a cascading error "
                    f"when creating its config buffer: "
                    f"{create_result.get('error', 'unknown error when creating its config buffer')}"
                )

            # Only register hooks and functions on a genuine first add — replays find
            # them already present and skip safely.
            if name not in factory._functions:
                # Guard the pattern buffer against direct drops — remove via the catalog.
                guard_result = factory.register_buffer_update_hook(
                    pattern_buffer,
                    "drop_guard",
                    _drop_guard,
                )
                if not guard_result.get("ok"):
                    return (
                        f"condition '{name}' could not be added: "
                        f"drop guard could not be registered on '{pattern_buffer}': "
                        f"{guard_result.get('error')}"
                    )

                # Register the pattern management update hook on the new config buffer.
                hook_result = factory.register_buffer_update_hook(
                    pattern_buffer,
                    "factory",
                    make_json_codec(
                        lambda buf, old_json, new_json, patch: _pattern_update_hook(
                            factory, buf, old_json, new_json, patch
                        )
                    ),
                )
                if not hook_result.get("ok"):
                    return (
                        f"condition '{name}' could not be added: "
                        f"pattern hook could not be registered on '{pattern_buffer}': "
                        f"{hook_result.get('error')}"
                    )

                # Create and register the condition closure so the routing table can evaluate it.
                condition_fn = _condition_closure(pattern_buffer, factory._read_buffer)
                func_result = await factory.create_function(
                    name=name,
                    callable=condition_fn,
                    short_description=f"matches lines against patterns in '{pattern_buffer}'",
                    long_description=(
                        f"Evaluates the incoming text against all regex patterns registered in '{pattern_buffer}'. "
                        f"All patterns must fullmatch for this condition to return True."
                    ),
                )
                if not func_result.get("ok"):
                    return f"condition '{name}' could not be added: {func_result.get('error')}"

                # Guard the meta buffer against direct drops — remove via the conditions list.
                meta_name = f"function:{name}:meta"

                def _meta_drop_guard(buf, old_text, start, end, new_text):
                    if new_text is None:
                        return (
                            f"Buffer '{meta_name}' is managed by RegexConditionManager. "
                            "Remove entries via system:list:regex_conditions instead."
                        )
                    return None

                reg_result = factory.register_buffer_update_hook(
                    meta_name,
                    "regex_condition_meta_drop_guard",
                    _meta_drop_guard,
                )
                if not reg_result.get("ok"):
                    return (
                        f"condition '{name}' could not be added: "
                        f"meta drop guard could not be registered on '{meta_name}': "
                        f"{reg_result.get('error')}"
                    )

        # Remove — the condition is absent from the new state.
        elif name in removed:
            # Derive the pattern buffer name from the condition name — always deterministic.
            pattern_buffer_name = _pattern_buffer_name(name)
            meta_name = f"function:{name}:meta"

            # Unregister our drop guard — we are authorized to drop, the sentinel must not block us.
            factory.unregister_buffer_update_hook(pattern_buffer_name, "drop_guard")
            factory.unregister_buffer_update_hook(meta_name, "regex_condition_meta_drop_guard")

            # Drop the function and its config buffer — the condition is being removed.
            drop_fn_result = await factory.drop_function(name)
            if not drop_fn_result.get("ok"):
                return f"condition '{name}' could not be removed: {drop_fn_result.get('error')}"
            drop_buf_result = await factory.drop_buffer(pattern_buffer_name)
            if not drop_buf_result.get("ok"):
                return f"condition '{name}' could not be removed: {drop_buf_result.get('error')}"

    return True


class RegexConditionFactory(FunctionManager, AgenticObject):
    f"""Regex-based pattern conditions. See {_REGEX_DOC_BUFFER} for details."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Seed the registry buffer with an empty conditions list.
        import json
        cond_result = self._create_buffer(
            _CONDITION_LIST_BUFFER,
            text=json.dumps({"conditions": {}}) + "\n",
        )
        if not cond_result.get("ok"):
            raise RuntimeError(f"failed to create condition list buffer: {cond_result.get('error')}")

        # Guard the conditions list against direct drops — the factory must remove entries
        # via the catalog; agents must not drop the catalog buffer.
        def _conditions_drop_guard(buf, old_text, start, end, new_text):
            if new_text is None:
                return "This buffer is protected by the RegexConditionManager."
            return None

        hook_result = self.register_buffer_update_hook(
            _CONDITION_LIST_BUFFER,
            "drop_guard",
            _conditions_drop_guard,
        )
        if not hook_result.get("ok"):
            raise RuntimeError(f"failed to register drop_guard on conditions list: {hook_result.get('error')}")

        # Register the factory's update hook so add/remove operations on conditions are intercepted.
        hook_result = self.register_buffer_update_hook(
            _CONDITION_LIST_BUFFER,
            "factory",
            make_json_codec(
                lambda buf, old_json, new_json, patch: _condition_list_update_hook(
                    self, buf, old_json, new_json, patch
                )
            ),
        )
        if not hook_result.get("ok"):
            raise RuntimeError(f"failed to register factory hook: {hook_result.get('error')}")

        # Seed the schema documentation buffer.
        self._create_buffer(_REGEX_DOC_BUFFER, text=_REGEX_DOC)
