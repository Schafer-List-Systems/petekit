"""Regex-based condition factory for stream buffer routing tables."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from peteos import AgenticObject, sandbox
from ...function.function_manager import FunctionManager
from ...utils.text_codecs import make_json_codec
from ...utils.text_formatters import format_dict_list_for_buffer


_CONDITION_LIST_BUFFER = "system:list:regex_conditions"
_REGEX_DOC_BUFFER = "doc:regex"


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
                "buffer": "regex:<condition_name>",
                "patterns": N
            }
        }
    }

regex:<condition_name>        — per-condition pattern dictionary (JSON)
    {
        "name": "<name>",
        "patterns": {
            "<pattern_name>": {
                "regex": "<regex>",
                "sample_line": "..."
            }
        }
    }

## Usage

A condition evaluates to True when the incoming text matches ANY of its patterns
using fullmatch semantics — the entire line must match (^...$).

To add a condition, add an entry to the conditions dictionary in the condition buffer:
    "conditions": {"<condition_name>": {"buffer": "regex:<condition_name>"}}
- The buffer name defaults to regex:<condition_name> if omitted — an empty dict auto-provisions it.
- The regex:<condition_name> buffer is created automatically!

To remove a condition, remove its entry from the conditions dictionary.

To add or update a pattern in the pattern dictionary of regex:<condition_name>, write an entry:
    "<pattern_name>": {"regex": "<regex>", "sample_line": "<example>"}
    The regex is validated against sample_line immediately — a mismatch is rejected.

To remove a pattern, remove its entry from the pattern dictionary.
"""

PatternId: type = int


@dataclass
class RegexPattern:
    """
    A single regex pattern associated with the exact line it was created from.

    Internal:
    - id: auto-incrementing integer, unique within the group
    - pattern: compiled re.Pattern (fullmatch semantics)
    - sample_line: the line that originally triggered creation of this pattern
    """
    id: PatternId
    pattern: re.Pattern
    sample_line: str


class RegexCondition:
    """A named collection of regex patterns. All patterns must match for the table to fire."""

    def __init__(self) -> None:
        self._patterns: dict[str, RegexPattern] = {}


def _condition_buffer_name(condition_name: str) -> str:
    # Derive the canonical buffer name for a single regex condition.
    return f"regex:{condition_name}"


def _condition_closure(config_buffer: str, read_buffer_fn: callable) -> callable:
    # Build a closure that reads the named config buffer and fullmatches text against all patterns.
    def condition(stream: str, text: str, metadata: dict) -> bool:
        import json
        read_result = read_buffer_fn(config_buffer)
        if not read_result.get("ok"):
            return False
        try:
            data = json.loads(read_result["content"])
        except json.JSONDecodeError:
            return False

        # Iterate over pattern entry values — patterns is now a dict keyed by pattern name.
        for pat in data.get("patterns", {}).values():
            compiled = re.compile(pat.get("regex", ""))
            if compiled.fullmatch(text):
                return True

        return False

    return condition


def _guard_pattern_value(op_type: str, value) -> re.Pattern | str:
    # Reject non-dict values and values missing the required 'regex' key.
    if not isinstance(value, dict) or "regex" not in value:
        return f"regex {op_type} must supply a dict with 'regex' key, got: {type(value).__name__} — see doc:regex"

    # Reject entries missing 'sample_line' — validation requires a sample to test against.
    if "sample_line" not in value:
        return f"regex {op_type} must supply 'sample_line' — the regex cannot be validated without a sample — see doc:regex"

    # Validate the regex against its sample_line; propagate validation errors upward.
    validation = _validate_pattern_against_sample(value["regex"], value["sample_line"])
    if isinstance(validation, str):
        return f"{validation} — see doc:regex"

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


def _pattern_update_hook(factory, buf, old_json: dict, new_json: dict, patch) -> bool | str:
    # Compute the diff between old and new patterns to find which are added, removed, or changed.
    added, removed, changed = _diff_keys(
        old_json.get("patterns", {}),
        new_json.get("patterns", {}),
    )

    # Collect all touched pattern names - skipping validation for removals
    touched_patterns = added | changed

    # Validate each touched pattern against its sample_line in the new state.
    patterns = new_json.get("patterns", {})
    for name in sorted(touched_patterns):
        pattern_entry = patterns.get(name)

        # Reject the patch if the regex does not match its sample_line.
        validation = _guard_pattern_value("replace", pattern_entry)
        if isinstance(validation, str):
            return validation

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
            # Resolve the config buffer name from the entry, defaulting to regex:{name} if omitted.
            entry = conditions[name]
            if not isinstance(entry, dict):
                return f"condition '{name}' must be a dict, got: {type(entry).__name__}"
            config_buffer = entry.get("buffer")
            if not config_buffer:
                config_buffer = _condition_buffer_name(name)

            # Provision the config buffer with an empty patterns list.
            import json
            init_content = json.dumps({"name": name, "patterns": {}}) + "\n"
            create_result = await factory.create_buffer(config_buffer, text=init_content)
            if not create_result.get("ok"):
                return (
                    f"condition '{name}' could not be created due to a cascading error "
                    f"when creating its config buffer: "
                    f"{create_result.get('error', 'unknown error when creating its config buffer')}"
                )

            # Register the pattern management update hook on the new config buffer.
            hook_result = factory.register_buffer_update_hook(
                config_buffer,
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
                    f"pattern hook could not be registered on '{config_buffer}': "
                    f"{hook_result.get('error')}"
                )

            # Create and register the condition closure so the routing table can evaluate it by name.
            condition_fn = _condition_closure(config_buffer, factory._read_buffer)
            func_result = await factory.create_function(
                name=name,
                callable=condition_fn,
                short_description=f"matches lines against patterns in '{config_buffer}'",
                long_description=(
                    f"Evaluates the incoming text against all regex patterns registered in '{config_buffer}'. "
                    f"All patterns must fullmatch for this condition to return True."
                ),
            )
            if not func_result.get("ok"):
                return f"condition '{name}' could not be added: {func_result.get('error')}"

        # Remove — the condition is absent from the new state.
        elif name in removed:
            # Remove — the condition is absent from the new state.
            # Resolve the actual buffer name from the old state in case a custom buffer was used.
            old_entry = old_json.get("conditions", {}).get(name, {})
            buf_name = old_entry.get("buffer") or _condition_buffer_name(name)

            # Drop the function and its config buffer — the condition is being removed.
            drop_fn_result = await factory.drop_function(name)
            if not drop_fn_result.get("ok"):
                return f"condition '{name}' could not be removed: {drop_fn_result.get('error')}"
            drop_buf_result = await factory.drop_buffer(buf_name)
            if not drop_buf_result.get("ok"):
                return f"condition '{name}' could not be removed: {drop_buf_result.get('error')}"

    return True


class RegexConditionFactory(FunctionManager, AgenticObject):
    """Regex-based pattern conditions. See doc:regex for details."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._conditions: dict[str, RegexCondition] = {}

        # Seed the registry buffer with an empty conditions list.
        import json
        cond_result = self._create_buffer(
            _CONDITION_LIST_BUFFER,
            text=json.dumps({"conditions": {}}) + "\n",
        )
        if not cond_result.get("ok"):
            raise RuntimeError(f"failed to create condition list buffer: {cond_result.get('error')}")

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

    async def _handle_add(self, idx: int, value) -> bool | str:
        # Validate the incoming value is a dict with name and config_buffer keys.
        # Reject early if the shape does not match what the factory needs to provision the condition.
        if not isinstance(value, dict):
            return (
                f"conditions list entry must be a dict "
                f"with keys 'name' (string) and 'config_buffer' (string), "
                f"got: {type(value).__name__}"
            )
        name = value.get("name")
        config_buffer = value.get("config_buffer")
        if not name or not config_buffer:
            return (
                f"conditions list entries must be objects "
                f"with 'name' and 'config_buffer' keys. "
                f"Got: {value}"
            )

        # Provision the config buffer with an empty patterns list.
        import json
        init_content = json.dumps({"name": name, "patterns": {}}) + "\n"
        create_result = await self.create_buffer(config_buffer, text=init_content)
        if not create_result.get("ok"):
            return (
                f"condition '{name}' could not be created due to a cascading error "
                f"when creating its config buffer: "
                f"{create_result.get('error', 'unknown error when creating its config buffer')}"
            )

        # Register the pattern management update hook on the new config buffer.
        hook_result = self.register_buffer_update_hook(
            config_buffer,
            "factory",
            make_json_codec(
                lambda buf, old_json, new_json, patch: _pattern_update_hook(
                    self, buf, old_json, new_json, patch
                )
            ),
        )
        if not hook_result.get("ok"):
            return (
                f"condition '{name}' could not be added: "
                f"pattern hook could not be registered on '{config_buffer}': "
                f"{hook_result.get('error')}"
            )

        # Create and register the condition closure so the routing table can evaluate it by name.
        condition_fn = _condition_closure(config_buffer, self._read_buffer)
        func_result = await self.create_function(
            name=name,
            callable=condition_fn,
            short_description=f"matches lines against patterns in '{config_buffer}'",
            long_description=(
                f"Evaluates the incoming text against all regex patterns registered in '{config_buffer}'. "
                f"All patterns must fullmatch for this condition to return True."
            ),
        )
        if not func_result.get("ok"):
            return f"condition '{name}' could not be added: {func_result.get('error')}"

        return True

    async def _handle_remove(self, idx: int) -> bool | str:
        # Read the conditions list buffer to find the condition at idx.
        import json
        read_result = self._read_buffer(_CONDITION_LIST_BUFFER)
        if not read_result.get("ok"):
            return f"could not read conditions list: {read_result.get('error')}"
        try:
            conditions_list = json.loads(read_result["content"])
        except json.JSONDecodeError as e:
            return f"conditions list is not valid JSON: {e.msg}"

        conditions = conditions_list.get("conditions", [])
        if idx < 0 or idx >= len(conditions):
            return f"condition index {idx} is out of range (list has {len(conditions)} entries)"
        entry = conditions[idx]
        name = entry.get("name")
        config_buffer = entry.get("buffer")
        if not name or not config_buffer:
            return f"condition entry at index {idx} is malformed: {entry}"

        drop_fn_result = await self.drop_function(name)
        if not drop_fn_result.get("ok"):
            return f"condition '{name}' could not be removed: {drop_fn_result.get('error')}"

        drop_buf_result = await self.drop_buffer(config_buffer)
        if not drop_buf_result.get("ok"):
            return f"condition '{name}' could not be removed: {drop_buf_result.get('error')}"

        return True
