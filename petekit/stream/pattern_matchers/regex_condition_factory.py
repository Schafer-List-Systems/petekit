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

_REGEX_DOC = """\
## Schema

system:list:regex_conditions  — catalog of all condition descriptors (JSON list)
    [{"name": "...", "buffer": "regex:...", "patterns": N}, ...]

regex:<condition_name>        — per-condition pattern store (JSON)
    {"name": "<name>", "patterns": [{"pattern": "<regex>", "sample_line": "..."}]}

## Usage

A condition evaluates to True when the incoming text matches ANY of its patterns
using fullmatch semantics — the entire line must match (^...$).

Update the system:list:regex_conditions buffer to add an empty RegEx condition:
  {"name": "<name>", "config_buffer": "regex:<name>"}

The factory automatically provisions the regex:<name> buffer on successful addition.

Update the regex:<name> buffer to add/update a pattern to a RegEx condition:
  {"pattern": "<regex>", "sample_line": "<example>"}
  The pattern is validated against sample_line immediately — a mismatch is rejected.
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


def _list_conditions(registry: dict[str, RegexCondition]) -> list[dict]:
    # Serialize the condition registry into discoverable summary records for the agent.
    return [
        {
            "name": name,
            "buffer": _condition_buffer_name(name),
            "patterns": len(cond._patterns),
        }
        for name, cond in registry.items()
    ]


def _refresh_conditions_buffer(factory, registry: dict) -> dict[str, Any]:
    # Persist the condition registry summary into the system:list:regex_conditions buffer.
    return factory.create_buffer(
        _CONDITION_LIST_BUFFER,
        text=format_dict_list_for_buffer(_list_conditions(registry)),
        overwrite=True,
    )


def _refresh_condition_buffer(factory, registry: dict, condition_name: str) -> dict[str, Any]:
    # Guard: reject unknown condition names to keep the agent-facing view consistent.
    cond = registry.get(condition_name)
    if cond is None:
        return {"ok": False, "error": f"condition '{condition_name}' not found"}
    
    # Serialize the RegexCondition state into JSON for agent inspection.
    import json
    patterns = [
        {"id": p.id, "pattern": p.pattern.pattern, "sample_line": p.sample_line}
        for p in cond._patterns.values()
    ]
    text = json.dumps({"name": condition_name, "patterns": patterns}, indent=2)
    return factory.create_buffer(_condition_buffer_name(condition_name), text=text, overwrite=True)


def _parse_condition_index(path: str) -> int:
    # Extract the integer index from a condition-level patch path like "/conditions/0".
    if not path.startswith("/conditions/"):
        raise ValueError(f"path '{path}' does not reference a condition")
    idx_str = path[len("/conditions/"):].split("/")[0]
    return int(idx_str)


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
        for pat in data.get("patterns", []):
            compiled = re.compile(pat.get("pattern", ""))
            if compiled.fullmatch(text):
                return True
        return False
    return condition


def _guard_pattern_value(op_type: str, value) -> re.Pattern | str:
    # Validate that value is a dict with 'pattern' and 'sample_line'; compile and validate the pattern.
    # Returns the compiled re.Pattern on success, or an error string on any failure.
    if not isinstance(value, dict) or "pattern" not in value:
        return f"pattern {op_type} must supply a dict with 'pattern' key, got: {type(value).__name__} — see doc:regex"
    if "sample_line" not in value:
        return f"pattern {op_type} must supply 'sample_line' — the pattern cannot be validated without a sample — see doc:regex"
    validation = _validate_pattern_against_sample(value["pattern"], value["sample_line"])
    if isinstance(validation, str):
        return f"{validation} — see doc:regex"
    return validation


def _validate_pattern_against_sample(pattern: str, sample_line: str) -> re.Pattern | str:
    # Fail-fast: compile the pattern first so a SyntaxError is caught before matching.
    try:
        compiled = re.compile(pattern)
    except re.error as e:
        return f"invalid regex in pattern: {e.msg} at position {e.pos}"

    # Guard: the pattern must fullmatch the entire sample_line — partial matches are rejected.
    m = compiled.fullmatch(sample_line)
    if m is None:
        return (
            f"sample_line does not match pattern '{pattern}'. "
            f"Match failed — pattern did not match the entire line.\n"
            f"  sample_line: {sample_line}\n"
            f"  pattern: {pattern}\n"
            f"Hint: pattern requires a full match (^...$). Adjust the pattern or sample_line."
        )

    return compiled


def _pattern_update_hook(factory, buf, old_json: dict, new_json: dict, patch) -> bool | str:
    # Walk each patch operation and dispatch pattern add/remove/replace within a condition's config buffer.
    # Only /patterns patches are accepted. The buffer name encodes the condition name.
    try:
        for op in patch:
            op_type = op.get("op")
            path = op.get("path", "")
            value = op.get("value")

            # Guard: reject any patch targeting a non-pattern path.
            if not path.startswith("/patterns"):
                return f"unsupported patch path: {path} — only /patterns/* is allowed"

            # Dispatch by operation type.
            if op_type == "add":
                validation = _guard_pattern_value("add", value)
                if isinstance(validation, str):
                    return validation

            elif op_type == "replace":
                validation = _guard_pattern_value("replace", value)
                if isinstance(validation, str):
                    return validation

            elif op_type == "remove":
                # Removals are always safe — the list preserves valid patterns.
                continue

            else:
                return f"unsupported patch operation: {op_type} — see doc:regex"

    except Exception as e:
        return str(e)

    return True


async def _factory_update_hook(factory, buf, old_json: dict, new_json: dict, patch) -> bool | str:
    # Walk each patch operation and dispatch to its handler.
    # A malformed path or out-of-range index rejects the entire batch in one exception.
    try:
        for op in patch:
            op_type = op.get("op")
            path = op.get("path", "")
            value = op.get("value")

            if op_type == "add":
                idx = _parse_condition_index(path)
                result = await factory._handle_add(idx, value)

            elif op_type == "replace":
                return "conditions list entries are immutable — drop and re-add to replace"

            elif op_type == "remove":
                idx = _parse_condition_index(path)
                result = await factory._handle_remove(idx)

            else:
                result = f"unsupported patch operation: {op_type}"

            if result is not True:
                return result

    except Exception as e:
        return str(e)

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
            text=json.dumps({"conditions": []}) + "\n",
        )
        if not cond_result.get("ok"):
            raise RuntimeError(f"failed to create condition list buffer: {cond_result.get('error')}")

        # Register the factory's update hook so add/remove operations on conditions are intercepted.
        hook_result = self.register_buffer_update_hook(
            _CONDITION_LIST_BUFFER,
            "factory",
            make_json_codec(
                lambda buf, old_json, new_json, patch: _factory_update_hook(
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
            return f"conditions list entry must be a dict with keys 'name' (string) and 'config_buffer' (string), got: {type(value).__name__}"
        name = value.get("name")
        config_buffer = value.get("config_buffer")
        if not name or not config_buffer:
            return f"conditions list entries must be objects with 'name' and 'config_buffer' keys. Got: {value}"

        # Provision the config buffer with an empty patterns list.
        import json
        init_content = json.dumps({"name": name, "patterns": []}) + "\n"
        create_result = await self.create_buffer(config_buffer, text=init_content)
        if not create_result.get("ok"):
            return f"condition '{name}' could not be created due to a cascading error when creating its config buffer: {create_result.get('error', 'unknown error when creating its config buffer')}"

        # Register the pattern management update hook on the new config buffer.
        hook_result = self.register_buffer_update_hook(
            config_buffer,
            "factory",
            make_json_codec(lambda buf, old_json, new_json, patch: _pattern_update_hook(self, buf, old_json, new_json, patch)),
        )
        if not hook_result.get("ok"):
            return f"condition '{name}' could not be added: pattern hook could not be registered on '{config_buffer}': {hook_result.get('error')}"

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

        if idx < 0 or idx >= len(conditions_list):
            return f"condition index {idx} is out of range (list has {len(conditions_list)} entries)"
        entry = conditions_list[idx]
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

    #@sandbox(name="make_regex_condition")
    #def _make_regex_condition(self, stream_buffer: str, rule_name: str) -> Callable[[BufferEntry, Buffer], dict | None]:
    #    """Return a condition that matches buffer entries against the pattern group for the given stream and rule."""
    #    def condition(entry: BufferEntry, buf: Buffer) -> dict | None:
    #        key = (stream_buffer, rule_name)
    #        group = self._regex_groups.get(key, {})
    #        for pattern_id, regex_pattern in group.items():
    #            # fullmatch = entire line must match (^...$ semantics)
    #            if regex_pattern.pattern.fullmatch(entry.data):
    #                return {"matched": rule_name, "pattern_id": pattern_id}
    #        return None
    #    return condition

    #@tool
    #def add_regex_pattern(
    #    self,
    #    stream_buffer: str,
    #    rule_name: str,
    #    pattern: str,
    #    sample_line: str,
    #) -> str:
    #    """
    #    Add a regex pattern to an existing rule. The pattern must fully match the
    #    sample_line. Returns a confirmation or a detailed error if the regex is
    #    invalid or the sample_line does not match. If this is the first pattern
    #    for the rule, the condition is auto-attached.
    #    """
    #    try:
    #        compiled = re.compile(pattern)
    #    except re.error as e:
    #        return f"Error: invalid regex '{pattern}' at position {e.pos}: {e.msg}"
    #    m = compiled.match(sample_line)
    #    if m is None or m.end() != len(sample_line):
    #        pos = m.end() if m is not None else 0
    #        marker = " " * pos + "^"
    #        return (
    #            f"Error: sample_line does not match pattern '{pattern}'. "
    #            f"Match failed at position {pos}.\n"
    #            f"  {sample_line}\n"
    #            f"  {marker}\n"
    #            f"Hint: pattern requires a full match (^...$). Adjust the pattern or sample_line."
    #        )
    #    key = (stream_buffer, rule_name)
    #    was_empty = key not in self._regex_groups or not self._regex_groups[key]
    #    if was_empty:
    #        try:
    #            self._get_stream_rule(stream_buffer, rule_name)
    #        except KeyError:
    #            return (
    #                f"Error: rule '{rule_name}' not found on stream '{stream_buffer}'. "
    #                f"Create it first with _register_stream_rule(..., action=...)."
    #            )
    #        self._regex_groups[key] = {}
    #    pattern_id = self._next_id
    #    self._next_id += 1
    #    self._regex_groups[key][pattern_id] = RegexPattern(
    #        id=pattern_id,
    #        pattern=compiled,
    #        sample_line=sample_line,
    #    )
    #    if was_empty:
    #        rule = self._get_stream_rule(stream_buffer, rule_name)
    #        rule.condition = self._make_regex_condition(stream_buffer, rule_name)
    #        return f"Condition activated on rule '{rule_name}'."
    #    return f"Pattern {pattern_id} added to rule '{rule_name}'."

    #@tool
    #def remove_regex_pattern(
    #    self,
    #    stream_buffer: str,
    #    rule_name: str,
    #    pattern_id: int,
    #) -> str:
    #    """
    #    Remove a pattern from a rule by its ID. If this was the last pattern,
    #    the condition is auto-cleared from the rule.
    #    """
    #    key = (stream_buffer, rule_name)
    #    if key not in self._regex_groups or pattern_id not in self._regex_groups[key]:
    #        return f"Error: pattern {pattern_id} not found in group for rule '{rule_name}' on stream '{stream_buffer}'."
    #    del self._regex_groups[key][pattern_id]
    #    if not self._regex_groups[key]:
    #        del self._regex_groups[key]
    #        rule = self._get_stream_rule(stream_buffer, rule_name)
    #        rule.condition = None
    #        return f"Condition cleared from rule '{rule_name}' (no patterns left)."
    #    return f"Pattern {pattern_id} removed from rule '{rule_name}'."

    #@tool
    #def list_regex_patterns(
    #    self,
    #    stream_buffer: str,
    #    rule_name: str,
    #) -> list[dict]:
    #    """List all patterns for a rule with their IDs, patterns, and sample lines. Returns an empty list if none exist."""
    #    key = (stream_buffer, rule_name)
    #    group = self._regex_groups.get(key, {})
    #    if not group:
    #        return []
    #    return [
    #        {
    #            "id": rp.id,
    #            "pattern": rp.pattern.pattern,
    #            "sample_line": rp.sample_line,
    #        }
    #        for rp in group.values()
    #    ]


