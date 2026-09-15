"""Regex-based condition provider for stream buffer rules."""

from __future__ import annotations

import re
from dataclasses import dataclass

from peteos import tool, sandbox

from .stream_buffer_manager import BufferEntry, Buffer, Rule, StreamBufferManager
from .stream_observer import StreamObserver


PatternId: type = int


@dataclass
class RegexPattern:
    """
    A single regex pattern with auto-incrementing ID and optional sample line.

    - id: auto-incrementing integer, unique within the group
    - pattern: compiled re.Pattern (fullmatch semantics)
    - sample_line: the line that originally triggered creation of this pattern
      (useful for the agent to understand what the pattern is supposed to match)
    """
    id: PatternId
    pattern: re.Pattern
    sample_line: str


class RegexStreamObserver(StreamObserver):
    """
    Provides regex-based conditions for StreamBufferManager rules.

    Responsibility (separation of concerns):
    - OWNS ONLY the condition logic (pattern matching)
    - DOES NOT own actions — those belong to StreamBufferManager/StreamObserver

    Internal state:
    - _regex_groups: dict[(stream_buffer, rule_name), dict[pattern_id, RegexPattern]]
      Pattern groups keyed by (stream_buffer, rule_name). A group is created on
      first add_regex_pattern call.
    - _next_id: PatternId
      Global auto-incrementing ID counter across all groups. IDs are unique
      across groups, not just within a group.

    How it works:
    - Each group maps pattern IDs to (compiled_pattern, sample_line)
    - The condition (closure over stream_buffer, rule_name) iterates the group's
      patterns and returns {"matched": rule_name, "pattern_id": id} on first
      full match, else None
    - Rule existence is a precondition — rules are NOT auto-created by this class
    - Condition is auto-attached to the rule when the first pattern is added
    - Condition is auto-cleared from the rule when the last pattern is removed
    """

    def __init__(self) -> None:
        super().__init__()
        # Keyed by (stream_buffer, rule_name) -> dict[pattern_id, RegexPattern]
        self._regex_groups: dict[tuple[str, str], dict[PatternId, RegexPattern]] = {}
        # Global auto-incrementing ID counter starting at 1
        self._next_id: PatternId = 1

    @sandbox(name="make_regex_condition")
    def _make_regex_condition(self, stream_buffer: str, rule_name: str) -> Callable[[BufferEntry, Buffer], dict | None]:
        """Return a condition that matches buffer entries against the pattern group for the given stream and rule."""
        def condition(entry: BufferEntry, buf: Buffer) -> dict | None:
            key = (stream_buffer, rule_name)
            group = self._regex_groups.get(key, {})
            for pattern_id, regex_pattern in group.items():
                # fullmatch = entire line must match (^...$ semantics)
                if regex_pattern.pattern.fullmatch(entry.data):
                    return {"matched": rule_name, "pattern_id": pattern_id}
            return None
        return condition

    @tool
    def add_regex_pattern(
        self,
        stream_buffer: str,
        rule_name: str,
        pattern: str,
        sample_line: str,
    ) -> str:
        """
        Add a regex pattern to a group's pattern list for an existing rule.

        - Rule must already exist in StreamBufferManager (precondition)
        - The pattern must be a valid regex — compiled on add, error if invalid
        - sample_line is required — the pattern must fully match it, error if not
        - Auto-increments a global ID for this pattern across all groups
        - The sample_line is stored as metadata (why this pattern was added)
        - If the group was empty before this add: condition is auto-attached to the
          rule, hint returned to agent ("Condition activated on rule '{rule_name}'")
        - If the group was non-empty: silent add, no hint returned
        """
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            return f"Error: invalid regex '{pattern}' at position {e.pos}: {e.msg}"
        m = compiled.match(sample_line)
        if m is None or m.end() != len(sample_line):
            pos = m.end() if m is not None else 0
            marker = " " * pos + "^"
            return (
                f"Error: sample_line does not match pattern '{pattern}'. "
                f"Match failed at position {pos}.\n"
                f"  {sample_line}\n"
                f"  {marker}\n"
                f"Hint: pattern requires a full match (^...$). Adjust the pattern or sample_line."
            )
        key = (stream_buffer, rule_name)
        was_empty = key not in self._regex_groups or not self._regex_groups[key]
        if was_empty:
            try:
                self._get_stream_rule(stream_buffer, rule_name)
            except KeyError:
                return (
                    f"Error: rule '{rule_name}' not found on stream '{stream_buffer}'. "
                    f"Create it first with _register_stream_rule(..., action=...)."
                )
            self._regex_groups[key] = {}
        pattern_id = self._next_id
        self._next_id += 1
        self._regex_groups[key][pattern_id] = RegexPattern(
            id=pattern_id,
            pattern=compiled,
            sample_line=sample_line,
        )
        if was_empty:
            rule = self._get_stream_rule(stream_buffer, rule_name)
            rule.condition = self._make_regex_condition(stream_buffer, rule_name)
            return f"Condition activated on rule '{rule_name}'."
        return f"Pattern {pattern_id} added to rule '{rule_name}'."

    @tool
    def remove_regex_pattern(
        self,
        stream_buffer: str,
        rule_name: str,
        pattern_id: int,
    ) -> str:
        """
        Remove a pattern from a group by its integer ID.

        - Returns error if the group or pattern_id does not exist
        - If removing leaves the group non-empty: silent remove, no hint
        - If removing leaves the group empty: condition is auto-cleared from the rule,
          hint returned to agent ("Condition cleared from rule '{rule_name}' (no patterns left)")
        """
        key = (stream_buffer, rule_name)
        if key not in self._regex_groups or pattern_id not in self._regex_groups[key]:
            return f"Error: pattern {pattern_id} not found in group for rule '{rule_name}' on stream '{stream_buffer}'."
        del self._regex_groups[key][pattern_id]
        if not self._regex_groups[key]:
            del self._regex_groups[key]
            rule = self._get_stream_rule(stream_buffer, rule_name)
            rule.condition = None
            return f"Condition cleared from rule '{rule_name}' (no patterns left)."
        return f"Pattern {pattern_id} removed from rule '{rule_name}'."

    @tool
    def list_regex_patterns(
        self,
        stream_buffer: str,
        rule_name: str,
    ) -> list[dict]:
        """
        List all patterns in a group with their IDs, pattern strings, and sample lines.

        Returns:
        - Empty list if the group does not exist
        - list of {id: int, pattern: str, sample_line: str | None}
        """
        key = (stream_buffer, rule_name)
        group = self._regex_groups.get(key, {})
        if not group:
            return []
        return [
            {
                "id": rp.id,
                "pattern": rp.pattern.pattern,
                "sample_line": rp.sample_line,
            }
            for rp in group.values()
        ]


