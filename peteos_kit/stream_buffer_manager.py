"""StreamBufferManager - manages rolling, append-only stream buffers on top of BufferManager."""

from __future__ import annotations

import bisect
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Coroutine

from peteos.oap.agentic_object import AgenticObject
from peteos.oap.decorators import tool
from .buffer_manager import Buffer, BufferEntry, BufferManager


@dataclass
class Rule:
    """A rule with a condition and an action, keyed by name in StreamBufferRules.rules.

    The condition returns None (no match) or a dict (matched, dict is signal metadata).
    Returning {} means matched with no metadata. Returning {"key": "val"} means matched
    with rich metadata that the action can use (e.g. matched text, entity key, etc.).

    The action is responsible for marking the entry as consumed (seen=True) once
    it has processed it. This ensures the entry is excluded from the unseen_count
    and the router only surfaces unconsumed entries to the agent for reasoning."""
    condition: Callable[["BufferEntry", "Buffer"], None | dict]
    action: Callable[["BufferEntry", str, dict], Coroutine[Any, Any, None]] | None = None


@dataclass
class StreamBufferRules:
    """Holds rules (keyed by name) and an optional fallback rule for a stream.

    Content lives in BufferManager._buffers (stream: prefix).

    Rule evaluation: each rule is checked in order; its action fires if condition
    is True. The fallback rule fires only when no other rule matched — this is the
    'unexpected data' path that surfaces to the agent for reasoning."""
    rules: dict[str, Rule] = field(default_factory=dict)
    fallback: Rule | None = None


class StreamBufferManager(BufferManager, AgenticObject):
    """
    You additionally manage named streams, stored as buffers with the 'stream:'
    prefix (e.g. 'stream:auth'). Streams are continuously appended to as data
    arrives. You can read, search, and otherwise interact with stream buffers
    using the buffer tools — they behave like any other named buffer.

    Timestamps shown in read_stream_buffer hints are rounded to 6 decimals.
    """

    # DESIGN: window_size (rolling trim) is deferred — buffers grow indefinitely for now.

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stream_buffer_rules: dict[str, StreamBufferRules] = {}
        self._stream_fallback_config: dict = {}

    def _create_stream(self, stream_buffer: str) -> StreamBufferRules:
        """Create a named stream. Raises KeyError if it already exists or name lacks 'stream:' prefix."""
        if not stream_buffer.startswith("stream:"):
            raise KeyError(f"Stream name '{stream_buffer}' must start with 'stream:' prefix.")
        if stream_buffer in self._stream_buffer_rules:
            raise KeyError(f"Stream '{stream_buffer}' already exists.")
        super().create_buffer(stream_buffer)
        rules = StreamBufferRules(rules={}, fallback=None)
        self._stream_buffer_rules[stream_buffer] = rules
        return rules

    def _drop_stream(self, stream_buffer: str) -> None:
        """Drop a stream and all its entries. Raises KeyError if it doesn't exist."""
        if stream_buffer not in self._stream_buffer_rules:
            raise KeyError(f"No stream named '{stream_buffer}'.")
        super().drop_buffer(stream_buffer)
        del self._stream_buffer_rules[stream_buffer]

    def _list_streams(self) -> dict[str, int]:
        """List all streams with their unseen counts."""
        return {
            name: sum(1 for e in self._buffers[name].lines if not e.seen)
            for name in self._stream_buffer_rules
        }

    def _get_stream_rules(self, stream_buffer: str) -> StreamBufferRules:
        """Get the StreamBufferRules for a stream. Raises KeyError if it doesn't exist."""
        if stream_buffer not in self._stream_buffer_rules:
            raise KeyError(f"No stream named '{stream_buffer}'.")
        return self._stream_buffer_rules[stream_buffer]

    def _timestamp_range_to_lines(self, buf: Buffer, start_time: float, end_time: float) -> tuple[int, int]:
        """Convert [start_time, end_time] to 1-based line range using binary search. Returns (lo, hi) or (-1, -1) if no entries match."""
        timestamps = [entry.timestamp for entry in buf.lines]
        lo = bisect.bisect_left(timestamps, start_time)
        hi = bisect.bisect_right(timestamps, end_time)
        if lo >= hi:
            return (-1, -1)
        return (lo + 1, hi)  # 1-based

    def _fmt_ts(self, ts: float, round_up: bool = False) -> str:
        """Format timestamp for agent-facing hints — up to 6 decimal places, trailing zeros stripped.

        Use round_up=True for upper bounds (end of range) to avoid excluding entries
        due to precision loss when the 7th+ digit rounds up. Use round_up=False (default)
        for lower bounds (start of range)."""
        s = f"{ts:.6f}"
        rounded = float(s)
        if round_up:
            if rounded < ts:
                ts = rounded + 0.000001
        else:
            if rounded > ts:
                ts = rounded - 0.000001
        return f"{ts:.6f}".rstrip("0").rstrip(".") or "0"

    @tool
    def read_stream_buffer(self, stream_buffer: str, start_time: float, end_time: float) -> str:
        """Read from a buffer within a time range [start_time, end_time] instead of using line numbers."""
        if stream_buffer not in self._stream_buffer_rules:
            raise KeyError(f"No stream named '{stream_buffer}'.")
        buf = self._buffers[stream_buffer]
        lo, hi = self._timestamp_range_to_lines(buf, start_time, end_time)
        if lo == -1:
            return f"No entries in '{stream_buffer}' between {start_time} and {end_time}."
        result = self._read_buffer(stream_buffer, start=lo, end=hi, show_timestamps=True)
        if result.kind == "content":
            return result.content  # type: ignore
        if result.kind == "error":
            return result.error  # type: ignore
        if result.kind == "skip":
            ts_start = buf.lines[result.line_range[0] - 1].timestamp
            ts_end = buf.lines[result.line_range[1] - 1].timestamp
            skip_msgs = []
            for line_idx, char_count in result.skip_lines:  # type: ignore
                ts = buf.lines[line_idx - 1].timestamp
                skip_msgs.append(f"{self._fmt_ts(ts)}({char_count} chars)")
            skip_lines = result.skip_lines  # type: ignore
            return (
                f"Range [{self._fmt_ts(ts_start)}–{self._fmt_ts(ts_end, round_up=True)}] is {result.total_chars} chars ({result.line_count} entries). "
                f"Heaviest lines ({len(skip_lines)} totaling {sum(c for _, c in skip_lines)} chars): {', '.join(skip_msgs)}. "
                f"Consider re-reading with a narrower time range to avoid them."
            )
        # kind == "bucket"
        ts_start = buf.lines[result.line_range[0] - 1].timestamp
        ts_end = buf.lines[result.line_range[1] - 1].timestamp
        result_bucket = result.bucket_info  # type: ignore
        bucket_msgs = []
        for b_start, b_end, b_chars in result_bucket:
            ts_b_start = buf.lines[b_start - 1].timestamp
            ts_b_end = buf.lines[b_end - 1].timestamp
            bucket_msgs.append(f"{self._fmt_ts(ts_b_start)}-{self._fmt_ts(ts_b_end, round_up=True)}:{b_chars}")
        return (
            f"Range [{self._fmt_ts(ts_start)}–{self._fmt_ts(ts_end, round_up=True)}] is {result.total_chars} chars ({result.line_count} entries) and exceeds the {BufferManager._MAX_CHUNK_CHARS}-char threshold. "
            f"Reduce the read time range to stay under the limit. "
            f"Bucket distribution (start-end:chars): {', '.join(bucket_msgs)}."
        )

    def _register_stream_rule(self, stream_buffer: str, name: str, rule: Rule) -> None:
        """Register a rule by name on a stream. Raises KeyError if the stream doesn't exist or name is already registered."""
        if stream_buffer not in self._stream_buffer_rules:
            raise KeyError(f"No stream named '{stream_buffer}'. Create it with _create_stream first.")
        br = self._stream_buffer_rules[stream_buffer]
        if name in br.rules:
            raise KeyError(f"Rule '{name}' already registered on '{stream_buffer}'.")
        br.rules[name] = rule

    def _exchange_stream_fallback_rule(self, stream_buffer: str, fallback: Rule) -> Rule | None:
        """Exchange the fallback rule on a stream. Returns the previous fallback (or None)."""
        if stream_buffer not in self._stream_buffer_rules:
            raise KeyError(f"No stream named '{stream_buffer}'. Create it with _create_stream first.")
        previous = self._stream_buffer_rules[stream_buffer].fallback
        self._stream_buffer_rules[stream_buffer].fallback = fallback
        return previous

    async def _append_stream_entry(self, stream_buffer: str, data: str) -> None:
        """
        Append a data entry to the named stream. Raises KeyError if the stream doesn't exist.

        Rule evaluation: each rule is checked in order; its action fires if condition is True.
        Multiple rules may fire simultaneously if multiple conditions are True.
        The fallback rule fires only when no other rule matched.

        The fallback is the 'unexpected data' path — entries that didn't match any known
        pattern are surfaced to the agent for reasoning.

        Design: entry.seen is not set to True here. The rule's action is responsible for
        marking the entry as consumed (e.g. via read_buffer or directly). This keeps the
        seen/unseen semantics aligned with "has an intelligent consumer processed this."
        Only entries with seen=False accumulate toward the unseen_count batch threshold.
        """
        if stream_buffer not in self._stream_buffer_rules:
            raise KeyError(f"No stream named '{stream_buffer}'. Create it with _create_stream first.")
        br = self._stream_buffer_rules[stream_buffer]
        buf = self._buffers[stream_buffer]
        now = time.time()
        entry = BufferEntry(timestamp=now, data=data, seen=False)
        buf.lines.append(entry)
        any_matched = False
        for name, rule in br.rules.items():
            signal = rule.condition(entry, buf)
            if signal is not None:
                any_matched = True
                if rule.action is not None:
                    await rule.action(entry, name, signal)
        if not any_matched and br.fallback is not None:
            if br.fallback.action is not None:
                await br.fallback.action(entry, stream_buffer, {})
