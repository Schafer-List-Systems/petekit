"""StreamBufferManager - manages rolling, append-only stream buffers on top of BufferManager."""

from __future__ import annotations

import bisect
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Coroutine

from peteos.oap.agentic_object import AgenticObject
from peteos.oap.decorators import sandbox, tool
from ..utils.text_formatters import format_dict_list_for_buffer
from ..text.buffer_manager import Buffer, BufferEntry, BufferManager


@dataclass
class StreamBufferHookError:
    timestamp: float = field(default_factory=time.time)  # when the error occurred
    error: str = ""  # exception message


@dataclass
class StreamBufferHook:
    # the async hook function, called with (entry, stream_buffer) after append
    callable_: Callable[[BufferEntry, str], Coroutine[Any, Any, None]]
    created_at: float = field(default_factory=time.time)  # registration timestamp
    fire_count: int = 0  # how many times this hook has fired
    errors: list[StreamBufferHookError] = field(default_factory=list)  # errors from fire-and-forget
    # higher = fires sooner; owner hooks use 10000+ so agent hooks (1..9999) always fire after
    priority: int = 0


@dataclass
class StreamBufferConfig:
    created_at: float = field(default_factory=time.time)  # stream creation timestamp
    hooks: dict[str, StreamBufferHook] = field(default_factory=dict)  # all hooks on this stream, keyed by name


def _fmt_ts(ts: float, round_up: bool = False) -> str:
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


class StreamBufferManager(BufferManager, AgenticObject):
    """
    - You can read buffers via read_buffer with int (line-based) or float (time-based) start/end.
      - Float start/end values trigger time-based reading on stream buffers.
      - Timestamps in skip/bucket messages are rounded to 6 decimal places.
    - The buffer "list:stream_buffers" is always up to date with all current stream buffers.
      Read it to get a JSON array of {name, created_at, hook_count} for each stream.
    """

    # DESIGN: window_size (rolling trim) is deferred — buffers grow indefinitely for now.

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.stream_buffer_configs: dict[str, StreamBufferConfig] = {}
        self._refresh_stream_buffers_buffer()

    @tool
    def create_buffer(self, name: str, text: str | None = None, overwrite: bool = False, stream: bool = False) -> dict[str, Any]:
        """Create a new named buffer, optionally populated with text.
        Pass stream=True to create a stream buffer with hook support."""
        if stream:
            if not name.startswith("stream:"):
                return {"ok": False, "error": "stream name must start with 'stream:' prefix"}
            if name in self.stream_buffer_configs and not overwrite:
                return {"ok": False, "error": f"stream '{name}' already exists. Use overwrite=True to replace it."}

        result = super().create_buffer(name, text=text, overwrite=overwrite)

        if not result.get("ok", False):
            return result

        if stream:
            self.stream_buffer_configs[name] = StreamBufferConfig()
            self._refresh_stream_buffers_buffer()
        return result

    @tool
    def drop_buffer(self, name: str) -> dict[str, Any]:
        """Delete a named buffer.
        Automatically cleans up stream metadata if it was a stream buffer."""
        is_stream = name in self.stream_buffer_configs  # detect before super() removes it

        result = super().drop_buffer(name)

        if not result.get("ok", False):
            return result

        if is_stream:
            del self.stream_buffer_configs[name]
            self._refresh_stream_buffers_buffer()
        return result

    @sandbox
    def list_stream_buffers(self) -> list[dict]:
        """Return structured records for all stream buffers: name, created_at, hook_count."""
        return [
            {
                "name": name,
                "created_at": cfg.created_at,
                "hook_count": len(cfg.hooks),
            }
            for name, cfg in self.stream_buffer_configs.items()
        ]

    def _refresh_stream_buffers_buffer(self) -> None:
        """Refresh the list:stream_buffers buffer, one JSON dict per line."""
        text = format_dict_list_for_buffer(self.list_stream_buffers())
        if "list:stream_buffers" not in self._buffers:
            self._create_buffer("list:stream_buffers", text=text)
        else:
            self.write_buffer("list:stream_buffers", text)

    def _resolve_time(self, ts: float, anchor: float) -> float:
        """Resolve a relative (negative) or absolute float timestamp to an absolute one.

        Negative values are offsets from `anchor` (e.g. last entry timestamp or now).
        Positive values (including 0.0) are absolute unix timestamps and are returned unchanged."""
        return anchor + ts if ts < 0 else ts

    @tool
    def read_buffer(self, name: str, start: int | float = 0, end: int | float | None = None, show_timestamps: bool = False, raw: bool = False) -> dict[str, Any] | str:
        """Read a range of lines from a buffer.
        Float values trigger time-based reading on stream buffers (int for line-based).
        Negative floats (e.g. -60.0) are relative to the last entry: -60.0 means '60s ago'.
        Set show_timestamps=True to prefix each line with its unix timestamp.
        Returns a dict with ok/error or ok/content on success.
        Set raw=True to get the raw string instead of a dict — errors always return dict."""
        time_based = isinstance(start, float) or isinstance(end, float)

        if time_based and name not in self.stream_buffer_configs:
            return {"ok": False, "error": f"No stream named '{name}'. Only stream buffers support time-based reading."}

        if time_based:
            buf = self._buffers[name]
            if not buf.lines:
                return {"ok": False, "error": f"Buffer '{name}' is empty."}
            anchor = buf.lines[-1].timestamp
            ts_list = [e.timestamp for e in buf.lines]

            if isinstance(start, float):
                resolved_start = self._resolve_time(start, anchor)
                start_idx = bisect.bisect_left(ts_list, resolved_start)
                if start_idx >= len(ts_list):
                    return {"ok": False, "error": f"No entries at or after {_fmt_ts(resolved_start)}."}
                start = start_idx

            if isinstance(end, float):
                resolved_end = self._resolve_time(end, anchor)
                end_idx = bisect.bisect_right(ts_list, resolved_end)
                if end_idx < 1:
                    return {"ok": False, "error": f"No entries at or before {_fmt_ts(resolved_end)}."}
                end = end_idx

            show_timestamps = True

        return super().read_buffer(name, start=start, end=end, show_timestamps=show_timestamps, raw=raw)

    @sandbox
    async def _set_stream_on_append_hook(
        self,
        stream_buffer: str,
        name: str,
        hook: Callable[[BufferEntry, str], Coroutine[Any, Any, None]] | None = None,
        priority: int = 0,
    ) -> dict[str, Any]:
        """Set or remove a named hook on a stream. Set hook to a callable to register; pass hook=None to remove the named hook. Hook is called with (entry, stream_buffer) after every append. Higher priority fires first."""
        if stream_buffer not in self.stream_buffer_configs:
            return {"ok": False, "error": f"no stream named '{stream_buffer}'", "stream_buffer": stream_buffer}
        if hook is None:
            if name in self.stream_buffer_configs[stream_buffer].hooks:
                del self.stream_buffer_configs[stream_buffer].hooks[name]
                return {"ok": True, "removed": name, "stream_buffer": stream_buffer}
            return {"ok": False, "error": f"no hook named '{name}' on '{stream_buffer}'", "stream_buffer": stream_buffer}
        self.stream_buffer_configs[stream_buffer].hooks[name] = StreamBufferHook(
            callable_=hook,
            priority=priority,
        )
        return {"ok": True, "hook_count": len(self.stream_buffer_configs[stream_buffer].hooks), "name": name, "stream_buffer": stream_buffer}

    async def _append_stream_entry(self, stream_buffer: str, data: str) -> None:
        """
        Append a data entry to the named stream. Raises KeyError if the stream doesn't exist.

        After appending, all registered hooks are fired in priority order (highest first).
        Each hook receives (entry, stream_buffer) and can perform I/O (write to process,
        send over socket, etc.). Hooks are fire-and-forget; exceptions are not propagated.

        Design: entry.seen is not set to True here. Consuming an entry (marking it seen)
        is the responsibility of whatever reads it (e.g. read_buffer).
        """
        if stream_buffer not in self.stream_buffer_configs:
            raise KeyError(f"No stream named '{stream_buffer}'. Create it with create_buffer(..., stream=True) first.")
        buf = self._buffers[stream_buffer]
        now = time.time()
        entry = BufferEntry(timestamp=now, data=data, seen=False)
        buf.lines.append(entry)
        cfg = self.stream_buffer_configs.get(stream_buffer)
        if cfg:
            ordered = sorted(cfg.hooks.values(), key=lambda h: h.priority, reverse=True)
            for sh in ordered:
                try:
                    await sh.callable_(entry, stream_buffer)
                    sh.fire_count += 1
                except Exception as e:
                    sh.errors.append(StreamBufferHookError(timestamp=time.time(), error=str(e)))
