"""StreamBufferManager - manages rolling, append-only stream buffers on top of BufferManager."""

from __future__ import annotations

import asyncio
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
    # the hook function, called with (stream_buffer, text, metadata) after each write
    callable_: Callable[[str, str, dict[str, Any]], None] | Callable[[str, str, dict[str, Any]], Coroutine[Any, Any, None]]
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
    """Some of you buffers are special: they are streams.
    - You can access stream buffers with float (time-based) start/end arguments. Naturally, writing is append-only.
    - Float start/end values trigger time-based reading on stream buffers. Example: start=-10.0 means to see the last 10 seconds.
    - Timestamps in skip/bucket messages are rounded to 6 decimal places.
    - The buffers "system:list:stream_buffers" and "system:list:stream_buffer_hooks"
      are always up to date with all current stream buffers and their hooks including metadata.
    - Hooks in "system:list:stream_buffer_hooks" are grouped by stream and fire in the order they appear (priority descending, highest first).
    """

    # DESIGN: window_size (rolling trim) is deferred — buffers grow indefinitely for now.

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.stream_buffer_configs: dict[str, StreamBufferConfig] = {}
        self._refresh_stream_buffers_buffer()
        self._refresh_stream_buffer_hooks()

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
        """Refresh the system:list:stream_buffers buffer, one JSON dict per line."""
        text = format_dict_list_for_buffer(self.list_stream_buffers())
        self.create_buffer("system:list:stream_buffers", text=text, overwrite=True)

    def _refresh_stream_buffer_hooks(self) -> None:
        """Refresh the system:list:stream_buffer_hooks buffer with all hook states."""
        records = [
            {
                "stream": stream_name,
                "name": hook_name,
                "priority": hook.priority,
                "fire_count": hook.fire_count,
                "errors": [{"ts": e.timestamp, "msg": e.error} for e in hook.errors],
            }
            for stream_name, cfg in self.stream_buffer_configs.items()
            for hook_name, hook in cfg.hooks.items()
        ]
        records.sort(key=lambda r: (r["stream"], -r["priority"]))
        text = format_dict_list_for_buffer(records)
        self.create_buffer("system:list:stream_buffer_hooks", text=text, overwrite=True)

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
        Set show_timestamps=True to prefix each line with its timestamp.
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

    @tool
    async def write_buffer(self, name: str, text: str, start: int | None = None, end: int | None = None) -> dict[str, Any]:
        """Overwrite the text in the range [start, end) of a buffer.
        To overwrite the whole buffer, pass start=0.
        To insert at line N: pass start=N and end=N.
        Omit both start and end to append after the last line.
        Write-mode for streams is append-only.
        After appending, all registered hooks are fired once per batch."""
        
        # Make sure, streams are append-only
        if name in self.stream_buffer_configs and (start is not None or end is not None):
            return {"ok": False, "error": f"Cannot write into the middle of stream '{name}'. Streams are append-only."}
        
        # write the data into the buffer
        result = super().write_buffer(name, text, start, end)
        if not result.get("ok") or not name in self.stream_buffer_configs:
            return result
        
        cfg = self.stream_buffer_configs.get(name)
        ordered = sorted(cfg.hooks.values(), key=lambda h: h.priority, reverse=True)
        for sh in ordered:
            try:
                hook_result = sh.callable_(name, text, result)
                if asyncio.iscoroutine(hook_result):
                    await hook_result
                sh.fire_count += 1
            except Exception as e:
                sh.errors.append(StreamBufferHookError(timestamp=time.time(), error=str(e)))

        # return the result from the super().write_buffer() call
        return result

    @sandbox
    def _set_stream_on_append_hook(
        self,
        stream_buffer: str,
        name: str,
        hook: Callable[[str, str, dict[str, Any]], None] | Callable[[str, str, dict[str, Any]], Coroutine[Any, Any, None]] | None = None,
        priority: int = 0,
    ) -> dict[str, Any]:
        """Set or remove a named hook on a stream. Set hook to a callable to register; omit hook to remove the named hook. The hook is called with (stream_buffer, text, metadata) after every write. Higher priority fires first. Both async and sync callables are supported."""
        if stream_buffer not in self.stream_buffer_configs:
            return {"ok": False, "error": f"no stream named '{stream_buffer}'", "stream_buffer": stream_buffer}
        if hook is None:
            if name in self.stream_buffer_configs[stream_buffer].hooks:
                del self.stream_buffer_configs[stream_buffer].hooks[name]
                self._refresh_stream_buffer_hooks()
                return {"ok": True, "removed": name, "stream_buffer": stream_buffer}
            return {"ok": False, "error": f"no hook named '{name}' on '{stream_buffer}'", "stream_buffer": stream_buffer}
        self.stream_buffer_configs[stream_buffer].hooks[name] = StreamBufferHook(
            callable_=hook,
            priority=priority,
        )
        self._refresh_stream_buffer_hooks()
        return {"ok": True, "hook_count": len(self.stream_buffer_configs[stream_buffer].hooks), "name": name, "stream_buffer": stream_buffer}


