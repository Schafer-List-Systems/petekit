"""StreamObserver - fallback trigger for unexpected stream data via agentic reasoning."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from peteos import AgenticObject, tool
from peteos.conversation import ContentPart, Message
from peteos.engine import Runner

from .stream_buffer_manager import BufferEntry, Rule, StreamBufferManager


@dataclass
class _FallbackConfig:
    """Per-stream fallback configuration."""
    batch_size: int = 1
    interval_secs: float | None = None
    invoke_on_empty: bool = False
    observing_runner: Runner | None = None
    _timer: threading.Timer | None = field(default=None, init=False, repr=False)


class StreamObserver(StreamBufferManager, AgenticObject):
    """
    You can observe a stream to get regular updates in time intervals or upon a number of (unexpected) messages.
    - Use `observe_steam` to observe an existing stream or change the configuration of observation.
    - Use `deobserve_stream` when you no longer want to be informed about unexpected messages in a stream.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stream_fallback_config: dict[str, _FallbackConfig] = {}

    def _start_timer(self, stream: str) -> None:
        """Start or restart the periodic fallback timer for a stream. Raises KeyError if not configured."""
        if stream not in self._stream_fallback_config:
            raise KeyError(f"No stream '{stream}' configured for fallback.")
        self._stop_timer(stream)
        cfg = self._stream_fallback_config[stream]
        interval = cfg.interval_secs
        if interval is None or interval <= 0:
            return

        def _timer_tick() -> None:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self._fallback_action(None, stream, {})))

        cfg._timer = threading.Timer(interval, _timer_tick)
        cfg._timer.daemon = True
        cfg._timer.start()

    def _stop_timer(self, stream: str) -> None:
        """Stop the periodic fallback timer for a stream. Raises KeyError if not configured."""
        if stream not in self._stream_fallback_config:
            raise KeyError(f"No stream '{stream}' configured for fallback.")
        cfg = self._stream_fallback_config[stream]
        if cfg._timer is not None:
            cfg._timer.cancel()
            cfg._timer = None

    async def _fallback_action(self, entry: BufferEntry | None, stream: str, signal: dict) -> None:
        """Fallback action — enqueue notification for the agent about unexpected entries.

        Called by StreamBufferManager when the fallback condition passes (entry is provided)
        or by the timer directly (entry is None). Re-arms the timer after firing if configured.
        Skips invocation if no unseen entries and invoke_on_empty is False.

        The signal dict is ignored for the fallback (fallback has no user-provided metadata)."""
        cfg = self._stream_fallback_config.get(stream)
        if cfg is None:
            return
        buf = self._buffers.get(stream)
        if buf is None:
            return
        unseen = [e for e in buf.lines if not e.seen]
        if not unseen and not cfg.invoke_on_empty:
            return
        now = time.time()
        if unseen:
            ts_start = unseen[0].timestamp
            ts_end = unseen[-1].timestamp
            prompt = (
                f"Stream '{stream}' has {len(unseen)} unexpected entries "
                f"between {self._fmt_ts(ts_start)} and {self._fmt_ts(ts_end, round_up=True)}. "
                f"Current time: {self._fmt_ts(now, round_up=True)}."
            )
        else:
            last_ts = buf.lines[-1].timestamp if buf.lines else 0.0
            prompt = (
                f"Stream '{stream}' has 0 unexpected entries. "
                f"Current time: {self._fmt_ts(now, round_up=True)}. "
                f"Last entry at {self._fmt_ts(last_ts, round_up=True)}. "
                f"Nothing new for {self._fmt_ts(now - last_ts)} seconds."
            )
        if cfg.observing_runner:
            msg = Message.create(
                role="user",
                content_parts=[ContentPart.create_text(prompt)]
            )
            await cfg.observing_runner.queue_message(msg)
        else:
            # TODO: possible deadlock when runner is None (timer path) — investigate passing runner through _timer_tick
            await self.invoke_agent(prompt)
        if cfg.interval_secs is not None:
            self._start_timer(stream)

    @tool
    def observe_stream(
        self,
        stream: str,
        batch_size: int = 1,
        interval_secs: float | None = None,
        invoke_on_empty: bool = False,
        runner: Runner | None = None,
    ) -> str:
        """Observe an existing "stream:" buffer for unexpected entries."""
        if stream not in self._buffers:
            return f"Stream '{stream}' does not exist."
        if stream not in self._stream_fallback_config:
            cfg = _FallbackConfig()
            self._stream_fallback_config[stream] = cfg
        else:
            cfg = self._stream_fallback_config[stream]
        cfg.batch_size = max(1, batch_size)
        cfg.invoke_on_empty = invoke_on_empty
        if interval_secs is not None:
            cfg.interval_secs = max(0.0, interval_secs)
        else:
            cfg.interval_secs = None
            self._stop_timer(stream)
        cfg.observing_runner = runner

        def condition(entry: BufferEntry, buf) -> None | dict:
            unseen = [e for e in buf.lines if not e.seen]
            if len(unseen) >= cfg.batch_size:
                return {"unseen_count": len(unseen), "batch_size": cfg.batch_size}
            return None

        self._exchange_stream_fallback_rule(
            stream,
            Rule(condition=condition, action=self._fallback_action),
        )

        if cfg.interval_secs is not None and cfg.interval_secs > 0:
            self._start_timer(stream)
        return f"Now observing stream '{stream}'. Notified every {cfg.batch_size} new entries."

    @tool
    def deobserve_stream(self, stream: str) -> str:
        """Stop observing a stream."""
        if stream not in self._stream_fallback_config:
            return f"Stream '{stream}' is not being observed."
        self._stop_timer(stream)
        del self._stream_fallback_config[stream]
        self._exchange_stream_fallback_rule(stream, None)
        return f"Stopped observing '{stream}'. The stream is still running."
