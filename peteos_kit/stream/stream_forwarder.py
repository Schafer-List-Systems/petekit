"""Simple stream forwarding action provider for stream buffer rules."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Coroutine

from peteos import AgenticObject, sandbox

from ..stream_buffer_manager import BufferEntry, Rule, StreamBufferManager


class StreamForwarder(StreamBufferManager, AgenticObject):
    """
    You forward matching stream entries to other streams. Use this to build
    the action side of a rule, then combine it with a condition from another
    observer (e.g. RegexStreamObserver).
    """

    @sandbox(name="make_stream_forward_action")
    def _make_stream_forward_action(
        self,
        target_stream: str,
    ) -> Callable[[BufferEntry, str, dict], Coroutine[Any, Any, None]]:
        """
        Return an action that forwards the matching BufferEntry to the target stream.

        Action signature matches Rule.action: Callable[[BufferEntry, str, dict], Coroutine].
        - Appends entry.data to target_stream via _append_stream_entry
        """
        async def action(entry: BufferEntry, rule_name: str, signal: dict) -> None:
            await self._append_stream_entry(target_stream, entry.data)
        return action
