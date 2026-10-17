"""Text codec hooks for buffer update hooks."""

from __future__ import annotations

import asyncio
from typing import Callable
import json

import jsonpatch

from petekit.text.buffer_manager import Buffer, BufferManager, BeforeUpdateBuffer, AfterUpdateBuffer
from petekit.text.buffer import _resolve_line_range


def _describe_write_scenario(buffer: Buffer, start: int, end: int | None) -> str:
    total = len(buffer.lines)
    resolved = _resolve_line_range(total, start, end)
    if isinstance(resolved, dict):
        return "You wrote an unknown range."
    resolved_start, resolved_end = resolved
    if resolved_start >= total:
        return "You appended new text at the end of the buffer."
    if resolved_start == 0 and resolved_end >= total:
        return "You replaced the entire buffer."
    if resolved_start < resolved_end:
        boundary = buffer.lines[resolved_start].data if resolved_start < total else ""
        return f"You replaced lines {resolved_start}–{resolved_end - 1} in the buffer (overwritten text started with: {boundary!r})."
    return f"You inserted text at line {resolved_start}."


def make_json_codec(inner: Callable[..., bool | str]) -> BeforeUpdateBuffer:
    """Build an UpdateHook that validates and diffs JSON-formatted buffer content.
    The inner callable receives (buf, old_json, new_json, patch) and returns bool|str.
    Supports both sync and async inner callables.
    """

    async def codec(
        buffer_manager: BufferManager,
        old_buffer: Buffer,
        new_buffer: Buffer,
        args: dict[str, Any],
    ) -> bool | str | None:
        # Extract the mutation range and text from args.
        start = args.get("start", 0)
        end = args.get("end", len(old_buffer.lines))
        old_text = old_buffer.get_text()
        new_text = new_buffer.get_text()

        # Validate old_text — reject if the buffer is not valid JSON.
        try:
            old_json = json.loads(old_text)
        except json.JSONDecodeError:
            return "old text in buffer is not valid JSON — codec cannot diff against corrupted state"

        # Parse new_text into JSON — skip if new_text is absent (drop sentinel path).
        new_json = None
        if new_text is not None:
            try:
                new_json = json.loads(new_text)
            except json.JSONDecodeError as e:
                # Report the write scenario and the parse error for the agent to act on.
                scenario = _describe_write_scenario(old_buffer, start, end)
                return (
                    f"{scenario}\n"
                    f"The resulting text is not valid JSON: {e.msg} at line {e.lineno}, char {e.colno}"
                )

        # Compute the RFC 6906 JSON Patch diff between the two parsed structures.
        patch = jsonpatch.make_patch(old_json, new_json)

        # Delegate to the inner codec — await if the inner is async.
        result = inner(old_buffer, old_json, new_json, patch)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    return codec
