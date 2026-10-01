"""Text codec hooks for buffer update hooks."""

from typing import Callable
import json

import jsonpatch

from petekit.text.buffer_manager import Buffer, UpdateHook, _resolve_line_range


def _describe_write_scenario(buf: Buffer, start: int, end: int | None) -> str:
    total = len(buf.lines)
    resolved = _resolve_line_range(total, start, end)
    if isinstance(resolved, dict):
        return "You wrote an unknown range."
    resolved_start, resolved_end = resolved
    if resolved_start >= total:
        return "You appended new text at the end of the buffer."
    if resolved_start == 0 and resolved_end >= total:
        return "You replaced the entire buffer."
    if resolved_start < resolved_end:
        boundary = buf.lines[resolved_start].data if resolved_start < total else ""
        return f"You replaced lines {resolved_start}–{resolved_end - 1} in the buffer (overwritten text started: {boundary!r})."
    return f"You inserted text at line {resolved_start}."


def make_json_codec(inner: Callable[..., bool | str]) -> UpdateHook:
    """Build an UpdateHook that validates and diffs JSON-formatted buffer content.
    The inner callable receives (buf, old_json, new_json, patch) and returns bool|str.
    """

    def codec(buf: Buffer, old_text: str | None, start: int, end: int, new_text: str | None) -> bool | str:
        # Reconstruct the complete old buffer by replacing the [start, end) slice with the original slice text.
        # This gives the codec the full buffer state for a fair comparison.
        prefix_lines = [buf.lines[i].data for i in range(0, start)]
        suffix_lines = [buf.lines[i].data for i in range(end, len(buf.lines))]
        full_old = "\n".join(prefix_lines + [old_text or ""] + suffix_lines) + "\n"

        # Validate old_text: parse as JSON; if malformed, the buffer was already in an invalid state.
        # TODO(design): decide how to handle pre-existing malformed JSON in the buffer.
        try:
            old_json = json.loads(full_old)
        except json.JSONDecodeError:
            return "old text in buffer is not valid JSON — codec cannot diff against corrupted state"

        # When new_text is None the buffer is being cleared/dropped — skip new_json parsing.
        new_json = None
        if new_text is not None:
            full_new = "\n".join(prefix_lines + [new_text] + suffix_lines) + "\n"
            try:
                new_json = json.loads(full_new)
            except json.JSONDecodeError as e:
                scenario = _describe_write_scenario(buf, start, end)
                return (
                    f"{scenario}\n"
                    f"The resulting text is not valid JSON: {e.msg} at line {e.lineno}, char {e.colno}"
                )

        # Compute the RFC 6906 JSON Patch diff between the two parsed structures.
        patch = jsonpatch.make_patch(old_json, new_json)

        # Delegate to the inner codec with the full old/new structures and the structured diff.
        return inner(buf, old_json, new_json, patch)

    return codec
