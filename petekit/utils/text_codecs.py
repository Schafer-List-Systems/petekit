"""Text codec hooks for buffer update hooks."""

from typing import Callable
import json

import jsonpatch

from petekit.text.buffer_manager import Buffer, UpdateHook


def make_json_codec(inner: Callable[..., bool | str]) -> UpdateHook:
    """Build an UpdateHook that validates and diffs JSON-formatted buffer content.
    The inner callable receives (buf, old_json, new_json, patch) and returns bool|str.
    """

    def codec(buf: Buffer, old_text: str | None, start: int, end: int, new_text: str | None) -> bool | str:
        # Reconstruct the complete old buffer by replacing the [start, end) slice with the original slice text.
        # This gives the codec the full buffer state for a fair comparison.
        prefix_lines = [buf.lines[i].data for i in range(0, start)]
        suffix_lines = [buf.lines[i].data for i in range(end, len(buf.lines))]
        full_old = "\n".join(prefix_lines + [old_text] + suffix_lines) + "\n"

        # Reconstruct the complete new buffer by splicing the new_text into the same position.
        full_new = "\n".join(prefix_lines + [new_text] + suffix_lines) + "\n"

        # Validate new_text: parse as JSON; reject with line/char error hints on failure.
        try:
            new_json = json.loads(full_new)
        except json.JSONDecodeError as e:
            return f"new text is not valid JSON: {e.msg} at line {e.lineno}, char {e.colno}"

        # Validate old_text: parse as JSON; if malformed, the buffer was already in an invalid state.
        # TODO(design): decide how to handle pre-existing malformed JSON in the buffer.
        try:
            old_json = json.loads(full_old)
        except json.JSONDecodeError:
            return "old text in buffer is not valid JSON — codec cannot diff against corrupted state"

        # Compute the RFC 6906 JSON Patch diff between the two parsed structures.
        patch = jsonpatch.make_patch(old_json, new_json)

        # Delegate to the inner codec with the full old/new structures and the structured diff.
        return inner(buf, old_json, new_json, patch)

    return codec
