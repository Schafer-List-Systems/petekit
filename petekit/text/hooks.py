"""Predefined buffer hooks for common validation and transformation tasks."""

from __future__ import annotations

import json
import logging
from typing import Any

from .buffer import Buffer
from .buffer_manager import BufferManager
from .buffer import BeforeUpdateBuffer


def is_json(
    buffer_manager: BufferManager,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> bool | str | None:
    """Reject the update if the buffer content is not valid JSON.

    The new buffer text is parsed as JSON. If parsing fails, the update is
    rejected with an error message describing the parse failure.
    Returns None (allow) if the text is valid JSON.
    """
    # Extract the buffer text for JSON validation.
    text = new_buffer.get_text()

    # Parse the text as JSON — reject with details on parse failure.
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        return (
            f"buffer content is not valid JSON: "
            f"line {e.lineno}, char {e.colno}: {e.msg}"
        )

    # The text is valid JSON — allow the update.
    return None


def has_trailing_newline(
    buffer_manager: BufferManager,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> bool | str | None:
    """Reject the update if the buffer does not end with a trailing newline.

    A buffer ends with a trailing newline when its last entry is an empty
    string (BufferEntry with data=""). If the last line has content, the
    buffer is considered incomplete and the update is rejected.
    Returns None (allow) if the last entry is empty, i.e. the buffer has
    a trailing newline.
    """
    # Guard: reject if the buffer has no lines — cannot determine trailing newline.
    if not new_buffer.lines:
        return "buffer has no lines — cannot determine trailing newline"

    # Check whether the last entry is the empty in-progress line.
    last_entry = new_buffer.lines[-1]
    if last_entry.data != "":
        return (
            f"buffer does not end with a trailing newline "
            f"(last line {len(new_buffer.lines) - 1!r} has content: {last_entry.data!r}). "
            f"Append a newline before updating."
        )

    # The last entry is empty — buffer ends with newline, allow the update.
    return None
