"""StreamBufferManager - extends OptionBufferManager with time-based reading.

A buffer becomes a stream when an option buffer named "BUFFER[stream]" exists.
Time-based reading (read_buffer with float start/end) is available on such buffers.

Writing to a stream is append-only — enforced by a before_update hook that checks
new_text.startswith(old_text). The update is rejected if the content check fails.
"""

from __future__ import annotations

import bisect
import logging
from typing import Any

from ..text.buffer_manager import (
    Buffer,
    BufferManager,
    HookPoint,
)
from ..text.option_buffer_manager import (
    OptionBufferManager,
    _origin_name_from_option,
    _option_name_from_option,
)
from peteos.oap.agentic_object import AgenticObject

_logger = logging.getLogger("stream_buffer_manager")

_STREAM_SUFFIX = "stream"
_STREAM_APPEND_ONLY_HOOK = "stream_buffer_manager.append_only"
_STREAM_DOC_BUFFER = "doc:stream_buffers"

_STREAM_DOC = """\
## Stream Buffers

A buffer becomes a stream when an option buffer "BUFFER[stream]" is created.
Streams support time-based reading and enforce append-only writes.

### Option Buffer

Create an option buffer "BUFFER[stream]" (no JSON content needed). This makes
the buffer a stream — it then supports time-based reading and enforces that
writes can only append (not replace or remove existing content).

### Time-Based Reading

Read a time range using float start and end values:
`read_buffer(name, start=-60.0, end=-10.0)` reads from 60 seconds ago to 10 seconds ago.
Negative floats are relative to the last entry. Positive floats are absolute Unix timestamps.

### Append-Only Enforcement

The "BUFFER[stream]" option buffer registers a pre-update hook that rejects any write
that removes or replaces existing content. The update must be a pure append —
new text must start with the old text.
"""


def _fmt_ts(ts: float, round_up: bool = False) -> str:
    """Format timestamp for agent-facing hints — up to 6 decimal places, trailing zeros stripped."""
    # Round to 6 decimal places and strip trailing zeros.
    s = f"{ts:.6f}"
    rounded = float(s)

    # Compensate for the rounding direction to avoid excluding entries
    # due to precision loss when the 7th digit rounds up.
    if round_up:
        if rounded < ts:
            ts = rounded + 0.000001
    else:
        if rounded > ts:
            ts = rounded - 0.000001

    # Strip trailing zeros and periods from the formatted string.
    result = f"{ts:.6f}".rstrip("0").rstrip(".") or "0"
    return result


def _resolve_time(timestamp: float, anchor: float) -> float:
    """Resolve a relative (negative) or absolute float timestamp to an absolute one."""
    # Negative values are offsets from anchor; positive values are returned unchanged.
    resolved = anchor + timestamp if timestamp < 0 else timestamp
    return resolved


def _time_ref_to_line(buf: Buffer, ts_ref: float, anchor: float, mode: str = "start") -> int:
    """Convert a float timestamp reference to a line index.

    Raises:
        ValueError: If the buffer has no entries or the timestamp is out of range.
    """
    # Empty buffers have no time anchors — reject immediately.
    if not buf.lines:
        raise ValueError("Buffer has no entries.")

    # Build a sorted list of modification timestamps so bisect can locate the right index.
    ts_list = [e.modified_at for e in buf.lines]

    # Resolve the timestamp, handling relative (negative) and absolute (positive) references.
    resolved = _resolve_time(ts_ref, anchor)

    # Choose bisect_left for "start" (lower bound) or bisect_right for "end" (upper bound),
    # then locate the index of the first entry at or past the resolved timestamp.
    bisect_fn = bisect.bisect_left if mode == "start" else bisect.bisect_right
    idx = bisect_fn(ts_list, resolved)

    # Reject out-of-range: "start" index cannot be past the last entry,
    # and "end" index cannot be before the first entry.
    if mode != "end" and idx >= len(ts_list):
        raise ValueError(f"No entries at or after {_fmt_ts(resolved)}.")
    if idx < 1 and mode == "end":
        raise ValueError(f"No entries at or before {_fmt_ts(resolved)}.")

    return idx


# -------------------------------------------------------------------------------------------------------------------- #
# before_update hook — enforce append-only semantics                                                          #
# -------------------------------------------------------------------------------------------------------------------- #

def _enforce_append_only(
    buffer_manager: BufferManager,
    old_buffer: Buffer,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Reject any update that is not a pure append.

    The update is append-only when new_buffer's full text starts with
    old_buffer's full text — no prior content is removed or replaced.
    """
    # Capture the full text of both states to compare containment.
    old_text = old_buffer.get_text()
    new_text = new_buffer.get_text()

    # Reject if new text does not start with old text — content was removed or replaced.
    if not new_text.startswith(old_text):
        return (
            f"Stream '{old_buffer.name}' requires append-only writes. "
            f"The update removes or replaces existing content — rejected."
        )


# -------------------------------------------------------------------------------------------------------------------- #
# Guard and registration                                                                       #
# -------------------------------------------------------------------------------------------------------------------- #

def _is_stream_option(name: str) -> bool:
    """Return True if name is a valid stream option buffer (has the [stream] suffix)."""
    # Extract the option type from the bracketed suffix and compare to "stream".
    return _option_name_from_option(name) == _STREAM_SUFFIX


def _stream_option_name(origin_name: str) -> str:
    """Return the option buffer name for a stream attached to origin_name."""
    # Build the bracketed suffix form of the option buffer name.
    return f"{origin_name}[{_STREAM_SUFFIX}]"


def on_create_stream_option_guard(
    buffer_manager: BufferManager,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Validate the stream option buffer before it is created.

    Validates:
    - The origin buffer exists
    - The origin buffer is not itself a stream (no nested streams)
    - The JSON content is parseable (empty is fine)
    """
    # Guard: this hook should only fire for NAME[stream] buffers — any other name is a bug in the pattern registration.
    if not _is_stream_option(buffer.name):
        _logger.error(f"[on_create_stream_option_guard] hook fired for non-stream buffer '{buffer.name}' — expected 'NAME[stream]'")
        return f"internal error: hook fired for non-stream buffer '{buffer.name}'"

    # Derive the origin name from the option buffer name.
    origin_name = _origin_name_from_option(buffer.name)
    if origin_name is None:
        return f"Cannot derive origin from '{buffer.name}'."

    # Reject if the origin buffer does not exist yet.
    if buffer_manager.get_buffer(origin_name) is None:
        return f"Origin buffer '{origin_name}' does not exist."

    # Reject if the origin buffer is itself a stream — streams cannot nest.
    if _is_stream_option(origin_name):
        return f"Origin buffer '{origin_name}' is already a stream — streams cannot be nested."


def on_stream_option_created(
    buffer_manager: BufferManager,
    option_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Register the append-only enforcement hook on the origin buffer.

    This fires after AFTER_CREATE_BUFFER when a "BUFFER[stream]" option buffer is created.
    """
    # Guard: this hook should only fire for NAME[stream] buffers — any other name is a bug in the pattern registration.
    if not _is_stream_option(option_buffer.name):
        _logger.error(f"[on_stream_option_created] hook fired for non-stream buffer '{option_buffer.name}' — expected 'NAME[stream]'")
        return

    # Derive the origin name and resolve the origin buffer.
    origin_name = _origin_name_from_option(option_buffer.name)

    # Guard: a valid stream option always has a derivable origin name.
    if origin_name is None:
        _logger.error(f"[on_stream_option_created] cannot derive origin from '{option_buffer.name}'")
        return

    # Resolve the origin buffer — it exists since the pre-create guard passed.
    origin_buffer = buffer_manager.get_buffer(origin_name)

    # Guard: the origin buffer must exist.
    if origin_buffer is None:
        _logger.error(f"[on_stream_option_created] origin '{origin_name}' not found for '{option_buffer.name}'")
        return

    # Register the append-only enforcement hook on the origin's before_update.
    # This fires before every write and rejects any non-append operation.
    origin_buffer.register_hook(
        HookPoint.BEFORE_UPDATE_BUFFER,
        _STREAM_APPEND_ONLY_HOOK,
        _enforce_append_only,
    )


def on_stream_option_dropped(
    buffer_manager: BufferManager,
    option_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Unregister the append-only enforcement hook when the option buffer is dropped.

    This fires on AFTER_DROP_BUFFER after the option buffer is already removed.
    """
    # Guard: this hook should only fire for NAME[stream] buffers — any other name is a bug in the pattern registration.
    if not _is_stream_option(option_buffer.name):
        _logger.error(f"[on_stream_option_dropped] hook fired for non-stream buffer '{option_buffer.name}' — expected 'NAME[stream]'")
        return

    # Derive the origin name from the (now orphaned) option buffer name.
    origin_name = _origin_name_from_option(option_buffer.name)
    if origin_name is None:
        _logger.error(f"[on_stream_option_dropped] cannot derive origin from '{option_buffer.name}'")
        return

    # Resolve the origin buffer — it must still exist since the origin was not yet dropped.
    origin_buffer = buffer_manager.get_buffer(origin_name)
    if origin_buffer is None:
        _logger.error(f"[on_stream_option_dropped] origin '{origin_name}' not found")
        return

    # Unregister the append-only enforcement hook — the stream is no longer protected.
    origin_buffer.unregister_hook(
        HookPoint.BEFORE_UPDATE_BUFFER,
        _STREAM_APPEND_ONLY_HOOK,
    )


# -------------------------------------------------------------------------------------------------------------------- #
# StreamBufferManager                                                                         #
# -------------------------------------------------------------------------------------------------------------------- #

class StreamBufferManager(OptionBufferManager, AgenticObject):
    """You can turn any buffer into a stream with time-based reading by creating a BUFFER[stream] option buffer.
    See "doc:stream_buffers" for details.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Initialize the OptionBufferManager base, which seeds pattern hooks
        # for origin-existence validation and cascade-drop on origin deletion.
        super().__init__(**kwargs)

        # Seed the stream documentation buffer so agents can discover the
        # stream option and its JSON shape without consulting the source.
        self._create_buffer(_STREAM_DOC_BUFFER, text=_STREAM_DOC)

        # Register the stream-specific pattern hooks: guard on creation,
        # seed the append-only hook on creation, and unregister on drop.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "stream_option_guard",
            r"^.+\[stream\]$",
            on_create_stream_option_guard,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "stream_option_created",
            r"^.+\[stream\]$",
            on_stream_option_created,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            "stream_option_dropped",
            r"^.+\[stream\]$",
            on_stream_option_dropped,
        )

    async def create_stream_buffer(
        self,
        name: str,
        text: str | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Create a buffer and its [stream] option buffer in one call.

        This is a convenience wrapper around two create_buffer calls.
        The origin buffer and the "name[stream]" option buffer are created together.
        Returns the result of the origin buffer creation.
        """
        # Create the origin buffer first.
        result = await self.create_buffer(name, text=text, overwrite=overwrite)
        if not result.get("ok"):
            return result

        # Create the [stream] option buffer to enable stream behaviour.
        stream_option_result = await self.create_buffer(
            _stream_option_name(name),
            text="{}",
            overwrite=overwrite,
        )

        # If the option buffer creation fails, surface the error.
        if not stream_option_result.get("ok"):
            return stream_option_result

        return result

    async def read_buffer(
        self,
        name: str,
        start: int | float = 0,
        end: int | float | str = "end",
        show_timestamps: bool = False,
        show_line_numbers: bool = False,
        raw: bool = False,
    ) -> dict[str, Any] | str:
        """Read a range of lines from a buffer.

        Omit start to read from the beginning; omit end to read to the last line.
        Pass integer start/end for line-based reading on any buffer.
        Pass float start/end for time-based reading — available only on streams
        (buffers with a BUFFER[stream] option buffer). Negative floats are
        relative to the last entry (e.g. -60.0 means '60 seconds ago');
        positive floats are absolute Unix timestamps.
        Set raw=True to get the raw string instead of a dict.
        """
        # Delegate immediately for non-time-based reading — no time conversion needed.
        if not isinstance(start, float) and not isinstance(end, float):
            return await super().read_buffer(
                name,
                start=start,
                end=end,
                show_timestamps=show_timestamps,
                show_line_numbers=show_line_numbers,
                raw=raw,
            )

        # Detect whether this buffer is a stream by checking for the [stream] option buffer.
        if not self.is_stream_buffer(name):
            return {"ok": False, "error": f"Time-based reading requires a stream buffer. '{name}' has no [stream] option."}

        # Resolve the buffer — it must exist for time-based reading to work.
        buf = self._buffers.get(name)
        if buf is None:
            return {"ok": False, "error": f"No buffer named '{name}'."}

        # Reject time queries on empty buffers — there are no timestamps to reference.
        if not buf.lines:
            return {"ok": False, "error": f"Buffer '{name}' is empty."}

        # Use the last entry's modified_at as the anchor for resolving relative timestamps.
        anchor = buf.lines[-1].modified_at

        # Convert float start to a line index; propagate errors from out-of-range timestamps.
        if isinstance(start, float):
            try:
                start = _time_ref_to_line(buf, start, anchor, "start")
            except ValueError as e:
                return {"ok": False, "error": str(e)}

        # Convert float end to a line index; propagate errors from out-of-range timestamps.
        if isinstance(end, float):
            try:
                end = _time_ref_to_line(buf, end, anchor, "end")
            except ValueError as e:
                return {"ok": False, "error": str(e)}

        # Auto-enable timestamps for time-based reads so callers can see when each entry arrived.
        if not raw:
            show_timestamps = True

        # Delegate to the base read_buffer with resolved integer indices.
        return await super().read_buffer(
            name,
            start=start,
            end=end,
            show_timestamps=show_timestamps,
            show_line_numbers=show_line_numbers,
            raw=raw,
        )

    def is_stream_buffer(self, buffer_name: str) -> bool:
        """Return True if 'buffer_name' is a stream buffer (has a "buffer_name[stream]" option buffer)."""
        # Detect stream status dynamically by checking for the option buffer's existence.
        # No registry needed — the option buffer itself is the stream marker.
        return self.get_buffer(_stream_option_name(buffer_name)) is not None
