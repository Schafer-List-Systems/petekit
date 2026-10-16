"""NotificationManager — fires agent notifications on stream buffers based on [notifications] option buffers."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from peteos import AgenticObject

from .stream_buffer_manager import StreamBufferManager
from ..text.buffer_manager import (
    Buffer,
    BufferManager,
    HookPoint,
    _sanitize_buffer_name,
)


_logger = logging.getLogger("notification_manager")

_MIN_NOTIFICATION_INTERVAL = 20.0
_MAX_NOTIFICATION_INTERVAL = 600.0


# -------------------------------------------------------------------------------------------------------------------- #
# Utility                                                                       #
# -------------------------------------------------------------------------------------------------------------------- #

def _most_recent_persistent_thread(obj: AgenticObject) -> str | None:
    """Return the persistent_thread_id of the session with the newest directory mtime.

    Iterates over active sessions in _oap_session_store, compares their session_dir
    mtimes, and returns the persistent_thread_id whose session was used most recently.
    Returns None if no sessions are active.
    """
    # Get the mapping from persistent_thread_id to session_uuid.
    thread_store = obj._oap_json["thread_store"]
    if not thread_store:
        return None

    # Build a reverse map: session_uuid -> persistent_thread_id.
    session_to_thread: dict[str, str] = {uuid: ptid for ptid, uuid in thread_store.items()}

    # Scan each active session, tracking whichever has the most recent session_dir mtime.
    best_ptid: str | None = None
    best_mtime = -1.0
    for session in obj._oap_session_store.values():
        # Skip sessions whose directory no longer exists on disk.
        if not session.session_dir.exists():
            continue
        try:
            mtime = session.session_dir.stat().st_mtime
            # Update best candidate if this session is more recent.
            if mtime > best_mtime:
                best_mtime = mtime
                best_ptid = session_to_thread.get(session.uuid)
        except OSError:
            _logger.error(f"[_most_recent_persistent_thread] cannot stat session dir '{session.session_dir}'")
            pass

    # The persistent_thread_id of the session with the highest mtime, or None.
    return best_ptid


def _notifications_option_name(origin_buffer_name: str) -> str:
    """Return the [notifications] option buffer name for a given buffer name."""
    return f"{origin_buffer_name}[notifications]"


# -------------------------------------------------------------------------------------------------------------------- #
# Notification firing                                                             #
# -------------------------------------------------------------------------------------------------------------------- #

async def notify_buffer(
    processor: "NotificationManager",
    origin_buffer_name: str,
) -> None:
    """Check batch threshold and fire notification for a single buffer.

    Reads the [notifications] option buffer JSON fresh on every call to get
    the current configuration. Marks all checked entries as seen before firing.
    """
    # Look up the [notifications] option buffer for this stream.
    notifications_option_name = _notifications_option_name(origin_buffer_name)
    notifications_option = processor.get_buffer(notifications_option_name)
    if notifications_option is None:
        _logger.error(f"[notify_buffer] notifications option not found for '{origin_buffer_name}' — dropped before fire")
        return

    # Parse the notification configuration from the option buffer JSON.
    try:
        config = json.loads(notifications_option.get_text())
    except json.JSONDecodeError:
        _logger.error(f"[notify_buffer] '{notifications_option_name}' is not valid JSON")
        return

    # Clamp batch_size to at least 1 — notification fires only when this many entries accumulate.
    batch_size = max(1, int(config.get("batch_size", 1)))

    # Collect entries that have not been seen by the agent yet (agent_at is None).
    now = time.time()
    buffer = processor.get_buffer(origin_buffer_name)
    if buffer is None:
        _logger.error(f"[notify_buffer] stream buffer '{origin_buffer_name}' not found — dropped before fire")
        return
    unseen = [e for e in buffer.lines if e.agent_at is None]

    # Not enough new entries to fire — skip this tick without notifying.
    if not unseen or len(unseen) < batch_size:
        return

    # Build the notification prompt with entry count and timestamp range.
    ts_start = unseen[0].modified_at
    ts_end = unseen[-1].modified_at
    prompt = (
        f"[NOTIFICATION] Stream '{origin_buffer_name}' has {len(unseen)} new entries "
        f"between {ts_start:.2f} and {ts_end:.2f}. "
    )

    # Mark all checked entries as seen before firing — prevents double-notification.
    for entry in unseen:
        entry.agent_at = now

    # Resolve the persistent thread target: use the explicit value from the config if set,
    # otherwise fall back to the most recently active session.
    persistent_thread_id = config.get("persistent_thread_id")
    if "persistent_thread_id" not in config:
        persistent_thread_id = _most_recent_persistent_thread(processor)

    # Deliver the notification to the agent via the resolved persistent thread.
    try:
        await processor.notify_agent(prompt, persistent_thread_id=persistent_thread_id)
    except Exception as exc:
        processor.write_buffer("system:log:notification_manager", f"NOTIFICATION ERROR: {exc}")


# -------------------------------------------------------------------------------------------------------------------- #
# Notification timer loop                                                        #
# -------------------------------------------------------------------------------------------------------------------- #

async def _notification_loop(
    processor: "NotificationManager",
    origin_buffer_name: str,
) -> None:
    """Per-buffer notification timer loop. Runs until cancelled or config is dropped.

    Reads the interval from the [notifications] option buffer JSON on every tick
    so config changes take effect without restarting the timer.
    """
    try:
        while True:
            # Look up the [notifications] option buffer to get the current interval.
            notifications_option_name = _notifications_option_name(origin_buffer_name)
            notifications_option = processor.get_buffer(notifications_option_name)
            if notifications_option is None:
                _logger.error(f"[_notification_loop] '[notifications]' option for '{origin_buffer_name}' disappeared — timer exiting")
                break

            # Parse the interval from JSON, capped between min and max.
            try:
                config = json.loads(notifications_option.get_text())
            except json.JSONDecodeError:
                _logger.error(f"[_notification_loop] '{notifications_option_name}' is not valid JSON — using defaults")
                config = {}
            interval_secs = max(
                _MIN_NOTIFICATION_INTERVAL,
                min(_MAX_NOTIFICATION_INTERVAL, float(config.get("interval_secs", _MIN_NOTIFICATION_INTERVAL))),
            )

            # Sleep the configured interval, then fire a notification if the buffer still exists.
            await asyncio.sleep(interval_secs)
            if processor.get_buffer(origin_buffer_name) is None:
                _logger.error(f"[_notification_loop] stream buffer '{origin_buffer_name}' disappeared — timer exiting")
                break
            await notify_buffer(processor, origin_buffer_name)

    except asyncio.CancelledError:
        pass  # Clean shutdown when the timer task is cancelled.


# -------------------------------------------------------------------------------------------------------------------- #
# Pattern hooks                                                                  #
# -------------------------------------------------------------------------------------------------------------------- #

def _validate_notifications_config(
    origin_buffer_name: str,
    config: dict[str, Any],
) -> str | None:
    """Validate the notifications config dict.

    Returns None if valid, or an error string describing the first validation failure.
    """
    # Guard: the config must be a JSON object, not a primitive or array.
    if not isinstance(config, dict):
        return f"'{origin_buffer_name}[notifications]' config must be a JSON object"

    # Validate batch_size — must be an integer of at least 1 when provided.
    raw_batch = config.get("batch_size")
    if raw_batch is not None:
        try:
            batch_size = int(raw_batch)
        except (TypeError, ValueError):
            return f"'{origin_buffer_name}[notifications]' batch_size={raw_batch!r} is not a valid int"
        if batch_size < 1:
            return f"'{origin_buffer_name}[notifications]' batch_size={batch_size} must be at least 1"

    # Validate interval_secs — must be a float in the allowed range when provided.
    raw_interval = config.get("interval_secs")
    if raw_interval is not None:
        try:
            interval_secs = float(raw_interval)
        except (TypeError, ValueError):
            return f"'{origin_buffer_name}[notifications]' interval_secs={raw_interval!r} is not a valid float"
        if not (_MIN_NOTIFICATION_INTERVAL <= interval_secs <= _MAX_NOTIFICATION_INTERVAL):
            return (
                f"'{origin_buffer_name}[notifications]' interval_secs={interval_secs} "
                f"must be between {_MIN_NOTIFICATION_INTERVAL} and {_MAX_NOTIFICATION_INTERVAL}"
            )

    # Validate notify_on_empty — must be a bool when provided.
    notify_on_empty = config.get("notify_on_empty")
    if notify_on_empty is not None and not isinstance(notify_on_empty, bool):
        return f"'{origin_buffer_name}[notifications]' notify_on_empty={notify_on_empty!r} must be a bool"

    # Validate persistent_thread_id — optional, but if present must be a string or null.
    ptid = config.get("persistent_thread_id")
    if ptid is not None and not isinstance(ptid, str):
        return f"'{origin_buffer_name}[notifications]' persistent_thread_id={ptid!r} must be a string or omitted"

    # All fields valid — allow the update.
    return None


async def before_update_notification_option(
    processor: BufferManager,
    old_option_buffer: Buffer,
    new_option_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Validate the [notifications] option buffer JSON before the update is applied.

    The BEFORE_UPDATE_BUFFER hook runs before the option buffer content is replaced.
    Return None or True to allow the update; return an error string to block it.
    """
    # Guard: parse the buffer name from the option buffer name.
    if not new_option_buffer.name.endswith("[notifications]"):
        return None  # Not a notifications option buffer — allow.
    origin_buffer_name = new_option_buffer.name[: -len("[notifications]")]
    if not origin_buffer_name:
        _logger.error(f"[before_update_notification_option] cannot parse origin buffer name from '{new_option_buffer.name}' — update blocked")
        return f"before_update_notification_option: cannot parse origin buffer name from '{new_option_buffer.name}'"

    # Guard: the proposed new buffer content must be valid JSON.
    try:
        new_text = new_option_buffer.get_text()
        new_config = json.loads(new_text)
    except json.JSONDecodeError:
        _logger.error(f"[before_update_notification_option] '{new_option_buffer.name}' is not valid JSON — update blocked")
        return f"'{new_option_buffer.name}' is not valid JSON — update blocked"

    # Validate the notifications config fields.
    validation_error = _validate_notifications_config(origin_buffer_name, new_config)
    if validation_error is not None:
        return validation_error

    # Validation passed — allow the update.
    return True


async def after_create_notification_option(
    processor: BufferManager,
    option_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Start a per-buffer notification timer when a [notifications] option buffer is created."""
    # Guard: parse the buffer name from the option buffer name.
    # E.g. "STREAM[notifications]" -> origin_buffer_name = "STREAM".
    if not option_buffer.name.endswith("[notifications]"):
        _logger.error(f"[after_create_notification_option] '{option_buffer.name}' is not a [notifications] option buffer")
        return
    origin_buffer_name = option_buffer.name[: -len("[notifications]")]
    if not origin_buffer_name:
        _logger.error(f"[after_create_notification_option] cannot parse origin buffer name from '{option_buffer.name}'")
        return

    # Start the per-buffer notification timer and store the task.
    task = asyncio.create_task(_notification_loop(processor, origin_buffer_name))
    processor._stream_buffer_notifications_loop[origin_buffer_name] = task


async def after_drop_notification_option(
    processor: BufferManager,
    option_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Cancel the per-buffer notification timer when a [notifications] option buffer is dropped."""
    # Guard: parse the origin buffer name from the option buffer name.
    if not option_buffer.name.endswith("[notifications]"):
        return
    origin_buffer_name = option_buffer.name[: -len("[notifications]")]

    # Guard: the origin buffer name was empty after stripping the suffix.
    if not origin_buffer_name:
        _logger.error(f"[after_drop_notification_option] cannot parse origin buffer name from '{option_buffer.name}'")
        return

    # Cancel and remove the timer task for this buffer.
    task = processor._stream_buffer_notifications_loop.pop(origin_buffer_name, None)
    if task is not None:
        task.cancel()


# -------------------------------------------------------------------------------------------------------------------- #
# NotificationManager class                                                      #
# -------------------------------------------------------------------------------------------------------------------- #

class NotificationManager(StreamBufferManager, AgenticObject):
    """You are a notification manager that fires agent notifications on stream buffers.

    Create a notification option buffer to enable notifications for a stream:

        STREAM[notifications] = {"batch_size": 1, "interval_secs": 60.0}

    Dropping the [notifications] option buffer cancels the notification timer.
    No [notifications] option buffer means no notifications on that stream.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Initialize the StreamBufferManager base (OptionBufferManager inside).
        super().__init__(**kwargs)

        # Per-origin-buffer notification timer tasks: origin_buffer_name -> asyncio.Task.
        self._stream_buffer_notifications_loop: dict[str, asyncio.Task] = {}

        # Register the [notifications] pattern hooks on the BufferManager.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "create_notification_option",
            r"^.+\[notifications\]$",
            after_create_notification_option,
        )
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_UPDATE_BUFFER,
            "before_update_notification_option",
            r"^.+\[notifications\]$",
            before_update_notification_option,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            "after_drop_notification_option",
            r"^.+\[notifications\]$",
            after_drop_notification_option,
        )
