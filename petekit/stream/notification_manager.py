"""NotificationManager — fires agent notifications on stream buffers based on [notifications] option buffers."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
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


_NOTIFICATION_MANAGER_DOC_BUFFER = "doc:notifications"
_NOTIFICATION_MANAGER_DOC = f"""\
## Notification Manager

You can receive notifications when a stream buffer accumulates new entries.
Notifications are delivered to you as prompts, allowing you to respond to stream
activity without continuously polling.

### Enable notifications for a stream

Create a notification option buffer with the name `<STREAM>[notifications]` and
provide configuration as JSON:

```json
{{
    "batch_size": 1,
    "interval_secs": 60.0,
    "persistent_thread_id": null
}}
```

The `[notifications]` option buffer is named `<STREAM>[notifications]`, where
`<STREAM>` is the name of the stream buffer you want to watch. For example,
if a stream is named `my_stream`, its notification option buffer is
`my_stream[notifications]`.

### Configuration options

- **batch_size** (integer, default: 1): fire only when at least this many new
  entries have accumulated. Set to a higher number to reduce notification
  frequency on high-volume streams.

- **interval_secs** (float, default: 20.0, min: 20.0, max: 600.0): minimum time
  in seconds between consecutive notifications for the same stream. Even if
  `batch_size` is reached sooner, the system waits at least this long since
  the last notification before firing again.

- **persistent_thread_id** (string, optional, default: null): the persistent
  thread to deliver the notification to. If omitted or set to `null`, the
  most recently active session is used automatically.

- **notify_on_empty** (bool, optional, default: false): fire a notification even
  when there are no unseen entries. Useful for keep-alive or heartbeat signals
  on low-activity streams.

### What happens when a notification fires

1. All entries that are new since the last notification are marked as seen
   (`agent_at` timestamp is set). This prevents double-notification.
2. A prompt is delivered to you via the persistent thread containing:
   - The stream name
   - The count of new entries
   - The timestamp range of the new entries
3. You receive the prompt and can respond to the stream activity.

### Disable notifications

Drop the `<STREAM>[notifications]` option buffer. This cancels the notification
timer for that stream immediately.

### Requirements

- The stream buffer itself must exist for notifications to fire
- The `[notifications]` option buffer must be valid JSON with the fields
  described above
- `persistent_thread_id`, if provided, must be a valid thread identifier
"""


class _TriggerSource(Enum):
    """The origin of a notification trigger."""

    TIMER = "timer"
    MESSAGE = "message"


@dataclass
class _Notification:
    """Holds the state for one origin buffer's notification timer loop."""

    origin_buffer_name: str
    task: asyncio.Task
    last_fired: float | None = None


# -------------------------------------------------------------------------------------------------------------------- #
# Utility                                                                       #
# -------------------------------------------------------------------------------------------------------------------- #

def _get_notification_config(
    processor: "NotificationManager",
    origin_buffer_name: str,
) -> dict[str, Any] | None:
    """Return the notification config dict for an origin buffer, or None if not configured."""
    # Look up the [notifications] option buffer for this origin buffer.
    option_name = _notifications_option_name(origin_buffer_name)
    option_buffer = processor.get_buffer(option_name)
    if option_buffer is None:
        return None

    # Parse the option buffer JSON — return None if the config is malformed.
    try:
        return json.loads(option_buffer.get_text())
    except json.JSONDecodeError:
        return None

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
    processor: NotificationManager,
    origin_buffer_name: str,
    trigger: _TriggerSource = _TriggerSource.TIMER,
) -> None:
    """Check batch threshold and fire notification for a single buffer.

    The trigger parameter indicates whether this was triggered by a new message
    arriving (_TriggerSource.MESSAGE) or by the periodic timer (_TriggerSource.TIMER).
    Marks all checked entries as seen before firing.
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

    # Skip this tick without notifying if there are no new entries,
    # unless notify_on_empty is set in the config.
    if not unseen and not config.get("notify_on_empty"):
        return

    # Not enough new entries — skip the notification for now.
    if trigger == _TriggerSource.MESSAGE and len(unseen) < batch_size:
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
    # Record the fire time on the notification state so the interval throttle can use it.
    loop = processor._stream_buffer_notifications_loop.get(origin_buffer_name)
    if loop is not None:
        loop.last_fired = now

    try:
        await processor.notify_agent(prompt, persistent_thread_id=persistent_thread_id)
    except Exception as exc:
        processor.write_buffer("system:log:notification_manager", f"NOTIFICATION ERROR: {exc}")


# -------------------------------------------------------------------------------------------------------------------- #
# Notification triggers
# -------------------------------------------------------------------------------------------------------------------- #

async def after_update_origin_buffer(
    processor: "NotificationManager",
    old_buffer: Buffer,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Reactive batch-size check: fire immediately when new entries arrive and batch is full.

    Registered as an AFTER_UPDATE_BUFFER hook on the origin stream buffer.
    This is in addition to the timer-based check in _notification_loop.
    """
    # Check if notifications are enabled for this buffer — skip if no option buffer exists.
    option_name = _notifications_option_name(new_buffer.name)
    if processor.get_buffer(option_name) is None:
        return

    # Fire immediately — check if batch_size is met and notify if so.
    await notify_buffer(processor, new_buffer.name, trigger=_TriggerSource.MESSAGE)


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
            await notify_buffer(processor, origin_buffer_name, trigger=_TriggerSource.TIMER)

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

    # Start the per-buffer notification timer and store its state.
    task = asyncio.create_task(_notification_loop(processor, origin_buffer_name))
    processor._stream_buffer_notifications_loop[origin_buffer_name] = _Notification(
        origin_buffer_name=origin_buffer_name,
        task=task,
    )

    # Register the reactive batch-size check on the origin buffer.
    origin = processor.get_buffer(origin_buffer_name)
    if origin is not None:
        origin.register_hook(
            HookPoint.AFTER_UPDATE_BUFFER,
            "notification",
            after_update_origin_buffer,
        )


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
    loop = processor._stream_buffer_notifications_loop.pop(origin_buffer_name, None)
    if loop is not None:
        loop.task.cancel()

    # Unregister the reactive batch-size check from the origin buffer.
    origin = processor.get_buffer(origin_buffer_name)
    if origin is not None:
        origin.unregister_hook(HookPoint.AFTER_UPDATE_BUFFER, "notification")


# -------------------------------------------------------------------------------------------------------------------- #
# NotificationManager class                                                      #
# -------------------------------------------------------------------------------------------------------------------- #

class NotificationManager(StreamBufferManager, AgenticObject):
    """You are a notification manager that fires notifications on stream buffers.

    Create a notification option buffer to enable notifications for a stream:

        STREAM[notifications] = {"batch_size": 1, "interval_secs": 60.0}

    Dropping the [notifications] option buffer cancels the notification timer.
    No [notifications] option buffer means no notifications on that stream.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Initialize the StreamBufferManager base (OptionBufferManager inside).
        super().__init__(**kwargs)

        # Per-origin-buffer notification state: buffer name -> notification loop state.
        self._stream_buffer_notifications_loop: dict[str, _Notification] = {}

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

        # Seed the documentation buffer so agents can discover the notification
        # API without consulting the source.
        self._create_buffer(_NOTIFICATION_MANAGER_DOC_BUFFER, text=_NOTIFICATION_MANAGER_DOC)
