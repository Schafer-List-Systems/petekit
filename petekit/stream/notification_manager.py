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


_NOTIFICATION_ERRORS_LOG_BUFFER = "system:log:notification_errors"
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
    "message_watermark": 1,
    "interval_secs": 60.0
}}
```

The `[notifications]` option buffer is named `<STREAM>[notifications]`, where
`<STREAM>` is the name of the stream buffer you want to watch. For example,
if a stream is named `my_stream`, its notification option buffer is
`my_stream[notifications]`.

### Configuration options

- **message_watermark** (integer, default: 1): fire when at least this many new
  entries have accumulated. This is a threshold — the minimum needed to trigger
  a notification. When messages arrive and the watermark is met before the
  interval expires, the notification fires right away instead of waiting for
  the next timer tick.

- **interval_secs** (float, default: 20.0, min: 20.0, max: 600.0): a ceiling on
  notification frequency. If messages arrive faster than this interval, you
  wait until the interval passes before the next notification. If the stream
  goes quiet, the interval ensures you still get notified once the watermark
  is eventually met.

- **persistent_thread_id** (string, optional): which conversation receives the
  notification. If omitted, your current conversation receives it automatically.
  If set to `null`, the notification goes to a fresh ephemeral context.

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
    task: asyncio.Task | None
    last_fired: float | None = None
    last_trigger: _TriggerSource | None = None


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
    # Load the notification config — skip if the option buffer is missing or malformed.
    config = _get_notification_config(processor, origin_buffer_name)
    if config is None:
        _logger.error(f"[notify_buffer] notifications option not found or not valid JSON for '{origin_buffer_name}' — dropped before fire")
        return

    # Get the notification state for this origin buffer.
    notification = processor._stream_buffer_notifications.get(origin_buffer_name)
    if notification is None:
        _logger.error(f"[notify_buffer] Corrupt notification configuration for '{origin_buffer_name}'")
        return

    # Clamp message_watermark to at least 1 — notification fires only when this many entries accumulate.
    message_watermark = max(1, int(config.get("message_watermark", 1)))

    # Collect entries that have not been seen by the agent yet (agent_at is None).
    now = time.time()
    buffer = processor.get_buffer(origin_buffer_name)
    if buffer is None:
        _logger.error(f"[notify_buffer] stream buffer '{origin_buffer_name}' not found — dropped before fire")
        return
    unseen = [e for e in buffer.lines if e.agent_at is None]

    try:
        # Skip this tick without notifying if there are no new entries,
        # unless notify_on_empty is set in the config.
        if not unseen and not config.get("notify_on_empty"):
            return

        # Only for message-triggered events: check threshold and flood prevention.
        if trigger == _TriggerSource.MESSAGE:
            # Not enough new entries — skip the notification for now.
            if len(unseen) < message_watermark:
                return

            # Flood check — only fire if enough time has passed since the last actual fire.
            if notification.last_fired is not None:
                elapsed = now - notification.last_fired
                if elapsed < config.get("interval_secs", _MIN_NOTIFICATION_INTERVAL):
                    return

            # Proceed: cancel any pending timer since we will fire now.
            if notification.task is not None:
                notification.task.cancel()

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
        # otherwise use the current active thread.
        persistent_thread_id = config.get("persistent_thread_id")
        if "persistent_thread_id" not in config:
            persistent_thread_id = processor.get_current_thread_id()

        # Deliver the notification to the agent via the resolved persistent thread.
        try:
            # Record the fire time — this marks the last actual notification, not an early return.
            if notification is not None:
                notification.last_fired = now

            # Prompt the agent.
            await processor.notify_agent(prompt, persistent_thread_id=persistent_thread_id)
        except Exception as exc:
            # Never write to the log buffer if the origin is the log buffer — that would loop forever.
            if origin_buffer_name != _NOTIFICATION_ERRORS_LOG_BUFFER:
                await processor.write_buffer(_NOTIFICATION_ERRORS_LOG_BUFFER, f"NOTIFICATION ERROR: {exc}")
            else:
                _logger.error(f"[notify_buffer] NOTIFICATION ERROR: {exc}")

        # Re-arm the timer for the next periodic check.
        _rearm_timer(processor, origin_buffer_name)

    finally:
        # Always record which trigger source was used for this cycle.
        notification.last_trigger = trigger


# -------------------------------------------------------------------------------------------------------------------- #
# Notification triggers
# -------------------------------------------------------------------------------------------------------------------- #

def _rearm_timer(processor: NotificationManager, origin_buffer_name: str) -> None:
    """Rearm the one-shot notification timer for an origin buffer.

    Called at the end of notify_buffer for TIMER-triggered events, and when
    a [notifications] option buffer is first created.
    """
    # Guard: the notification state must exist for this origin buffer.
    notification = processor._stream_buffer_notifications.get(origin_buffer_name)
    if notification is None:
        return

    # Cancel any existing pending timer — this call supersedes it.
    if notification.task is not None:
        notification.task.cancel()

    # Schedule the one-shot notification timer.
    notification.task = asyncio.create_task(_notification_timer(processor, origin_buffer_name))


async def after_update_origin_buffer(
    processor: NotificationManager,
    old_buffer: Buffer,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Batch-size check: fire immediately when new entries arrive and batch is full.

    Registered as an AFTER_UPDATE_BUFFER hook on the origin stream buffer.
    This is in addition to the timer-based check in _notification_timer.
    """
    # Check if notifications are enabled for this buffer — skip if no option buffer exists.
    option_name = _notifications_option_name(new_buffer.name)
    if processor.get_buffer(option_name) is None:
        return

    # Fire immediately — check if message_watermark is met and notify if so.
    await notify_buffer(processor, new_buffer.name, trigger=_TriggerSource.MESSAGE)


async def _notification_timer(
    processor: NotificationManager,
    origin_buffer_name: str,
) -> None:
    """One-shot notification timer for a single origin buffer.

    Fires once after the configured interval, then exits. The caller (notify_buffer)
    re-arms the timer at the end of a TIMER-triggered notification.
    """
    # Load the notification config — use defaults if the option buffer is missing or malformed.
    config = _get_notification_config(processor, origin_buffer_name)
    if config is None:
        _logger.error(f"[_notification_timer] '[notifications]' option for '{origin_buffer_name}' disappeared or is not valid JSON — timer exiting")
        return

    # Parse the interval from JSON, capped between min and max.
    interval_secs = max(
        _MIN_NOTIFICATION_INTERVAL,
        min(_MAX_NOTIFICATION_INTERVAL, float(config.get("interval_secs", _MIN_NOTIFICATION_INTERVAL))),
    )

    # Sleep the configured interval, then fire a notification if the buffer still exists.
    await asyncio.sleep(interval_secs)
    if processor.get_buffer(origin_buffer_name) is None:
        _logger.error(f"[_notification_timer] stream buffer '{origin_buffer_name}' disappeared — timer exiting")
        return

    # Reset the task to None — the timer has fired and will be re-armed by notify_buffer.
    notification = processor._stream_buffer_notifications.get(origin_buffer_name)
    if notification is not None:
        notification.task = None

    await notify_buffer(processor, origin_buffer_name, trigger=_TriggerSource.TIMER)


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

    # Validate message_watermark — must be an integer of at least 1 when provided.
    raw_batch = config.get("message_watermark")
    if raw_batch is not None:
        try:
            message_watermark = int(raw_batch)
        except (TypeError, ValueError):
            return f"'{origin_buffer_name}[notifications]' message_watermark={raw_batch!r} is not a valid int"
        if message_watermark < 1:
            return f"'{origin_buffer_name}[notifications]' message_watermark={message_watermark} must be at least 1"

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


async def before_create_notification_option(
    processor: BufferManager,
    option_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Resolve the persistent_thread_id at creation time if omitted.

    If the proposed config has no persistent_thread_id, the current thread's ID
    is injected so the notification is delivered to the active conversation.
    """
    # Guard: parse the buffer name from the option buffer name.
    if not option_buffer.name.endswith("[notifications]"):
        return None
    origin_buffer_name = option_buffer.name[: -len("[notifications]")]
    if not origin_buffer_name:
        return None

    # Parse the proposed buffer content as JSON.
    try:
        config = json.loads(option_buffer.get_text())
    except json.JSONDecodeError:
        return None

    # Only inject if persistent_thread_id is absent.
    if "persistent_thread_id" not in config:
        current = processor.get_current_thread_id()
        if current is not None:
            config["persistent_thread_id"] = current
            # Patch the buffer's lines in place before creation commits.
            patched_text = json.dumps(config, indent=2) + "\n"
            option_buffer.set_text(patched_text, modified_at=time.time())

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
    task = asyncio.create_task(_notification_timer(processor, origin_buffer_name))
    processor._stream_buffer_notifications[origin_buffer_name] = _Notification(
        origin_buffer_name=origin_buffer_name,
        task=task,
    )

    # Register the watermark check on the origin buffer.
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
    notification = processor._stream_buffer_notifications.pop(origin_buffer_name, None)
    if notification is not None:
        notification.task.cancel()

    # Unregister the watermark check from the origin buffer.
    origin = processor.get_buffer(origin_buffer_name)
    if origin is not None:
        origin.unregister_hook(HookPoint.AFTER_UPDATE_BUFFER, "notification")


# -------------------------------------------------------------------------------------------------------------------- #
# NotificationManager class                                                      #
# -------------------------------------------------------------------------------------------------------------------- #

class NotificationManager(StreamBufferManager, AgenticObject):
    """You are a notification manager that fires notifications on stream buffers.

    Create a notification option buffer to enable notifications for a stream:

        STREAM[notifications] = {"message_watermark": 1, "interval_secs": 60.0}

    Dropping the [notifications] option buffer cancels the notification timer.
    No [notifications] option buffer means no notifications on that stream.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Initialize the StreamBufferManager base (OptionBufferManager inside).
        super().__init__(**kwargs)

        # Per-origin-buffer notification state: buffer name -> notification loop state.
        self._stream_buffer_notifications: dict[str, _Notification] = {}

        # Register the [notifications] pattern hooks on the BufferManager.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "resolve_notification_thread",
            r"^.+\[notifications\]$",
            before_create_notification_option,
        )
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
