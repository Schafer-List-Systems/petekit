from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from peteos.oap.agentic_object import AgenticObject


from petekit.function.function_caller import FunctionCaller
from petekit.text.buffer_manager import Buffer, BufferManager, HookPoint
from petekit.text.option_buffer_manager import OptionBufferManager


_logger = logging.getLogger("croner")


# -------------------------------------------------------------------------------------------------------------------- #
# Constants                                                                                                          #
# -------------------------------------------------------------------------------------------------------------------- #


_CRONER_DOC_BUFFER = "doc:cron"

_CRONER_DOC = f"""\
## Croner

The Croner attaches time-based actions to any buffer via [timer], [alarm], and
[schedule] option buffers. Create `<buffer-name>[timer]`, `[alarm]`, or `[schedule]`
to schedule a timed action on that buffer. Only one option buffer per trigger type
is allowed per origin buffer.

### Buffer Naming

- `<buffer-name>[timer]`  --  a recurring or one-shot timer
- `<buffer-name>[alarm]`  --  a wall-clock alarm
- `<buffer-name>[schedule]`  --  a crontab-style schedule

### Creating a Scheduled Action

Create the option buffer on the target buffer to enable the trigger.
The target buffer must exist first. The option buffer holds a flat JSON object
with an `action` and one trigger specifier (`timer`, `alarm`, or `schedule`).

### Trigger Types

`timer`  --  Relative time in seconds. The value is the number of seconds until
the job fires. By default the job fires once and is removed. Set `repeat: true`
to make it a recurring timer:

```json
{{"action": "!read", "timer": 60, "repeat": true}}
```

`alarm`  --  Wall-clock alarm. The `time` is in 24h format (HH:MM:SS). The alarm
fires daily at that time:

```json
{{"action": "!read", "time": "08:00:00"}}
```

`schedule`  --  Crontab-style field matching. Fire when the current datetime matches
all specified fields. Use `*` as wildcard for "any". Fields: `second`, `minute`,
`hour`, `day`, `month`, `weekday`:

```json
{{"action": "!read", "second": 0, "minute": 0, "hour": 9, "weekday": 1}}
```

### Built-In Actions

The `action` can be the name of a custom function from the FunctionManager,
or one of the two built-in actions:

- `!read`  --  reads the origin buffer, discarding the result but triggering the read hooks
- `!touch`  --  writes an empty string to the origin buffer, triggering the update hooks

### Custom Action Signature

The `action` is a function name string from the FunctionManager. It is called
with `buffer_name` as the sole keyword argument:

- `buffer_name: str`  --  the name of the buffer this trigger is attached to

### Deleting a Scheduled Action

Drop the option buffer (`[timer]`, `[alarm]`, or `[schedule]`) to remove the trigger.

### Constraints

- The target buffer must exist before the [timer], [alarm], or [schedule] option buffer can be created.
- Dropping the target buffer automatically drops its option buffers.
- Exactly one trigger key (`timer`, `alarm`, or `schedule`) is required in the JSON.
- The `action` must be a function known to the FunctionManager.
"""


# -------------------------------------------------------------------------------------------------------------------- #
# Data Classes                                                                                                       #
# -------------------------------------------------------------------------------------------------------------------- #


@dataclass
class _Timer:
    origin: str
    timeout: float | None = None
    repeat: bool = True
    task: asyncio.Task | None = None


@dataclass
class _Alarm:
    origin: str
    timeout: float | None = None
    task: asyncio.Task | None = None


@dataclass
class _Schedule:
    origin: str
    timeout: float | None = None
    task: asyncio.Task | None = None


# -------------------------------------------------------------------------------------------------------------------- #
# Utils                                                                                                             #
# -------------------------------------------------------------------------------------------------------------------- #


def _read_option_buffer_config(croner: Croner, option_buffer_name: str) -> dict[str, Any] | None:
    """Read and parse the JSON config from an option buffer.

    Returns the parsed config dict on success, or None if the buffer is absent,
    empty, or contains invalid JSON.
    """
    # Locate the option buffer by name.
    option_buffer = croner.get_buffer(option_buffer_name)
    if option_buffer is None:
        _logger.error(f"[Croner] cannot locate {option_buffer_name}  --  discarding rearm")
        return None

    # Extract the text from the option buffer.
    option_buffer_text = option_buffer.get_text()
    if not option_buffer_text:
        _logger.error(f"[Croner] {option_buffer_name} is empty  --  discarding rearm")
        return None

    # Parse the JSON config from the option buffer text.
    try:
        config = json.loads(option_buffer_text)
    except json.JSONDecodeError:
        _logger.error(f"[Croner] {option_buffer_name} has invalid JSON  --  discarding rearm")
        return None

    return config


async def _fire_action(croner: Croner, job: _Timer | _Alarm | _Schedule) -> None:
    """Fire the action configured on the job.

    Reads the option buffer config, dispatches to a built-in action, or calls the
    function via the FunctionManager.
    """
    # Determine the trigger type suffix and option buffer name.
    if isinstance(job, _Timer):
        suffix = "[timer]"
    elif isinstance(job, _Alarm):
        suffix = "[alarm]"
    else:
        suffix = "[schedule]"
    option_buffer_name = f"{job.origin}{suffix}"

    # Read the option buffer config to get the action.
    config = _read_option_buffer_config(croner, option_buffer_name)
    if config is None:
        return

    # Guard against a missing or empty action field.
    action = config.get("action")
    if not action:
        _logger.error(f"[Croner] no action in {option_buffer_name}")
        return

    # Dispatch to a built-in action or treat the action as a custom function name.
    if action == "!read":
        # Read the origin buffer and discard the result, triggering read hooks.
        await croner.read_buffer(job.origin)

    elif action == "!touch":
        # Write an empty string to the origin buffer, triggering update hooks.
        await croner.write_buffer(job.origin, text="")

    else:
        # Call the custom function with the origin buffer name as the argument.
        await croner.call_function(action, kwargs={"buffer_name": job.origin})


async def _wait_for(croner: Croner, job: _Timer | _Alarm | _Schedule) -> None:
    """The event loop for a timer, alarm, or schedule task.

    Sleeps for the job timeout, resets state on normal return, and rearms.
    On cancellation, resets state but does not rearm.
    """
    try:
        # Save the active task reference to guard the reset against race conditions.
        active_task = job.task

        # Sleep until the next fire time.
        await asyncio.sleep(job.timeout or 0)

        # Fire the action as a detached task so cancellation does not interrupt it.
        asyncio.create_task(_fire_action(croner, job))

        # On normal return: reset state and rearm.
        job.timeout = None
        job.task = None

        # Drop one-shot event buffers.
        if isinstance(job, _Timer) and not job.repeat:
            await croner.drop_buffer(f"{job.origin}[timer]")
        
        # Rearm recurring events.
        else:
            _dispatch_rearm(croner, job)

    except asyncio.CancelledError:
        # On cancellation: reset only if the job still holds the task we started with.
        if job.task is active_task:
            job.timeout = None
            job.task = None


def _rearm_timer(croner: Croner, job: _Timer) -> None:
    """Rearm the timer task from the job object.

    Reads the [timer] config, sets the timeout on the job, and starts a new task.
    """
    # Read and parse the [timer] config, then abort if unavailable.
    config = _read_option_buffer_config(croner, f"{job.origin}[timer]")
    if config is None:
        return

    # Assign the interval timeout and schedule the event loop as a new task.
    job.timeout = config.get("timer", 0)
    job.task = asyncio.create_task(_wait_for(croner, job))


def _rearm_alarm(croner: Croner, job: _Alarm) -> None:
    """Rearm the alarm task from the job object.

    Reads the [alarm] config, computes the timeout until the next fire time, and
    starts a new asyncio task.
    """
    # Read and parse the [alarm] config, then abort if unavailable.
    config = _read_option_buffer_config(croner, f"{job.origin}[alarm]")
    if config is None:
        return

    # Parse the wall-clock time from the alarm config.
    time_str = config.get("time", "")
    try:
        import datetime
        fire_time = datetime.datetime.strptime(time_str, "%H:%M:%S").time()
    except ValueError:
        _logger.error(f"[Croner] invalid time format in [alarm] for {job.origin}  --  discarding rearm")
        return

    # Compute the target datetime by combining today with the fire time.
    now = datetime.datetime.now()
    target = datetime.datetime.combine(now.date(), fire_time)

    # Compute the timeout as the difference from now to the target.
    timeout = (target - now).total_seconds()
    if timeout < 0:
        timeout += 86400

    # Assign the timeout and schedule the event loop as a new task.
    job.timeout = timeout
    job.task = asyncio.create_task(_wait_for(croner, job))


def _rearm_schedule(croner: Croner, job: _Schedule) -> None:
    """Rearm the schedule task from the job object.

    Reads the [schedule] config, computes the timeout until the next fire time, and
    starts a new asyncio task.
    """
    # Read and parse the [schedule] config, then abort if unavailable.
    config = _read_option_buffer_config(croner, f"{job.origin}[schedule]")
    if config is None:
        return

    # Assemble the six crontab fields into a cron expression string for croniter.
    fields = (
        config.get("second", "*"),
        config.get("minute", "*"),
        config.get("hour", "*"),
        config.get("day", "*"),
        config.get("month", "*"),
        config.get("weekday", "*"),
    )
    cron_expr = " ".join(str(f) for f in fields)

    # Use croniter to find the next occurrence from now and compute the timeout.
    import datetime
    from croniter import Croniter
    try:
        cron = Croniter(cron_expr, date=datetime.datetime.now())
        next_fire = cron.get_next(datetime.datetime)
    except Exception:
        _logger.error(f"[Croner] cannot parse or resolve schedule for {job.origin}  --  discarding rearm")
        return

    # Assign the timeout and schedule the event loop as a new task.
    job.timeout = (next_fire - datetime.datetime.now()).total_seconds()
    job.task = asyncio.create_task(_wait_for(croner, job))


def _dispatch_rearm(croner: Croner, job: _Timer | _Alarm | _Schedule) -> None:
    """Dispatch to the correct rearm function based on the job type.

    Cancels any already-running task and replaces it with a fresh one.
    """
    # Determine the event type string for the log message.
    if isinstance(job, _Timer):
        event_type = "timer"
    elif isinstance(job, _Alarm):
        event_type = "alarm"
    else:
        event_type = "schedule"

    # Cancel any already-running task so a reconfiguration can take over.
    if job.task is not None:
        job.task.cancel()

    # Route to the correct rearm function with the job object.
    if isinstance(job, _Timer):
        _rearm_timer(croner, job)
    elif isinstance(job, _Alarm):
        _rearm_alarm(croner, job)
    else:
        _rearm_schedule(croner, job)


# -------------------------------------------------------------------------------------------------------------------- #
# Buffer hooks                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #


def _before_update_timer(
    croner: Croner,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Guard the [timer] option buffer against invalid JSON.

    Required: action (string), timer (positive int).
    Optional: repeat (bool).
    """
    # Accept empty buffer as a no-op.
    text = new_buffer.get_text()
    if not text or not text.strip():
        return None

    # Parse the JSON content.
    try:
        job = json.loads(text)
    except json.JSONDecodeError:
        return "timer buffer is not valid JSON"

    # Reject non-dict content.
    if not isinstance(job, dict):
        return "timer buffer must be a JSON object"

    # Reject missing or empty action.
    if not isinstance(job.get("action"), str) or not job["action"]:
        return "action is required and must be a non-empty string"

    # Reject non-positive or non-integer timer value.
    if not isinstance(job.get("timer"), int) or job["timer"] <= 0:
        return "timer must be a positive integer"

    # Reject non-boolean repeat value.
    if job.get("repeat") is not None and not isinstance(job["repeat"], bool):
        return "repeat must be a boolean"

    return None


def _before_update_alarm(
    croner: Croner,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Guard the [alarm] option buffer against invalid JSON.

    Required: action (string), time (string HH:MM:SS).
    Optional: date (string YYYY-MM-DD).
    """
    # Accept empty buffer as a no-op.
    text = new_buffer.get_text()
    if not text or not text.strip():
        return None

    # Parse the JSON content.
    try:
        job = json.loads(text)
    except json.JSONDecodeError:
        return "alarm buffer is not valid JSON"

    # Reject non-dict content.
    if not isinstance(job, dict):
        return "alarm buffer must be a JSON object"

    # Reject missing or empty action.
    if not isinstance(job.get("action"), str) or not job["action"]:
        return "action is required and must be a non-empty string"

    # Reject missing or non-string time field.
    if not isinstance(job.get("time"), str):
        return "alarm requires a time field (HH:MM:SS)"

    # Reject non-string date field.
    if job.get("date") is not None and not isinstance(job["date"], str):
        return "alarm.date must be a string (YYYY-MM-DD) or absent"

    return None


def _before_update_schedule(
    croner: Croner,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Guard the [schedule] option buffer against invalid JSON.

    Required: action (string), crontab fields (second, minute, hour, day, month, weekday).
    All fields must be '*' or a non-negative integer.
    """
    # Accept empty buffer as a no-op.
    text = new_buffer.get_text()
    if not text or not text.strip():
        return None

    # Parse the JSON content.
    try:
        job = json.loads(text)
    except json.JSONDecodeError:
        return "schedule buffer is not valid JSON"

    # Reject non-dict content.
    if not isinstance(job, dict):
        return "schedule buffer must be a JSON object"

    # Reject missing or empty action.
    if not isinstance(job.get("action"), str) or not job["action"]:
        return "action is required and must be a non-empty string"

    # Validate each crontab field is '*' or a non-negative integer.
    valid_fields = {"second", "minute", "hour", "day", "month", "weekday"}
    for fk, fv in job.items():
        if fk not in valid_fields:
            continue
        if not (isinstance(fv, str) or isinstance(fv, int)):
            return f"schedule.{fk} must be a string or int"
        if isinstance(fv, str) and fv != "*":
            return f"schedule.{fk} must be '*' or an integer"
        if isinstance(fv, int) and fv < 0:
            return f"schedule.{fk} must be non-negative"

    # Reject any unknown top-level key.
    unknown = {k for k in job if k not in {"action"} and k not in valid_fields}
    if unknown:
        return f"schedule has unknown fields: {', '.join(sorted(unknown))}"

    return None


async def _after_update_timer(
    croner: Croner,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after a [timer] buffer is created or updated.

    Creates the job if absent, then rearms it.
    """
    # Derive the origin buffer name from the timer option buffer name.
    origin = new_buffer.name.removesuffix("[timer]")

    # Create the timer job if absent, or fetch the existing one.
    if origin not in croner._cron_timers:
        timer_job = _Timer(origin=origin)
        croner._cron_timers[origin] = timer_job
    else:
        timer_job = croner._cron_timers[origin]

    # Keep the job repeat flag in sync with the current buffer config.
    timer_config = _read_option_buffer_config(croner, new_buffer.name)
    if timer_config is None:
        return

    # Apply the repeat flag from the timer config to the job.
    timer_job.repeat = timer_config.get("repeat", False)

    # Rearm the timer job.
    _dispatch_rearm(croner, timer_job)


async def _after_update_alarm(
    croner: Croner,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after an [alarm] buffer is created or updated.

    Creates the job if absent, then rearms it.
    """
    # Derive the origin buffer name from the alarm option buffer name.
    origin = new_buffer.name.removesuffix("[alarm]")

    # Create the alarm job if it does not yet exist in the dictionary.
    if origin not in croner._cron_alarms:
        croner._cron_alarms[origin] = _Alarm(origin=origin)

    # Rearm the alarm job.
    _dispatch_rearm(croner, croner._cron_alarms[origin])


async def _after_update_schedule(
    croner: Croner,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after a [schedule] buffer is created or updated.

    Creates the job if absent, then rearms it.
    """
    # Derive the origin buffer name from the schedule option buffer name.
    origin = new_buffer.name.removesuffix("[schedule]")

    # Create the schedule job if it does not yet exist in the dictionary.
    if origin not in croner._cron_schedules:
        croner._cron_schedules[origin] = _Schedule(origin=origin)

    # Rearm the schedule job.
    _dispatch_rearm(croner, croner._cron_schedules[origin])


async def _after_drop_cron(
    croner: Croner,
    old_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Cancel and remove the task when the option buffer is dropped.

    Extracts the trigger type from the dropped buffer name ([timer], [alarm], or
    [schedule]), looks up the corresponding task in the dictionaries, cancels it,
    and removes the entry.
    """
    # Determine which trigger type this is and select the correct task dictionary.
    if old_buffer.name.endswith("[timer]"):
        tasks = croner._cron_timers
    elif old_buffer.name.endswith("[alarm]"):
        tasks = croner._cron_alarms
    elif old_buffer.name.endswith("[schedule]"):
        tasks = croner._cron_schedules
    else:
        return

    # Derive the origin buffer name by stripping the exact trigger suffix.
    if old_buffer.name.endswith("[timer]"):
        origin = old_buffer.name.removesuffix("[timer]")
    elif old_buffer.name.endswith("[alarm]"):
        origin = old_buffer.name.removesuffix("[alarm]")
    else:
        origin = old_buffer.name.removesuffix("[schedule]")

    # Cancel the running task and remove the entry from the dictionary.
    entry = tasks.get(origin)
    if entry is not None and entry.task is not None:
        entry.task.cancel()
    tasks.pop(origin, None)


# -------------------------------------------------------------------------------------------------------------------- #
# Croner                                                                                                              #
# -------------------------------------------------------------------------------------------------------------------- #


class Croner(FunctionCaller, OptionBufferManager, AgenticObject):
    """I can set up timers, alarms and schedule events on buffers.
    The doc:cron buffer holds the details."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Per-origin task dictionaries keyed by origin buffer name.
        self._cron_timers: dict[str, _Timer] = {}
        self._cron_alarms: dict[str, _Alarm] = {}
        self._cron_schedules: dict[str, _Schedule] = {}

        # Seed the doc buffer with the module documentation.
        self._create_buffer(_CRONER_DOC_BUFFER, text=_CRONER_DOC)

        # Register per-trigger validation hooks for update and create.
        # CREATE fires 3 args (bm, new_buffer, args) but UPDATE needs 4 (bm, old, new, args).
        # The lambda bridges the gap by inserting None for old_buffer.
        # First register the timer guard hooks on [timer] buffers.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_UPDATE_BUFFER,
            "timer_guard",
            r"^.+\[timer\]$",
            _before_update_timer,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "timer_guard",
            r"^.+\[timer\]$",
            lambda bm, new_buffer, args: _before_update_timer(bm, None, new_buffer, args),
            apply_to_existing=True,
        )

        # Then register the alarm guard hooks on [alarm] buffers.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_UPDATE_BUFFER,
            "alarm_guard",
            r"^.+\[alarm\]$",
            _before_update_alarm,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "alarm_guard",
            r"^.+\[alarm\]$",
            lambda bm, new_buffer, args: _before_update_alarm(bm, None, new_buffer, args),
            apply_to_existing=True,
        )

        # Finally register the schedule guard hooks on [schedule] buffers.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_UPDATE_BUFFER,
            "schedule_guard",
            r"^.+\[schedule\]$",
            _before_update_schedule,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "schedule_guard",
            r"^.+\[schedule\]$",
            lambda bm, new_buffer, args: _before_update_schedule(bm, None, new_buffer, args),
            apply_to_existing=True,
        )

        # Register the after-update hooks for timer, alarm, and schedule to start or
        # restart the respective tasks when the option buffer is written.
        # First register the timer after-update hooks.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_UPDATE_BUFFER,
            "after_timer",
            r"^.+\[timer\]$",
            _after_update_timer,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "after_timer",
            r"^.+\[timer\]$",
            lambda bm, new_buffer, args: _after_update_timer(bm, None, new_buffer, args),
            apply_to_existing=True,
        )

        # Then register the alarm after-update hooks.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_UPDATE_BUFFER,
            "after_alarm",
            r"^.+\[alarm\]$",
            _after_update_alarm,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "after_alarm",
            r"^.+\[alarm\]$",
            lambda bm, new_buffer, args: _after_update_alarm(bm, None, new_buffer, args),
            apply_to_existing=True,
        )

        # Finally register the schedule after-update hooks.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_UPDATE_BUFFER,
            "after_schedule",
            r"^.+\[schedule\]$",
            _after_update_schedule,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "after_schedule",
            r"^.+\[schedule\]$",
            lambda bm, new_buffer, args: _after_update_schedule(bm, None, new_buffer, args),
            apply_to_existing=True,
        )

        # Cancel and remove the task when the option buffer is dropped.
        # One shared hook inspects the buffer name to determine the trigger type.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            "after_drop_cron",
            r"^.+\[(timer|alarm|schedule)\]$",
            _after_drop_cron,
            apply_to_existing=True,
        )

