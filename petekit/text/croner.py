from __future__ import annotations

import asyncio
import datetime
import json
import logging
from dataclasses import dataclass
from typing import Any
from typing import Callable

from croniter.croniter import croniter

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

**Warning** -- timers are not persisted across system restarts. When the system
restarts, the timer starts fresh from its original duration -- the intended fire
time is lost. If the exact fire time matters, use `alarm` or `schedule` instead.

`alarm`  --  Wall-clock alarm. The `time` is in 24h format (HH:MM:SS). The alarm
fires daily at that time:

```json
{{"action": "!read", "time": "08:00:00"}}
```

`schedule`  --  Crontab-style field matching. Fire when the current datetime matches
all specified fields. **All six fields are required**: `second`, `minute`, `hour`,
`day`, `month`, `weekday`. Each supports: a plain value, `*` (any), `1,3,5` (list),
`1-5` (range), `*/5` (step). `weekday 0 = Sunday`.

```json
{{"action": "!read", "second": 0, "minute": 0, "hour": 9, "day": "*", "month": "*", "weekday": 1}}
```

Examples:
- `second: 0, minute: 5`  --  every hour at minute 5, second 0
- `minute: "*/5"`  --  every 5 minutes
- `minute: "1,3,5"`  --  at minute 1, 3, and 5
- `hour: "1-5"`  --  between hours 1 and 5 (inclusive)
- `weekday: 0`  --  Sunday only
- `weekday: 1`  --  Monday only
- `minute: 0, hour: "9-17", weekday: "1-5"`  --  weekdays 9am-5pm, on the hour

### Built-In Actions

The `action` can be the name of a custom function from the FunctionManager,
or one of the built-in actions:

- `!read`  --  reads the origin buffer, discarding the result but triggering the read hooks
- `!touch`  --  writes an empty string to the origin buffer, triggering the update hooks
- `!notify`  --  delivers a notification to you -- see `!notify` below

#### Built-In Action: !notify

Set `action` to `!notify` and provide the notification text via `action_config`:

```json
{{
  "action": "!notify",
  "action_config": {{
    "prompt": "the timer has fired",
    "description": "alerts me when the timer fires"
  }}
}}
```

**What it does** -- delivers the prompt text to you when the trigger fires.
If `prompt` was not configured, a default hint naming the buffer and the
current date and time is used instead.

**Configure it** -- `action_config` holds:

- `prompt` -- optional. The notification text delivered to you.
  If omitted, a default hint is used instead, naming the buffer and the
  current date and time.
- `description` -- optional. A hint explaining why this trigger exists.
- `persistent_thread_id` -- optional. If omitted, the current conversation
  thread is inferred automatically. Set this only if you know what you
  are doing -- a thread ID string redirects the notification to that
  specific conversation instead.

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
- The `action` must be a function known to the FunctionManager or a built-in action (!notify, !read, !touch).
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


def _get_timer_timeout(
    config: dict[str, Any],
) -> float | str:
    """Extract the timer interval from the config.

    Returns the interval in seconds on success, or an error string if
    the value is absent or not a positive number.
    """
    # Read the interval from the config.
    interval = config.get("timer", None)

    # Guard against a missing interval.
    if interval is None:
        return "[Croner] timer interval is missing from config"

    # Guard against a non-numeric or non-positive value.
    if not isinstance(interval, (int, float)) or interval <= 0:
        return f"[Croner] timer interval must be a positive number, got {interval!r}"

    # Return the interval as a float.
    return float(interval)


def _get_alarm_timeout(
    config: dict[str, Any],
    origin: str,
) -> float | str:
    """Compute the timeout in seconds until the next alarm fires.

    Reads the time field from the config, combines it with today's date,
    and returns the positive difference in seconds. Returns an error
    string on failure.
    """
    # Parse the wall-clock time from the alarm config.
    time_str = config.get("time", "")
    try:
        fire_time = datetime.datetime.strptime(time_str, "%H:%M:%S").time()
    except ValueError:
        return f"[Croner] invalid time format in [alarm] for {origin}"

    # Compute the target datetime by combining today with the fire time.
    now = datetime.datetime.now()
    target = datetime.datetime.combine(now.date(), fire_time)

    # Compute the timeout as the difference from now to the target.
    timeout = (target - now).total_seconds()
    if timeout < 0:
        timeout += 86400

    return timeout


def _get_schedule_timeout(
    config: dict[str, Any],
    origin: str,
) -> float | str:
    """Compute the timeout in seconds until the next scheduled fire.

    Assembles the crontab fields into an expression, runs it through croniter,
    and returns the interval in seconds until the next match. Returns an
    error string on failure.
    """
    # Assemble the six crontab fields into a cron expression string for croniter.
    # Each field is required -- a missing field is a config error, not a silent default.
    fields = (
        config.get("second"),
        config.get("minute"),
        config.get("hour"),
        config.get("day"),
        config.get("month"),
        config.get("weekday"),
    )
    if None in fields:
        return f"[Croner] schedule config for {origin} is missing one or more required fields (second, minute, hour, day, month, weekday)"
    cron_expr = " ".join(str(f) for f in fields)

    # Use croniter to find the next occurrence from now and compute the timeout.
    try:
        cron = croniter(cron_expr, start_time=datetime.datetime.now(), second_at_beginning=True)
        next_fire = cron.get_next(datetime.datetime)
    except Exception as e:
        return f"[Croner] cannot parse or resolve schedule for {origin}  --  {e}"

    return (next_fire - datetime.datetime.now()).total_seconds()


def _validate_and_prepare_action_config(
    config_buffer: Buffer,
    croner: Croner,
) -> str | None:
    """Validate and prepare the action_config from the config buffer.

    Reads the buffer content, parses it as JSON, and for the !notify
    action injects the current thread ID if persistent_thread_id is absent.
    For any other action, validates that action_config is a dict if present.
    Returns None on success or an error string on failure.
    """
    # Read the config buffer content fresh.
    text = config_buffer.get_text()
    if not text or not text.strip():
        _logger.error(f"[_validate_and_prepare_action_config] '{config_buffer.name}' is empty")
        return "config buffer is empty"

    # Parse the JSON content.
    try:
        config = json.loads(text)
    except json.JSONDecodeError as e:
        _logger.error(f"[_validate_and_prepare_action_config] '{config_buffer.name}' contains malformed JSON: {e}")
        return f"config buffer is not valid JSON: {e}"

    # Guard against non-dict content.
    if not isinstance(config, dict):
        _logger.error(f"[_validate_and_prepare_action_config] '{config_buffer.name}' must be a JSON object")
        return "config buffer must be a JSON object"

    # Extract the action name to decide how to validate action_config.
    action = config.get("action")

    # For !notify, inject the current thread ID if absent.
    if action == "!notify":
        # Validate that action_config exists and is a dict.
        action_config = config.get("action_config", {})
        if not isinstance(action_config, dict):
            return f"[!notify] action_config must be a dictionary -- got {type(action_config).__name__}"

        # Nothing to do if persistent_thread_id is already set.
        if "persistent_thread_id" in action_config:
            return None

        # Resolve the current conversation thread.
        current_thread_id = croner.get_current_thread_id()
        if current_thread_id is None:
            return f"[!notify] cannot determine current thread ID -- update rejected (not inside a running session?)"

        # Inject the resolved thread ID so the notification goes to the current conversation.
        action_config["persistent_thread_id"] = current_thread_id

        # Patch the buffer text in place before the update commits.
        patched_text = json.dumps(config, indent=2) + "\n"
        config_buffer.set_text(patched_text)
        return None

    # For any other action, validate action_config is a dict if present.
    action_config = config.get("action_config")
    if action_config is not None and not isinstance(action_config, dict):
        return f"[{action}] action_config must be a dictionary -- got {type(action_config).__name__}"

    return None


async def _do_notify(
    croner: Croner,
    job: _Timer | _Alarm | _Schedule,
) -> str | None:
    """Fire the !notify action for the given job.

    Reads the action_config from the job's option buffer, extracts the prompt,
    resolves the persistent_thread_id, and delivers the notification via
    the croner. Returns an error string on failure, None on success.
    """
    # Determine the config buffer name from the job type.
    if isinstance(job, _Timer):
        config_name = f"{job.origin}[timer]"
    elif isinstance(job, _Alarm):
        config_name = f"{job.origin}[alarm]"
    else:
        config_name = f"{job.origin}[schedule]"

    # Read the config from the option buffer.
    config = _read_option_buffer_config(croner, config_name)
    if config is None:
        return f"[!notify] cannot read config from {config_name}"

    # Extract the action_config.
    action_config = config.get("action_config")
    if not isinstance(action_config, dict):
        return f"[!notify] action_config must be a dictionary -- got {type(action_config).__name__}"

    # Extract the prompt -- use a default hint if absent.
    prompt = action_config.get("prompt")
    if not prompt:
        now = datetime.datetime.now()
        prompt = f"[{job.origin}] fired at {now.strftime('%Y-%m-%d %H:%M:%S')}"

    # Extract the persistent_thread_id -- may be absent (already injected by the validation helper).
    persistent_thread_id = action_config.get("persistent_thread_id")

    # Deliver the notification via the croner.
    try:
        await croner.notify_agent(prompt, persistent_thread_id=persistent_thread_id)
    except Exception as e:
        return f"[!notify] failed to deliver notification: {e}"

    return None


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

    elif action == "!notify":
        # Deliver the notification using the dedicated helper.
        error = await _do_notify(croner, job)
        if error:
            _logger.error(f"[_fire_action] !notify failed for {job.origin}: {error}")

    else:
        # Call the custom function with the origin buffer name as the argument.
        await croner.call_function(action, buffer_name=job.origin)

    # Drop one-shot event buffers.
    if isinstance(job, _Timer) and not job.repeat:
        await croner.drop_buffer(f"{job.origin}[timer]")
        

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

        # Rearm recurring events.
        if not isinstance(job, _Timer) or job.repeat:
            _dispatch_rearm(croner, job)

    except asyncio.CancelledError:
        # On cancellation: reset only if the job still holds the task we started with.
        if job.task is active_task:
            job.timeout = None
            job.task = None


def _rearm_timer(
    croner: Croner,
    job: _Timer,
) -> str | None:
    """Rearm the timer task from the job object.

    Reads the [timer] config, sets the timeout on the job, and starts a new task.
    Returns an error string on failure, None on success.
    """
    # Read and parse the [timer] config, then abort if unavailable.
    config = _read_option_buffer_config(croner, f"{job.origin}[timer]")
    if config is None:
        return "[Croner] timer config not found"

    # Delegate the interval extraction to the dedicated helper.
    interval_or_error = _get_timer_timeout(config)
    if not isinstance(interval_or_error, float):
        return interval_or_error

    # Assign the interval timeout and schedule the event loop as a new task.
    job.timeout = interval_or_error
    job.task = asyncio.create_task(_wait_for(croner, job))


def _rearm_alarm(
    croner: Croner,
    job: _Alarm,
) -> str | None:
    """Rearm the alarm task from the job object.

    Reads the [alarm] config, computes the timeout until the next fire time, and
    starts a new asyncio task. Returns an error string on failure, None on success.
    """
    # Read and parse the [alarm] config, then abort if unavailable.
    config = _read_option_buffer_config(croner, f"{job.origin}[alarm]")
    if config is None:
        return "[Croner] alarm config not found"

    # Delegate the timeout computation to the dedicated helper.
    timeout_or_error = _get_alarm_timeout(config, job.origin)
    if not isinstance(timeout_or_error, float):
        return timeout_or_error

    # Assign the timeout and schedule the event loop as a new task.
    job.timeout = timeout_or_error
    job.task = asyncio.create_task(_wait_for(croner, job))


def _rearm_schedule(
    croner: Croner,
    job: _Schedule,
) -> str | None:
    """Rearm the schedule task from the job object.

    Reads the [schedule] config, computes the timeout until the next fire time, and
    starts a new asyncio task. Returns an error string on failure, None on success.
    """
    # Read and parse the [schedule] config, then abort if unavailable.
    config = _read_option_buffer_config(croner, f"{job.origin}[schedule]")
    if config is None:
        return "[Croner] schedule config not found"

    # Delegate the timeout computation to the dedicated helper.
    timeout_or_error = _get_schedule_timeout(config, job.origin)
    if not isinstance(timeout_or_error, float):
        return timeout_or_error

    # Assign the timeout and schedule the event loop as a new task.
    job.timeout = timeout_or_error
    job.task = asyncio.create_task(_wait_for(croner, job))


def _dispatch_rearm(
    croner: Croner,
    job: _Timer | _Alarm | _Schedule,
) -> str | None:
    """Dispatch to the correct rearm function based on the job type.

    Cancels any already-running task and replaces it with a fresh one.
    Returns an error string if the rearm failed, None on success.
    """
    # Route to the correct rearm function and return any error.
    if isinstance(job, _Timer):
        return _rearm_timer(croner, job)
    if isinstance(job, _Alarm):
        return _rearm_alarm(croner, job)
    else:
        return _rearm_schedule(croner, job)


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

    # Validate and prepare the action_config from the buffer.
    prepare_error = _validate_and_prepare_action_config(new_buffer, croner)
    if prepare_error:
        return prepare_error

    # Validate the timer interval using the dedicated helper.
    interval_or_error = _get_timer_timeout(job)
    if not isinstance(interval_or_error, float):
        return interval_or_error

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

    # Validate and prepare the action_config from the buffer.
    prepare_error = _validate_and_prepare_action_config(new_buffer, croner)
    if prepare_error:
        return prepare_error

    # Validate the alarm time using the dedicated helper.
    origin = new_buffer.name.removesuffix("[alarm]")
    timeout_or_error = _get_alarm_timeout(job, origin)
    if not isinstance(timeout_or_error, float):
        return timeout_or_error

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

    # Validate and prepare the action_config from the buffer.
    prepare_error = _validate_and_prepare_action_config(new_buffer, croner)
    if prepare_error:
        return prepare_error

    # Validate the crontab fields using the dedicated helper.
    origin = new_buffer.name.removesuffix("[schedule]")
    timeout_or_error = _get_schedule_timeout(job, origin)
    if not isinstance(timeout_or_error, float):
        return timeout_or_error

    # Reject any unknown top-level key.
    valid_fields = {"second", "minute", "hour", "day", "month", "weekday", "action_config"}
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
    error = _dispatch_rearm(croner, timer_job)
    if error:
        _logger.error(error)


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
    error = _dispatch_rearm(croner, croner._cron_alarms[origin])
    if error:
        _logger.error(error)


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
    error = _dispatch_rearm(croner, croner._cron_schedules[origin])
    if error:
        _logger.error(error)


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

