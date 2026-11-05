from __future__ import annotations

import datetime
import json
import logging
import re
from pathlib import Path
from typing import Any

from .buffer_manager import (
    Buffer,
    BufferManager,
    HookPoint,
)

_logger = logging.getLogger("option_buffer_manager")

_OPTION_BUFFER_PATTERN = re.compile(r"^(.+)\[(.+)\]$")
_OPTION_BUFFER_DOC_BUFFER = "doc:option_buffers"
_OPTION_BUFFER_DOC = """\
## Option Buffers

Option buffers are subordinate buffers that exist only alongside an origin buffer.
Their names carry a trailing [...]: ORIGIN[option].

## Invariants

- An option buffer may only be created when its origin buffer already exists.
- Dropping an origin buffer automatically drops all its option buffers.

## System Buffers

system:status:buffers  — live listing of all buffers (including option buffers)
"""


# -------------------------------------------------------------------------------------------------------------------- #
# Utils                                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #


def _origin_name_from_option_buffer_name(name: str) -> str | None:
    """Return the origin buffer name if name is an option buffer (has [...]), else None."""
    match = _OPTION_BUFFER_PATTERN.fullmatch(name)
    if not match:
        return None
    return match.group(1)


def _option_type_from_option_buffer_name(name: str) -> str | None:
    """Return the option type (the content inside [...]) if name is an option buffer, else None."""
    match = _OPTION_BUFFER_PATTERN.fullmatch(name)
    if not match:
        return None
    return match.group(2)


def _read_option_buffer_config(option_buffer_manager: OptionBufferManager, option_buffer_name: str) -> dict[str, Any] | None:
    """Read and parse the JSON config from an option buffer.

    Returns the parsed config dict on success, or None if the buffer is absent,
    empty, or contains invalid JSON.
    """
    # Locate the option buffer by name.
    option_buffer = option_buffer_manager.get_buffer(option_buffer_name)
    if option_buffer is None:
        _logger.error(f"[OptionBufferManager] cannot locate {option_buffer_name}")
        return None

    # Extract the text from the option buffer.
    option_buffer_text = option_buffer.get_text()
    if not option_buffer_text:
        _logger.error(f"[OptionBufferManager] {option_buffer_name} is empty")
        return None

    # Parse the JSON config from the option buffer text.
    try:
        config = json.loads(option_buffer_text)
    except json.JSONDecodeError:
        _logger.error(f"[OptionBufferManager] {option_buffer_name} has invalid JSON")
        return None

    return config


def validate_and_prepare_action_config(
    option_buffer_manager: OptionBufferManager,
    config_buffer: Buffer,
    buffer_config: dict,
    config: dict,
) -> str | None:
    """Validate and prepare the action_config for a single action.

    The `config_buffer` holds one or more actions as a JSON object.
    The `buffer_config` is the full parsed config of that buffer.
    The `config` is the sub-dict for the specific action being processed.
    When this function updates `config`, it writes the modified
    `buffer_config` back to `config_buffer` to persist the change.
    Returns None on success or an error string on failure.
    """
    # Extract the action name to decide how to validate action_config.
    action = config.get("name")
    if not action:
        return f"[OptionBufferManager] action config is missing a 'name' field"

    # For !notify, inject the current thread ID if absent.
    if action == "!notify":
        # Fetch the existing action_config or create a new one and store it in config
        # so that the subsequent mutation actually persists into the buffer.
        action_config = config.get("action_config")
        if action_config is None:
            action_config = {}
            config["action_config"] = action_config

        # Validate that action_config is a dict.
        if not isinstance(action_config, dict):
            return f"[!notify] action_config must be a dictionary -- got {type(action_config).__name__}"

        # Nothing to do if persistent_thread_id is already set.
        if "persistent_thread_id" in action_config:
            return None

        # Resolve the current conversation thread.
        current_thread_id = option_buffer_manager.get_current_thread_id()
        if current_thread_id is None:
            return f"[!notify] cannot determine current thread ID -- update rejected (not inside a running session?)"

        # Inject the resolved thread ID so the notification goes to the current conversation.
        action_config["persistent_thread_id"] = current_thread_id

        # Patch the buffer text in place before the update commits.
        patched_text = json.dumps(buffer_config, indent=2) + "\n"
        config_buffer.set_text(patched_text)
        return None

    # For any other action, validate action_config is a dict if present.
    action_config = config.get("action_config")
    if action_config is not None and not isinstance(action_config, dict):
        return f"[{action}] action_config must be a dictionary -- got {type(action_config).__name__}"

    return None


async def _do_notify(
    option_buffer_manager: OptionBufferManager,
    action_config: dict[str, Any],
    origin: str,
) -> str | None:
    """Fire the !notify action for the given action_config.

    Extracts the prompt, resolves the persistent_thread_id, and delivers
    the notification via the option_buffer_manager. Returns an error
    string on failure, None on success.
    """
    # Extract the prompt -- use a default hint if absent.
    prompt = action_config.get("prompt")
    if not prompt:
        # Build a two-line default: timestamp header, then the buffer content or a
        # truncation hint when the content exceeds the max-chunk threshold.
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

        # Read the origin buffer to include its content in the notification.
        buffer = option_buffer_manager.get_buffer(origin)
        if buffer is not None:
            content = buffer.get_text() or ""
        else:
            _logger.error(f"[_do_notify] origin buffer '{origin}' not found")
            content = ""

        # Choose the full content or a truncation hint based on the max-chunk threshold.
        if len(content) > option_buffer_manager._MAX_CHUNK_CHARS:
            # Content exceeds the threshold -- provide a hint instead of the full content.
            prompt = f"[notification] fired at {timestamp} on buffer '{origin}' (content exceeds {option_buffer_manager._MAX_CHUNK_CHARS} chars)"
        else:
            prompt = f"[notification] fired at {timestamp} on buffer '{origin}'\n\n{content}"

    # Extract the persistent_thread_id -- may already be injected.
    persistent_thread_id = action_config.get("persistent_thread_id")

    # Deliver the notification via the option_buffer_manager.
    try:
        await option_buffer_manager.notify_agent(prompt, persistent_thread_id=persistent_thread_id)
    except Exception as e:
        return f"[!notify] failed to deliver notification: {e}"

    return None


async def _fire_action(
    option_buffer_manager: OptionBufferManager,
    config: dict[str, Any],
    origin: str,
) -> None:
    """Fire the action described by the action sub-dict on the origin buffer.

    Reads the action field from config, then dispatches to a built-in action
    (!read, !touch, !notify) or calls a custom function via the
    option_buffer_manager's function manager.
    """
    # Guard against a missing or empty action field.
    action = config.get("name")
    if not action:
        _logger.error(f"[OptionBufferManager] no action in config for '{origin}'")
        return

    # Dispatch to a built-in action or treat the action as a custom function name.
    if action == "!read":
        result = await option_buffer_manager.read_buffer(origin)
        if not isinstance(result, dict) or not result.get("ok"):
            _logger.error(f"[_fire_action] !read failed for '{origin}'")

    elif action == "!touch":
        result = await option_buffer_manager.write_buffer(origin, text="")
        if not isinstance(result, dict) or not result.get("ok"):
            _logger.error(f"[_fire_action] !touch failed for '{origin}'")

    elif action == "!notify":
        action_config = config.get("action_config")
        if isinstance(action_config, dict):
            error = await _do_notify(option_buffer_manager, action_config, origin)
            if error:
                _logger.error(f"[_fire_action] !notify failed for '{origin}': {error}")

    else:
        result = await option_buffer_manager.call_function(action, buffer_name=origin)
        if not isinstance(result, dict) or not result.get("ok"):
            _logger.error(f"[_fire_action] custom action '{action}' failed for '{origin}'")


# -------------------------------------------------------------------------------------------------------------------- #
# Hooks                                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #


def _reject_creation_if_origin_missing(
    buffer_manager: BufferManager,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Pre-create guard: reject option buffer creation if the origin does not exist."""
    origin_name = _origin_name_from_option_buffer_name(buffer.name)
    if origin_name is None:
        return None  # Not an option buffer — allow creation to proceed.

    # Reject if the origin buffer does not exist in the manager.
    if buffer_manager.get_buffer(origin_name) is None:
        return (
            f"Cannot create option buffer '{buffer.name}': "
            f"origin buffer '{origin_name}' does not exist."
        )
    return None  # Origin exists — allow creation.


def _register_origin_cascade_drop(
    buffer_manager: BufferManager,
    buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Post-create hook: register a cascade-drop hook on the origin buffer so the option is dropped when the origin is dropped."""
    origin_name = _origin_name_from_option_buffer_name(buffer.name)
    if origin_name is None:
        _logger.error(f"[_register_origin_cascade_drop] buffer '{buffer.name}' does not match the option buffer pattern — hook should not have been invoked")
        return None  # Not an option buffer — nothing to register.

    # Resolve the origin buffer — it must exist since the pre-create guard passed.
    origin = buffer_manager.get_buffer(origin_name)
    if origin is None:
        _logger.error(f"[_register_origin_cascade_drop] origin '{origin_name}' not found for option buffer '{buffer.name}'")
        return None

    # Capture the full option buffer name in the closure — this is what gets dropped.
    option_name = buffer.name

    async def _cascade_drop_option(
        hook_buffer_manager: BufferManager,
        origin_buffer: Buffer,
        args: dict[str, Any],
    ) -> None:
        """Drop the option buffer when the origin buffer is dropped."""
        if hook_buffer_manager.get_buffer(option_name) is not None:
            await hook_buffer_manager.drop_buffer(option_name)

    # Register the cascade-drop hook on the origin buffer's AFTER_DROP_BUFFER hook point.
    origin.register_hook(
        HookPoint.AFTER_DROP_BUFFER,
        f"option_cascade_{option_name}",
        _cascade_drop_option,
    )


class OptionBufferManager(BufferManager):
    """Option buffers (named ORIGIN[option]) always require an existing origin buffer, and are dropped automatically when the origin is dropped. See doc:option_buffers for details."""

    def __init__(self, **kwargs: Any) -> None:
        # Initialize the base buffer manager.
        super().__init__(**kwargs)

        # Seed the schema documentation buffer.
        self._create_buffer(_OPTION_BUFFER_DOC_BUFFER, text=_OPTION_BUFFER_DOC)

        # Register the pattern hooks that enforce the option-buffer/origin invariant.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "option_buffer_requires_origin",
            r"^.+\[.+\]$",
            _reject_creation_if_origin_missing,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "option_buffer_cascade_drop",
            r"^.+\[.+\]$",
            _register_origin_cascade_drop,
        )
