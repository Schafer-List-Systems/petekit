from __future__ import annotations

import functools
import inspect
import json
import logging
import time
from typing import Any
from typing import Callable
from typing import Dict

from .buffer_manager import (
    Buffer,
    BufferManager,
    HookPoint,
)
from .option_buffer_manager import (
    OptionBufferManager,
    _origin_name_from_option_buffer_name,
    _option_type_from_option_buffer_name,
)

from petekit.function.function_manager import FunctionManager
from peteos.oap.agentic_object import AgenticObject

_logger = logging.getLogger("smart_buffer_manager")

_SMART_BUFFER_DOC_BUFFER = "doc:buffer_hooks"

_BEFORE_UPDATE_SUFFIX = "before_update"
_AFTER_UPDATE_SUFFIX = "after_update"
_BEFORE_DROP_SUFFIX = "before_drop"
_AFTER_DROP_SUFFIX = "after_drop"

_SMART_BUFFER_PREFIX = "smart_buffer_manager."
_SMART_BUFFER_MANAGER_HOOK_NAME = "smart_buffer_manager"
_SMART_BUFFER_HOOK_PATTERN = r"^.+\[(before_update|after_update|before_drop|after_drop)\]$"

_OPTION_TYPE_TO_HOOK_POINT: dict[str, HookPoint] = {
    _BEFORE_UPDATE_SUFFIX: HookPoint.BEFORE_UPDATE_BUFFER,
    _AFTER_UPDATE_SUFFIX: HookPoint.AFTER_UPDATE_BUFFER,
    _BEFORE_DROP_SUFFIX: HookPoint.BEFORE_DROP_BUFFER,
    _AFTER_DROP_SUFFIX: HookPoint.AFTER_DROP_BUFFER,
}


_SMART_BUFFER_DOC = f"""\
## Smart Buffer Manager

You can register reactions on update or drop events that fire on buffers.
Reactions execute before or after the event occurs.
A reaction can reject the event from happening.

You register a reaction by creating an option buffer on the origin buffer.
The four option buffers per origin are `ORIGIN[{_BEFORE_UPDATE_SUFFIX}]`,
`ORIGIN[{_AFTER_UPDATE_SUFFIX}]`, `ORIGIN[{_BEFORE_DROP_SUFFIX}]`, and `ORIGIN[{_AFTER_DROP_SUFFIX}]`.
Each holds a JSON dictionary mapping reaction keys to their configuration.

### Reaction Types

A reaction is either a **hook function** or a **built-in notification**.

#### Hook Function

A named function from the function manager. The option buffer maps the
function name to a description string:

```json
{{
  "my_listener": "alerts the team of changes"
}}
```

The function must be registered in the function manager before use.

#### Built-In Notification

The special key `!notify` delivers the configured prompt to you. The option buffer
maps `!notify` to a notification config dict:

```json
{{
  "!notify": {{
    "prompt": "the origin buffer has changed",
    "description": "notifies the agent of changes to this buffer"
  }}
}}
```

- `prompt` -- the notification text delivered to you
- `persistent_thread_id` -- if omitted, your current conversation receives
  the prompt automatically. Set this only if you know what you are doing --
  `null` redirects the prompt to a fresh ephemeral context; a thread ID
  string redirects it to that specific conversation instead.
- `description` -- a hint explaining why this reaction exists

### Behaviour

`{_BEFORE_UPDATE_SUFFIX}` -- pre-update: fires before the origin buffer is updated.
A hook function can inspect the old and new buffers and reject the update.
You cannot use `!notify` here -- it has no way to block the event.

`{_AFTER_UPDATE_SUFFIX}`  -- post-update: fires after the origin buffer is updated.
You can use a hook function or `!notify` here.

`{_BEFORE_DROP_SUFFIX}`   -- pre-drop: fires before the origin buffer is dropped.
A hook function can inspect the buffer and reject the drop.
You cannot use `!notify` here -- it has no way to block the event.

`{_AFTER_DROP_SUFFIX}`    -- post-drop: fires after the origin buffer is dropped.
You can use a hook function or `!notify` here.

### Function Signatures (Hook Function Reactions)

Hook functions registered via option buffers must follow one of these signatures:

#### UPDATE hooks -- fire on buffer updates (4 parameters)

Hook fires with: `(buffer_manager, old_buffer, new_buffer, args)`

Valid signatures:

  - `(buffer_manager, Buffer, Buffer, Dict)`  -- receives Buffer objects directly
  - `(buffer_manager, str, str, Dict)`        -- receives buffer text as strings

#### DROP hooks -- fire on buffer drops (3 parameters)

Hook fires with: `(buffer_manager, buffer, args)`

Valid signatures:

  - `(buffer_manager, Buffer, Dict)`  -- receives Buffer object directly
  - `(buffer_manager, str, Dict)`     -- receives empty string (buffer already dropped)

Functions are looked up via the function manager -- register your function there
before referencing it in an option buffer.

### Setup Example

Suppose you have a dashboard view that needs to stay in sync with `MYBUFFER`
and an audit log that must record every change to it.

1. Ensure `update_dashboard` and `log_change` exist in the function manager.
2. Create the option buffer `MYBUFFER[after_update]` with the function name as key:

```json
{{
  "update_dashboard": "refreshes the dashboard view after changes",
  "log_change": "records the change in the audit log"
}}
```

3. The system registers each function as a post-update hook on `MYBUFFER` automatically.

You can also attach a notification reaction:

```json
{{
  "!notify": {{
    "prompt": "MYBUFFER was updated",
    "description": "alerts me whenever this buffer changes"
  }}
}}
```

To remove a reaction, remove its entry from the option buffer.
"""


# -------------------------------------------------------------------------------------------------------------------- #
# Utils                                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #


def _make_notify_hook_for_update(
    option_buffer: Buffer,
) -> Callable:
    """Factory that creates a notify hook for update hook points.

    The returned closure has signature (bm, old_buffer, new_buffer, args) -> None.
    It reads the `!notify` config from the option buffer at fire time and
    delivers the notification to the agent.
    """
    async def notify_hook_for_update(
        smart_buffer_manager: Any,
        old_buffer: Buffer,
        new_buffer: Buffer,
        args: dict[str, Any],
    ) -> None:
        # Read the option buffer content fresh and parse it as JSON.
        try:
            config = json.loads(option_buffer.get_text())
        except json.JSONDecodeError as e:
            _logger.error(f"[notify_hook_for_update] failed to parse option buffer '{option_buffer.name}' as JSON: {e}")
            return

        # Extract the `!notify` reaction config from the parsed JSON.
        reaction_config = config.get("!notify")
        if reaction_config is None:
            _logger.error(f"[notify_hook_for_update] '!notify' key not found in option buffer '{option_buffer.name}'")
            return

        # Reject if `!notify` is present but not a dictionary.
        if not isinstance(reaction_config, dict):
            _logger.error(f"[notify_hook_for_update] '!notify' in '{option_buffer.name}' must be a dict -- got {type(reaction_config).__name__}")
            return

        # Extract the prompt -- this is required.
        prompt = reaction_config.get("prompt")
        if not prompt:
            _logger.error(f"[notify_hook_for_update] 'prompt' not found in '!notify' config of '{option_buffer.name}'")
            return

        # Use the configured persistent_thread_id, or None if absent.
        persistent_thread_id = reaction_config.get("persistent_thread_id")
        try:
            await smart_buffer_manager.notify_agent(prompt, persistent_thread_id=persistent_thread_id)
        except Exception as exc:
            _logger.error(f"[notify_hook_for_update] failed to notify agent: {exc}")

    return notify_hook_for_update


def _make_notify_hook_for_create_or_drop(
    option_buffer: Buffer,
) -> Callable:
    """Factory that creates a notify hook for create or drop hook points.

    The returned closure has signature (bm, buffer, args) -> None.
    It reads the `!notify` config from the option buffer at fire time and
    delivers the notification to the agent.
    """
    async def notify_hook_for_create_or_drop(
        smart_buffer_manager: Any,
        buffer: Buffer,
        args: dict[str, Any],
    ) -> None:
        # Read the option buffer content fresh and parse it as JSON.
        try:
            config = json.loads(option_buffer.get_text())
        except json.JSONDecodeError as e:
            _logger.error(f"[notify_hook_for_create_or_drop] failed to parse option buffer '{option_buffer.name}' as JSON: {e}")
            return

        # Extract the `!notify` reaction config from the parsed JSON.
        reaction_config = config.get("!notify")
        if reaction_config is None:
            _logger.error(f"[notify_hook_for_create_or_drop] '!notify' key not found in option buffer '{option_buffer.name}'")
            return

        # Reject if `!notify` is present but not a dictionary.
        if not isinstance(reaction_config, dict):
            _logger.error(f"[notify_hook_for_create_or_drop] '!notify' in '{option_buffer.name}' must be a dict -- got {type(reaction_config).__name__}")
            return

        # Extract the prompt -- this is required.
        prompt = reaction_config.get("prompt")
        if not prompt:
            _logger.error(f"[notify_hook_for_create_or_drop] 'prompt' not found in '!notify' config of '{option_buffer.name}'")
            return

        # Use the configured persistent_thread_id, or None if absent.
        persistent_thread_id = reaction_config.get("persistent_thread_id")
        try:
            await smart_buffer_manager.notify_agent(prompt, persistent_thread_id=persistent_thread_id)
        except Exception as exc:
            _logger.error(f"[notify_hook_for_create_or_drop] failed to notify agent: {exc}")

    return notify_hook_for_create_or_drop


def _unregister_reaction(
    origin_buffer: Buffer,
    hook_point: HookPoint,
    reaction_name: str,
) -> None:
    """Unregister a single reaction hook from the origin buffer."""
    origin_buffer.unregister_hook(hook_point, f"{_SMART_BUFFER_PREFIX}{reaction_name}")


def _register_reaction(
    smart_buffer_manager: Any,
    origin_buffer: Buffer,
    hook_point: HookPoint,
    reaction_name: str,
    option_buffer: Buffer,
) -> None:
    """Resolve a reaction name to its callable and register it as a hook on the origin buffer.

    For the built-in `!notify` key, the factory creates a notify hook and registers
    it directly. For named functions, the function manager resolves the callable.
    """
    # Resolve the callable for this reaction.
    if reaction_name == "!notify":
        # The built-in `!notify` uses a factory that reads the option buffer at fire time.
        if hook_point in (HookPoint.BEFORE_UPDATE_BUFFER, HookPoint.AFTER_UPDATE_BUFFER):
            callable_hook = _make_notify_hook_for_update(option_buffer)
        else:
            callable_hook = _make_notify_hook_for_create_or_drop(option_buffer)

        # Register the notify hook directly on the origin buffer.
        origin_buffer.register_hook(hook_point, f"{_SMART_BUFFER_PREFIX}{reaction_name}", callable_hook)
    else:
        # Look up the registered function from the function manager.
        result = smart_buffer_manager.get_function(reaction_name)
        if not result["ok"]:
            _logger.error(f"[_register_reaction] function '{reaction_name}' not found in function manager -- this should never happen")
            return
        callable_hook = result["callable"]

        # Register the resolved callable as a hook on the origin buffer.
        _register_buffer_hook(origin_buffer, hook_point, reaction_name, callable_hook)


def _get_option_buffer_content(
    option_buffer: Buffer | None,
) -> dict[str, Any]:
    """Return the parsed JSON content of an option buffer, or an empty dict if the buffer does not exist."""
    if option_buffer is None:
        return {}
    return json.loads(option_buffer.get_text())


def _compute_hook_diffs(
    old_option_content: dict[str, Any],
    new_option_content: dict[str, Any],
) -> tuple[set[str], set[str]]:
    """Compute which hook functions were added and which were removed between two option buffer states.

    Takes the old and new parsed JSON content from option buffers.
    Returns the set of functions removed (old \\ new) and the set of functions added (new \\ old).
    """
    # Compute the keys to remove (old \ new) and add (new \ old).
    remove_hooks = set(old_option_content) - set(new_option_content)
    add_hooks = set(new_option_content) - set(old_option_content)
    return remove_hooks, add_hooks


def _validate_hook_signature(
    params: list[tuple[str, Any]],
    data_param_indices: list[int],
    args_param_index: int,
    expected_param_count: int,
    function_name: str,
) -> str | None:
    """Validate that a hook callable has the expected signature.

    Args:
        params: List of (name, Parameter) tuples from inspect.signature().
        data_param_indices: Which param indices carry the Buffer or str data.
            All of these must be annotated as either Buffer or str, and all of them
            must be the SAME type (all Buffer or all str).
        args_param_index: Which param index carries the dict args. Must be annotated
            as dict (or a string variant thereof).
        expected_param_count: Total number of parameters expected.
        function_name: Name of the function for use in error messages.

    Returns:
        None if the signature is valid, an error string describing the mismatch otherwise.
    """
    # Validate total param count.
    if len(params) != expected_param_count:
        return f"[{function_name}] expected {expected_param_count} parameters, got {len(params)}"

    # Validate the args param is annotated as dict.
    args_annotation = params[args_param_index][1].annotation
    if args_annotation not in (dict, "dict", Dict, "Dict"):
        return f"[{function_name}] type hint for parameter {args_param_index + 1} is {args_annotation!r} -- expected dict"

    # Collect the annotations of the data params.
    data_annotations = {params[i][1].annotation for i in data_param_indices}

    # All data params must be either Buffer or str -- not a mix of both.
    if not data_annotations.issubset({Buffer, "Buffer"}) and not data_annotations.issubset({str, "str"}):
        # Format the unexpected annotations for the error message.
        seen = ", ".join(f"'{a}'" for a in sorted(data_annotations))
        return f"[{function_name}] data params {data_param_indices} have types [{seen}] -- expected all Buffer or all str"

    return None


def _get_data_param_mode(
    params: list[tuple[str, Any]],
    data_param_indices: list[int],
) -> str:
    """Return 'buffer' if data params are Buffer-annotated, 'str' if str-annotated.

    Call this only after _validate_hook_signature has confirmed all data params share one type.
    """
    # Inspect the first data param to determine the shared mode.
    first_annotation = params[data_param_indices[0]][1].annotation
    return "buffer" if first_annotation in (Buffer, "Buffer") else "str"


def _register_buffer_hook(
    origin_buffer: Buffer,
    hook_point: HookPoint,
    function_name: str,
    callable_hook: Callable,
) -> str | None:
    # Inspect the callable's signature to decide direct registration or text-wrapping.
    sig = inspect.signature(callable_hook)
    params = list(sig.parameters.items())

    # Validate the signature -- fail fast if unexpected.
    # Delegate to the generic validator with the right param layout for each hook family.
    if hook_point in (HookPoint.BEFORE_UPDATE_BUFFER, HookPoint.AFTER_UPDATE_BUFFER):
        # UPDATE: data params are [1, 2] (old and new buffer), args is [3].
        error = _validate_hook_signature(
            params=params,
            data_param_indices=[1, 2],
            args_param_index=3,
            expected_param_count=4,
            function_name=function_name,
        )
        if error:
            raise RuntimeError(f"[{function_name}] signature mismatch at registration -- this should have been caught by the guard")

        # Determine the data param mode to decide direct vs. wrapped registration.
        data_mode = _get_data_param_mode(params, [1, 2])

        # Buffer-annotated: register directly -- no conversion needed.
        if data_mode == "buffer":
            origin_buffer.register_hook(hook_point, f"{_SMART_BUFFER_PREFIX}{function_name}", callable_hook)

        # str-annotated: wrap to extract buffer text before forwarding.
        else:
            # Wrap the str-annotated callable so it receives text instead of buffer objects.
            def wrapped_update_hook(bm: Any, old_buffer: Buffer, new_buffer: Buffer, args: dict[str, Any]) -> Any:
                return callable_hook(bm, old_buffer.get_text(), new_buffer.get_text(), args)
            origin_buffer.register_hook(hook_point, f"{_SMART_BUFFER_PREFIX}{function_name}", wrapped_update_hook)

    elif hook_point in (HookPoint.BEFORE_DROP_BUFFER, HookPoint.AFTER_DROP_BUFFER):
        # DROP: data param is [1] (the buffer being dropped), args is [2].
        error = _validate_hook_signature(
            params=params,
            data_param_indices=[1],
            args_param_index=2,
            expected_param_count=3,
            function_name=function_name,
        )
        if error:
            raise RuntimeError(f"[{function_name}] signature mismatch at registration -- this should have been caught by the guard")

        # Determine the data param mode to decide direct vs. wrapped registration.
        data_mode = _get_data_param_mode(params, [1])

        # Buffer-annotated: register directly -- buffer is still valid at fire time.
        if data_mode == "buffer":
            origin_buffer.register_hook(hook_point, f"{_SMART_BUFFER_PREFIX}{function_name}", callable_hook)

        # str-annotated: the buffer is already gone -- pass empty string as the data value.
        else:
            # Wrap the str-annotated callable -- buffer is gone at fire time, so forward empty string.
            def wrapped_drop_hook(bm: Any, buffer: Buffer, args: dict[str, Any]) -> Any:
                return callable_hook(bm, "", args)
            origin_buffer.register_hook(hook_point, f"{_SMART_BUFFER_PREFIX}{function_name}", wrapped_drop_hook)


# -------------------------------------------------------------------------------------------------------------------- #
# Hooks                                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #


def before_update_option_buffer(
    smart_buffer_manager: Any,
    old_option_buffer: Buffer | None,
    new_option_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Validate an option buffer update before it is applied.

    Checks that the new option buffer contains valid JSON and that every
    function named in the config is registered in the function manager and
    has a signature compatible with the target hook point.
    Returns an error message if validation fails (blocking the update), or
    None to allow the update to proceed.
    """
    # Read the new buffer content for validation.
    new_text = new_option_buffer.get_text()
    if not new_text:
        return  # Empty buffer -- allow.

    # Parse the JSON config to get the function names.
    try:
        config = json.loads(new_text)
    except json.JSONDecodeError as e:
        return f"[{new_option_buffer.name}] contains malformed JSON: {e} -- rejecting update"

    # Reject if any named function is not registered.
    for reaction_name in config:
        # The built-in `!notify` key is always allowed -- it does not need to be registered.
        if reaction_name == "!notify":
            continue

        # Reject the update if the named function is not registered in the function manager.
        result = smart_buffer_manager.get_function(reaction_name)
        if not result["ok"]:
            return f"Function '{reaction_name}' is not registered. Cannot apply before_update."

    # Get the old and new option buffer content as dictionaries.
    old_option_content = _get_option_buffer_content(old_option_buffer)
    new_option_content = _get_option_buffer_content(new_option_buffer)

    # If `!notify` is in the config and has no persistent_thread_id, inject the current thread ID.
    # Reject the update if no current thread ID is available (outside a running session).
    if "!notify" in new_option_content:
        reaction_config = new_option_content["!notify"]

        # Reject if the reaction config is not a dictionary.
        if not isinstance(reaction_config, dict):
            _logger.error(f"[before_update_option_buffer] '!notify' in '{new_option_buffer.name}' must be a dict -- got {type(reaction_config).__name__}")
            return f"[!notify] reaction config must be a dictionary -- got {type(reaction_config).__name__}"

        # Only inject if the config has no explicit thread ID.
        if "persistent_thread_id" not in reaction_config:
            # Resolve the current conversation thread.
            current = smart_buffer_manager.get_current_thread_id()
            if current is None:
                _logger.error(f"[before_update_option_buffer] cannot determine current thread ID for '!notify' in '{new_option_buffer.name}' -- update rejected (not inside a running session?)")
                return f"[!notify] cannot determine current thread ID -- update rejected (not inside a running session?)"

            # Inject the resolved thread ID so the notification goes to the current conversation.
            reaction_config["persistent_thread_id"] = current

            # Patch the buffer's lines in place before the update commits.
            patched_text = json.dumps(new_option_content, indent=2) + "\n"
            new_option_buffer.set_text(patched_text)

    # Validate signatures of all functions being registered.
    # Compute which functions are being added (new \ old) vs removed (old \ new).
    remove_hooks, add_hooks = _compute_hook_diffs(old_option_content, new_option_content)
    if not add_hooks:
        return  # No new functions -- nothing to validate.

    # Determine the hook point for this option type.
    option_type = _option_type_from_option_buffer_name(new_option_buffer.name)
    hook_point = _OPTION_TYPE_TO_HOOK_POINT[option_type]

    # For each function, verify the signature matches what this hook point expects.
    for reaction_name in add_hooks:
        # The built-in `!notify` key requires no function registration -- only config validation.
        if reaction_name == "!notify":
            # Extract the `!notify` config dict from the option buffer content.
            new_option_content = _get_option_buffer_content(new_option_buffer)
            reaction_config = new_option_content.get("!notify", {})

            # Reject if the config is not a dictionary -- it must be a JSON object.
            if not isinstance(reaction_config, dict):
                return f"[!notify] reaction config must be a dictionary -- got {type(reaction_config).__name__}"

            # Validate that only the expected keys are present -- prompt, persistent_thread_id, description.
            for key in reaction_config:
                if key not in ("prompt", "persistent_thread_id", "description"):
                    return f"[!notify] unexpected key '{key}' -- expected prompt, persistent_thread_id, or description"

            # Config is valid -- skip further function-based validation since `!notify` is a built-in.
            continue

        # The `!notify` key is not registered in the function manager -- it is a built-in.
        # Look up the registered function and extract its callable and signature for validation.
        result = smart_buffer_manager.get_function(reaction_name)
        callable_hook = result["callable"]
        sig = inspect.signature(callable_hook)
        params = list(sig.parameters.items())

        # Validate the signature against the hook point's expectations.
        if hook_point in (HookPoint.BEFORE_UPDATE_BUFFER, HookPoint.AFTER_UPDATE_BUFFER):
            # UPDATE hooks: params are (bm, old_buffer, new_buffer, args) -- data params are [1, 2].
            error = _validate_hook_signature(
                params=params,
                data_param_indices=[1, 2],
                args_param_index=3,
                expected_param_count=4,
                function_name=reaction_name,
            )
            if error:
                return error

        # Validate for DROP hooks.
        elif hook_point in (HookPoint.BEFORE_DROP_BUFFER, HookPoint.AFTER_DROP_BUFFER):
            # DROP hooks: params are (bm, buffer, args) -- data param is [1].
            error = _validate_hook_signature(
                params=params,
                data_param_indices=[1],
                args_param_index=2,
                expected_param_count=3,
                function_name=reaction_name,
            )
            if error:
                return error


def after_drop_origin_buffer(
    buffer_manager: BufferManager,
    origin_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    # Clean up all option buffers.
    for suffix in (_BEFORE_UPDATE_SUFFIX, _AFTER_UPDATE_SUFFIX, _BEFORE_DROP_SUFFIX, _AFTER_DROP_SUFFIX):
        # Get the option buffer.
        option_name = f"{origin_buffer.name}{suffix}"
        option_buffer = buffer_manager.get_buffer(option_name)

        # Guard: skip this suffix if no option buffer exists for it.
        if option_buffer is None:
            _logger.error(f"[after_drop_origin_buffer] option buffer '{option_name}' not found -- skipping")
            continue

        # Unregister the dropped hook from the option buffer.
        # If the hook is still active, dropping the option buffer would trigger after_drop_option_buffer,
        # which in turn would try to update the already-dropped origin buffer -- causing a crash or silent failure.
        # Silencing the hook first breaks that cascade.
        option_buffer.unregister_hook(HookPoint.AFTER_DROP_BUFFER, _SMART_BUFFER_MANAGER_HOOK_NAME)

        # Drop the option buffer.
        buffer_manager.drop_buffer(option_name)


def after_update_option_buffer(
    smart_buffer_manager: Any,
    old_option_buffer: Buffer | None,
    new_option_buffer: Buffer | None,
    args: dict[str, Any],
) -> None:
    """Read the option buffer and apply its configuration as hooks on the origin buffer.

    On creation old_option_buffer is None -- nothing to unregister.
    On update the old_option_buffer config is unregistered before the new one is registered.
    """
    # Guard -- validate this is actually a smart option buffer.
    option_buffer_name = (new_option_buffer.name if new_option_buffer is not None else old_option_buffer.name)
    option_type = _option_type_from_option_buffer_name(option_buffer_name)
    if option_type is None:
        _logger.error(f"[_update_origin_hooks] '{option_buffer_name}' is not a smart option buffer")
        return

    # Resolve the origin name from the option buffer name.
    origin_name = _origin_name_from_option_buffer_name(option_buffer_name)
    if origin_name is None:
        _logger.error(f"[_update_origin_hooks] cannot derive origin from '{option_buffer_name}'")
        return

    # Retrieve the origin buffer -- it must exist for hooks to be registered on it.
    origin_buffer = smart_buffer_manager.get_buffer(origin_name)
    if origin_buffer is None:
        _logger.warning(f"[_update_origin_hooks] origin '{origin_name}' not found")
        return

    # Get the old and new option buffer content.
    old_option_content = _get_option_buffer_content(old_option_buffer)
    new_option_content = _get_option_buffer_content(new_option_buffer)

    # Compute the keys to remove (old \ new) and add (new \ old).
    remove_hooks, add_hooks = _compute_hook_diffs(old_option_content, new_option_content)

    # Get the hook point on which to register / unregister the hooks.
    hook_point = _OPTION_TYPE_TO_HOOK_POINT[option_type]

    # Unregister hooks which have been removed from the option buffer.
    for reaction_name in remove_hooks:
        _unregister_reaction(origin_buffer, hook_point, reaction_name)

    # Register hooks which have been added from the option buffer.
    for reaction_name in add_hooks:
        _register_reaction(smart_buffer_manager, origin_buffer, hook_point, reaction_name, new_option_buffer)


# onBufferCreatedHook -- apply the new option buffer config to the origin immediately.
def after_create_option_buffer(
    buffer_manager: BufferManager,
    option_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    # Get the origin buffer to this option buffer.
    origin_name = _origin_name_from_option_buffer_name(option_buffer.name)
    origin_buffer = buffer_manager.get_buffer(origin_name)
    if origin_buffer is None:
        _logger.error(f"[after_create_option_buffer] origin '{origin_name}' not found")
        return

    # Register dropped sentinel on origin buffer, which drops the option buffer too.
    origin_buffer.register_hook(
        HookPoint.AFTER_DROP_BUFFER,
        _SMART_BUFFER_MANAGER_HOOK_NAME,
        after_drop_origin_buffer,
    )

    # Seed the initial hooks from the option buffer.
    after_update_option_buffer(buffer_manager, None, option_buffer, {})


# -------------------------------------------------------------------------------------------------------------------- #
# Main Class                                                                                                           #
# -------------------------------------------------------------------------------------------------------------------- #


class SmartBufferManager(FunctionManager, OptionBufferManager, AgenticObject):
    """I can register hooks (functions) on before or after changes to buffers..
    "doc:smart_buffers" holds details."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Seed the schema documentation buffer.
        self._create_buffer(_SMART_BUFFER_DOC_BUFFER, text=_SMART_BUFFER_DOC)

        # Register guard on create to validate before the buffer is created.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            _SMART_BUFFER_MANAGER_HOOK_NAME,
            _SMART_BUFFER_HOOK_PATTERN,
            lambda bm, buf, args: before_update_option_buffer(bm, None, buf, args),
            apply_to_existing=True,
        )

        # Register handler to seed and update hooks when the option buffer is created or changed.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            _SMART_BUFFER_MANAGER_HOOK_NAME,
            _SMART_BUFFER_HOOK_PATTERN,
            after_create_option_buffer,
            apply_to_existing=True,
        )

        # Register update guard on the option buffer, which checks for valid option buffer schema and existing functions.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_UPDATE_BUFFER,
            _SMART_BUFFER_MANAGER_HOOK_NAME,
            _SMART_BUFFER_HOOK_PATTERN,
            before_update_option_buffer,
            apply_to_existing=True,
        )

        # Register update hook on the option buffer, which updates the hooks on the origin buffer.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_UPDATE_BUFFER,
            _SMART_BUFFER_MANAGER_HOOK_NAME,
            _SMART_BUFFER_HOOK_PATTERN,
            after_update_option_buffer,
            apply_to_existing=True,
        )

        # Register dropped sentinel on the option buffer, which unregisters the hooks from the origin.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            _SMART_BUFFER_MANAGER_HOOK_NAME,
            _SMART_BUFFER_HOOK_PATTERN,
            lambda bm, buf, args: after_update_option_buffer(bm, buf, None, args),
            apply_to_existing=True,
        )
