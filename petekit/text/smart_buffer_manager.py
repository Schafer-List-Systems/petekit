from __future__ import annotations

import functools
import inspect
import json
import logging
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
    _origin_name_from_option,
    _option_name_from_option,
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

_OPTION_TYPE_TO_HOOK_POINT: dict[str, HookPoint] = {
    _BEFORE_UPDATE_SUFFIX: HookPoint.ON_UPDATE_BUFFER,
    _AFTER_UPDATE_SUFFIX: HookPoint.ON_BUFFER_UPDATED,
    _BEFORE_DROP_SUFFIX: HookPoint.ON_DROP_BUFFER,
    _AFTER_DROP_SUFFIX: HookPoint.ON_BUFFER_DROPPED,
}


_SMART_BUFFER_DOC = f"""\
## Smart Buffer Manager

You can register hook functions on update or drop events that fire on buffers.
Those functions execute before or after the event occurs.
If registered before the event, you can even reject the event from happening.

You can register a hook by creating an option buffer on the origin buffer you want to apply the hook on.
The four option buffers per origin are `ORIGIN[{_BEFORE_UPDATE_SUFFIX}]`,
`ORIGIN[{_AFTER_UPDATE_SUFFIX}]`, `ORIGIN[{_BEFORE_DROP_SUFFIX}]`, and `ORIGIN[{_AFTER_DROP_SUFFIX}]`.
Each holds a JSON dictionary mapping function names to reason strings.

You can edit the option buffer at any time — the hooks update automatically to match the new configuration.
If you drop the option buffer, the hooks are automatically removed from the origin buffer.

### Behaviour

`{_BEFORE_UPDATE_SUFFIX}`  — pre-update: fires before the origin buffer is updated.
The hook function can inspect the old and new buffers and reject the update by returning an error message.

`{_AFTER_UPDATE_SUFFIX}`  — post-update: registers each named function as a post-update
hook on the origin, so it fires after the origin is updated.

`{_BEFORE_DROP_SUFFIX}`  — pre-update: fires before the origin buffer is dropped.
The hook function can inspect the buffer and reject the drop by returning an error message.

`{_AFTER_DROP_SUFFIX}`  — post-drop: registers each named function as a post-drop hook
on the origin, so it fires after the origin is dropped.

### Function Signatures

Functions registered via option buffers must follow one of these signatures:

#### UPDATE hooks — fire on buffer updates (4 parameters)

Hook fires with: `(buffer_manager, old_buffer, new_buffer, args)`

Valid signatures:

  - `(buffer_manager, Buffer, Buffer, Dict)`  — receives Buffer objects directly
  - `(buffer_manager, str, str, Dict)`         — receives buffer text as strings

#### DROP hooks — fire on buffer drops (3 parameters)

Hook fires with: `(buffer_manager, buffer, args)`

Valid signatures:

  - `(buffer_manager, Buffer, Dict)`  — receives Buffer object directly
  - `(buffer_manager, str, Dict)`     — receives empty string (buffer already dropped)

Functions are looked up via the function manager — register your function there
before referencing it in an option buffer.

### Setup Example

To attach a hook to `MYBUFFER`:

1. Ensure your function exists in the function manager.
2. Create the option buffer `MYBUFFER[after_update]` with the function name as key:

```json
{{"my_listener": "alerts the team of changes"}}
```

3. The system registers `my_listener` as a post-update hook on `MYBUFFER` automatically.

To remove the hook, remove that entry from the hook option buffer.
"""


def _compute_hook_diffs(
    old_option_buffer: Buffer | None,
    new_option_buffer: Buffer | None,
) -> tuple[set[str], set[str]]:
    """Compute which hook functions were added and which were removed between two option buffer states.

    Parses the old and new option buffers as JSON dictionaries.
    Returns the set of functions removed (old \\ new) and the set of functions added (new \\ old).
    """
    # Parse the old and new JSON buffer.
    old_option_json = json.loads(old_option_buffer.get_text()) if old_option_buffer else {}
    new_option_json = json.loads(new_option_buffer.get_text()) if new_option_buffer else {}

    # Compute the keys to remove (old \ new) and add (new \ old).
    remove_hooks = set(old_option_json) - set(new_option_json)
    add_hooks = set(new_option_json) - set(old_option_json)
    return remove_hooks, add_hooks


def on_update_option_guard(
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
        return  # Empty buffer — allow.

    # Parse the JSON config to get the function names.
    try:
        config = json.loads(new_text)
    except json.JSONDecodeError as e:
        return f"[{new_option_buffer.name}] contains malformed JSON: {e} — rejecting update"

    # Reject if any named function is not registered.
    for function_name in config:
        result = smart_buffer_manager.get_function(function_name)
        if not result["ok"]:
            return f"Function '{function_name}' is not registered. Cannot apply before_update."

    # Validate signatures of all functions being registered.
    # Compute which functions are being added (new \ old) vs removed (old \ new).
    remove_hooks, add_hooks = _compute_hook_diffs(old_option_buffer, new_option_buffer)
    if not add_hooks:
        return  # No new functions — nothing to validate.

    # Determine the hook point for this option type.
    option_type = _option_name_from_option(new_option_buffer.name)
    hook_point = _OPTION_TYPE_TO_HOOK_POINT[option_type]

    # For each function, verify the signature matches what this hook point expects.
    for function_name in add_hooks:
        result = smart_buffer_manager.get_function(function_name)
        callable_hook = result["callable"]
        sig = inspect.signature(callable_hook)
        params = list(sig.parameters.items())

        # For UPDATE hooks — valid signatures are (bm, Buffer, Buffer, Dict) or (bm, str, str, Dict).
        if hook_point in (HookPoint.ON_UPDATE_BUFFER, HookPoint.ON_BUFFER_UPDATED):
            # Need exactly 4 parameters: (bm, old_buffer, new_buffer, args).
            if len(params) != 4:
                return f"[{function_name}] expected 4 parameters, got {len(params)}"

            # Fourth param must be a dict.
            if params[3][1].annotation not in (dict, "dict", Dict, "Dict"):
                return f"[{function_name}] type hint for parameter 4 is {params[3][1].annotation} — expected dict"

            # Parameters 2 and 3 must both be Buffer, or both be str.
            if not ((params[1][1].annotation in (Buffer, "Buffer") and params[2][1].annotation in (Buffer, "Buffer")) or
                    (params[1][1].annotation in (str, "str") and params[2][1].annotation in (str, "str"))):
                return f"[{function_name}] type hints: param2={params[1][1].annotation}, param3={params[2][1].annotation} — expected both Buffer or both str"

        # For DROP hooks — valid signatures are (bm, Buffer, Dict) or (bm, str, Dict).
        elif hook_point in (HookPoint.ON_DROP_BUFFER, HookPoint.ON_BUFFER_DROPPED):
            # Need exactly 3 parameters.
            if len(params) != 3:
                return f"[{function_name}] expected 3 parameters, got {len(params)}"

            # Third param must be a dict.
            if params[2][1].annotation not in (dict, "dict", Dict, "Dict"):
                return f"[{function_name}] type hint for parameter 3 is {params[2][1].annotation} — expected dict"

            # Parameter 2 must be Buffer or str.
            if not (params[1][1].annotation in (Buffer, "Buffer") or params[1][1].annotation in (str, "str")):
                return f"[{function_name}] type hint for parameter 2 is {params[1][1].annotation} — expected Buffer or str"


def _register_buffer_hook(
    origin_buffer: Buffer,
    hook_point: HookPoint,
    function_name: str,
    callable_hook: Callable,
) -> str | None:
    # Inspect the user-facing parameter annotation to decide whether the callable
    # accepts a Buffer directly or needs a text adapter.
    sig = inspect.signature(callable_hook)
    params = list(sig.parameters.items())

    # Distinguish between UPDATE hooks and DROP hooks
    if hook_point in (HookPoint.ON_UPDATE_BUFFER, HookPoint.ON_BUFFER_UPDATED):
        # If the signature is (BufferManager, Buffer, Buffer, Dict), then we register the hook directly
        if len(params) == 4 and params[1][1].annotation in (Buffer, "Buffer") and params[2][1].annotation in (Buffer, "Buffer"):
            origin_buffer.register_hook(hook_point, f"{_SMART_BUFFER_PREFIX}{function_name}", callable_hook)

        # Otherwise, the signature must be (BufferManager, str, str, Dict),
        # in which case we wrap the hook to pass the text of the buffers.
        elif len(params) == 4 and params[1][1].annotation in (str, "str") and params[2][1].annotation in (str, "str"):
            def wrapped_update_hook(bm: Any, old_buffer: Buffer, new_buffer: Buffer, args: dict[str, Any]) -> Any:
                # Use empty string for drop hooks since the buffer is already gone.
                return callable_hook(bm, old_buffer.get_text(), new_buffer.get_text(), args)

            # Register the update hook wrapper.
            origin_buffer.register_hook(hook_point, f"{_SMART_BUFFER_PREFIX}{function_name}", wrapped_update_hook)

        else:
            raise RuntimeError(f"[{function_name}] signature mismatch at registration — this should have been caught by the guard")

    elif hook_point in (HookPoint.ON_DROP_BUFFER, HookPoint.ON_BUFFER_DROPPED):
        # If the signature is (BufferManager, Buffer, Dict), register direct.
        if len(params) == 3 and params[1][1].annotation in (Buffer, "Buffer"):
            origin_buffer.register_hook(hook_point, f"{_SMART_BUFFER_PREFIX}{function_name}", callable_hook)

        # Otherwise, the signature must be (BufferManager, str, Dict),
        # in which case we wrap the hook to pass empty string (buffer is already dropped).
        elif len(params) == 3 and params[1][1].annotation in (str, "str"):
            def wrapped_drop_hook(bm: Any, buffer: Buffer, args: dict[str, Any]) -> Any:
                return callable_hook(bm, "", args)

            # Register the drop hook wrapper.
            origin_buffer.register_hook(hook_point, f"{_SMART_BUFFER_PREFIX}{function_name}", wrapped_drop_hook)

        else:
            raise RuntimeError(f"[{function_name}] signature mismatch at registration — this should have been caught by the guard")


def on_option_updated(
    smart_buffer_manager: Any,
    old_option_buffer: Buffer | None,
    new_option_buffer: Buffer | None,
    args: dict[str, Any],
) -> None:
    """Read the option buffer and apply its configuration as hooks on the origin buffer.

    On creation old_option_buffer is None — nothing to unregister.
    On update the old_option_buffer config is unregistered before the new one is registered.
    """
    # Guard — validate this is actually a smart option buffer.
    option_buffer_name = new_option_buffer.name
    option_type = _option_name_from_option(option_buffer_name)
    if option_type is None:
        _logger.error(f"[_update_origin_hooks] '{option_buffer_name}' is not a smart option buffer")
        return

    # Resolve the origin name from the option buffer name.
    origin_name = _origin_name_from_option(option_buffer_name)
    if origin_name is None:
        _logger.error(f"[_update_origin_hooks] cannot derive origin from '{option_buffer_name}'")
        return

    # Retrieve the origin buffer — it must exist for hooks to be registered on it.
    origin_buffer = smart_buffer_manager.get_buffer(origin_name)
    if origin_buffer is None:
        _logger.warning(f"[_update_origin_hooks] origin '{origin_name}' not found")
        return

    # Compute the keys to remove (old \ new) and add (new \ old).
    remove_hooks, add_hooks = _compute_hook_diffs(old_option_buffer, new_option_buffer)

    # Get the hook point on which to register / unregister the hooks.
    hook_point = _OPTION_TYPE_TO_HOOK_POINT[option_type]

    # Unregister hooks which have been removed from the option buffer.
    for function_name in remove_hooks:
        origin_buffer.unregister_hook(hook_point, f"{_SMART_BUFFER_PREFIX}{function_name}")

    # Register hooks which have been added from the option buffer.
    for function_name in add_hooks:
        result = smart_buffer_manager.get_function(function_name)
        # Skip if the function is not registered — this should never happen.
        if not result["ok"]:
            _logger.error(f"[_update_origin_hooks] function '{function_name}' not found in function manager — this should never happen")
            continue
        callable_hook = result["callable"]

        # Register the callable as a hook on the origin buffer.
        _register_buffer_hook(origin_buffer, hook_point, function_name, callable_hook)


def on_origin_buffer_dropped(
    buffer_manager: BufferManager,
    origin_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    # Clean up all option buffers.
    for suffix in (_BEFORE_UPDATE_SUFFIX, _AFTER_UPDATE_SUFFIX, _BEFORE_DROP_SUFFIX, _AFTER_DROP_SUFFIX):
        # Get the option buffer.
        option_name = f"{origin_buffer.name}{suffix}"
        option_buffer = buffer_manager.get_buffer(option_name)
        if option_buffer is None:
            continue

        # Unregister the dropped hook from the option buffer.
        option_buffer.unregister_hook(HookPoint.ON_BUFFER_DROPPED, _SMART_BUFFER_MANAGER_HOOK_NAME)

        # Drop the option buffer.
        buffer_manager.drop_buffer(option_name)


def on_option_buffer_dropped(
    buffer_manager: BufferManager,
    option_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    # Remove all hooks from the origin buffer
    on_option_updated(buffer_manager, option_buffer, None, args)


# onCreateBuffer hook — validate the option buffer before it is created.
def on_create_hook_option_guard(
    buffer_manager: BufferManager,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    return on_update_option_guard(buffer_manager, None, buffer, args)


# onBufferCreatedHook — apply the new option buffer config to the origin immediately.
def on_hook_option_created(
    buffer_manager: BufferManager,
    option_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    # Get the origin buffer to this option buffer.
    origin_name = _origin_name_from_option(option_buffer.name)
    origin_buffer = buffer_manager.get_buffer(origin_name)
    if origin_buffer is None:
        _logger.error(f"[on_hook_option_created] origin '{origin_name}' not found")
        return

    # Register dropped sentinel on origin buffer, which drops the option buffer too.
    origin_buffer.register_hook(
        HookPoint.ON_BUFFER_DROPPED,
        _SMART_BUFFER_MANAGER_HOOK_NAME,
        on_origin_buffer_dropped,
    )

    # Register update guard on the option buffer, which checks for valid option buffer schema and existing functions.
    option_buffer.register_hook(
        HookPoint.ON_UPDATE_BUFFER,
        _SMART_BUFFER_MANAGER_HOOK_NAME,
        on_update_option_guard,
    )

    # Register update hook on the option buffer, which updates the hooks on the origin buffer.
    option_buffer.register_hook(
        HookPoint.ON_BUFFER_UPDATED,
        _SMART_BUFFER_MANAGER_HOOK_NAME,
        on_option_updated,
    )

    # Register dropped sentinel on the option buffer, which unregisters the hooks from the origin.
    option_buffer.register_hook(
        HookPoint.ON_BUFFER_DROPPED,
        _SMART_BUFFER_MANAGER_HOOK_NAME,
        on_option_buffer_dropped,
    )

    # Register inital set of hooks
    on_option_updated(buffer_manager, None, option_buffer, {})


class SmartBufferManager(FunctionManager, OptionBufferManager, AgenticObject):
    """Buffer manager that turns specific option buffers into active hooks on their origin buffer.
    
    See `doc:smart_buffers` for the full schema and behaviour."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Seed the schema documentation buffer.
        self._create_buffer(_SMART_BUFFER_DOC_BUFFER, text=_SMART_BUFFER_DOC)

        # Register guard on create to validate before the buffer is created.
        self.register_buffer_pattern_hook(
            HookPoint.ON_CREATE_BUFFER,
            _SMART_BUFFER_MANAGER_HOOK_NAME,
            r"^.+\[(before_update|after_update|before_drop|after_drop)\]$",
            on_create_hook_option_guard,
        )

        # Register handler to seed and update hooks when the option buffer is created or changed.
        self.register_buffer_pattern_hook(
            HookPoint.ON_BUFFER_CREATED,
            _SMART_BUFFER_MANAGER_HOOK_NAME,
            r"^.+\[(before_update|after_update|before_drop|after_drop)\]$",
            on_hook_option_created,
        )
