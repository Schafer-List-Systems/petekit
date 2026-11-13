"""Function manager -- runtime Callable management built on top of BufferManager."""

from __future__ import annotations

import asyncio
import hmac
import inspect
import json
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Any, Callable

from peteos import tool, sandbox
from peteos.oap.agentic_object import AgenticObject

from petekit.text.buffer_manager import Buffer, BufferManager, HookPoint, AfterUpdateBuffer, BeforeUpdateBuffer, _sanitize_buffer_name

_logger = logging.getLogger(__name__)

_CALLABLES_LIST_BUFFER = "sys/callables.json"
_FUNCTION_MANAGER_DOC_BUFFER = "doc:function_manager"


_FUNCTION_MANAGER_DOC = """\
## FunctionManager -- runtime Callable management

The FunctionManager holds registered callables in `sys/callables.json`. That buffer
is the authoritative list of every live callable -- its name, signature, and module.

Do not edit this buffer manually -- it is managed automatically by the FunctionManager
and protected against unauthorized modification.
"""


# -------------------------------------------------------------------------------------------------------------------- #
# Data Classes                                                                                                         #
# -------------------------------------------------------------------------------------------------------------------- #


class ManagedCallable:
    """A stable callable wrapper that defers to the FunctionManager on every call.

    The wrapper holds no direct reference to the underlying callable, so recompiles
    of the function with the same interface are transparent to existing callers.
    Reference counting is managed via __init__ (increment) and __del__ (decrement).
    """

    def __init__(self, manager: "FunctionManager", name: str, callable: Callable) -> None:
        """Initialize with a FunctionManager reference, the function name, and the callable to wrap.

        Copies all introspection attributes from the callable so callers using
        inspect.getfullargspec, inspect.signature, etc. get the real interface.
        The name is required -- lambdas and some callables have no __name__.
        """
        # Set the manager, name, reference counter, and initial version to zero.
        self._manager = manager
        self.__name__ = name
        self._reference_count = 0
        self._version = 0

        # Swap the callable and refresh all fields via _update_callable -- sets version to 1.
        self._update_callable(callable)

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward the call to the current callable in the FunctionManager."""
        # Call the underlying wrapped callable with the caller's arguments.
        result = self.__wrapped__(*args, **kwargs)

        # Await the result if it is a coroutine (i.e. the wrapped callable is an async def).
        if inspect.iscoroutine(result):
            result = await result

        return result

    def acquire(self) -> None:
        """Acquire a callable reference to this managed callable."""
        # Increment the reference counter.
        self._reference_count += 1

        # Schedule the callables list to be updated without blocking the caller.
        loop = asyncio.get_running_loop()
        loop.create_task(self._manager._update_callable_list(self))

    def release(self) -> None:
        """Release a callable reference to this managed callable."""
        # Decrement the reference counter.
        self._reference_count -= 1

        # Schedule the callables list to be updated without blocking the caller.
        loop = asyncio.get_running_loop()
        loop.create_task(self._manager._update_callable_list(self))

    def _update_callable(self, new_callable: Callable) -> None:
        """Swap the underlying callable and refresh all introspection fields."""
        # Increment the version counter.
        self._version += 1

        # Swap the underlying callable on the existing wrapper.
        self.__wrapped__ = new_callable

        # Refresh introspection fields on the wrapper to reflect the new callable.
        self.__doc__ = getattr(new_callable, "__doc__", "") or ""
        self.__signature__ = inspect.signature(new_callable)
        self.__annotations__ = getattr(new_callable, "__annotations__", {}) or {}
        self.__module__ = getattr(new_callable, "__module__", None)
        self.__qualname__ = getattr(new_callable, "__qualname__", self.__name__)


class CallableRef:
    """A stable callable wrapper that defers to the FunctionManager on every call.

    The wrapper holds no direct reference to the underlying callable, so recompiles
    of the function with the same interface are transparent to existing callers.
    Reference counting is managed via __init__ (increment) and __del__ (decrement).
    """

    def __init__(self, managed: ManagedCallable) -> None:
        """Initialize from a ManagedCallable.

        Stores the managed callable reference and copies all introspection
        attributes from it so callers using inspect.getfullargspec,
        inspect.signature, etc. get the real interface.
        """
        # Acquire a reference to the managed callable.
        managed.acquire()

        # Copy all introspection attributes from the managed callable.
        self._managed = managed
        self._update_ref()

    def _update_ref(self) -> bool:
        """Sync all introspection fields and the version from the managed callable.

        Returns True when the version differed and an update was performed.
        Returns False when the version already matched and no update was needed.
        """
        # Check whether the managed callable has been updated since this ref was created.
        if self._version == self._managed._version:
            return False

        # Copy every introspection field and the version from the managed callable.
        self.__name__ = self._managed.__name__
        self.__doc__ = self._managed.__doc__
        self.__signature__ = self._managed.__signature__
        self.__annotations__ = self._managed.__annotations__
        self.__module__ = self._managed.__module__
        self.__qualname__ = self._managed.__qualname__
        self._version = self._managed._version

        return True

    def __del__(self) -> None:
        """Release the reference to the managed callable when garbage collected."""
        self._managed.release()

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward the call to the managed callable."""
        return await self._managed(*args, **kwargs)


# -------------------------------------------------------------------------------------------------------------------- #
# Utils                                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #


def _compute_mac(fm: "FunctionManager", data: str) -> str:
    """Produce an HMAC over the given data using the function manager's secret signing key."""
    return hmac.new(
        fm._function_manager_signing_key.encode(),
        data.encode(),
        "sha256",
    ).hexdigest()


def _sign_callables_list(fm: "FunctionManager", data: dict[str, Any]) -> dict[str, Any]:
    """Compute an HMAC over the stringified callables list and return it with the raw data.

    The MAC is computed over the compact (no indentation) JSON string so callers
    that read the data field get the original object, not a re-serialized version.
    """
    # Dump the callables list to a compact string and compute the HMAC.
    signed = json.dumps(data, separators=(",", ":"))
    mac = _compute_mac(fm, signed)

    # Return the raw data alongside the HMAC -- not the stringified version.
    return {"data": data, "mac": mac}


def _check_signatures_compatible(old_sig: inspect.Signature, new_sig: inspect.Signature) -> str | None:
    """Check that two signatures are compatible for a managed callable update.

    Returns None when compatible. Returns an error string describing the conflict
    when the signatures are incompatible -- callers may depend on the old interface.

    Compatibility rules:
    - The number of positional and required arguments must be the same.
    - Keyword argument names must match.
    - A kind can be added (e.g. POSITIONAL_ONLY -> POSITIONAL_OR_KEYWORD) but not removed.
    """
    # Count positional-only and positional-or-keyword params in each signature.
    old_positional = [p for p, pv in old_sig.parameters.items() if pv.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    new_positional = [p for p, pv in new_sig.parameters.items() if pv.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]

    # Reject when the number of positional arguments differs.
    if len(old_positional) != len(new_positional):
        return f"positional arg count changed: {len(old_positional)} -> {len(new_positional)}"

    # Reject when the number of required arguments differs.
    old_required = [p for p, pv in old_sig.parameters.items() if pv.default is inspect.Parameter.empty and p != "self"]
    new_required = [p for p, pv in new_sig.parameters.items() if pv.default is inspect.Parameter.empty and p != "self"]
    if len(old_required) != len(new_required):
        return f"required arg count changed: {len(old_required)} -> {len(new_required)}"

    # Reject when keyword argument names differ.
    old_kw = {p for p, pv in old_sig.parameters.items() if pv.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD)}
    new_kw = {p for p, pv in new_sig.parameters.items() if pv.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD)}
    if old_kw != new_kw:
        added = new_kw - old_kw
        removed = old_kw - new_kw
        if removed:
            return f"keyword argument(s) removed: {removed}"
        if added:
            return f"keyword argument(s) added: {added}"

    # Check that no parameter kind was removed -- a kind can only be widened.
    for pname in old_sig.parameters:
        # Skip self -- it's bound automatically and never caller-provided.
        if pname == "self":
            continue

        # Skip params that no longer exist in the new signature.
        if pname not in new_sig.parameters:
            continue

        # Extract the kind from each signature for comparison.
        old_kind = old_sig.parameters[pname].kind
        new_kind = new_sig.parameters[pname].kind

        # Reject when VAR_POSITIONAL becomes anything other than itself or VAR_KEYWORD.
        if old_kind == inspect.Parameter.VAR_POSITIONAL:
            if new_kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                return f"'{pname}' kind narrowed: VAR_POSITIONAL -> {new_kind.name}"

        # Reject when VAR_KEYWORD becomes anything other than itself or VAR_POSITIONAL.
        if old_kind == inspect.Parameter.VAR_KEYWORD:
            if new_kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                return f"'{pname}' kind narrowed: VAR_KEYWORD -> {new_kind.name}"

        # Reject when KEYWORD_ONLY loses its keyword-only nature.
        if old_kind == inspect.Parameter.KEYWORD_ONLY:
            if new_kind not in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                return f"'{pname}' kind narrowed: {old_kind.name} -> {new_kind.name}"

        # Reject when POSITIONAL_ONLY loses its positional-only nature.
        if old_kind == inspect.Parameter.POSITIONAL_ONLY:
            if new_kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                return f"'{pname}' kind narrowed: {old_kind.name} -> {new_kind.name}"

    return None


def _callable_wrapper_to_dict(wrapper: ManagedCallable) -> dict[str, Any]:
    """Serialize a ManagedCallable into a dictionary with all its introspection fields."""
    # Collect all introspection fields into a dict for serialization.
    return {
        "name": wrapper.__name__,
        "doc": wrapper.__doc__,
        "module": wrapper.__module__,
        "qualname": wrapper.__qualname__,
        "signature": str(wrapper.__signature__),
        "reference_count": wrapper._reference_count,
    }


def _recreate_callables_list(manager: "FunctionManager") -> dict[str, Any]:
    """Rebuild the callables list dictionary from the current registry state."""
    # Start with an empty dictionary keyed by function name.
    callables_dict: dict[str, Any] = {}

    # Convert every registered wrapper to a dict and add it to the result.
    for wrapper in manager._callables.values():
        wrapper_dict = _callable_wrapper_to_dict(wrapper)
        callables_dict[wrapper_dict["name"]] = wrapper_dict

    return _sign_callables_list(manager, callables_dict)


def _get_callables_list(manager: "FunctionManager") -> dict[str, Any]:
    """Get the callables list dict from the buffer or recreate it if absent."""
    # Get the callables list buffer.
    callables_buffer = manager.get_buffer(_CALLABLES_LIST_BUFFER)

    # When the buffer is absent, recreate the callables dict from the current registry.
    if callables_buffer is None:
        return _recreate_callables_list(manager)

    # The buffer is present. Read the raw text from the buffer.
    callables_text = callables_buffer.get_text()

    # Parse and return the data field from the signed JSON buffer.
    try:
        obj = json.loads(callables_text)
        return obj.get("data", {})
    except json.JSONDecodeError as e:
        _logger.error(f"[_get_callables_list] failed to parse '{_CALLABLES_LIST_BUFFER}': {e}")

        # We could not parse the callables list, so we recreate it.
        return _recreate_callables_list(manager)


async def _update_callable_list(manager: "FunctionManager", wrapper: ManagedCallable) -> None:
    """Register a function wrapper in the manager's callables list."""
    # Get the callables list dictionary from the buffer or recreate it.
    callables_list = _get_callables_list(manager)

    # Convert the wrapper to a dict and add it to the callables list keyed by name.
    wrapper_dict = _callable_wrapper_to_dict(wrapper)
    callables_list[wrapper_dict["name"]] = wrapper_dict

    # Sign the callables list with an HMAC.
    signed_callables_list = _sign_callables_list(manager, callables_list)

    # Persist the signed callables list to the buffer.
    callables_json = json.dumps(signed_callables_list, indent=2)
    await manager.create_buffer(_CALLABLES_LIST_BUFFER, text=callables_json, overwrite=True)


async def _remove_callable_from_list(manager: "FunctionManager", wrapper: ManagedCallable) -> None:
    """Remove a function wrapper from the manager's callables list."""
    # Get the callables list dictionary from the buffer.
    callables_list = _get_callables_list(manager)

    # Remove the wrapper entry from the callables list by name.
    callables_list.pop(wrapper.__name__, None)

    # Sign the callables list with an HMAC.
    signed_callables_list = _sign_callables_list(manager, callables_list)

    # Persist the signed callables list to the buffer.
    callables_json = json.dumps(signed_callables_list, indent=2)
    await manager.create_buffer(_CALLABLES_LIST_BUFFER, text=callables_json, overwrite=True)


# -------------------------------------------------------------------------------------------------------------------- #
# Hooks                                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #

def _before_update_callable_list(
    manager: "FunctionManager",
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Empty hook stub called before the callables list is updated."""
    # Get the new buffer text.
    new_text = new_buffer.get_text()
    if not new_text:
        return "missing buffer text"

    # Parse the new_text as JSON.
    try:
        obj = json.loads(new_text)
    except json.JSONDecodeError as e:
        return f"failed to parse callables list: {e}"

    # Create the signed payload from the data field.
    data = obj.get("data", {})
    expect = _sign_callables_list(manager, data)

    # Compare the newly created signature against the one provided.
    if not hmac.compare_digest(expect["mac"], obj.get("mac", "")):
        return "This buffer is protected!"

    return None


def _before_drop_callable_list(
    manager: "FunctionManager",
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Refuse to drop the callables list buffer."""
    return "This buffer is protected!"


# -------------------------------------------------------------------------------------------------------------------- #
# FunctionManager                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #


class FunctionManager(BufferManager, AgenticObject):
    """I manage a namespace of runtime-created Callables via `register_function` and `unregister_function`."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._callables: dict[str, ManagedCallable] = {}
        self._function_manager_signing_key: str = secrets.token_hex(32)

        # Create the initial callables list buffer.
        callables_json = json.dumps(_recreate_callables_list(self), indent=2, sort_keys=True)
        callables_result = self._create_buffer(_CALLABLES_LIST_BUFFER, text=callables_json)
        if not isinstance(callables_result, dict) or not callables_result.get("ok"):
            raise RuntimeError(f"failed to create callables list buffer: {callables_result.get('error')}")

        # Guard the callables list buffer against unsigned writes.
        callables_buffer = self.get_buffer(_CALLABLES_LIST_BUFFER)
        callables_hook_result = callables_buffer.register_hook(
            HookPoint.BEFORE_UPDATE_BUFFER,
            "before_update_callable_list",
            _before_update_callable_list,
        )
        if not isinstance(callables_hook_result, dict) or not callables_hook_result.get("ok"):
            raise RuntimeError(f"failed to register before_update_callable_list on {_CALLABLES_LIST_BUFFER}: {callables_hook_result.get('error')}")

        # Guard the callables list buffer against drops.
        callables_drop_result = callables_buffer.register_hook(
            HookPoint.BEFORE_DROP_BUFFER,
            "before_drop_callable_list",
            _before_drop_callable_list,
        )
        if not isinstance(callables_drop_result, dict) or not callables_drop_result.get("ok"):
            raise RuntimeError(f"failed to register before_drop_callable_list on {_CALLABLES_LIST_BUFFER}: {callables_drop_result.get('error')}")

        # Seed the function manager documentation buffer.
        self._create_buffer(_FUNCTION_MANAGER_DOC_BUFFER, text=_FUNCTION_MANAGER_DOC)

    async def register_function(self, callable: Callable, name: str | None = None) -> dict[str, Any]:
        """Create a ManagedCallable and register it in the functions registry."""
        # Derive the function name from the callable when not provided.
        if name is None:
            name = callable.__name__

        # Reject when the name already exists in the registry.
        if name in self._callables:
            return {"ok": False, "error": f"function '{name}' already registered"}

        # Create the wrapper and store it in the registry.
        wrapper = ManagedCallable(self, name, callable)
        self._callables[name] = wrapper

        # Notify the system that a new callable was registered.
        await _update_callable_list(self, wrapper)

        return {"ok": True}

    async def update_registered_function(self, callable: Callable, name: str | None = None) -> dict[str, Any]:
        """Replace the wrapper for an existing function with a new callable."""
        # Derive the function name from the callable when not provided.
        if name is None:
            name = callable.__name__

        # Reject when the name does not exist in the registry.
        if name not in self._callables:
            return {"ok": False, "error": f"function '{name}' not registered"}

        # Get the wrapper from the registry.
        wrapper = self._callables[name]

        # When there are active callable refs, check that the new signature is compatible
        # with the old one so existing callers are not broken by the update.
        if wrapper._reference_count > 0:
            old_sig = inspect.signature(wrapper.__wrapped__)
            new_sig = inspect.signature(callable)
            compatible_error = _check_signatures_compatible(old_sig, new_sig)
            if compatible_error is not None:
                return {"ok": False, "error": f"function '{name}' signature incompatible: {compatible_error}"}

        # Update the wrapper with the new callable and persist to the list buffer.
        wrapper._update_callable(callable)
        await _update_callable_list(self, wrapper)

        return {"ok": True}

    async def unregister_function(self, name: str) -> dict[str, Any]:
        """Remove a function wrapper from the functions registry."""
        # Reject when the name is not in the callables registry.
        if name not in self._callables:
            return {"ok": False, "error": f"function '{name}' not registered"}

        # Get the wrapper to check the reference counter.
        wrapper = self._callables[name]

        # Reject when the reference counter is larger than zero -- active references must be released first.
        if wrapper._reference_count > 0:
            return {"ok": False, "error": f"function '{name}' has active references"}

        # Remove the wrapper from the callables registry.
        del self._callables[name]

        # Remove the callable from the callables list buffer.
        await _remove_callable_from_list(self, wrapper)

        return {"ok": True}

    def acquire_function(self, name: str) -> CallableRef | dict[str, Any]:
        """Return a CallableRef for the named function, creating it if needed."""
        # Reject when the name is not in the callables registry.
        if name not in self._callables:
            return {"ok": False, "error": f"No function named '{name}'."}

        # Create and return a CallableRef for the managed callable.
        return CallableRef(self._callables[name])

    @sandbox
    async def create_function(
        self,
        name: str,
        callable: Callable,
        short_description: str = "",
        long_description: str = "",
    ) -> dict[str, Any]:
        # Guard: reject duplicate names outright -- the namespace is flat and must stay unique.
        # Overwrite the docstring when a long description is available.
        if long_description:
            callable.__doc__ = long_description

        # Delegate to register_function.
        return await self.register_function(callable, name)

    @tool
    async def drop_function(self, name: str) -> dict[str, Any]:
        # Delegate to unregister_function.
        return await self.unregister_function(name)

    @sandbox
    def get_function(self, name: str) -> dict[str, Any]:
        # Look up the callable by name; return an error if not found.
        if name not in self._callables:
            return {"ok": False, "error": f"No function named '{name}'."}
        wrapper = self._callables[name]
        sig = wrapper.__signature__
        required = [p for p, pv in sig.parameters.items() if pv.default is inspect.Parameter.empty and p != "self"]
        return_annotation = sig.return_annotation
        return {
            "ok": True,
            "name": wrapper.__name__,
            "callable": wrapper,
            "args": list(sig.parameters.keys()),
            "required_args": required,
            "returns": str(return_annotation) if return_annotation is not inspect.Parameter.empty else "None",
        }

    def has_function(self, name: str) -> bool:
        """Return True when a function with the given name is registered."""
        return name in self._callables
