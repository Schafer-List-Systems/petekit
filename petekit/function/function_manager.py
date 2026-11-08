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

_FUNCTION_STATUS_BUFFER = "system:status:function"
_FUNCTION_STATUS_NAME_BASE = "system:status:function:"
_CALLABLES_LIST_BUFFER = "sys/callables.json"


# -------------------------------------------------------------------------------------------------------------------- #
# Data Classes                                                                                                         #
# -------------------------------------------------------------------------------------------------------------------- #


@dataclass
class FunctionEntry:
    """A registered callable with its runtime metadata and introspectable description."""
    name: str
    callable: Callable
    short_description: str
    long_description: str
    signature: inspect.Signature
    return_annotation: Any
    required_args: list[str]  # names of params without defaults
    created_at: float
    modified_at: float


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

    def __del__(self) -> None:
        """Decrement the reference counter when the wrapper is garbage collected."""
        raise NotImplementedError("stub only")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward the call to the current callable in the FunctionManager."""
        raise NotImplementedError("stub only")

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

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward the call to the managed callable."""
        return self._managed(*args, **kwargs)


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


def _sign_payload(fm: "FunctionManager", data: Any) -> str:
    """Serialize data as HMAC-signed JSON with indent=2."""
    signed = json.dumps(data, sort_keys=True)
    mac = _compute_mac(fm, signed)
    return json.dumps({"data": data, "mac": mac}, indent=2)


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


def _recreate_callable_list(manager: "FunctionManager") -> dict[str, Any]:
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
        return _recreate_callable_list(manager)

    # The buffer is present. Read the raw text from the buffer.
    callables_text = callables_buffer.get_text()

    # Parse and return the JSON list from the buffer text.
    try:
        return json.loads(callables_text)
    except json.JSONDecodeError as e:
        _logger.error(f"[_get_callables_list] failed to parse '{_CALLABLES_LIST_BUFFER}': {e}")
        
        # We could not parse the callables list, so we recreate it.
        return _recreate_callable_list(manager)


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


def _make_stable_wrapper(manager: "FunctionManager", func_name: str) -> Callable:
    """Build a stable delegating wrapper whose identity is fixed but delegates to the live sandbox proxy.

    The wrapper is registered once and never replaced. Every call forwards to the
    current sandbox proxy, so callers holding a reference are never stale.
    """
    # Stable wrapper -- delegates to the live sandbox proxy on every call.
    # The proxy from getattr already has sandbox bound via closure capture.
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        sandbox = manager._oap_sandbox_builder.get_sandbox(freeze_namespaces=False)
        proxy = getattr(sandbox, func_name)
        return proxy(*args, **kwargs)

    # Mirror signature from the proxy so inspect.signature sees the real interface.
    sandbox = manager._oap_sandbox_builder.get_sandbox(freeze_namespaces=False)
    proxy = getattr(sandbox, func_name)
    wrapper.__signature__ = inspect.signature(proxy)
    wrapper.__name__ = proxy.__name__
    wrapper.__doc__ = getattr(proxy, "__doc__", "")

    return wrapper


def _function_status_name(name: str) -> str:
    # Derive the canonical status buffer name for a single function.
    return f"{_FUNCTION_STATUS_NAME_BASE}{_sanitize_buffer_name(name)}"


def _make_readonly_sentinel(fm: "FunctionManager") -> BeforeUpdateBuffer:
    """Factory that produces a pre-update hook verifying the MAC on every buffer write.

    The sentinel rejects any write whose content does not carry a valid HMAC signed
    by the function manager, preventing external modifications to protected buffers.
    """
    def readonly_sentinel(
        buffer_manager: BufferManager,
        buffer: Buffer,
        new_buffer: Buffer,
        args: dict[str, Any],
    ) -> str | None:
        # Extract the new text from the hook args.
        new_text = args.get("new_text")
        if not new_text:
            return None

        # Parse the new_text as JSON and verify the HMAC signature.
        # Reject if the signature does not match -- buffer was modified externally.
        try:
            obj = json.loads(new_text)
            signed = obj.get("data", "")
            mac = obj.get("mac", "")
            expect = _compute_mac(fm, json.dumps(signed, sort_keys=True))
            if not hmac.compare_digest(mac, expect):
                return "This buffer is protected!"
        except Exception:
            return "This buffer is protected!"
        return None
    return readonly_sentinel


def _build_function_meta(entry: FunctionEntry) -> dict[str, Any]:
    # Collect the full function metadata as a dict for JSON serialization.
    return {
        "name": entry.name,
        "short_description": entry.short_description,
        "long_description": entry.long_description,
        "signature": {
            "parameters": [
                {
                    "name": p.name,
                    "type": str(p.annotation) if p.annotation is not inspect.Parameter.empty else "Any",
                    "default": repr(p.default) if p.default is not inspect.Parameter.empty else None,
                }
                for p in entry.signature.parameters.values()
            ],
            "return": (
                str(entry.return_annotation)
                if entry.return_annotation is not inspect.Parameter.empty
                else "None"
            ),
        },
        "created_at": entry.created_at,
        "modified_at": entry.modified_at,
    }


async def _refresh_list_buffer(fm: "FunctionManager") -> None:
    # Collect each function's name, args, and short description for the catalog.
    records = []
    for entry in fm._functions.values():
        arg_summary = ", ".join(p.name for p in entry.signature.parameters.values())
        records.append({
            "name": entry.name,
            "args": f"({arg_summary})",
            "description": entry.short_description,
        })
    # Write the signed JSON catalog to the _FUNCTION_STATUS_BUFFER, replacing existing content.
    text = _sign_payload(fm, records)
    await fm.create_buffer(_FUNCTION_STATUS_BUFFER, text=text, overwrite=True)


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
    """I manage a namespace of runtime-created Callables via `create_function` and `drop_function`.
    The "system:status:function" buffer catalogs all registered functions.
    Each function is documented in a system:status:function:<name> buffer for introspection.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._functions: dict[str, FunctionEntry] = {}
        self._callables: dict[str, ManagedCallable] = {}
        self._function_manager_signing_key: str = secrets.token_hex(32)

        # Create the initial callables list buffer.
        callables_json = json.dumps(_recreate_callable_list(self), indent=2, sort_keys=True)
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

        # Create the initial catalog buffer.
        init_result = self._create_buffer(_FUNCTION_STATUS_BUFFER, text=_sign_payload(self, []))
        if not isinstance(init_result, dict) or not init_result.get("ok"):
            raise RuntimeError(f"failed to create function list buffer: {init_result.get('error')}")

        # Guard the catalog buffer against unsigned writes.
        hook_result = self.get_buffer(_FUNCTION_STATUS_BUFFER).register_hook(
            HookPoint.BEFORE_UPDATE_BUFFER,
            "readonly_sentinel",
            _make_readonly_sentinel(self),
        )
        if not isinstance(hook_result, dict) or not hook_result.get("ok"):
            raise RuntimeError(f"failed to register readonly_sentinel on {_FUNCTION_STATUS_BUFFER}: {hook_result.get('error')}")

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

    @sandbox
    async def create_function(
        self,
        name: str,
        callable: Callable,
        short_description: str = "",
        long_description: str = "",
    ) -> dict[str, Any]:
        # Guard: reject duplicate names outright -- the namespace is flat and must stay unique.
        if name in self._functions:
            return {
                "ok": False,
                "error": f"Function '{name}' already registered.",
            }

        # Sanitize the name by replacing colons and slashes with underscores,
        # then verify it is a valid Python identifier.
        sanitized = name.replace(":", "_").replace("/", "_")
        if not sanitized.isidentifier():
            return {
                "ok": False,
                "error": f"'{name}' is not a valid Python identifier.",
            }

        # Extract metadata from the callable; fall back to its docstring when descriptions are absent.
        sig = inspect.signature(callable)
        doc = callable.__doc__ or ""
        if not short_description:
            short_description = doc.split("\n")[0].strip() if doc else ""
        if not long_description:
            long_description = doc.strip() if doc else ""

        # Assemble the function entry with all metadata, timestamps, and introspection data.
        # self is never caller-provided — it's bound by the sandbox automatically.
        required = [p for p, pv in sig.parameters.items() if pv.default is inspect.Parameter.empty and p != "self"]
        now = time.time()
        entry = FunctionEntry(
            name=name,
            callable=callable,
            short_description=short_description,
            long_description=long_description,
            signature=sig,
            return_annotation=sig.return_annotation,
            required_args=required,
            created_at=now,
            modified_at=now,
        )

        # Mirror the function entry into a signed JSON buffer the agent can read;
        # create_buffer with overwrite=True fires existing buffer hooks automatically.
        buf_result = await self.create_buffer(
            _function_status_name(name),
            text=_sign_payload(self, _build_function_meta(entry)),
            overwrite=True,
        )
        if not isinstance(buf_result, dict) or not buf_result.get("ok"):
            return {
                "ok": False,
                "error": f"Cascaded error due to buffer creation for function description: {buf_result.get('error')}",
            }

        # Guard the new meta buffer against unsigned writes.
        hook_result = self.get_buffer(_function_status_name(name)).register_hook(
            HookPoint.BEFORE_UPDATE_BUFFER,
            "readonly_sentinel",
            _make_readonly_sentinel(self),
        )
        if not isinstance(hook_result, dict) or not hook_result.get("ok"):
            return {
                "ok": False,
                "error": f"Function '{name}' created but could not be protected: {hook_result.get('error')}",
            }

        # Register AFTER_DROP_BUFFER to cascade drop to the code buffer.
        from petekit.function.function_compiler import _CODE_BUFFER_PREFIX
        code_buffer_name = _CODE_BUFFER_PREFIX + name

        async def _on_function_dropped(bm: "BufferManager", buf: "Buffer", args: dict[str, Any]) -> str | None:
            # Drop the code buffer when the function is dropped -- no-op if already gone.
            drop_result = await bm.drop_buffer(code_buffer_name)
            if not drop_result.get("ok"):
                _logger.warning(f"[on_function_dropped] cascade to drop_buffer('{code_buffer_name}') failed: {drop_result.get('error')}")
            return None

        drop_cascade = self.get_buffer(_function_status_name(name)).register_hook(
            HookPoint.AFTER_DROP_BUFFER,
            "on_function_dropped",
            _on_function_dropped,
        )

        # Register the callable in the function namespace.
        self._functions[name] = entry

        # Refresh the global function list so the new entry appears in the catalog.
        await _refresh_list_buffer(self)

        return {
            "ok": True,
            "name": name,
            "args": list(entry.signature.parameters.keys()),
            "returns": str(entry.return_annotation) if entry.return_annotation is not inspect.Parameter.empty else "None",
        }

    @tool
    async def drop_function(self, name: str) -> dict[str, Any]:
        # Guard: reject unknown names to keep the namespace consistent.
        if name not in self._functions:
            return {"ok": False, "error": f"No function named '{name}'."}

        # Drop the function's buffer; a hook rejection prevents the drop and returns the error.
        buf_name = _function_status_name(name)

        drop_result = await self.drop_buffer(buf_name)
        if not drop_result.get("ok"):
            return drop_result

        # Remove the callable from the namespace now that the buffer has been dropped.
        del self._functions[name]

        # Refresh the global catalog so the removed function no longer appears.
        await _refresh_list_buffer(self)

        return {"ok": True, "dropped": name}

    @sandbox
    def get_function(self, name: str) -> dict[str, Any]:
        # Look up the callable by name; return an error if not found.
        if name not in self._functions:
            return {"ok": False, "error": f"No function named '{name}'."}
        entry = self._functions[name]
        return {
            "ok": True,
            "name": entry.name,
            "callable": entry.callable,
            "args": list(entry.signature.parameters.keys()),
            "required_args": entry.required_args,
            "returns": str(entry.return_annotation) if entry.return_annotation is not inspect.Parameter.empty else "None",
        }
