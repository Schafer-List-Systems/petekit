"""Function manager — runtime Callable management built on top of BufferManager."""

from __future__ import annotations

import hmac
import inspect
import json
import secrets
import time
from dataclasses import dataclass
from typing import Any, Callable

from peteos import tool, sandbox
from peteos.oap.agentic_object import AgenticObject

from petekit.text.buffer_manager import BufferManager

_FUNCTION_LIST_BUFFER = "system:list:functions"


@dataclass
class FunctionEntry:
    """A registered callable with its runtime metadata and introspectable description."""
    name: str
    callable: Callable
    short_description: str
    long_description: str
    signature: inspect.Signature
    return_annotation: Any
    created_at: float
    modified_at: float


def _build_buffer_name(name: str) -> str:
    # Prefix the function name so it lives in the function: namespace.
    return f"function:{name}:meta"


def _compute_mac(fm: "FunctionManager", data: str) -> str:
    """Produce an HMAC over the given data using the function manager's secret signing key."""
    return hmac.new(
        fm._function_manager_signing_key.encode(),
        data.encode(),
        "sha256",
    ).hexdigest()


def _make_readonly_sentinel(fm: "FunctionManager"):
    """Factory that produces a sentinel hook verifying the MAC on every buffer write.

    The sentinel rejects any write whose content does not carry a valid HMAC signed
    by the function manager, preventing external modifications to protected buffers.
    """
    def readonly_sentinel(buf: str, old_text: str, start: int, end: int, new_text: str) -> str | None:
        # Parse the new_text as JSON and verify the HMAC signature.
        try:
            obj = json.loads(new_text)
            signed = obj.get("data", "")
            mac = obj.get("mac", "")
            expect = _compute_mac(fm, json.dumps(signed, sort_keys=True))
            if not hmac.compare_digest(mac, expect):
                return "This buffer is protected by the FunctionManager!"
        except Exception:
            return "This buffer is protected by the FunctionManager!"
        return None
    return readonly_sentinel


def _sign_payload(fm: "FunctionManager", data: Any) -> str:
    """Serialize data as HMAC-signed JSON with indent=2."""
    signed = json.dumps(data, sort_keys=True)
    mac = _compute_mac(fm, signed)
    return json.dumps({"data": data, "mac": mac}, indent=2)


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
    # Write the signed JSON catalog to the system:list:functions buffer, replacing existing content.
    text = _sign_payload(fm, records)
    await fm.create_buffer(_FUNCTION_LIST_BUFFER, text=text, overwrite=True)


class FunctionManager(BufferManager, AgenticObject):
    """A BufferManager that also manages a namespace of runtime-created Callables.
    Each function lives in self._functions and is mirrored to a function:<name>:meta buffer
    (signed JSON) for agent introspection. A system:list:functions buffer catalogs all registered functions.
    All buffers are HMAC-protected against external writes.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._functions: dict[str, FunctionEntry] = {}
        self._function_manager_signing_key: str = secrets.token_hex(32)

        # Create the initial catalog buffer.
        init_result = self._create_buffer(_FUNCTION_LIST_BUFFER, text=_sign_payload(self, []))
        if not isinstance(init_result, dict) or not init_result.get("ok"):
            raise RuntimeError(f"failed to create function list buffer: {init_result.get('error')}")

        # Guard the catalog buffer against unsigned writes.
        hook_result = self.register_buffer_update_hook(
            _FUNCTION_LIST_BUFFER,
            "readonly_sentinel",
            _make_readonly_sentinel(self),
        )
        if not isinstance(hook_result, dict) or not hook_result.get("ok"):
            raise RuntimeError(f"failed to register readonly_sentinel on {_FUNCTION_LIST_BUFFER}: {hook_result.get('error')}")

    @sandbox
    async def create_function(
        self,
        name: str,
        callable: Callable,
        short_description: str = "",
        long_description: str = "",
    ) -> dict[str, Any]:
        # Guard: reject duplicate names outright — the namespace is flat and must stay unique.
        if name in self._functions:
            return {
                "ok": False,
                "error": f"Function '{name}' already registered.",
            }

        # Guard: prevent the reserved catalog buffer name from being used as a function name.
        if name == _FUNCTION_LIST_BUFFER:
            return {
                "ok": False,
                "error": f"'{name}' is a reserved name.",
            }

        # Extract metadata from the callable; fall back to its docstring when descriptions are absent.
        sig = inspect.signature(callable)
        doc = callable.__doc__ or ""
        if not short_description:
            short_description = doc.split("\n")[0].strip() if doc else ""
        if not long_description:
            long_description = doc.strip() if doc else ""

        # Assemble the function entry with all metadata, timestamps, and introspection data.
        now = time.time()
        entry = FunctionEntry(
            name=name,
            callable=callable,
            short_description=short_description,
            long_description=long_description,
            signature=sig,
            return_annotation=sig.return_annotation,
            created_at=now,
            modified_at=now,
        )

        # Mirror the function entry into a signed JSON buffer the agent can read;
        # create_buffer with overwrite=True fires existing buffer hooks automatically.
        buf_result = await self.create_buffer(
            _build_buffer_name(name),
            text=_sign_payload(self, _build_function_meta(entry)),
            overwrite=True,
        )
        if not isinstance(buf_result, dict) or not buf_result.get("ok"):
            return {
                "ok": False,
                "error": f"Cascaded error due to buffer creation for function description: {buf_result.get('error')}",
            }

        # Guard the new meta buffer against unsigned writes.
        hook_result = self.register_buffer_update_hook(
            _build_buffer_name(name),
            "readonly_sentinel",
            _make_readonly_sentinel(self),
        )
        if not isinstance(hook_result, dict) or not hook_result.get("ok"):
            return {
                "ok": False,
                "error": f"Function '{name}' created but could not be protected: {hook_result.get('error')}",
            }

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
        buf_name = _build_buffer_name(name)

        # Unregister the write guard before dropping — the sentinel would block the drop.
        self.unregister_buffer_update_hook(buf_name, "readonly_sentinel")

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
            "returns": str(entry.return_annotation) if entry.return_annotation is not inspect.Parameter.empty else "None",
        }
