"""Function manager — runtime Callable management built on top of BufferManager."""

from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from peteos import tool, sandbox
from peteos.oap.agentic_object import AgenticObject

from petekit.text.buffer_manager import Buffer, BufferEntry, BufferManager
from petekit.utils.text_formatters import format_dict_list_for_buffer

_FUNCTION_LIST_BUFFER = "function:list"


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
    return f"function:{name}"


def _signature_str(sig: inspect.Signature) -> str:
    params = []
    for p in sig.parameters.values():
        if p.default is inspect.Parameter.empty:
            params.append(p.name)
        else:
            params.append(f"{p.name}={p.default!r}")
    return ", ".join(params)


def _return_str(ann: Any) -> str:
    if ann is inspect.Parameter.empty:
        return "None"
    return str(ann)


def _param_type_str(param: inspect.Parameter) -> str:
    if param.annotation is inspect.Parameter.empty:
        return "Any"
    return str(param.annotation)


def _build_function_buffer(entry: FunctionEntry) -> str:
    # Serialize the full function entry into a text format the agent can read.
    # This includes name, short/long description, parameter details,
    # and return annotation — everything needed for the agent to use the function.
    lines = [
        f"# Function: {entry.name}",
        f"# Short: {entry.short_description}",
        f"# Created: {entry.created_at}",
        f"# Modified: {entry.modified_at}",
        "",
        f"## Description",
        entry.long_description,
        "",
        "## Signature",
        f"def {entry.name}({_signature_str(entry.signature)}) -> {_return_str(entry.return_annotation)}:",
        "",
        "## Parameters",
    ]
    for param_name, param in entry.signature.parameters.items():
        lines.append(f"#   {param_name}: {_param_type_str(param)}")
        if param.default is inspect.Parameter.empty:
            lines.append(f"#     default: (no default)")
        else:
            lines.append(f"#     default: {param.default!r}")
    lines.extend([
        "",
        f"## Returns: {_return_str(entry.return_annotation)}",
    ])
    return "\n".join(lines) + "\n"


def _refresh_list_buffer(fm: FunctionManager) -> asyncio.Task:
    # Collect each function's name, arg summary, and short description for the catalog.
    records = []
    for entry in fm._functions.values():
        # Build an arg summary so the agent can see the function signature at a glance.
        arg_summary = ", ".join(
            p.name for p in entry.signature.parameters.values()
        )
        records.append({
            "name": entry.name,
            "args": f"({arg_summary})",
            "description": entry.short_description,
        })

    # Format and persist the flat catalog to the function:list buffer for agent browsing.
    text = format_dict_list_for_buffer(records)
    loop = asyncio.get_running_loop()
    return loop.create_task(fm.write_buffer(_FUNCTION_LIST_BUFFER, text=text, start=0))


class FunctionManager(BufferManager, AgenticObject):
    """A BufferManager that also manages a namespace of runtime-created Callables.
    Each function lives in self._functions and is mirrored to a function:<name> buffer
    for agent introspection. A function:list buffer catalogs all registered functions
    grouped by category.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._functions: dict[str, FunctionEntry] = {}
        self._create_buffer(_FUNCTION_LIST_BUFFER, text="[]")

    @tool
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

        # Mirror the function entry into a text buffer the agent can read and search;
        # create_buffer with overwrite=True fires existing buffer hooks automatically.
        buf_result = await self.create_buffer(_build_buffer_name(name), text=_build_function_buffer(entry), overwrite=True)
        if not buf_result.get("ok"):
            return {
                "ok": False,
                "error": f"Cascaded error due to buffer creation for function description: {buf_result.get("error")}",
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

