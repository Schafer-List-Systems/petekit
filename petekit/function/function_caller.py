"""FunctionCaller — FunctionManager subclass that provides runtime function invocation."""

from __future__ import annotations

import inspect
from typing import Any

from peteos import tool
from peteos.oap.agentic_object import AgenticObject

from petekit.function.function_manager import FunctionManager
from petekit.text.buffer_manager import BufferManager

_CALL_RESULT_BUFFER_NAME_BASE = "tmp:call:"


def _next_call_id(caller) -> int:
    # Atomically grab the next call ID for a result buffer.
    call_id = caller._function_call_id_counter
    caller._function_call_id_counter += 1
    return call_id


def _call_result_buffer_name(call_id: int) -> str:
    # Derive the canonical result buffer name for the given call ID.
    return f"{_CALL_RESULT_BUFFER_NAME_BASE}{call_id}"


def _spill_to_buffer(caller, value: Any) -> str:
    # Serialize the value to a string for buffering.
    result_text = str(value)

    # Allocate a unique call ID and derive the tmp buffer name.
    call_id = _next_call_id(caller)
    buf_name = _call_result_buffer_name(call_id)

    # Write to the buffer and validate the result.
    create_result = AgenticObject.create_buffer(caller, buf_name, text=result_text)
    if not isinstance(create_result, dict) or not create_result.get("ok"):
        raise RuntimeError(f"Failed to spill result to {buf_name}: {create_result.get('error')}")
    return buf_name


class FunctionCaller(FunctionManager, AgenticObject):
    """FunctionManager extension that provides a tool for calling registered functions by name.
    Large results are spilled to a temporary result buffer instead of being returned inline."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._function_call_id_counter: int = 0

    @tool
    async def call_function(
        self,
        name: str,
        kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call a registered function by name with the given keyword arguments.
        If the result is larger than 8000 chars it is written to a tmp buffer instead.
        Returns a dict with ok/error, plus either result=... or result_buffer=tmp:call:<id>.
        Awaitable results are automatically awaited."""
        # Resolve the callable — get_function is inherited from FunctionManager.
        resolve_result = self.get_function(name)
        if not isinstance(resolve_result, dict) or not resolve_result.get("ok"):
            return resolve_result  # error dict

        func = resolve_result["callable"]
        call_kwargs = kwargs if kwargs is not None else {}

        # Guard: reject unknown argument names.
        for k in call_kwargs:
            if k not in resolve_result["args"]:
                return {"ok": False, "error": f"Unknown argument '{k}' for '{name}'."}

        # Guard: require all non-default parameters.
        for pname in resolve_result["required_args"]:
            if pname not in call_kwargs:
                return {"ok": False, "error": f"Missing required argument '{pname}' for '{name}'."}

        # Call and await — non-usage exceptions propagate.
        result = func(**call_kwargs)
        if inspect.iscoroutine(result):
            result = await result

        # Spill to a buffer if the serialized result is too large; otherwise return inline.
        result_str = str(result)
        if len(result_str) > BufferManager._MAX_CHUNK_CHARS:
            buf_name = _spill_to_buffer(self, result)
            return {"ok": True, "result_buffer": buf_name}
        return {"ok": True, "result": result}
