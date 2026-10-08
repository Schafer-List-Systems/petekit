"""FunctionCompiler — FunctionManager subclass that compiles source-code strings into live callables."""

from __future__ import annotations

import ast
from typing import Any, Callable

from peteos import tool, sandbox
from peteos.oap.agentic_object import AgenticObject
from peteos.sandbox import SandboxBuilder

from petekit.function.function_manager import FunctionManager

_CODE_BUFFER_PREFIX = "system:code:"


def _code_buffer_name(name: str) -> str:
    """Return the source-code buffer name for the given function name."""
    return f"{_CODE_BUFFER_PREFIX}{name}"


def _compile_function(compiler: "FunctionCompiler", func_name: str, source: str) -> Callable | str:
    """Validate source in a trial builder, then commit to the OAP sandbox builder."""
    # Parse and require exactly one FunctionDef as the sole top-level construct.
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}"

    # The source must contain exactly one top-level node, and it must be a FunctionDef.
    top_level_nodes = list(ast.iter_child_nodes(tree))
    if len(top_level_nodes) != 1:
        return f"Source must contain exactly one top-level construct, found {len(top_level_nodes)}."
    if not isinstance(top_level_nodes[0], ast.FunctionDef):
        node_type = type(top_level_nodes[0]).__name__
        return f"Top-level construct must be a function definition, found '{node_type}'."
    if top_level_nodes[0].name != func_name:
        return f"Function name must be '{func_name}', found '{top_level_nodes[0].name}'."

    # Phase 1: trial compilation — mutate a throwaway child builder, not the real one.
    trial_name = f"{compiler._oap_sandbox_builder._name}_trial"
    trial = SandboxBuilder(trial_name, base=compiler._oap_sandbox_builder)

    # Attempt to add to the trial builder — catches all compilation errors.
    try:
        entries = trial.add_source_code(source)
    except Exception as e:
        return f"Compilation failed: {e}"
    if not entries:
        return f"add_source_code returned no entries for '{func_name}'."

    # Phase 2: commit to the real instance-level sandbox builder.
    try:
        compiler._oap_sandbox_builder.add_source_code(source)
    except Exception as e:
        return f"Failed to register '{func_name}' in sandbox: {e}"

    # Return the proxy from the real sandbox.
    sandbox = compiler._oap_sandbox_builder.get_sandbox(freeze_namespaces=False)
    proxy = getattr(sandbox, func_name)
    return proxy


def _make_stable_wrapper(compiler: "FunctionCompiler", func_name: str) -> Callable:
    """Build a stable delegating wrapper whose identity is fixed but delegates to the live sandbox proxy.

    The wrapper is registered once and never replaced. Every call forwards to the
    current sandbox proxy, so callers holding a reference are never stale.
    """
    import inspect

    # Stable wrapper — delegates to the live sandbox function on every call.
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        sandbox = compiler._oap_sandbox_builder.get_sandbox(freeze_namespaces=False)
        func = getattr(sandbox, func_name)
        return func(*args, **kwargs)

    # Steal signature from the live proxy so introspection sees the real interface.
    sandbox = compiler._oap_sandbox_builder.get_sandbox(freeze_namespaces=False)
    proxy = getattr(sandbox, func_name)
    wrapper.__code__ = proxy.__code__
    wrapper.__signature__ = inspect.signature(proxy)
    wrapper.__name__ = proxy.__name__
    wrapper.__doc__ = getattr(proxy, "__doc__", "")

    return wrapper


def _make_compilation_sentinel(compiler: "FunctionCompiler", name: str):
    """Factory producing a sentinel hook that recompiles the live callable on every source write.

    On failure the write is rejected and the error is surfaced to the caller.
    On success the sandbox is updated — the stable wrapper in _functions is untouched.
    """
    def compilation_sentinel(buf: str, old_text: str, start: int, end: int, new_text: str) -> str | None:
        # Delegate to the two-phase compiler — trial, then commit on success.
        result = _compile_function(compiler, name, new_text)
        if isinstance(result, str):
            return result  # error — reject the write, surface the message
        return None  # success — sandbox already updated, wrapper in _functions is untouched

    return compilation_sentinel


class FunctionCompiler(FunctionManager, AgenticObject):
    """FunctionManager that compiles source-code strings into live callables.

    create_function accepts a `code` kwarg holding Python source. That source is stored
    in a system:code:{name} buffer. Every write to that buffer is intercepted by a
    compilation sentinel — on failure the write is rejected with the compile error.
    The live callable is updated in-place on each successful write.

    Usage:
        await fm.create_function("greet", code=\"\"\"
            def greet(name: str) -> str:
                return f"hello {name}"
        \"\"\")

    The function name used in the source must match the `name` argument.
    The source must contain exactly one top-level `def`.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    @tool
    async def create_function(
        self,
        name: str,
        short_description: str = "",
        long_description: str = "",
        *,
        code: str,
    ) -> dict[str, Any]:
        # Compile and register in sandbox — build the stable delegating wrapper.
        compile_result = _compile_function(self, name, code)
        if isinstance(compile_result, str):
            return {"ok": False, "error": compile_result}

        # Build stable wrapper — its identity is fixed, it delegates to the live sandbox proxy.
        stable_wrapper = _make_stable_wrapper(self, name)

        # Register the stable wrapper in the namespace — parent runs all guards first.
        create_result = await super().create_function(
            name,
            stable_wrapper,
            short_description,
            long_description,
        )
        if not isinstance(create_result, dict) or not create_result.get("ok"):
            return create_result

        # Write the source to the code buffer, replacing any existing content.
        buf_result = await self.create_buffer(
            _code_buffer_name(name),
            text=code,
            overwrite=True,
        )
        if not isinstance(buf_result, dict) or not buf_result.get("ok"):
            # Roll back: unregister from parent.
            await super().drop_function(name)
            return {
                "ok": False,
                "error": f"Function '{name}' could not be created: {buf_result.get('error')}",
            }

        # Register the compilation sentinel so every source write is recompiled.
        hook_result = self.register_buffer_update_hook(
            _code_buffer_name(name),
            "compilation_sentinel",
            _make_compilation_sentinel(self, name),
        )
        if not isinstance(hook_result, dict) or not hook_result.get("ok"):
            # Roll back: unregister from parent.
            await super().drop_function(name)
            return {
                "ok": False,
                "error": f"Function '{name}' created but compilation hook could not be registered: {hook_result.get('error')}",
            }

        return create_result

    @tool
    async def drop_function(self, name: str) -> dict[str, Any]:
        # Drop the code buffer first — no sentinel blocks this since we unguard first below.
        buf_name = _code_buffer_name(name)
        self.unregister_buffer_update_hook(buf_name, "compilation_sentinel")
        await self.drop_buffer(buf_name)

        # Mirror to parent — drops the introspection buffer.
        return await super().drop_function(name)
