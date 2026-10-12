"""FunctionCompiler — FunctionManager subclass that compiles source-code strings into live callables."""

from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
from typing import Any, Callable

from peteos import tool, sandbox
from peteos.oap.agentic_object import AgenticObject, RestoreHook
from peteos.sandbox import SandboxBuilder

from petekit.function.function_manager import FunctionManager
from petekit.text.buffer_manager import HookPoint

_CODE_BUFFER_PREFIX = "system:code:"
_logger = logging.getLogger(__name__)


# --- Persistence helpers (module-level to keep the class footprint small) ---

def _get_persistence_dir(agent: AgenticObject) -> Path:
    """Return the FunctionCompiler persistence directory inside the agent's instance dir."""
    return agent.instance_dir / "function_compiler"


def _persist_function(agent: AgenticObject, name: str, source: str, short: str, long: str) -> None:
    """Write function source and metadata to disk for persistence."""
    import json

    # Ensure the persistence directory exists.
    dir = _get_persistence_dir(agent)
    dir.mkdir(parents=True, exist_ok=True)

    # Raw source file — IDE/debugging friendly, direct Python syntax
    (dir / f"{name}.py").write_text(source)

    # Metadata JSON
    meta = {
        "name": name,
        "short_description": short,
        "long_description": long,
    }
    (dir / f"{name}.json").write_text(json.dumps(meta, indent=2))


def _drop_persisted_function(agent: AgenticObject, name: str) -> None:
    """Remove persisted source and metadata for the named function."""
    dir = _get_persistence_dir(agent)

    # Remove source and metadata files; missing_ok=True handles the already-gone case.
    (dir / f"{name}.py").unlink(missing_ok=True)
    (dir / f"{name}.json").unlink(missing_ok=True)


def _overwrite_persisted_source(agent: AgenticObject, name: str, source: str) -> None:
    """Overwrite the persisted source file after a successful in-memory recompilation."""
    dir = _get_persistence_dir(agent)
    # Write only the source file — descriptions live in .json and are not updated on edit.
    (dir / f"{name}.py").write_text(source)


def _restore_functions(agent: AgenticObject) -> None:
    """Re-compile all persisted functions from disk on agent restore()."""
    dir = _get_persistence_dir(agent)

    # Nothing to restore if the directory was never created.
    if not dir.exists():
        return

    # Iterate all persisted metadata files and replay create_function for each.
    for json_file in dir.glob("*.json"):

        # Extract function name and skip if already live.
        name = json_file.stem
        if name in agent._functions:
            continue

        # Replay create_function for this function — loads source, recompiles, re-registers.
        try:
            meta = json.loads(json_file.read_text())
            source = (dir / f"{name}.py").read_text()
            agent.create_function(
                name=name,
                code=source,
                short_description=meta.get("short_description", ""),
                long_description=meta.get("long_description", ""),
            )
            _logger.info(
                "[function_compiler] restored function '%s' for %s",
                name,
                agent.__class__.__name__,
            )

        # Log failures so the maintainer knows what failed and why.
        except Exception as e:
            _logger.error(
                "[function_compiler] failed to restore function '%s' for %s: %s",
                name,
                agent.__class__.__name__,
                e,
            )


# --- Internal helpers ---

def _code_buffer_name(name: str) -> str:
    """Return the source-code buffer name for the given function name."""
    return f"{_CODE_BUFFER_PREFIX}{name}"


def _compile_function(
    compiler: "FunctionCompiler",
    func_name: str,
    source: str,
    source_path: "Path | None" = None,
) -> Callable | str:
    """Validate source in a trial builder, then commit to the OAP sandbox builder.

    When source_path is given the Path is passed to add_source_code so the source
    file is imported as a module and debuggers can resolve breakpoints in it.
    """
    # Parse and require exactly one FunctionDef as the sole top-level construct.
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg}"

    # The source must contain exactly one top-level node, and it must be a FunctionDef.
    top_level_nodes = list(ast.iter_child_nodes(tree))
    if len(top_level_nodes) != 1:
        return f"Source must contain exactly one top-level construct, found {len(top_level_nodes)}."
    if not isinstance(top_level_nodes[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        node_type = type(top_level_nodes[0]).__name__
        return f"Top-level construct must be a function definition, found '{node_type}'."
    if top_level_nodes[0].name != func_name:
        return f"Function name must be '{func_name}', found '{top_level_nodes[0].name}'."

    # Phase 1: trial compilation — mutate a throwaway child builder, not the real one.
    trial_name = f"{compiler._oap_sandbox_builder._name}_trial"
    trial = SandboxBuilder(trial_name, base=compiler._oap_sandbox_builder)

    # Attempt to add to the trial builder — catches all compilation errors.
    try:
        entries = trial.add_source_code(source_path if source_path is not None else source)
    except Exception as e:
        return f"Compilation failed: {e}"
    if not entries:
        return f"add_source_code returned no entries for '{func_name}'."

    # Phase 2: commit to the real instance-level sandbox builder.
    try:
        compiler._oap_sandbox_builder.add_source_code(source_path if source_path is not None else source)
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

    # Stable wrapper — delegates to the live sandbox proxy on every call.
    # The proxy from getattr already has sandbox bound via closure capture.
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        sandbox = compiler._oap_sandbox_builder.get_sandbox(freeze_namespaces=False)
        proxy = getattr(sandbox, func_name)
        return proxy(*args, **kwargs)

    # Mirror signature from the proxy so inspect.signature sees the real interface.
    sandbox = compiler._oap_sandbox_builder.get_sandbox(freeze_namespaces=False)
    proxy = getattr(sandbox, func_name)
    wrapper.__signature__ = inspect.signature(proxy)
    wrapper.__name__ = proxy.__name__
    wrapper.__doc__ = getattr(proxy, "__doc__", "")

    return wrapper


def _make_compilation_sentinel(compiler: "FunctionCompiler", name: str):
    """Factory producing a sentinel hook that recompiles the live callable on every source write.

    On failure the write is rejected and the error is surfaced to the caller.
    On success the sandbox and disk are both updated.
    """
    def compilation_sentinel(buf: str, old_text: str, start: int, end: int, new_text: str) -> str | None:
        # Write the updated source to disk first — the file on disk is the debugger's view of the code.
        _overwrite_persisted_source(compiler, name, new_text)
        source_path = _get_persistence_dir(compiler) / f"{name}.py"

        # Delegate to the two-phase compiler — trial, then commit on success.
        result = _compile_function(compiler, name, new_text, source_path)
        if isinstance(result, str):
            return result  # error — reject the write, surface the message

        return None  # success — sandbox and disk both updated

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

        # Register restore hook so persisted functions are re-compiled after a restart.
        self._oap_restore_hooks["function_compiler"] = RestoreHook(
            callable_=_restore_functions,
            dependencies=[],
        )

    @tool
    async def create_function(
        self,
        name: str,
        short_description: str = "",
        long_description: str = "",
        *,
        code: str,
    ) -> dict[str, Any]:
        # Persist to disk first — the file on disk is the debugger's view of the code.
        _persist_function(self, name, code, short_description, long_description)
        source_path = _get_persistence_dir(self) / f"{name}.py"

        # Compile and register in sandbox — build the stable delegating wrapper.
        compile_result = _compile_function(self, name, code, source_path)
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
        hook_result = self.register_buffer_hook(
            HookPoint.ON_UPDATE_BUFFER,
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
        self.unregister_buffer_hook(HookPoint.ON_UPDATE_BUFFER, buf_name, "compilation_sentinel")
        await self.drop_buffer(buf_name)

        # Mirror to parent — drops the introspection buffer.
        result = await super().drop_function(name)

        # Remove persisted source and metadata from disk.
        _drop_persisted_function(self, name)

        return result
