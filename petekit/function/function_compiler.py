"""FunctionCompiler — FunctionManager subclass that compiles source-code strings into live callables."""

from __future__ import annotations

import ast
import inspect
import logging
from pathlib import Path
from typing import Any, Callable

from peteos.oap.agentic_object import AgenticObject
from peteos.sandbox import SandboxBuilder

from petekit.function.function_manager import FunctionManager
from petekit.text.buffer_manager import Buffer, HookPoint, _get_persistence_dir

_CODE_BUFFER_PREFIX = "system:code:"
_FUNCTION_COMPILER_DOC_BUFFER = "doc:function_compiler"
_logger = logging.getLogger(__name__)


_FUNCTION_COMPILER_DOC = """\
## Schema

Each compiled function lives in a `system:code:<name>` buffer. The buffer holds the
Python source. The function is automatically compiled and registered when the buffer
is created, and re-compiled when the buffer is updated.

## Create a function

Create a `system:code:<name>` buffer with your Python source. The function name in the
source must match the buffer name suffix. Only one top-level `def` is allowed.

Result: `{"ok": true, "name": "...", "args": [...], "returns": "..."}`

On error: `{"ok": false, "error": "SyntaxError at line 2: ..."}`

The first line of the docstring becomes the short description. The full docstring
is the long description. The function is immediately callable.

## Update a function

Overwrite the `system:code:<name>` buffer with new source. The function is re-compiled
automatically. If the source is invalid the update is rejected with a compile error
and the previous source remains active.

## Drop a function

Drop the `system:code:<name>` buffer. The function and its introspection buffer are
removed automatically.
"""


# --- Internal helpers ---

def _validate_and_trial_compile(
    compiler: "FunctionCompiler",
    source: str,
) -> str:
    """Validate source structure and run a trial compilation without committing to the real sandbox.

    Parses the AST, enforces a single FunctionDef, and compiles into a throwaway builder.
    Returns the function name on success. Raises on any validation or compilation failure.
    """
    # Parse and require exactly one FunctionDef as the sole top-level construct.
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"SyntaxError at line {e.lineno}: {e.msg}")

    # The source must contain exactly one top-level node, and it must be a FunctionDef.
    top_level_nodes = list(ast.iter_child_nodes(tree))
    if len(top_level_nodes) != 1:
        raise ValueError(f"Source must contain exactly one top-level construct, found {len(top_level_nodes)}.")
    if not isinstance(top_level_nodes[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        node_type = type(top_level_nodes[0]).__name__
        raise ValueError(f"Top-level construct must be a function definition, found '{node_type}'.")

    # Derive the canonical function name from the AST — the single source of truth.
    func_name = top_level_nodes[0].name

    # Trial compilation — mutate a throwaway child builder, not the real one.
    trial_name = f"{compiler._oap_sandbox_builder._name}_trial"
    trial = SandboxBuilder(trial_name, base=compiler._oap_sandbox_builder)

    # Attempt to add to the trial builder — catches all compilation errors.
    try:
        entries = trial.add_source_code(source)
    except Exception as e:
        raise ValueError(f"Compilation failed: {e}")
    if not entries:
        raise ValueError(f"add_source_code returned no entries for '{func_name}'.")

    return func_name


def _compile_function(
    compiler: "FunctionCompiler",
    source: str,
    buffer_name: str | None = None,
) -> tuple[str, Callable]:
    """Validate and compile source into a sandbox proxy.

    Parses the source to extract the function name from the AST. Requires exactly one
    FunctionDef as the sole top-level construct. Commits to the OAP sandbox on success.

    When buffer_name is given, reads the persisted source file from the BufferManager's
    persistence directory so the file is imported as a module and debuggers can resolve
    breakpoints in it. This avoids a redundant in-memory string as the compilation input.

    Returns (func_name, proxy) on success. Raises on any validation or compilation failure.
    """
    # Validation and trial compilation — raises on failure.
    func_name = _validate_and_trial_compile(compiler, source)

    # Derive the source path from the persisted buffer file when a buffer name is given.
    if buffer_name is not None:
        source_path: Path = _get_persistence_dir(compiler) / buffer_name
    else:
        source_path = None

    # Phase 2: commit to the real instance-level sandbox builder.
    compiler._oap_sandbox_builder.add_source_code(source_path if source_path is not None else source)

    # Return name and proxy from the real sandbox.
    sandbox = compiler._oap_sandbox_builder.get_sandbox(freeze_namespaces=False)
    proxy = getattr(sandbox, func_name)
    return (func_name, proxy)


def _make_stable_wrapper(compiler: "FunctionCompiler", func_name: str) -> Callable:
    """Build a stable delegating wrapper whose identity is fixed but delegates to the live sandbox proxy.

    The wrapper is registered once and never replaced. Every call forwards to the
    current sandbox proxy, so callers holding a reference are never stale.
    """
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


def _make_on_update_code(compiler: "FunctionCompiler", func_name: str):
    """Factory producing a pre-update hook that validates source or handles the drop sentinel.

    When new_text is None the drop sentinel is active — unregisters hooks and delegates
    to the parent drop_function. Otherwise validates AST structure, enforces a single
    function def, and runs a trial compilation before allowing the update.
    """
    async def on_update_code(buf: Buffer, old_text: str | None, start: int, end: int, new_text: str | None) -> str | None:
        # Drop sentinel — new_text is None means the buffer is being dropped.
        if new_text is None:
            buf_name = buf.name

            # Unregister our own hooks before the buffer is removed.
            compiler.unregister_buffer_hook(HookPoint.ON_UPDATE_BUFFER, buf_name, "on_update_code")
            compiler.unregister_buffer_hook(HookPoint.ON_BUFFER_UPDATED, buf_name, "on_code_updated")

            # Delegate to parent — drops the introspection buffer and the callable.
            await compiler.drop_function(func_name)
            return None  # allow the drop

        # Validate structure and run trial compilation — raises on any error.
        try:
            compiled_name = _validate_and_trial_compile(compiler, new_text)
        except ValueError as e:
            return str(e)

        # Name from AST must match the buffer we're registered on.
        if compiled_name != func_name:
            return f"Function name must be '{func_name}', found '{compiled_name}'"

        return None  # validation passed — allow the buffer update

    return on_update_code


def _make_on_code_updated(compiler: FunctionCompiler, func_name: str, buf_name: str):
    """Factory producing a post-update hook that commits validated source to the sandbox.

    Called after the buffer update has already succeeded. Reads the new content from the
    persistence path, registers it in the real sandbox, then refreshes the function list.
    """
    async def on_code_updated(buf: Buffer, old_text: str | None, start: int, end: int, new_text: str | None) -> None:
        # Read the new source from the persisted buffer file — the authoritative content after the update.
        source_path = _get_persistence_dir(compiler) / buf_name
        if not source_path.exists():
            return

        # Commit to the real instance-level sandbox builder.
        try:
            compiler._oap_sandbox_builder.add_source_code(source_path)
        except Exception:
            return  # best-effort — log and continue

        # Refresh the function catalog so the updated entry is visible.
        from petekit.function.function_manager import _refresh_list_buffer
        await _refresh_list_buffer(compiler)

    return on_code_updated


# --- Hook handlers ---

async def _on_create_code(
    compiler: FunctionCompiler,
    buffer_name: str,
    args: dict[str, Any],
) -> str | None:
    """ON_CREATE_BUFFER hook for the system:code: scope — compiles and registers a function from source text."""
    # Parse the source from the create args.
    source: str = args.get("text", "")
    if not source.strip():
        return "source buffer is empty"

    # Validate and compile — raises on any error.
    try:
        func_name, proxy = _compile_function(compiler, source)
    except ValueError as e:
        return str(e)

    # Extract docstring for descriptions.
    doc = proxy.__doc__ or ""
    short_description = doc.split("\n")[0].strip() if doc else ""
    long_description = doc.strip() if doc else ""

    # Build stable wrapper — its identity is fixed, it delegates to the live sandbox proxy.
    stable_wrapper = _make_stable_wrapper(compiler, func_name)

    # Register in the function namespace via the parent.
    create_result = await super(FunctionCompiler, compiler).create_function(
        func_name,
        stable_wrapper,
        short_description,
        long_description,
    )
    if not isinstance(create_result, dict) or not create_result.get("ok"):
        # Roll back: remove the compiled source from the sandbox since manager registration failed.
        compiler._oap_sandbox_builder.remove_source_code(func_name)
        return create_result.get("error", f"failed to create function '{func_name}'")

    _logger.info(
        "[function_compiler] registered function '%s' for %s",
        func_name,
        compiler.__class__.__name__,
    )
    return None


async def _on_code_created(
    compiler: FunctionCompiler,
    buffer_name: str,
    args: dict[str, Any],
) -> str | None:
    """ON_BUFFER_CREATED hook for the system:code: scope — registers pre/post update hooks on the new buffer."""
    # Derive the function name from the buffer name — prefix confirmed by scope registration.
    func_name = buffer_name[len(_CODE_BUFFER_PREFIX) :]

    # Register the on_update_code guard as a pre-update hook — validates source before mutation.
    guard_result = compiler.register_buffer_hook(
        HookPoint.ON_UPDATE_BUFFER,
        buffer_name,
        "on_update_code",
        _make_on_update_code(compiler, func_name),
    )
    if not isinstance(guard_result, dict) or not guard_result.get("ok"):
        return f"failed to register on_update_code on {buffer_name}: {guard_result.get('error')}"

    # Register the on_code_updated post-update hook — commits to sandbox and refreshes the catalog.
    commit_result = compiler.register_buffer_hook(
        HookPoint.ON_BUFFER_UPDATED,
        buffer_name,
        "on_code_updated",
        _make_on_code_updated(compiler, func_name, buffer_name),
    )
    if not isinstance(commit_result, dict) or not commit_result.get("ok"):
        return f"failed to register on_code_updated on {buffer_name}: {commit_result.get('error')}"

    return None


class FunctionCompiler(FunctionManager, AgenticObject):
    """I compile Python source-code strings into live callable functions.

    See doc:function_compiler for the full schema and usage guide.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Register the ON_CREATE_BUFFER scope hook — compiles and registers functions on buffer creation.
        on_create_result = self.register_buffer_hook(
            HookPoint.ON_CREATE_BUFFER,
            _CODE_BUFFER_PREFIX,
            "on_create_code",
            _on_create_code,
        )
        if not isinstance(on_create_result, dict) or not on_create_result.get("ok"):
            raise RuntimeError(f"failed to register on_create_code hook: {on_create_result.get('error')}")

        # Register the ON_BUFFER_CREATED scope hook — registers pre/post update hooks on new code buffers.
        on_created_result = self.register_buffer_hook(
            HookPoint.ON_BUFFER_CREATED,
            _CODE_BUFFER_PREFIX,
            "on_code_created",
            _on_code_created,
        )
        if not isinstance(on_created_result, dict) or not on_created_result.get("ok"):
            raise RuntimeError(f"failed to register on_code_created hook: {on_created_result.get('error')}")

        # Seed the function compiler documentation buffer.
        self._create_buffer(_FUNCTION_COMPILER_DOC_BUFFER, text=_FUNCTION_COMPILER_DOC)
