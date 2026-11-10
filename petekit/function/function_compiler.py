"""FunctionCompiler -- FunctionManager subclass that compiles source-code strings into live callables."""

from __future__ import annotations

import ast
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Callable

from peteos.oap.agentic_object import AgenticObject
from peteos.sandbox import SandboxBuilder

from petekit.function.function_manager import FunctionManager, _check_signatures_compatible
from petekit.text.buffer_manager import Buffer, BufferManager, HookPoint, AfterBufferDropped, AfterUpdateBuffer, BeforeDropBuffer, BeforeUpdateBuffer, _get_persistence_dir
from petekit.text.option_buffer_manager import OptionBufferManager, _origin_name_from_option_buffer_name

_CODE_BUFFER_PREFIX = "system:code:"
_CODE_BUFFER_SUFFIX = "[python]"
_CODE_BUFFER_OPTION_BUFFER_PATTERN = rf".+\[{_CODE_BUFFER_SUFFIX[1:-1]}\]$"
_FUNCTION_COMPILER_DOC_BUFFER = "doc:function_compiler"
_logger = logging.getLogger(__name__)


_FUNCTION_COMPILER_DOC = """\
## Compile Python source into live callable functions

The FunctionCompiler turns arbitrary source code buffers into registered, callable
functions. Source is validated, compiled, and wired into the sandbox on every change.

## Create a function

**Step 1** — Create a buffer holding your Python source. A `.py` suffix is
recommended. Only one top-level `def` is allowed.

```
buffer: my_module.py
content: def greet(name): return f"Hello, {name}!"
```

**Step 2** — Create the `[python]` option buffer on that source to trigger compilation
and registration. The config can be an empty `{}` — the origin is derived from the
option buffer name automatically.

Buffer name: `my_module.py[python]`
Config: `{}`

The function is compiled and registered. Its name is whatever you called it
in the Python source — in the example above: `greet`.

## Update a function

Overwrite the source buffer (`my_module.py`) with new Python code. The `[python]`
option buffer update triggers re-validation, signature-compatibility checking, and
re-compilation. If the new signature is incompatible with existing callers the update
is rejected. The function must not be renamed — the name in the source must match the
registered name.

## Drop a function

Drop the `[python]` option buffer. The function is unregistered. Active `CallableRef`
references block the drop until they are released.

## Constraints

- The source must define exactly one function at the top level.
- Function names are fixed for the lifetime of the registration — renaming is
  rejected at the `before_update` hook.
- Signature changes are accepted only when they are backward-compatible
  (no required parameter removed, no parameter kind narrowed).
"""


# -------------------------------------------------------------------------------------------------------------------- #
# Utils                                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #


def _get_code_config(buffer: Buffer) -> dict[str, Any] | str:
    """Read and parse the JSON config from a code option buffer.

    Returns the parsed config dict on success, or an error string if the buffer is
    absent, empty, or contains invalid JSON.
    """
    # Read the raw text from the buffer.
    buffer_text = buffer.get_text()
    if not buffer_text:
        return f"[FunctionCompiler] buffer '{buffer.name}' is empty"

    # Parse the JSON config from the buffer text.
    try:
        config = json.loads(buffer_text)
    except json.JSONDecodeError as e:
        return f"[FunctionCompiler] buffer '{buffer.name}' has invalid JSON: {e}"

    return config


def _validate_and_trial_compile(
    compiler: "FunctionCompiler",
    source: str,
) -> tuple[str, inspect.Signature]:
    """Validate source structure and run a trial compilation without committing to the real sandbox.

    Parses the AST, enforces a single FunctionDef, and compiles into a throwaway builder.
    Returns (func_name, signature) on success. Raises on any validation or compilation failure.
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

    # Derive the canonical function name from the AST -- the single source of truth.
    func_name = top_level_nodes[0].name

    # Trial compilation -- mutate a throwaway child builder, not the real one.
    trial_name = f"{compiler._oap_sandbox_builder._name}_trial"
    trial = SandboxBuilder(trial_name, base=compiler._oap_sandbox_builder)

    # Attempt to add to the trial builder -- catches all compilation errors.
    try:
        entries = trial.add_source_code(source)
    except Exception as e:
        raise ValueError(f"Compilation failed: {e}")
    if not entries:
        raise ValueError(f"add_source_code returned no entries for '{func_name}'.")

    # Extract and return the signature from the trial sandbox proxy.
    trial_sandbox = trial.get_sandbox(freeze_namespaces=False)
    proxy = getattr(trial_sandbox, func_name)
    return (func_name, inspect.signature(proxy))


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
    # Validation and trial compilation -- raises on failure.
    func_name, _sig = _validate_and_trial_compile(compiler, source)
    del _sig  # signature from trial compile is not needed in _compile_function

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


async def _register_function_with_hooks(
    compiler: FunctionCompiler,
    proxy: Callable,
    function_name: str,
    origin_buffer: Buffer,
    origin_name: str,
) -> str | None:
    """Register a compiled proxy in the function manager and attach update and drop hooks on the origin buffer."""
    # Enlist the function in the function manager so callers can acquire and call it.
    # Side effect: the callable becomes visible to get_function and callable refs are created.
    register_result = await compiler.register_function(proxy, function_name)
    if not register_result.get("ok"):
        return f"register failed for '{function_name}': {register_result.get('error')}"

    # Reject any source change that would rename the function or break existing callers
    # with an incompatible signature -- must happen before the buffer update is accepted.
    before_hook = origin_buffer.register_hook(
        HookPoint.BEFORE_UPDATE_BUFFER,
        "function_compiler:before_update_code",
        _before_update_code,
    )
    if not before_hook.get("ok"):
        return f"failed to register before_update_code on '{origin_name}': {before_hook.get('error')}"

    # Recompile the function and replace the registered callable after every source update.
    # This hook fires after the buffer update has already succeeded -- it performs the swap.
    after_hook = origin_buffer.register_hook(
        HookPoint.AFTER_UPDATE_BUFFER,
        "function_compiler:after_update_code",
        _after_update_code,
    )
    if not after_hook.get("ok"):
        return f"failed to register after_update_code on '{origin_name}': {after_hook.get('error')}"

    # Block the buffer drop when the function has active callable refs -- callers may depend on it.
    # The drop is rejected until every reference has been released.
    before_drop_hook = origin_buffer.register_hook(
        HookPoint.BEFORE_DROP_BUFFER,
        "function_compiler:before_drop_code",
        _before_drop_code,
    )
    if not before_drop_hook.get("ok"):
        return f"failed to register before_drop_code on '{origin_name}': {before_drop_hook.get('error')}"

    # Unregister the function from the function manager after the buffer is dropped.
    # Side effect: the callable is removed from the manager; callable refs become invalid.
    after_drop_hook = origin_buffer.register_hook(
        HookPoint.AFTER_DROP_BUFFER,
        "function_compiler:after_drop_code",
        _after_drop_code,
    )
    if not after_drop_hook.get("ok"):
        return f"failed to register after_drop_code on '{origin_name}': {after_drop_hook.get('error')}"

    return None


def _make_stable_wrapper(compiler: "FunctionCompiler", func_name: str) -> Callable:
    """Build a stable delegating wrapper whose identity is fixed but delegates to the live sandbox proxy.

    The wrapper is registered once and never replaced. Every call forwards to the
    current sandbox proxy, so callers holding a reference are never stale.
    """
    # Stable wrapper -- delegates to the live sandbox proxy on every call.
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


# -------------------------------------------------------------------------------------------------------------------- #
# Hooks                                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #


def _make_on_update_code(compiler: "FunctionCompiler", func_name: str) -> BeforeUpdateBuffer:
    """Factory producing a pre-update hook that validates source code on every buffer mutation.

    Validates AST structure, enforces a single function def, and runs a trial compilation
    before allowing the update. Drop is handled by BEFORE_DROP_BUFFER / AFTER_DROP_BUFFER hooks.
    """
    async def on_update_code(
        buffer_manager: BufferManager,
        old_buffer: Buffer,
        new_buffer: Buffer,
        args: dict[str, Any],
    ) -> bool | str | None:
        new_text = new_buffer.get_text()
        # Validate structure and run trial compilation -- raises on any error.
        try:
            compiled_name, _ = _validate_and_trial_compile(compiler, new_text)
        except ValueError as e:
            return str(e)

        # Name from AST must match the buffer we're registered on.
        if compiled_name != func_name:
            return f"Function name must be '{func_name}', found '{compiled_name}'"

        return None  # validation passed -- allow the buffer update

    return on_update_code


def _make_on_code_drop(compiler: "FunctionCompiler", func_name: str) -> BeforeDropBuffer:
    """Factory producing the BEFORE_DROP_BUFFER hook for system:code:.* buffers.
    Cascades the drop to remove the callable from the function namespace. The drop
    itself is allowed to proceed; this hook only performs the side effect.
    """
    async def on_code_drop(
        buffer_manager: BufferManager,
        buffer: Buffer,
        args: dict[str, Any],
    ) -> str | None:
        # Delegate to drop_function -- removes the introspection buffer and the callable.
        drop_result = await compiler.drop_function(func_name)
        # Log if the cascade failed, but still allow the buffer drop.
        if not drop_result.get("ok"):
            _logger.warning(f"[on_code_drop] cascade to drop_function('{func_name}') failed: {drop_result.get('error')}")
        return None  # always allow the drop

    return on_code_drop


def _make_on_code_dropped(compiler: "FunctionCompiler", func_name: str) -> AfterBufferDropped:
    """Factory producing the AFTER_DROP_BUFFER safety-net hook for system:code:.* buffers.

    Fires after the buffer is dropped. If the function is still registered, the normal
    cascade in on_code_drop failed somehow -- this logs the anomaly but does not
    re-attempt the cascade to avoid re-entrancy issues.
    """
    def on_code_dropped(
        buffer_manager: BufferManager,
        buffer: Buffer,
        args: dict[str, Any],
    ) -> None:
        # Check whether the function is still registered -- it should have been removed by on_code_drop.
        buffer_name = buffer.name
        if compiler.has_function(func_name):
            _logger.warning(f"[on_code_dropped] buffer '{buffer_name}' dropped but func '{func_name}' still in namespace -- normal cascade may have failed")
        else:
            _logger.debug(f"[on_code_dropped] buffer '{buffer_name}' dropped, func '{func_name}' already gone (normal)")

    return on_code_dropped


def _make_on_code_updated(compiler: FunctionCompiler, func_name: str, buf_name: str) -> AfterUpdateBuffer:
    """Factory producing a post-update hook that commits validated source to the sandbox.

    Called after the buffer update has already succeeded. Reads the new content from the
    persistence path, registers it in the real sandbox, then refreshes the function list.
    """
    async def on_code_updated(
        buffer_manager: BufferManager,
        buffer: Buffer,
        new_buffer: Buffer,
        args: dict[str, Any],
    ) -> None:
        # Read the new source from the persisted buffer file -- the authoritative content after the update.
        source_path = _get_persistence_dir(compiler) / buf_name
        if not source_path.exists():
            return

        # Commit to the real instance-level sandbox builder.
        try:
            compiler._oap_sandbox_builder.add_source_code(source_path)
        except Exception:
            return  # best-effort -- log and continue

    return on_code_updated


# -------------------------------------------------------------------------------------------------------------------- #
# Code Option Buffer Hooks                                                                                             #
# -------------------------------------------------------------------------------------------------------------------- #


async def _before_create_code_option(
    compiler: FunctionCompiler,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """BEFORE_UPDATE_BUFFER hook for [python] option buffers -- validate source before mutation."""
    # Get the JSON config from the code option buffer.
    config = _get_code_config(new_buffer)
    if isinstance(config, str):
        return config

    # Derive the origin buffer name from the option buffer name.
    origin_name = _origin_name_from_option_buffer_name(new_buffer.name)
    if origin_name is None:
        return f"[_before_update_code_option] cannot derive origin from '{new_buffer.name}'"

    # Read the source code from the origin buffer.
    origin_buffer = compiler.get_buffer(origin_name)
    if origin_buffer is None:
        return f"[_before_update_code_option] origin buffer '{origin_name}' not found"

    # Extract the source text -- the function body to be validated.
    source = origin_buffer.get_text()
    if not source:
        return f"[_before_update_code_option] source in '{origin_name}' is empty"

    # Validate and trial-compile -- return an error when the source is invalid.
    try:
        func_name, _ = _validate_and_trial_compile(compiler, source)
    except ValueError as e:
        return str(e)

    # Check for naming conflicts -- new functions must not already exist.
    if compiler.has_function(func_name):
        return f"[_before_update_code_option] function '{func_name}' already exists"

    # Persist the function name in the buffer config so it survives restarts.
    config["function_name"] = func_name
    new_buffer.set_text(json.dumps(config, indent=2) + "\n")


async def _before_update_code_option(
    compiler: FunctionCompiler,
    old_buffer: Buffer,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """BEFORE_UPDATE_BUFFER hook for [python] option buffers -- validate source before mutation."""
    # Get the JSON config from the code option buffer.
    config = _get_code_config(new_buffer)
    if isinstance(config, str):
        return config

    # Derive the origin buffer name from the option buffer name.
    origin_name = _origin_name_from_option_buffer_name(new_buffer.name)
    if origin_name is None:
        return f"[_before_update_code_option] cannot derive origin from '{new_buffer.name}'"

    # Read the source code from the origin buffer.
    origin_buffer = compiler.get_buffer(origin_name)
    if origin_buffer is None:
        return f"[_before_update_code_option] origin buffer '{origin_name}' not found"

    # Extract the source text -- the function body to be validated.
    source = origin_buffer.get_text()
    if not source:
        return f"[_before_update_code_option] source in '{origin_name}' is empty"

    # Validate and trial-compile -- return an error when the source is invalid.
    try:
        func_name, _ = _validate_and_trial_compile(compiler, source)
    except ValueError as e:
        return str(e)

    # Read the old config to compare function names.
    old_config = _get_code_config(old_buffer)
    if isinstance(old_config, str):
        return f"[_before_update_code_option] could not read old config: {old_config}"

    # Reject the update when the function name in the source differs from the persisted one.
    old_func_name = old_config.get("function_name")
    if old_func_name is not None and old_func_name != func_name:
        return f"[_before_update_code_option] renaming functions is not allowed ('{old_func_name}' -> '{func_name}')"

    # Persist the function name in the buffer config so it survives restarts.
    config["function_name"] = func_name
    new_buffer.set_text(json.dumps(config, indent=2) + "\n")

    return None


async def _after_update_code_option(
    compiler: FunctionCompiler,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """AFTER_UPDATE_BUFFER hook for code option buffers -- recompile from the origin buffer."""
    # Parse the configuration from the new buffer.
    try:
        config = json.loads(new_buffer.get_text())
    except json.JSONDecodeError as e:
        _logger.error(f"[_after_update_code_option] option buffer is not valid JSON: {e.msg} — buffer: {new_buffer.name}")
        return None

    # Derive the origin buffer name from the option buffer configuration.
    origin_name = _origin_name_from_option_buffer_name(new_buffer.name)
    if not origin_name:
        _logger.error(f"[_after_update_code_option] option buffer is missing 'origin' field: {new_buffer.name}")
        return None

    # Get the origin buffer.
    origin_buffer = compiler.get_buffer(origin_name)
    if origin_buffer is None:
        _logger.error(f"[_after_update_code_option] origin buffer not found: {origin_name}")
        return None

    # Extract the source code from the origin buffer.
    source = origin_buffer.get_text()
    if not source.strip():
        _logger.error(f"[_after_update_code_option] origin buffer is empty: {origin_name}")
        return None

    # Validate and compile -- raises on any error.
    try:
        func_name, proxy = _compile_function(compiler, source)
    except ValueError as e:
        _logger.error(f"[_after_update_code_option] compile failed for '{origin_name}': {e}")
        return None

    # Determine whether this is an initial registration or an update.
    # Initial registration -- the buffer is new or the old config is absent.
    function_name = config.get("function_name")
    if old_buffer is None:
        hook_error = await _register_function_with_hooks(compiler, proxy, function_name, origin_buffer, origin_name)
        if hook_error is not None:
            _logger.error(f"[_after_update_code_option] {hook_error}")
        return None

    # Try to retrieve the old config to check whether this is an update.
    try:
        json.loads(old_buffer.get_text())
    except json.JSONDecodeError:
        hook_error = await _register_function_with_hooks(compiler, proxy, function_name, origin_buffer, origin_name)
        if hook_error is not None:
            _logger.error(f"[_after_update_code_option] {hook_error}")
        return None


async def _before_update_code(
    compiler: FunctionCompiler,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """BEFORE_UPDATE_BUFFER hook for code buffers -- validate the update does not rename the function."""
    # Derive the config buffer name from the code buffer name.
    config_buffer_name = new_buffer.name + _CODE_BUFFER_SUFFIX

    # Get the config buffer and read the function name from it.
    config_buffer = compiler.get_buffer(config_buffer_name)
    if config_buffer is None:
        return f"[_before_update_code] config buffer not found: {config_buffer_name}"

    config = _get_code_config(config_buffer)
    if isinstance(config, str):
        return f"[_before_update_code] could not read config: {config}"

    # Retrieve the function name from the config.
    func_name = config.get("function_name")
    if not func_name:
        return f"[_before_update_code] config is missing 'function_name': {config_buffer_name}"

    # Extract the new source and trial-compile to get the proposed new function name and signature.
    new_source = new_buffer.get_text()
    try:
        new_func_name, new_sig = _validate_and_trial_compile(compiler, new_source)
    except ValueError as e:
        return str(e)

    # Reject when the new source defines a function with a different name.
    if new_func_name != func_name:
        return f"[_before_update_code] renaming functions is not allowed ('{func_name}' -> '{new_func_name}')"

    # Get the signature of the currently registered callable.
    wrapper = compiler._callables.get(func_name)
    if wrapper is None:
        return f"[_before_update_code] function '{func_name}' not in callables"
    old_sig = wrapper.__signature__

    # Check that the new signature is compatible with the registered callable.
    compatible_error = _check_signatures_compatible(old_sig, new_sig)
    if compatible_error is not None:
        return f"[_before_update_code] function '{func_name}' signature incompatible: {compatible_error}"


async def _after_update_code(
    compiler: FunctionCompiler,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """AFTER_UPDATE_BUFFER hook for code buffers -- compile updated source and update the registered callable."""
    # Derive the config buffer name from the code buffer name.
    config_buffer_name = new_buffer.name + _CODE_BUFFER_SUFFIX

    # Get the config buffer and reject when it is absent.
    config_buffer = compiler.get_buffer(config_buffer_name)
    if config_buffer is None:
        _logger.error(f"[_after_update_code] config buffer not found: {config_buffer_name}")
        return None

    # Parse the config from the config buffer.
    config = _get_code_config(config_buffer)
    if isinstance(config, str):
        _logger.error(f"[_after_update_code] could not read config: {config}")
        return None

    # Retrieve the function name from the config.
    func_name = config.get("function_name")
    if not func_name:
        _logger.error(f"[_after_update_code] config is missing 'function_name': {config_buffer_name}")
        return None

    # Extract the updated source from the buffer.
    source = new_buffer.get_text()
    if not source.strip():
        _logger.error(f"[_after_update_code] source buffer is empty: {new_buffer.name}")
        return None

    # Compile the source to get the new proxy.
    try:
        _func_name, proxy = _compile_function(compiler, source)
    except ValueError as e:
        _logger.error(f"[_after_update_code] compile failed for '{func_name}': {e}")
        return None

    # Update the registered callable with the newly compiled proxy.
    update_result = await compiler.update_registered_function(proxy, func_name)
    if not update_result.get("ok"):
        _logger.error(f"[_after_update_code] update failed for '{func_name}': {update_result.get('error')}")


async def _before_drop_code(
    compiler: FunctionCompiler,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """BEFORE_DROP_BUFFER hook for code buffers -- reject drop when the function has active references."""
    # Derive the config buffer name from the code buffer name.
    config_buffer_name = buffer.name + _CODE_BUFFER_SUFFIX

    # Get the config buffer and reject when it is absent.
    config_buffer = compiler.get_buffer(config_buffer_name)
    if config_buffer is None:
        return None

    # Parse the config from the config buffer.
    config = _get_code_config(config_buffer)
    if isinstance(config, str):
        return None

    # Retrieve the function name from the config.
    func_name = config.get("function_name")
    if not func_name:
        return None

    # Check whether the function has active references.
    if not compiler.has_function(func_name):
        return None
    wrapper = compiler._callables.get(func_name)
    if wrapper is not None and wrapper._reference_count > 0:
        return f"[_before_drop_code] function '{func_name}' is still in use (ref_count={wrapper._reference_count})"

    return None


async def _after_drop_code(
    compiler: FunctionCompiler,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """AFTER_DROP_BUFFER hook for code buffers -- unregister the function from the function manager."""
    # Derive the config buffer name from the code buffer name.
    config_buffer_name = buffer.name + _CODE_BUFFER_SUFFIX

    # Get the config buffer.
    config_buffer = compiler.get_buffer(config_buffer_name)
    if config_buffer is None:
        _logger.error(f"[_after_drop_code] config buffer not found: {config_buffer_name}")
        return None

    # Parse the config from the config buffer.
    config = _get_code_config(config_buffer)
    if isinstance(config, str):
        _logger.error(f"[_after_drop_code] could not read config: {config}")
        return None

    # Retrieve the function name from the config.
    func_name = config.get("function_name")
    if not func_name:
        _logger.error(f"[_after_drop_code] config is missing 'function_name': {config_buffer_name}")
        return None

    # Unregister the function from the function manager.
    unregister_result = await compiler.unregister_function(func_name)
    if not unregister_result.get("ok"):
        _logger.error(f"[_after_drop_code] unregister failed for '{func_name}': {unregister_result.get('error')}")
        return None

    _logger.info(f"[>>] '{func_name}' unregistered after buffer drop")


async def _before_drop_code_option(
    compiler: FunctionCompiler,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """BEFORE_DROP_BUFFER hook for [python] option buffers -- reject drop when the function has active references."""
    # Parse the config from the option buffer.
    config = _get_code_config(buffer)
    if isinstance(config, str):
        return None

    # Retrieve the function name from the config.
    func_name = config.get("function_name")
    if not func_name:
        return None

    # Reject when the function has active callable refs.
    if not compiler.has_function(func_name):
        return None
    wrapper = compiler._callables.get(func_name)
    if wrapper is not None and wrapper._reference_count > 0:
        return f"[_before_drop_code_option] function '{func_name}' is still in use (ref_count={wrapper._reference_count})"

    return None


async def _after_drop_code_option(
    compiler: FunctionCompiler,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """AFTER_DROP_BUFFER hook for [python] option buffers -- unregister the function and clean up hooks from the origin buffer."""
    # Parse the config from the option buffer.
    config = _get_code_config(buffer)
    if isinstance(config, str):
        _logger.error(f"[_after_drop_code_option] could not read config: {config}")
        return None

    # Retrieve the function name from the config.
    func_name = config.get("function_name")
    if not func_name:
        _logger.error(f"[_after_drop_code_option] config is missing 'function_name'")
        return None

    # Derive the origin buffer name from the option buffer name.
    origin_name = _origin_name_from_option_buffer_name(buffer.name)
    if origin_name is None:
        _logger.error(f"[_after_drop_code_option] cannot derive origin from '{buffer.name}'")
        return None

    # Get the origin buffer to remove the hooks from it.
    origin_buffer = compiler.get_buffer(origin_name)
    if origin_buffer is not None:
        # Remove the hooks that were registered on the origin buffer at compile time.
        origin_buffer.unregister_hook(HookPoint.BEFORE_UPDATE_BUFFER, "function_compiler:before_update_code")
        origin_buffer.unregister_hook(HookPoint.AFTER_UPDATE_BUFFER, "function_compiler:after_update_code")
        origin_buffer.unregister_hook(HookPoint.BEFORE_DROP_BUFFER, "function_compiler:before_drop_code")
        origin_buffer.unregister_hook(HookPoint.AFTER_DROP_BUFFER, "function_compiler:after_drop_code")

    # Unregister the function from the function manager.
    unregister_result = await compiler.unregister_function(func_name)
    if not unregister_result.get("ok"):
        _logger.error(f"[_after_drop_code_option] unregister failed for '{func_name}': {unregister_result.get('error')}")
        return None

    _logger.info(f"[>>] '{func_name}' unregistered and hooks removed after option buffer drop")


async def _on_create_code(
    compiler: FunctionCompiler,
    buffer_name: str,
    args: dict[str, Any],
) -> str | None:
    """BEFORE_CREATE_BUFFER hook for the system:code: scope -- compiles and registers a function from source text."""
    # Parse the source from the create args.
    source: str = args.get("text", "")
    if not source.strip():
        return "source buffer is empty"

    # Validate and compile -- raises on any error.
    try:
        func_name, proxy = _compile_function(compiler, source)
    except ValueError as e:
        return str(e)

    # Extract docstring for descriptions.
    doc = proxy.__doc__ or ""
    short_description = doc.split("\n")[0].strip() if doc else ""
    long_description = doc.strip() if doc else ""

    # Build stable wrapper -- its identity is fixed, it delegates to the live sandbox proxy.
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
    buffer_manager: BufferManager,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """AFTER_CREATE_BUFFER hook for the system:code: scope -- registers pre/post update and drop hooks on the new buffer."""
    # Derive the function name from the buffer name -- prefix confirmed by scope registration.
    buffer_name = new_buffer.name
    func_name = buffer_name[len(_CODE_BUFFER_PREFIX) :]

    # Register the on_update_code guard as a pre-update hook -- validates source before mutation.
    guard_result = new_buffer.register_hook(
        HookPoint.BEFORE_UPDATE_BUFFER,
        "on_update_code",
        _make_on_update_code(buffer_manager, func_name),
    )
    if not isinstance(guard_result, dict) or not guard_result.get("ok"):
        return f"failed to register on_update_code on {buffer_name}: {guard_result.get('error')}"

    # Register the on_code_updated post-update hook -- commits to sandbox and refreshes the catalog.
    commit_result = new_buffer.register_hook(
        HookPoint.AFTER_UPDATE_BUFFER,
        "on_code_updated",
        _make_on_code_updated(buffer_manager, func_name, buffer_name),
    )
    if not isinstance(commit_result, dict) or not commit_result.get("ok"):
        return f"failed to register on_code_updated on {buffer_name}: {commit_result.get('error')}"

    # Register the on_code_drop pre-drop hook -- cascades buffer drop to the function namespace.
    drop_result = new_buffer.register_hook(
        HookPoint.BEFORE_DROP_BUFFER,
        "on_code_drop",
        _make_on_code_drop(buffer_manager, func_name),
    )
    if not isinstance(drop_result, dict) or not drop_result.get("ok"):
        return f"failed to register on_code_drop on {buffer_name}: {drop_result.get('error')}"

    # Register the on_code_dropped safety-net -- logs if the function is still registered after the drop.
    dropped_result = new_buffer.register_hook(
        HookPoint.AFTER_DROP_BUFFER,
        "on_code_dropped",
        _make_on_code_dropped(buffer_manager, func_name),
    )
    if not isinstance(dropped_result, dict) or not dropped_result.get("ok"):
        return f"failed to register on_code_dropped on {buffer_name}: {dropped_result.get('error')}"

    return None


# -------------------------------------------------------------------------------------------------------------------- #
# FunctionCompiler                                                                                                     #
# -------------------------------------------------------------------------------------------------------------------- #


class FunctionCompiler(FunctionManager, OptionBufferManager, AgenticObject):
    """I compile Python source-code strings into live callable functions.
    "doc:function_compiler" holds details.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Seed the function compiler documentation buffer.
        self._create_buffer(_FUNCTION_COMPILER_DOC_BUFFER, text=_FUNCTION_COMPILER_DOC)

        # Register the BEFORE_CREATE_BUFFER scope hook -- compiles and registers functions on buffer creation.
        on_create_result = self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "function_compiler:on_create_code",
            _CODE_BUFFER_PREFIX + ".*",
            _on_create_code,
        )
        if not isinstance(on_create_result, dict) or not on_create_result.get("ok"):
            raise RuntimeError(f"failed to register on_create_code hook: {on_create_result.get('error')}")

        # Register the AFTER_CREATE_BUFFER scope hook -- registers pre/post update hooks on new code buffers.
        on_created_result = self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "function_compiler:on_code_created",
            _CODE_BUFFER_PREFIX + ".*",
            _on_code_created,
        )
        if not isinstance(on_created_result, dict) or not on_created_result.get("ok"):
            raise RuntimeError(f"failed to register on_code_created hook: {on_created_result.get('error')}")

        # Register the before hooks for [python] option buffers -- validate source before mutation.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "function_compiler:python_code_option_before_create",
            _CODE_BUFFER_OPTION_BUFFER_PATTERN,
            _before_create_code_option,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_UPDATE_BUFFER,
            "function_compiler:python_code_option_before_update",
            _CODE_BUFFER_OPTION_BUFFER_PATTERN,
            _before_update_code_option,
            apply_to_existing=True,
        )

        # Register the after hooks for [python] option buffers -- compile and register after mutation.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_UPDATE_BUFFER,
            "function_compiler:python_code_option_after_update",
            _CODE_BUFFER_OPTION_BUFFER_PATTERN,
            _after_update_code_option,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "function_compiler:python_code_option_after_create",
            _CODE_BUFFER_OPTION_BUFFER_PATTERN,
            lambda bm, new_buffer, args: _after_update_code_option(bm, None, new_buffer, args),
            apply_to_existing=True,
        )

        # Register the drop hooks for [python] option buffers -- block drop when refs are active, then clean up.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_DROP_BUFFER,
            "function_compiler:python_code_option_before_drop",
            _CODE_BUFFER_OPTION_BUFFER_PATTERN,
            _before_drop_code_option,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            "function_compiler:python_code_option_after_drop",
            _CODE_BUFFER_OPTION_BUFFER_PATTERN,
            _after_drop_code_option,
            apply_to_existing=True,
        )
