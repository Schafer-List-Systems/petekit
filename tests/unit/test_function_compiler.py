"""Unit tests for FunctionCompiler — compiles source-code strings into live sandboxed callables."""

from __future__ import annotations

import asyncio
import inspect
import pytest
import sys

sys.path.insert(0, "/home/frygge/projects/AIOS/peteos-kit")
sys.path.insert(0, "/home/frygge/projects/private/petekit/src/peteos/peteos")

from petekit.function import FunctionCompiler


@pytest.fixture
async def fc():
    """FunctionCompiler test instance."""
    return FunctionCompiler()


class TestFunctionCompilerBasic:
    """Basic compilation and callable behavior."""

    async def test_create_buffer_compiles_source_into_callable(self, fc):
        """Source code compiles and the callable is registered in the sandbox."""
        result = await fc.create_buffer(
            "system:code:add",
            text="def add(a: int, b: int) -> int:\n    return a + b\n",
        )
        assert result["ok"] is True

        entry = fc.get_function("add")
        assert entry["ok"] is True
        func = entry["callable"]
        result = func(2, 3)
        assert result == 5

    async def test_callable_execution_returns_correct_value(self, fc):
        """The compiled callable executes and returns the expected value."""
        await fc.create_buffer(
            "system:code:add",
            text="def add(a: int, b: int) -> int:\n    return a + b\n",
        )
        entry = fc.get_function("add")
        assert entry["ok"] is True
        func = entry["callable"]
        result = func(3, 5)
        assert result == 8

    async def test_signature_matches_source_function(self, fc):
        """inspect.signature on the callable shows the source function's signature."""
        await fc.create_buffer(
            "system:code:greet",
            text="def greet(name: str, loud: bool = False) -> str:\n    prefix = 'HELLO' if loud else 'Hello'\n    return f'{prefix} {name}'\n",
        )
        entry = fc.get_function("greet")
        func = entry["callable"]
        sig = inspect.signature(func)

        params = list(sig.parameters.keys())
        assert params == ["name", "loud"]
        loud_param = sig.parameters["loud"]
        assert loud_param.default is False

    async def test_multiple_functions_coexist(self, fc):
        """Multiple functions can be compiled and coexist in the sandbox."""
        await fc.create_buffer(
            "system:code:add",
            text="def add(a: int, b: int) -> int:\n    return a + b\n",
        )
        await fc.create_buffer(
            "system:code:mul",
            text="def mul(a: int, b: int) -> int:\n    return a * b\n",
        )

        add_entry = fc.get_function("add")
        mul_entry = fc.get_function("mul")

        assert add_entry["callable"](2, 3) == 5
        assert mul_entry["callable"](2, 3) == 6


class TestFunctionCompilerErrors:
    """Error handling for invalid source code."""

    async def test_rejects_syntax_error(self, fc):
        """Source with syntax errors returns an error result."""
        result = await fc.create_buffer(
            "system:code:bad",
            text="def bad(x) return x  # missing colon and invalid syntax\n",
        )
        assert result["ok"] is False
        assert "SyntaxError" in result["error"]

    async def test_rejects_non_functiondef(self, fc):
        """Source that is not exactly one FunctionDef returns an error."""
        result = await fc.create_buffer(
            "system:code:notfunc",
            text="x = 1 + 2  # not a function\n",
        )
        assert result["ok"] is False

    async def test_rejects_multiple_definitions(self, fc):
        """Source with multiple top-level definitions returns an error."""
        result = await fc.create_buffer(
            "system:code:multi",
            text="def foo(): pass\ndef bar(): pass\n",
        )
        assert result["ok"] is False


class TestFunctionCompilerDropFunction:
    """Dropping functions removes them from the sandbox."""

    async def test_drop_function_removes_from_functions(self, fc):
        """After drop, get_function returns not-found for that name."""
        await fc.create_buffer(
            "system:code:to_drop",
            text="def to_drop(): pass\n",
        )
        assert fc.get_function("to_drop")["ok"] is True

        await fc.drop_function("to_drop")
        assert fc.get_function("to_drop")["ok"] is False

    async def test_drop_function_removes_code_buffer(self, fc):
        """After drop, the source code buffer is gone."""
        await fc.create_buffer(
            "system:code:with_code",
            text="def with_code(): pass\n",
        )
        buf_result = fc._read_buffer("system:code:with_code")
        assert buf_result["ok"] is True

        await fc.drop_function("with_code")
        buf_result = fc._read_buffer("system:code:with_code")
        assert buf_result["ok"] is False


class TestFunctionCompilerStableWrapper:
    """The stable wrapper preserves callability and signature stability."""

    async def test_wrapper_preserves_signature_after_creation(self, fc):
        """The wrapper's signature is correct immediately after creation."""
        await fc.create_buffer(
            "system:code:fn",
            text="def fn(x: int, msg: str = 'hi') -> str:\n    return f'{msg}: {x}'\n",
        )
        entry = fc.get_function("fn")
        func = entry["callable"]

        sig = inspect.signature(func)
        params = list(sig.parameters.keys())
        assert params == ["x", "msg"]
        # Annotations may be strings when serialized, check both forms.
        assert sig.parameters["x"].annotation in (int, "int")
        assert sig.parameters["msg"].annotation in (str, "str")

    async def test_wrapper_is_same_object_on_repeated_get_function(self, fc):
        """Calling get_function returns the same wrapper object each time."""
        await fc.create_buffer(
            "system:code:stable",
            text="def stable(): pass\n",
        )

        entry1 = fc.get_function("stable")
        entry2 = fc.get_function("stable")

        # The callable wrapper should be the same object
        # (this is what "stable" means — same identity)
        assert entry1["callable"] is entry2["callable"]

    async def test_sandbox_proxy_has_actual_function(self, fc):
        """The compiled function actually exists as an attribute on the sandbox."""
        await fc.create_buffer(
            "system:code:in_sandbox",
            text="def in_sandbox(): return 42\n",
        )

        sandbox = fc._oap_sandbox_builder.get_sandbox(freeze_namespaces=False)
        compiled = getattr(sandbox, "in_sandbox")

        # The compiled function should be a plain function, not a bound method
        # or a lambda from __getattr__
        assert callable(compiled)
        assert compiled() == 42
