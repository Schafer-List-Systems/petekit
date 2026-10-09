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

    async def test_create_function_compiles_source_into_callable(self, fc):
        """Source code compiles and the callable is registered in the sandbox."""
        result = await fc.create_function(
            name="add",
            code="""
def add(a: int, b: int) -> int:
    return a + b
""",
            short_description="Adds two integers.",
            long_description="Returns the sum of two integers.",
        )
        assert result["ok"] is True
        assert result["name"] == "add"
        assert result["args"] == ["a", "b"]
        assert result["returns"] == "int"

    async def test_callable_execution_returns_correct_value(self, fc):
        """The compiled callable executes and returns the expected value."""
        await fc.create_function(
            name="add",
            code="""
def add(a: int, b: int) -> int:
    return a + b
""",
        )
        entry = fc.get_function("add")
        assert entry["ok"] is True
        func = entry["callable"]
        result = func(3, 5)
        assert result == 8

    async def test_signature_matches_source_function(self, fc):
        """inspect.signature on the callable shows the source function's signature."""
        await fc.create_function(
            name="greet",
            code="""
def greet(name: str, loud: bool = False) -> str:
    '''Greet someone, optionally loudly.'''
    prefix = "HELLO" if loud else "Hello"
    return f\"{prefix} {name}\"
""",
        )
        entry = fc.get_function("greet")
        func = entry["callable"]
        sig = inspect.signature(func)

        # Verify parameter names and defaults
        params = list(sig.parameters.keys())
        assert params == ["name", "loud"]

        # Check that 'loud' has a default
        loud_param = sig.parameters["loud"]
        assert loud_param.default is False

        # Annotations may be strings when serialized, check both forms.
        name_param = sig.parameters["name"]
        assert name_param.annotation in (str, "str")
        assert loud_param.annotation in (str, "bool")
        assert sig.return_annotation in (str, "str")

    async def test_multiple_functions_coexist(self, fc):
        """Multiple functions can be compiled and coexist in the sandbox."""
        await fc.create_function(
            name="add",
            code="def add(a: int, b: int) -> int:\n    return a + b",
        )
        await fc.create_function(
            name="mul",
            code="def mul(a: int, b: int) -> int:\n    return a * b",
        )

        add_entry = fc.get_function("add")
        mul_entry = fc.get_function("mul")

        assert add_entry["callable"](2, 3) == 5
        assert mul_entry["callable"](2, 3) == 6


class TestFunctionCompilerErrors:
    """Error handling for invalid source code."""

    async def test_rejects_syntax_error(self, fc):
        """Source with syntax errors returns an error result."""
        result = await fc.create_function(
            name="bad",
            code="def bad(x) return x  # missing colon and invalid syntax",
        )
        assert result["ok"] is False
        assert "SyntaxError" in result["error"]

    async def test_rejects_non_functiondef(self, fc):
        """Source that is not exactly one FunctionDef returns an error."""
        result = await fc.create_function(
            name="notfunc",
            code="x = 1 + 2  # not a function",
        )
        assert result["ok"] is False

    async def test_rejects_multiple_definitions(self, fc):
        """Source with multiple top-level definitions returns an error."""
        result = await fc.create_function(
            name="multi",
            code="""
def foo(): pass
def bar(): pass
""",
        )
        assert result["ok"] is False

    async def test_rejects_name_mismatch(self, fc):
        """Source function name must match the requested name."""
        result = await fc.create_function(
            name="add",
            code="def sub(a, b): return a - b",  # 'sub' != 'add'
        )
        assert result["ok"] is False
        assert "name" in result["error"]


class TestFunctionCompilerDropFunction:
    """Dropping functions removes them from the sandbox."""

    async def test_drop_function_removes_from_functions(self, fc):
        """After drop, get_function returns not-found for that name."""
        await fc.create_function(
            name="to_drop",
            code="def to_drop(): pass",
        )
        assert fc.get_function("to_drop")["ok"] is True

        await fc.drop_function("to_drop")
        assert fc.get_function("to_drop")["ok"] is False

    async def test_drop_function_removes_code_buffer(self, fc):
        """After drop, the source code buffer is gone."""
        await fc.create_function(
            name="with_code",
            code="def with_code(): pass",
        )
        buf_result = fc._read_buffer(f"system:code:with_code")
        assert buf_result["ok"] is True

        await fc.drop_function("with_code")
        buf_result = fc._read_buffer(f"system:code:with_code")
        assert buf_result["ok"] is False


class TestFunctionCompilerStableWrapper:
    """The stable wrapper preserves callability and signature stability."""

    async def test_wrapper_preserves_signature_after_creation(self, fc):
        """The wrapper's signature is correct immediately after creation."""
        await fc.create_function(
            name="fn",
            code="""
def fn(x: int, msg: str = \"hi\") -> str:
    return f\"{msg}: {x}\"
""",
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
        await fc.create_function(
            name="stable",
            code="def stable(): pass",
        )

        entry1 = fc.get_function("stable")
        entry2 = fc.get_function("stable")

        # The callable wrapper should be the same object
        # (this is what "stable" means — same identity)
        assert entry1["callable"] is entry2["callable"]

    async def test_sandbox_proxy_has_actual_function(self, fc):
        """The compiled function actually exists as an attribute on the sandbox."""
        await fc.create_function(
            name="in_sandbox",
            code="def in_sandbox(): return 42",
        )

        sandbox = fc._oap_sandbox_builder.get_sandbox(freeze_namespaces=False)
        compiled = getattr(sandbox, "in_sandbox")

        # The compiled function should be a plain function, not a bound method
        # or a lambda from __getattr__
        assert callable(compiled)
        assert compiled() == 42
