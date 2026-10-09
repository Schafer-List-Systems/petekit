"""Unit tests for FunctionManager — runtime Callable management built on BufferManager."""

from __future__ import annotations

import pytest
import sys
import unittest

sys.path.insert(0, "/home/frygge/projects/AIOS/peteos-kit")
sys.path.insert(0, "/home/frygge/projects/private/petekit/src/peteos/peteos")

from petekit.function import FunctionCompiler, FunctionEntry, FunctionManager


@pytest.fixture
async def fm():
    return FunctionManager()


class TestFunctionManagerBasic:
    async def test_create_function_registers_and_returns_metadata(self, fm):
        def greet(name: str) -> str:
            """Greet someone by name."""
            return f"hello {name}"

        result = await fm.create_function(
            "greet",
            greet,
            short_description="Return a greeting.",
            long_description="Takes a name and returns a personalized greeting string.",
        )
        assert result["ok"]
        assert result["name"] == "greet"
        assert result["args"] == ["name"]
        assert result["returns"] == "str"

    async def test_create_function_stores_callable(self, fm):
        def add(a: int, b: int) -> int:
            return a + b

        await fm.create_function("add", add)
        entry = fm._functions["add"]
        assert entry.callable is add
        assert entry.name == "add"

    async def test_get_function_returns_callable(self, fm):
        def double(x: int) -> int:
            return x * 2

        await fm.create_function("double", double)
        result = fm.get_function("double")
        assert result["ok"]
        assert result["callable"] is double

    async def test_create_function_rejects_duplicate_name(self, fm):
        def one():
            pass

        def two():
            pass

        await fm.create_function("fn", one)
        result = await fm.create_function("fn", two)
        assert not result["ok"]
        assert "already registered" in result["error"]

    async def test_drop_function_removes_from_namespace(self, fm):
        def dummy():
            pass

        await fm.create_function("dummy", dummy)
        result = await fm.drop_function("dummy")
        assert result["ok"]
        assert "dummy" not in fm._functions

    async def test_drop_function_removes_buffer(self, fm):
        def temp():
            pass

        await fm.create_function("temp", temp)
        await fm.drop_function("temp")
        assert "system:status:function:temp" not in fm._buffers

    async def test_drop_unknown_is_error(self, fm):
        result = await fm.drop_function("does_not_exist")
        assert not result["ok"]
        assert "No function named" in result["error"]

    async def test_get_function_returns_error_for_unknown(self, fm):
        result = fm.get_function("unknown_func")
        assert not result["ok"]
        assert "No function named" in result["error"]

    async def test_function_buffer_contains_docstring(self, fm):
        import json
        def documented(x: int) -> int:
            """Returns the input unchanged."""
            return x

        await fm.create_function("documented", documented)
        content = await fm.read_buffer("system:status:function:documented", raw=True, show_line_numbers=False)
        obj = json.loads(content)
        assert obj["data"]["name"] == "documented"
        assert obj["data"]["long_description"] == "Returns the input unchanged."

    async def test_function_list_buffer_contains_functions(self, fm):
        import json
        def fn1():
            pass

        def fn2():
            pass

        await fm.create_function("fn1", fn1)
        await fm.create_function("fn2", fn2)
        content = await fm.read_buffer("system:status:function", raw=True, show_line_numbers=False)
        obj = json.loads(content)
        names = [r["name"] for r in obj["data"]]
        assert "fn1" in names
        assert "fn2" in names


class TestFunctionManagerHookGating:
    async def test_drop_function_blocked_by_hook(self):
        fm = FunctionManager()
        await fm.create_function("to_drop", lambda: None)
        fm.register_buffer_update_hook("system:status:function:to_drop", "reject", lambda *_: "blocked by hook")
        result = await fm.drop_function("to_drop")
        assert not result["ok"]
        assert "blocked by hook" in result["error"]
        assert "to_drop" in fm._functions

    async def test_drop_function_allows_without_hook(self):
        fm = FunctionManager()
        await fm.create_function("to_drop", lambda: None)
        result = await fm.drop_function("to_drop")
        assert result["ok"]
        assert "to_drop" not in fm._functions

    async def test_drop_function_calls_hook_with_full_content(self):
        fm = FunctionManager()
        calls = []

        def target():
            """Doc string."""
            pass

        def accepting_hook(buf, old_text, start, end, new_text):
            calls.append((buf, old_text, start, end, new_text))
            return True

        await fm.create_function("target", target)
        fm.register_buffer_update_hook("system:status:function:target", "spy", accepting_hook)
        await fm.drop_function("target")
        assert len(calls) == 1
        _, old_text, start, end, new_text = calls[0]
        assert "target" in old_text
        assert new_text is None
