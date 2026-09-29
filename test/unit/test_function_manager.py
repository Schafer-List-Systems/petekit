"""Unit tests for FunctionManager — runtime Callable management built on BufferManager."""

from __future__ import annotations

import sys
import unittest

sys.path.insert(0, "/home/frygge/projects/AIOS/peteos-kit")
sys.path.insert(0, "/home/frygge/projects/private/petekit/src/peteos/peteos")

from petekit.function import FunctionEntry, FunctionManager


class TestFunctionManagerBasic(unittest.TestCase):
    """Smoke tests for FunctionManager: create, get, drop, and buffer reflection."""

    def setUp(self):
        self.fm = FunctionManager()

    def test_create_function_registers_and_returns_metadata(self):
        def greet(name: str) -> str:
            """Greet someone by name."""
            return f"hello {name}"

        result = self.fm.create_function(
            "greet",
            greet,
            short_description="Return a greeting.",
            long_description="Takes a name and returns a personalized greeting string.",
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["name"], "greet")
        self.assertEqual(result["args"], ["name"])
        self.assertEqual(result["returns"], "str")

    def test_create_function_stores_callable(self):
        def add(a: int, b: int) -> int:
            return a + b

        self.fm.create_function("add", add)
        entry = self.fm._functions["add"]
        self.assertIs(entry.callable, add)
        self.assertEqual(entry.name, "add")

    def test_get_function_returns_callable(self):
        def double(x: int) -> int:
            return x * 2

        self.fm.create_function("double", double)
        result = self.fm.get_function("double")
        self.assertTrue(result["ok"])
        self.assertIs(result["callable"], double)

    def test_create_function_rejects_duplicate_name(self):
        def one():
            pass

        def two():
            pass

        self.fm.create_function("fn", one)
        result = self.fm.create_function("fn", two)
        self.assertFalse(result["ok"])
        self.assertIn("already registered", result["error"])

    def test_drop_function_removes_from_namespace(self):
        def dummy():
            pass

        self.fm.create_function("dummy", dummy)
        result = self.fm.drop_function("dummy")
        self.assertTrue(result["ok"])
        self.assertNotIn("dummy", self.fm._functions)

    def test_drop_function_removes_buffer(self):
        def temp():
            pass

        self.fm.create_function("temp", temp)
        self.fm.drop_function("temp")
        self.assertNotIn("function:temp", self.fm._buffers)

    def test_drop_unknown_is_error(self):
        result = self.fm.drop_function("does_not_exist")
        self.assertFalse(result["ok"])
        self.assertIn("No function named", result["error"])

    def test_get_function_returns_error_for_unknown(self):
        result = self.fm.get_function("unknown_func")
        self.assertFalse(result["ok"])
        self.assertIn("No function named", result["error"])

    def test_function_buffer_contains_docstring(self):
        def documented(x: int) -> int:
            """Returns the input unchanged."""
            return x

        self.fm.create_function("documented", documented)
        content = self.fm.read_buffer("function:documented", raw=True, show_line_numbers=False)
        self.assertIn("documented", content)
        self.assertIn("Returns the input unchanged", content)

    def test_function_list_buffer_contains_functions(self):
        def fn1():
            pass

        def fn2():
            pass

        self.fm.create_function("fn1", fn1)
        self.fm.create_function("fn2", fn2)
        content = self.fm.read_buffer("function:list", raw=True, show_line_numbers=False)
        self.assertIn("fn1", content)
        self.assertIn("fn2", content)


class TestFunctionManagerHookGating(unittest.TestCase):
    """Test that function create/drop respects update hooks on their buffers."""

    def setUp(self):
        self.fm = FunctionManager()
        self.calls: list = []

    def _accepting_hook(self, buf, old_text, start, end, new_text):
        self.calls.append((buf, old_text, start, end, new_text))
        return True

    def _rejecting_hook(self, buf, old_text, start, end, new_text):
        return "blocked by hook"

    def test_drop_function_blocked_by_hook(self):
        def to_drop():
            pass

        self.fm.create_function("to_drop", to_drop)
        self.fm.register_buffer_update_hook("function:to_drop", "reject", self._rejecting_hook)
        result = self.fm.drop_function("to_drop")
        self.assertFalse(result["ok"])
        self.assertIn("blocked by hook", result["error"])
        self.assertIn("to_drop", self.fm._functions)

    def test_drop_function_allows_without_hook(self):
        def to_drop():
            pass

        self.fm.create_function("to_drop", to_drop)
        result = self.fm.drop_function("to_drop")
        self.assertTrue(result["ok"])
        self.assertNotIn("to_drop", self.fm._functions)

    def test_drop_function_calls_hook_with_full_content(self):
        def target():
            """Doc string."""
            pass

        self.fm.create_function("target", target)
        self.fm.register_buffer_update_hook("function:target", "spy", self._accepting_hook)
        self.fm.drop_function("target")
        self.assertEqual(len(self.calls), 1)
        _, old_text, start, end, new_text = self.calls[0]
        self.assertIn("target", old_text)
        self.assertEqual(new_text, None)
