"""Unit tests for make_json_codec — pure unit tests, no BufferManager dependency."""

from __future__ import annotations

import sys
import unittest

import jsonpatch

sys.path.insert(0, "/home/frygge/projects/AIOS/peteos-kit")
sys.path.insert(0, "/home/frygge/projects/private/petekit/src/peteos/peteos")

from petekit.utils.text_codecs import make_json_codec


class MockLine:
    def __init__(self, data: str):
        self.data = data


class MockBuffer:
    def __init__(self, lines: list[MockLine]):
        self.lines = lines


class TestMakeJsonCodecUnit(unittest.TestCase):
    """Unit-test make_json_codec by calling the returned hook directly with mock objects."""

    def test_valid_json_calls_inner_with_parsed_json_and_patch(self):
        calls: list = []

        def inner(buf, old_json, new_json, patch):
            calls.append((old_json, new_json, list(patch)))
            return True

        buf = MockBuffer([MockLine('{"a": 1}')])
        hook = make_json_codec(inner)
        result = hook(buf, '{"a": 1}', 0, 1, '{"b": 2}')

        self.assertTrue(result)
        self.assertEqual(len(calls), 1)
        old_json, new_json, patch_list = calls[0]
        self.assertEqual(old_json, {"a": 1})
        self.assertEqual(new_json, {"b": 2})
        self.assertTrue(any(p["op"] == "add" for p in patch_list))

    def test_invalid_new_text_rejected_before_inner_called(self):
        calls: list = []

        def inner(buf, old_json, new_json, patch):
            calls.append(1)
            return True

        buf = MockBuffer([MockLine('{"a": 1}')])
        hook = make_json_codec(inner)
        result = hook(buf, '{"a": 1}', 0, 1, "not json")

        self.assertIsInstance(result, str)
        self.assertIn("not valid JSON", result)
        self.assertEqual(calls, [])

    def test_invalid_old_text_rejected_before_inner_called(self):
        def inner(buf, old_json, new_json, patch):
            return True

        buf = MockBuffer([MockLine("not json")])
        hook = make_json_codec(inner)
        result = hook(buf, "not json", 0, 1, '{"a": 2}')

        self.assertIsInstance(result, str)
        self.assertIn("not valid JSON", result)

    def test_inner_reject_string_propagates(self):
        def inner(buf, old_json, new_json, patch):
            return "nope"

        buf = MockBuffer([MockLine('{"a": 1}')])
        hook = make_json_codec(inner)
        result = hook(buf, '{"a": 1}', 0, 1, '{"a": 2}')

        self.assertEqual(result, "nope")

    def test_inner_false_rejected(self):
        def inner(buf, old_json, new_json, patch):
            return False

        buf = MockBuffer([MockLine('{"a": 1}')])
        hook = make_json_codec(inner)
        result = hook(buf, '{"a": 1}', 0, 1, '{"a": 2}')

        self.assertFalse(result)

    def test_empty_patch_when_identical(self):
        def inner(buf, old_json, new_json, patch):
            return list(patch) == []

        buf = MockBuffer([MockLine('{"a": 1}')])
        hook = make_json_codec(inner)
        result = hook(buf, '{"a": 1}', 0, 1, '{"a": 1}')

        self.assertTrue(result)

    def test_patch_contains_remove_and_add_operations(self):
        def inner(buf, old_json, new_json, patch):
            ops = {p["op"] for p in patch}
            return "remove" in ops and "add" in ops

        buf = MockBuffer([MockLine('{"a": 1, "b": 2}')])
        hook = make_json_codec(inner)
        result = hook(buf, '{"a": 1, "b": 2}', 0, 1, '{"a": 1, "c": 3}')

        self.assertTrue(result)

    def test_splice_preserves_prefix_and_suffix(self):
        def inner(buf, old_json, new_json, patch):
            return old_json == {"prefix": True} and new_json == {"prefix": True, "added": 1}

        buf = MockBuffer([MockLine('{"prefix": true}'), MockLine('{"keep": 2}')])
        hook = make_json_codec(inner)
        result = hook(buf, '{"prefix": true}', 0, 1, '{"added": 1}', )

        self.assertTrue(result)
