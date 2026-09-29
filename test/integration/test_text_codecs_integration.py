"""Unit tests for the JSON codec make_json_codec."""

from __future__ import annotations

import sys
import unittest

sys.path.insert(0, "/home/frygge/projects/AIOS/peteos-kit")
sys.path.insert(0, "/home/frygge/projects/private/petekit/src/peteos/peteos")

from petekit import BufferManager
from petekit.utils.text_codecs import make_json_codec


class TestMakeJsonCodec(unittest.TestCase):
    """Test the JSON codec factory and its UpdateHook output."""

    def setUp(self):
        self.bm = BufferManager()
        self.calls: list = []

    def _inner_accept(self, buf, old_json, new_json, patch):
        self.calls.append((buf, old_json, new_json, patch))
        return True

    def _inner_reject(self, buf, old_json, new_json, patch):
        return "inner rejected"

    def test_valid_new_json_passes_to_inner(self):
        self.bm.create_buffer("test", text='{"a": 1}')
        self.bm.register_buffer_update_hook(
            "test", "jc",
            make_json_codec(self._inner_accept),
        )
        result = self.bm.write_buffer("test", '{"b": 2}', start=0)
        self.assertTrue(result["ok"])
        self.assertEqual(len(self.calls), 1)
        buf, old_json, new_json, patch = self.calls[0]
        self.assertEqual(old_json, {"a": 1})
        self.assertEqual(new_json, {"b": 2})

    def test_invalid_new_json_rejected(self):
        self.bm.create_buffer("test", text='{"a": 1}')
        self.bm.register_buffer_update_hook(
            "test", "jc",
            make_json_codec(self._inner_accept),
        )
        result = self.bm.write_buffer("test", "not json", start=0)
        self.assertFalse(result["ok"])
        self.assertIn("not valid JSON", result["error"])

    def test_invalid_old_json_rejected(self):
        self.bm.create_buffer("test", text="not json either")
        self.bm.register_buffer_update_hook(
            "test", "jc",
            make_json_codec(self._inner_accept),
        )
        result = self.bm.write_buffer("test", '{"b": 2}', start=0)
        self.assertFalse(result["ok"])
        self.assertIn("not valid JSON", result["error"])

    def test_inner_reject_propagates(self):
        self.bm.create_buffer("test", text='{"a": 1}')
        self.bm.register_buffer_update_hook(
            "test", "jc",
            make_json_codec(self._inner_reject),
        )
        result = self.bm.write_buffer("test", '{"a": 2}', start=0)
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "inner rejected")

    def test_inner_accept_propagates(self):
        self.bm.create_buffer("test", text='{"a": 1}')
        self.bm.register_buffer_update_hook(
            "test", "jc",
            make_json_codec(self._inner_reject),
        )
        self.bm.unregister_buffer_update_hook("test", "jc")
        self.bm.register_buffer_update_hook(
            "test", "jc",
            make_json_codec(self._inner_accept),
        )
        result = self.bm.write_buffer("test", '{"a": 2}', start=0)
        self.assertTrue(result["ok"])

    def test_patch_contains_json_patch_operations(self):
        self.bm.create_buffer("test", text='{"a": 1, "b": 2}')
        self.bm.register_buffer_update_hook(
            "test", "jc",
            make_json_codec(self._inner_accept),
        )
        self.bm.write_buffer("test", '{"a": 1, "c": 3}', start=0)
        _, _, _, patch = self.calls[0]
        ops = [p["op"] for p in patch]
        self.assertIn("remove", ops)
        self.assertIn("add", ops)

    def test_patch_is_empty_when_identical(self):
        self.bm.create_buffer("test", text='{"a": 1}')
        self.bm.register_buffer_update_hook(
            "test", "jc",
            make_json_codec(self._inner_accept),
        )
        self.bm.write_buffer("test", '{"a": 1}', start=0)
        _, _, _, patch = self.calls[0]
        self.assertFalse(list(patch))

    def test_reject_blocks_write(self):
        self.bm.create_buffer("test", text='{"a": 1}')
        self.bm.register_buffer_update_hook(
            "test", "jc",
            make_json_codec(self._inner_reject),
        )
        result = self.bm.write_buffer("test", '{"a": 2}', start=0)
        self.assertFalse(result["ok"])
        content = self.bm.read_buffer("test", raw=True, show_line_numbers=False)
        self.assertEqual(content, '{"a": 1}')
