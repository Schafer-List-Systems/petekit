"""Unit tests for BufferManager, especially edit_buffer timestamp behavior."""

from __future__ import annotations

import sys
import time
import unittest

sys.path.insert(0, "/home/frygge/projects/AIOS/peteos-kit")
sys.path.insert(0, "/home/frygge/projects/private/petekit/src/peteos/peteos")

from petekit import BufferManager
from petekit.text.buffer_manager import Buffer, BufferEntry


class TestBufferManagerBasics(unittest.TestCase):
    """Basic create/read/write/drop operations."""

    def setUp(self):
        self.bm = BufferManager()

    def test_create_buffer_empty(self):
        result = self.bm.create_buffer("test")
        self.assertTrue(result["ok"])
        result2 = self.bm.create_buffer("test")
        self.assertFalse(result2["ok"])

    def test_create_buffer_with_text(self):
        result = self.bm.create_buffer("test", text="line1\nline2")
        self.assertTrue(result["ok"])
        content = self.bm.read_buffer("test", raw=True, show_line_numbers=False)
        self.assertEqual(content, "line1\nline2")

    def test_read_buffer_with_timestamps(self):
        self.bm.create_buffer("test", text="a\nb")
        result = self.bm.read_buffer("test", show_timestamps=True, show_line_numbers=False, raw=True)
        lines = result.strip().split("\n")
        for line in lines:
            self.assertRegex(line, r"^\(\d+\.\d+\)")
        self.assertEqual(len(lines), 2)

    def test_read_buffer_ranges(self):
        self.bm.create_buffer("test", text="a\nb\nc\nd\ne")
        result = self.bm.read_buffer("test", start=1, end=4, raw=True, show_line_numbers=False)
        self.assertEqual(result, "b\nc\nd")

    def test_write_buffer(self):
        self.bm.create_buffer("test")
        result = self.bm.write_buffer("test", "x\ny\nz")
        self.assertTrue(result["ok"])
        content = self.bm.read_buffer("test", raw=True, show_line_numbers=False)
        self.assertEqual(content, "x\ny\nz")

    def test_write_buffer_nonexistent(self):
        result = self.bm.write_buffer("nonexistent", "text")
        self.assertFalse(result["ok"])

    def test_drop_buffer(self):
        self.bm.create_buffer("test", text="data")
        result = self.bm.drop_buffer("test")
        self.assertTrue(result["ok"])
        result2 = self.bm.drop_buffer("test")
        self.assertFalse(result2["ok"])

    def test_list_buffers(self):
        self.bm.create_buffer("a", text="x")
        self.bm.create_buffer("b", text="y\nz")
        listing_raw = self.bm.read_buffer("system:list:buffers", raw=True)
        self.assertIn('"name": "a"', listing_raw)
        self.assertIn('"name": "b"', listing_raw)
        self.assertIn('"name": "system:list:buffers"', listing_raw)


class TestEditBufferTimestampSemantics(unittest.TestCase):
    """Test that edit_buffer assigns timestamps based on replacement regions, not content comparison."""

    def setUp(self):
        self.bm = BufferManager()

    def _make(self, text: str, timestamps: list[float] | None = None) -> None:
        """Create a buffer with explicit timestamps."""
        entries = []
        now = time.time()
        for i, line in enumerate(text.split("\n")):
            ts = timestamps[i] if timestamps and i < len(timestamps) else now
            entries.append(BufferEntry(data=line, timestamp=ts, seen=True))
        self.bm._buffers["t"] = Buffer(
            lines=entries,
            created_at=now,
            modified_at=now,
        )

    def _timestamps(self) -> list[float]:
        return [e.timestamp for e in self.bm._buffers["t"].lines]

    def test_simple_single_line_replace_preserves_untouched(self):
        self._make("A\nB\nC", timestamps=[1.0, 2.0, 3.0])
        self.bm.edit_buffer("t", "B", "X")
        result = self._timestamps()
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 1.0)  # untouched
        self.assertAlmostEqual(result[2], 3.0)  # untouched
        self.assertGreater(result[1], 3.0)  # changed (new timestamp)

    def test_single_line_expand_preserves_trailing(self):
        self._make("A\nB\nC", timestamps=[1.0, 2.0, 3.0])
        self.bm.edit_buffer("t", "B", "X\nY")
        result = self._timestamps()
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result[0], 1.0)  # untouched
        self.assertGreater(result[1], 2.0)  # new
        self.assertGreater(result[2], 2.0)  # new
        self.assertAlmostEqual(result[3], 3.0)  # untouched C (old line 2)

    def test_two_line_contract_to_one(self):
        self._make("A\nB\nC", timestamps=[1.0, 2.0, 3.0])
        self.bm.edit_buffer("t", "A\nB", "X")
        result = self._timestamps()
        self.assertEqual(len(result), 2)
        self.assertGreater(result[0], 2.0)  # X (touched)
        self.assertAlmostEqual(result[1], 3.0)  # C untouched

    def test_two_line_expand_from_three(self):
        self._make("A\nB\nC", timestamps=[1.0, 2.0, 3.0])
        self.bm.edit_buffer("t", "B\nC", "X\nY")
        result = self._timestamps()
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0], 1.0)  # A untouched
        self.assertGreater(result[1], 3.0)  # X touched
        self.assertGreater(result[2], 3.0)  # Y touched

    def test_replace_all_timestamps(self):
        self._make("A\nB\nA\nB", timestamps=[1.0, 2.0, 3.0, 4.0])
        self.bm.edit_buffer("t", "A", "X", replace_all=True)
        result = self._timestamps()
        self.assertEqual(len(result), 4)
        self.assertGreater(result[0], 4.0)  # X touched
        self.assertAlmostEqual(result[1], 2.0)  # B untouched
        self.assertGreater(result[2], 4.0)  # X touched
        self.assertAlmostEqual(result[3], 4.0)  # B untouched

    def test_multi_occurrence_replace_all(self):
        self._make("A\nB\nA\nB", timestamps=[1.0, 2.0, 3.0, 4.0])
        self.bm.edit_buffer("t", "A\nB", "X", replace_all=True)
        result = self._timestamps()
        self.assertEqual(len(result), 2)
        self.assertGreater(result[0], 4.0)  # first X
        self.assertGreater(result[1], 4.0)  # second X

    def test_replace_with_same_content_still_marks_as_touched(self):
        self._make("A\nB\nC", timestamps=[1.0, 2.0, 3.0])
        self.bm.edit_buffer("t", "B", "B")
        result = self._timestamps()
        self.assertAlmostEqual(result[0], 1.0)  # untouched
        self.assertGreater(result[1], 3.0)  # touched (same text but still replaced)
        self.assertAlmostEqual(result[2], 3.0)  # untouched

    def test_replace_nonexistent_returns_error(self):
        self._make("A\nB\nC")
        result = self.bm.edit_buffer("t", "Z", "X")
        self.assertFalse(result["ok"])
        self.assertIn("not found", result["error"])

    def test_replace_all_with_single_occurrence(self):
        self._make("A\nB\nC", timestamps=[1.0, 2.0, 3.0])
        self.bm.edit_buffer("t", "A", "X", replace_all=True)
        result = self._timestamps()
        self.assertEqual(len(result), 3)
        self.assertGreater(result[0], 3.0)  # touched
        self.assertAlmostEqual(result[1], 2.0)  # untouched
        self.assertAlmostEqual(result[2], 3.0)  # untouched

    def test_partial_match_without_replace_all_returns_error(self):
        self._make("A\nB\nA")
        result = self.bm.edit_buffer("t", "A", "X")
        self.assertFalse(result["ok"])
        self.assertIn("found multiple times", result["error"])
        self.assertIn("Set replace_all=True", result["error"])

    def test_empty_buffer_replace(self):
        self.bm._buffers["t"] = Buffer(
            lines=[],
            created_at=0.0,
            modified_at=0.0,
        )
        result = self.bm.edit_buffer("t", "A", "X")
        self.assertFalse(result["ok"])
        self.assertIn("not found", result["error"])

    def test_timestamp_precision_single_edit(self):
        t0 = time.time()
        self._make("A", timestamps=[t0])
        t1 = time.time()
        self.bm.edit_buffer("t", "A", "B")
        result = self._timestamps()
        self.assertGreaterEqual(result[0], t1)


class TestEditBufferReturnValues(unittest.TestCase):
    """Test edit_buffer return value hints."""

    def setUp(self):
        self.bm = BufferManager()

    def test_single_replace_hint(self):
        self.bm.create_buffer("t", text="A\nB")
        result = self.bm.edit_buffer("t", "A", "X")
        self.assertTrue(result["ok"])
        self.assertEqual(result["count"], 1)

    def test_replace_all_hint_count(self):
        self.bm.create_buffer("t", text="A\nA\nA")
        result = self.bm.edit_buffer("t", "A", "X", replace_all=True)
        self.assertTrue(result["ok"])
        self.assertEqual(result["count"], 3)

    def test_multiline_hint(self):
        self.bm.create_buffer("t", text="A\nB\nC")
        result = self.bm.edit_buffer("t", "A\nB", "X")
        self.assertTrue(result["ok"])
        self.assertEqual(result["count"], 1)


class TestDiffBuffers(unittest.TestCase):
    """Test diff_buffers uses BufferEntry correctly."""

    def setUp(self):
        self.bm = BufferManager()

    def test_diff_stores_entry_with_new_timestamp(self):
        self.bm.create_buffer("a", text="x\ny")
        self.bm.create_buffer("b", text="x\nz")
        result = self.bm.diff_buffers("a", "b")
        self.assertTrue(result["ok"])
        self.assertEqual(result["buffer"], "diff:a→b")
        diff_content = self.bm.read_buffer("diff:a→b", raw=True)
        self.assertIn("-y", diff_content)
        self.assertIn("+z", diff_content)

    def test_diff_identical_buffers(self):
        self.bm.create_buffer("a", text="x\ny")
        self.bm.create_buffer("b", text="x\ny")
        result = self.bm.diff_buffers("a", "b")
        self.assertTrue(result["ok"])
        self.assertTrue(result["identical"])


class TestEditBufferShowTimestamps(unittest.TestCase):
    """Test edit_buffer with show_timestamps in read."""

    def setUp(self):
        self.bm = BufferManager()

    def test_edit_preserves_timestamps_in_read_output(self):
        self.bm.create_buffer("t", text="A\nB\nC")
        self.bm.edit_buffer("t", "B", "X")
        result = self.bm.read_buffer("t", show_timestamps=True, show_line_numbers=False, raw=True)
        lines = result.strip().split("\n")
        for line in lines:
            self.assertRegex(line, r"^\(\d+\.\d+\)")


class TestEditBufferScopedRange(unittest.TestCase):
    """Test edit_buffer with explicit start/end range — only operates within the range."""

    def setUp(self):
        self.bm = BufferManager()

    def _make(self, text: str, timestamps: list[float] | None = None) -> None:
        entries = []
        now = time.time()
        for i, line in enumerate(text.split("\n")):
            ts = timestamps[i] if timestamps and i < len(timestamps) else now
            entries.append(BufferEntry(data=line, timestamp=ts, seen=True))
        self.bm._buffers["t"] = Buffer(
            lines=entries,
            created_at=now,
            modified_at=now,
        )

    def _timestamps(self) -> list[float]:
        return [e.timestamp for e in self.bm._buffers["t"].lines]

    def test_replace_only_within_range_untouched_outside(self):
        self._make("A\nB\nC\nD\nE", timestamps=[1.0, 2.0, 3.0, 4.0, 5.0])
        self.bm.edit_buffer("t", "B", "X", start=1, end=4)
        result = self._timestamps()
        self.assertEqual(len(result), 5)
        self.assertAlmostEqual(result[0], 1.0)  # A untouched (before range)
        self.assertGreater(result[1], 3.0)       # B replaced in range
        self.assertAlmostEqual(result[2], 3.0)  # C untouched (in range, unchanged)
        self.assertAlmostEqual(result[3], 4.0)  # D untouched (in range, unchanged)
        self.assertAlmostEqual(result[4], 5.0)  # E untouched (after range)

    def test_replace_all_within_range_only(self):
        self._make("A\nB\nA\nB\nA", timestamps=[1.0, 2.0, 3.0, 4.0, 5.0])
        self.bm.edit_buffer("t", "A", "X", start=1, end=4, replace_all=True)
        result = self._timestamps()
        self.assertEqual(len(result), 5)
        self.assertAlmostEqual(result[0], 1.0)  # A at index 0 outside range, untouched
        self.assertAlmostEqual(result[1], 2.0)  # B at index 1 outside range, untouched
        self.assertGreater(result[2], 5.0)  # A at index 2 in range replaced
        self.assertAlmostEqual(result[3], 4.0)  # B at index 3 outside range, untouched
        self.assertAlmostEqual(result[4], 5.0)  # A at index 4 outside range, untouched

    def test_replace_expand_within_range(self):
        self._make("A\nB\nC\nD", timestamps=[1.0, 2.0, 3.0, 4.0])
        self.bm.edit_buffer("t", "B", "X\nY", start=1, end=3)
        result = self._timestamps()
        self.assertEqual(len(result), 5)
        self.assertAlmostEqual(result[0], 1.0)  # A untouched
        self.assertGreater(result[1], 2.0)       # X touched
        self.assertGreater(result[2], 2.0)       # Y touched
        self.assertAlmostEqual(result[3], 3.0)  # C in range, untouched
        self.assertAlmostEqual(result[4], 4.0)  # D untouched (after range)

    def test_replace_outside_range_not_found(self):
        self._make("A\nB\nC", timestamps=[1.0, 2.0, 3.0])
        result = self.bm.edit_buffer("t", "A", "X", start=2, end=None)
        self.assertFalse(result["ok"])
        self.assertIn("not found", result["error"])

    def test_replace_expand_preserves_trailing_untouched(self):
        self._make("A\nB\nC", timestamps=[1.0, 2.0, 3.0])
        self.bm.edit_buffer("t", "B", "X\nY", start=1, end=3)
        result = self._timestamps()
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(result[0], 1.0)  # A untouched
        self.assertGreater(result[1], 2.0)       # X touched
        self.assertGreater(result[2], 2.0)       # Y touched
        self.assertAlmostEqual(result[3], 3.0)  # C untouched (after range)


class TestUpdateHooks(unittest.TestCase):
    """Test the update hook infrastructure: register, unregister, and pre-mutation gating."""

    def setUp(self):
        self.bm = BufferManager()
        self.calls: list = []

    def _accepting_hook(self, buf, old_text, start, end, new_text):
        self.calls.append((buf, old_text, start, end, new_text))
        return True

    def _rejecting_hook(self, buf, old_text, start, end, new_text):
        return "rejected by hook"

    def _reason_hook(self, buf, old_text, start, end, new_text):
        return "custom rejection reason"

    def test_register_hook_success(self):
        self.bm.create_buffer("test")
        result = self.bm.register_buffer_update_hook("test", "myhook", self._accepting_hook)
        self.assertTrue(result["ok"])
        self.assertEqual(result["hook"], "myhook")
        self.assertEqual(result["hook_count"], 1)

    def test_register_hook_no_overwrite(self):
        self.bm.create_buffer("test")
        self.bm.register_buffer_update_hook("test", "myhook", self._accepting_hook)
        result = self.bm.register_buffer_update_hook("test", "myhook", self._accepting_hook)
        self.assertFalse(result["ok"])
        self.assertIn("already registered", result["error"])

    def test_register_unknown_buffer(self):
        result = self.bm.register_buffer_update_hook("nonexistent", "hook", self._accepting_hook)
        self.assertFalse(result["ok"])
        self.assertIn("No buffer named", result["error"])

    def test_unregister_hook_success(self):
        self.bm.create_buffer("test")
        self.bm.register_buffer_update_hook("test", "myhook", self._accepting_hook)
        result = self.bm.unregister_buffer_update_hook("test", "myhook")
        self.assertTrue(result["ok"])
        self.assertEqual(result["removed"], "myhook")

    def test_unregister_unknown_silent(self):
        self.bm.create_buffer("test")
        result = self.bm.unregister_buffer_update_hook("test", "unknown")
        self.assertFalse(result["ok"])

    def test_write_buffer_hook_receives_correct_args(self):
        self.bm.create_buffer("test", text="A\nB\nC")
        self.bm.register_buffer_update_hook("test", "h", self._accepting_hook)
        self.bm.write_buffer("test", "X\nY", start=1, end=2)
        self.assertEqual(len(self.calls), 1)
        buf, old_text, start, end, new_text = self.calls[0]
        self.assertEqual(old_text, "B\n")  # line 1 replaced
        self.assertEqual(start, 1)
        self.assertEqual(end, 2)
        self.assertEqual(new_text, "X\nY\n")

    def test_edit_buffer_hook_receives_correct_args(self):
        self.bm.create_buffer("test", text="A\nB\nC")
        self.bm.register_buffer_update_hook("test", "h", self._accepting_hook)
        self.bm.edit_buffer("test", "B", "X", start=1, end=2)
        self.assertEqual(len(self.calls), 1)
        buf, old_text, start, end, new_text = self.calls[0]
        self.assertEqual(old_text, "B\n")
        self.assertIn("X\n", new_text)

    def test_drop_buffer_hook_receives_full_content_and_none(self):
        self.bm.create_buffer("test", text="A\nB\nC")
        self.bm.register_buffer_update_hook("test", "h", self._accepting_hook)
        self.bm.drop_buffer("test")
        self.assertEqual(len(self.calls), 1)
        buf, old_text, start, end, new_text = self.calls[0]
        self.assertIn("A\n", old_text)
        self.assertIn("B\n", old_text)
        self.assertEqual(new_text, None)

    def test_hook_accept_true_allows_write(self):
        self.bm.create_buffer("test", text="A\nB")
        self.bm.register_buffer_update_hook("test", "h", lambda *_: True)
        result = self.bm.write_buffer("test", "X", start=0)
        self.assertTrue(result["ok"])
        content = self.bm.read_buffer("test", raw=True, show_line_numbers=False)
        self.assertEqual(content, "X")

    def test_hook_accept_none_allows_write(self):
        self.bm.create_buffer("test", text="A\nB")
        self.bm.register_buffer_update_hook("test", "h", lambda *_: None)
        result = self.bm.write_buffer("test", "X")
        self.assertTrue(result["ok"])

    def test_hook_reject_blocks_write(self):
        self.bm.create_buffer("test", text="A\nB")
        self.bm.register_buffer_update_hook("test", "h", self._rejecting_hook)
        result = self.bm.write_buffer("test", "X")
        self.assertFalse(result["ok"])
        self.assertIn("rejected by hook", result["error"])
        content = self.bm.read_buffer("test", raw=True, show_line_numbers=False)
        self.assertEqual(content, "A\nB")  # unchanged

    def test_hook_reason_string_rejected(self):
        self.bm.create_buffer("test", text="A\nB")
        self.bm.register_buffer_update_hook("test", "h", self._reason_hook)
        result = self.bm.write_buffer("test", "X")
        self.assertFalse(result["ok"])
        self.assertIn("custom rejection reason", result["error"])

    def test_multiple_hooks_first_rejection_blocks(self):
        self.bm.create_buffer("test", text="A\nB")
        self.bm.register_buffer_update_hook("test", "first", self._accepting_hook)
        self.bm.register_buffer_update_hook("test", "second", self._rejecting_hook)
        result = self.bm.write_buffer("test", "X")
        self.assertFalse(result["ok"])
        self.assertIn("rejected by hook", result["error"])

    def test_hook_receives_buffer_reference(self):
        captured: list = []
        def spy(buf, old, start, end, new):
            captured.append(buf.modified_at)
            return True
        self.bm.create_buffer("test", text="X")
        self.bm.register_buffer_update_hook("test", "spy", spy)
        self.bm.write_buffer("test", "Y")
        self.assertGreater(captured[0], 0)

    def test_hook_reject_blocks_edit(self):
        self.bm.create_buffer("test", text="A\nB\nC")
        self.bm.register_buffer_update_hook("test", "h", self._rejecting_hook)
        result = self.bm.edit_buffer("test", "B", "X")
        self.assertFalse(result["ok"])
        content = self.bm.read_buffer("test", raw=True, show_line_numbers=False)
        self.assertEqual(content, "A\nB\nC")  # unchanged

    def test_hook_reject_blocks_drop(self):
        self.bm.create_buffer("test", text="A\nB")
        self.bm.register_buffer_update_hook("test", "h", self._rejecting_hook)
        result = self.bm.drop_buffer("test")
        self.assertFalse(result["ok"])
        self.assertIn("rejected by hook", result["error"])
        self.assertIn("test", self.bm._buffers)  # still exists

