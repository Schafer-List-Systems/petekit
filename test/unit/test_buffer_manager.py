"""Unit tests for BufferManager, especially edit_buffer timestamp behavior."""

from __future__ import annotations

import sys
import time
import unittest

sys.path.insert(0, "/home/frygge/projects/AIOS/peteos-kit")
sys.path.insert(0, "/home/frygge/projects/private/petekit/src/peteos/peteos")

from peteos_kit.buffer_manager import BufferManager, BufferEntry


class TestBufferManagerBasics(unittest.TestCase):
    """Basic create/read/write/drop operations."""

    def setUp(self):
        self.bm = BufferManager()

    def test_create_buffer_empty(self):
        result = self.bm.create_buffer("test")
        self.assertIn("created", result)
        result2 = self.bm.create_buffer("test")
        self.assertIn("already exists", result2)

    def test_create_buffer_with_text(self):
        result = self.bm.create_buffer("test", text="line1\nline2")
        self.assertIn("2 lines", result)
        content = self.bm.read_buffer("test")
        self.assertEqual(content, "line1\nline2")

    def test_read_buffer_with_timestamps(self):
        self.bm.create_buffer("test", text="a\nb")
        result = self.bm.read_buffer("test", show_timestamps=True)
        lines = result.strip().split("\n")
        for line in lines:
            self.assertRegex(line, r"^\(\d+\.\d+\)")
        self.assertEqual(len(lines), 2)

    def test_read_buffer_ranges(self):
        self.bm.create_buffer("test", text="a\nb\nc\nd\ne")
        result = self.bm.read_buffer("test", start=2, end=4)
        self.assertEqual(result, "b\nc\nd")

    def test_write_buffer(self):
        self.bm.create_buffer("test")
        result = self.bm.write_buffer("test", "x\ny\nz")
        self.assertIn("3 lines", result)
        self.assertEqual(self.bm.read_buffer("test"), "x\ny\nz")

    def test_write_buffer_nonexistent(self):
        result = self.bm.write_buffer("nonexistent", "text")
        self.assertIn("Error", result)

    def test_drop_buffer(self):
        self.bm.create_buffer("test", text="data")
        result = self.bm.drop_buffer("test")
        self.assertIn("dropped", result)
        result2 = self.bm.drop_buffer("test")
        self.assertIn("Error", result2)

    def test_list_buffers(self):
        self.bm.create_buffer("a", text="x")
        self.bm.create_buffer("b", text="y\nz")
        listing = self.bm.list_buffers()
        self.assertEqual(listing, {"a": 1, "b": 2})


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
        self.bm._buffers["t"] = __import__(
            "peteos_kit.buffer_manager", fromlist=["Buffer"]
        ).Buffer(
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
        self.assertIn("Error", result)
        self.assertIn("not found", result)

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
        self.assertIn("found multiple times", result)
        self.assertIn("Set replace_all=True", result)

    def test_empty_buffer_replace(self):
        self.bm._buffers["t"] = __import__(
            "peteos_kit.buffer_manager", fromlist=["Buffer"]
        ).Buffer(
            lines=[],
            created_at=0.0,
            modified_at=0.0,
        )
        result = self.bm.edit_buffer("t", "A", "X")
        self.assertIn("not found", result)

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
        self.assertIn("1 occurrence", result)

    def test_replace_all_hint_count(self):
        self.bm.create_buffer("t", text="A\nA\nA")
        result = self.bm.edit_buffer("t", "A", "X", replace_all=True)
        self.assertIn("3 occurrence", result)

    def test_multiline_hint(self):
        self.bm.create_buffer("t", text="A\nB\nC")
        result = self.bm.edit_buffer("t", "A\nB", "X")
        self.assertIn("1 occurrence", result)


class TestDiffBuffers(unittest.TestCase):
    """Test diff_buffers uses BufferEntry correctly."""

    def setUp(self):
        self.bm = BufferManager()

    def test_diff_stores_entry_with_new_timestamp(self):
        self.bm.create_buffer("a", text="x\ny")
        self.bm.create_buffer("b", text="x\nz")
        result = self.bm.diff_buffers("a", "b")
        self.assertIn("diff:a→b", result)
        diff_content = self.bm.read_buffer("diff:a→b")
        self.assertIn("-y", diff_content)
        self.assertIn("+z", diff_content)

    def test_diff_identical_buffers(self):
        self.bm.create_buffer("a", text="x\ny")
        self.bm.create_buffer("b", text="x\ny")
        result = self.bm.diff_buffers("a", "b")
        self.assertIn("identical", result)


class TestEditBufferShowTimestamps(unittest.TestCase):
    """Test edit_buffer with show_timestamps in read."""

    def setUp(self):
        self.bm = BufferManager()

    def test_edit_preserves_timestamps_in_read_output(self):
        self.bm.create_buffer("t", text="A\nB\nC")
        self.bm.edit_buffer("t", "B", "X")
        result = self.bm.read_buffer("t", show_timestamps=True)
        lines = result.strip().split("\n")
        # first line A should have its original timestamp, not the new one
        # last line C should have original
        # middle line X should have new
        for line in lines:
            self.assertRegex(line, r"^\(\d+\.\d+\)")
