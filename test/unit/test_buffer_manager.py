"""Unit tests for BufferManager, especially edit_buffer timestamp behavior."""

from __future__ import annotations

import time
import pytest

from petekit import BufferManager
from petekit.text.buffer_manager import Buffer, BufferEntry, _cluster_lines_to_ranges, _NO_BUFFER_


@pytest.fixture
async def bm():
    return BufferManager()


class TestBufferManagerBasics:
    async def test_create_buffer_empty(self, bm):
        result = await bm.create_buffer("test")
        assert result["ok"]
        result2 = await bm.create_buffer("test")
        assert not result2["ok"]

    async def test_create_buffer_with_text(self, bm):
        result = await bm.create_buffer("test", text="line1\nline2")
        assert result["ok"]
        content = await bm.read_buffer("test", raw=True, show_line_numbers=False)
        assert content == "line1\nline2"

    async def test_read_buffer_with_timestamps(self, bm):
        await bm.create_buffer("test", text="a\nb")
        result = await bm.read_buffer("test", show_timestamps=True, show_line_numbers=False, raw=True)
        lines = result.strip().split("\n")
        for line in lines:
            assert line.startswith("(")
        assert len(lines) == 2

    async def test_read_buffer_ranges(self, bm):
        await bm.create_buffer("test", text="a\nb\nc\nd\ne")
        result = await bm.read_buffer("test", start=1, end=4, raw=True, show_line_numbers=False)
        assert result == "b\nc\nd"

    async def test_write_buffer(self, bm):
        await bm.create_buffer("test")
        result = await bm.write_buffer("test", "x\ny\nz")
        assert result["ok"]
        content = await bm.read_buffer("test", raw=True, show_line_numbers=False)
        assert content == "x\ny\nz"

    async def test_write_buffer_nonexistent(self, bm):
        result = await bm.write_buffer("nonexistent", "text")
        assert not result["ok"]

    async def test_drop_buffer(self, bm):
        await bm.create_buffer("test", text="data")
        result = await bm.drop_buffer("test")
        assert result["ok"]
        result2 = await bm.drop_buffer("test")
        assert not result2["ok"]

    async def test_list_buffers(self, bm):
        await bm.create_buffer("a", text="x")
        await bm.create_buffer("b", text="y\nz")
        listing_raw = await bm.read_buffer("system:list:buffers", raw=True)
        assert '"name": "a"' in listing_raw
        assert '"name": "b"' in listing_raw
        assert '"name": "system:list:buffers"' in listing_raw


class TestEditBufferTimestampSemantics:
    def _make(self, bm, text: str, timestamps: list[float] | None = None) -> None:
        entries = []
        now = time.time()
        for i, line in enumerate(text.split("\n")):
            ts = timestamps[i] if timestamps and i < len(timestamps) else now
            entries.append(BufferEntry(data=line, modified_at=ts))
        bm._buffers["t"] = Buffer(
            name="t",
            lines=entries,
            created_at=now,
            modified_at=now,
        )

    def _timestamps(self, bm) -> list[float]:
        return [e.modified_at for e in bm._buffers["t"].lines]

    async def test_simple_single_line_replace_preserves_untouched(self, bm):
        self._make(bm, "A\nB\nC", timestamps=[1.0, 2.0, 3.0])
        await bm.edit_buffer("t", "B", "X")
        result = self._timestamps(bm)
        assert len(result) == 3
        assert abs(result[0] - 1.0) < 0.001  # untouched
        assert abs(result[2] - 3.0) < 0.001  # untouched
        assert result[1] > 3.0  # changed (new timestamp)

    async def test_single_line_expand_preserves_trailing(self, bm):
        self._make(bm, "A\nB\nC", timestamps=[1.0, 2.0, 3.0])
        await bm.edit_buffer("t", "B", "X\nY")
        result = self._timestamps(bm)
        assert len(result) == 4
        assert abs(result[0] - 1.0) < 0.001  # untouched
        assert result[1] > 2.0  # new
        assert result[2] > 2.0  # new
        assert abs(result[3] - 3.0) < 0.001  # untouched C (old line 2)

    async def test_two_line_contract_to_one(self, bm):
        self._make(bm, "A\nB\nC", timestamps=[1.0, 2.0, 3.0])
        await bm.edit_buffer("t", "A\nB", "X")
        result = self._timestamps(bm)
        assert len(result) == 2
        assert result[0] > 2.0  # X (touched)
        assert abs(result[1] - 3.0) < 0.001  # C untouched

    async def test_two_line_expand_from_three(self, bm):
        self._make(bm, "A\nB\nC", timestamps=[1.0, 2.0, 3.0])
        await bm.edit_buffer("t", "B\nC", "X\nY")
        result = self._timestamps(bm)
        assert len(result) == 3
        assert abs(result[0] - 1.0) < 0.001  # A untouched
        assert result[1] > 3.0  # X touched
        assert result[2] > 3.0  # Y touched

    async def test_replace_all_timestamps(self, bm):
        self._make(bm, "A\nB\nA\nB", timestamps=[1.0, 2.0, 3.0, 4.0])
        await bm.edit_buffer("t", "A", "X", replace_all=True)
        result = self._timestamps(bm)
        assert len(result) == 4
        assert result[0] > 4.0  # X touched
        assert abs(result[1] - 2.0) < 0.001  # B untouched
        assert result[2] > 4.0  # X touched
        assert abs(result[3] - 4.0) < 0.001  # B untouched

    async def test_multi_occurrence_replace_all(self, bm):
        self._make(bm, "A\nB\nA\nB", timestamps=[1.0, 2.0, 3.0, 4.0])
        await bm.edit_buffer("t", "A\nB", "X", replace_all=True)
        result = self._timestamps(bm)
        assert len(result) == 2
        assert result[0] > 4.0  # first X
        assert result[1] > 4.0  # second X

    async def test_replace_with_same_content_still_marks_as_touched(self, bm):
        self._make(bm, "A\nB\nC", timestamps=[1.0, 2.0, 3.0])
        await bm.edit_buffer("t", "B", "B")
        result = self._timestamps(bm)
        assert abs(result[0] - 1.0) < 0.001  # untouched
        assert result[1] > 3.0  # touched (same text but still replaced)
        assert abs(result[2] - 3.0) < 0.001  # untouched

    async def test_replace_nonexistent_returns_error(self, bm):
        self._make(bm, "A\nB\nC")
        result = await bm.edit_buffer("t", "Z", "X")
        assert not result["ok"]
        assert "not found" in result["error"]

    async def test_replace_all_with_single_occurrence(self, bm):
        self._make(bm, "A\nB\nC", timestamps=[1.0, 2.0, 3.0])
        await bm.edit_buffer("t", "A", "X", replace_all=True)
        result = self._timestamps(bm)
        assert len(result) == 3
        assert result[0] > 3.0  # touched
        assert abs(result[1] - 2.0) < 0.001  # untouched
        assert abs(result[2] - 3.0) < 0.001  # untouched

    async def test_partial_match_without_replace_all_returns_error(self, bm):
        self._make(bm, "A\nB\nA")
        result = await bm.edit_buffer("t", "A", "X")
        assert not result["ok"]
        assert "found multiple times" in result["error"]
        assert "Set replace_all=True" in result["error"]

    async def test_empty_buffer_replace(self, bm):
        bm._buffers["t"] = Buffer(
            name="t",
            lines=[],
            created_at=0.0,
            modified_at=0.0,
        )
        result = await bm.edit_buffer("t", "A", "X")
        assert not result["ok"]
        assert "empty" in result["error"].lower() or "nothing was replaced" in result["error"]

    async def test_timestamp_precision_single_edit(self, bm):
        t0 = time.time()
        self._make(bm, "A", timestamps=[t0])
        t1 = time.time()
        await bm.edit_buffer("t", "A", "B")
        result = self._timestamps(bm)
        assert result[0] >= t1


class TestEditBufferReturnValues:
    async def test_single_replace_hint(self, bm):
        await bm.create_buffer("t", text="A\nB")
        result = await bm.edit_buffer("t", "A", "X")
        assert result["ok"]
        assert result["count"] == 1

    async def test_replace_all_hint_count(self, bm):
        await bm.create_buffer("t", text="A\nA\nA")
        result = await bm.edit_buffer("t", "A", "X", replace_all=True)
        assert result["ok"]
        assert result["count"] == 3

    async def test_multiline_hint(self, bm):
        await bm.create_buffer("t", text="A\nB\nC")
        result = await bm.edit_buffer("t", "A\nB", "X")
        assert result["ok"]
        assert result["count"] == 1


class TestDiffBuffers:
    async def test_diff_stores_entry_with_new_timestamp(self, bm):
        await bm.create_buffer("a", text="x\ny")
        await bm.create_buffer("b", text="x\nz")
        result = await bm.diff_buffers("a", "b")
        assert result["ok"]
        assert result["buffer"] == "diff:a→b"
        diff_content = await bm.read_buffer("diff:a→b", raw=True)
        assert "-y" in diff_content
        assert "+z" in diff_content

    async def test_diff_identical_buffers(self, bm):
        await bm.create_buffer("a", text="x\ny")
        await bm.create_buffer("b", text="x\ny")
        result = await bm.diff_buffers("a", "b")
        assert result["ok"]
        assert result["identical"]


class TestEditBufferShowTimestamps:
    async def test_edit_preserves_timestamps_in_read_output(self, bm):
        await bm.create_buffer("t", text="A\nB\nC")
        await bm.edit_buffer("t", "B", "X")
        result = await bm.read_buffer("t", show_timestamps=True, show_line_numbers=False, raw=True)
        lines = result.strip().split("\n")
        for line in lines:
            assert line.startswith("(")


class TestEditBufferScopedRange:
    def _make(self, bm, text: str, timestamps: list[float] | None = None) -> None:
        entries = []
        now = time.time()
        for i, line in enumerate(text.split("\n")):
            ts = timestamps[i] if timestamps and i < len(timestamps) else now
            entries.append(BufferEntry(data=line, modified_at=ts))
        bm._buffers["t"] = Buffer(
            name="t",
            lines=entries,
            created_at=now,
            modified_at=now,
        )

    def _timestamps(self, bm) -> list[float]:
        return [e.modified_at for e in bm._buffers["t"].lines]

    async def test_replace_only_within_range_untouched_outside(self, bm):
        self._make(bm, "A\nB\nC\nD\nE", timestamps=[1.0, 2.0, 3.0, 4.0, 5.0])
        await bm.edit_buffer("t", "B", "X", start=1, end=4)
        result = self._timestamps(bm)
        assert len(result) == 5
        assert abs(result[0] - 1.0) < 0.001  # A untouched (before range)
        assert result[1] > 3.0       # B replaced in range
        assert abs(result[2] - 3.0) < 0.001  # C untouched (in range, unchanged)
        assert abs(result[3] - 4.0) < 0.001  # D untouched (in range, unchanged)
        assert abs(result[4] - 5.0) < 0.001  # E untouched (after range)

    async def test_replace_all_within_range_only(self, bm):
        self._make(bm, "A\nB\nA\nB\nA", timestamps=[1.0, 2.0, 3.0, 4.0, 5.0])
        await bm.edit_buffer("t", "A", "X", start=1, end=4, replace_all=True)
        result = self._timestamps(bm)
        assert len(result) == 5
        assert abs(result[0] - 1.0) < 0.001  # A at index 0 outside range, untouched
        assert abs(result[1] - 2.0) < 0.001  # B at index 1 outside range, untouched
        assert result[2] > 5.0  # A at index 2 in range replaced
        assert abs(result[3] - 4.0) < 0.001  # B at index 3 outside range, untouched
        assert abs(result[4] - 5.0) < 0.001  # A at index 4 outside range, untouched

    async def test_replace_expand_within_range(self, bm):
        self._make(bm, "A\nB\nC\nD", timestamps=[1.0, 2.0, 3.0, 4.0])
        await bm.edit_buffer("t", "B", "X\nY", start=1, end=3)
        result = self._timestamps(bm)
        assert len(result) == 5
        assert abs(result[0] - 1.0) < 0.001  # A untouched
        assert result[1] > 2.0       # X touched
        assert result[2] > 2.0       # Y touched
        assert abs(result[3] - 3.0) < 0.001  # C in range, untouched
        assert abs(result[4] - 4.0) < 0.001  # D untouched (after range)

    async def test_replace_outside_range_not_found(self, bm):
        self._make(bm, "A\nB\nC", timestamps=[1.0, 2.0, 3.0])
        result = await bm.edit_buffer("t", "A", "X", start=2, end=None)
        assert not result["ok"]
        assert "not found" in result["error"]

    async def test_replace_expand_preserves_trailing_untouched(self, bm):
        self._make(bm, "A\nB\nC", timestamps=[1.0, 2.0, 3.0])
        await bm.edit_buffer("t", "B", "X\nY", start=1, end=3)
        result = self._timestamps(bm)
        assert len(result) == 4
        assert abs(result[0] - 1.0) < 0.001  # A untouched
        assert result[1] > 2.0       # X touched
        assert result[2] > 2.0       # Y touched
        assert abs(result[3] - 3.0) < 0.001  # C untouched (after range)


class TestUpdateHooks:
    async def test_register_hook_success(self, bm):
        await bm.create_buffer("test")
        result = bm.register_buffer_update_hook("test", "myhook", lambda *_: True)
        assert result["ok"]
        assert result["hook"] == "myhook"
        assert result["hook_count"] == 1

    async def test_register_hook_no_overwrite(self, bm):
        await bm.create_buffer("test")
        bm.register_buffer_update_hook("test", "myhook", lambda *_: True)
        result = bm.register_buffer_update_hook("test", "myhook", lambda *_: True)
        assert not result["ok"]
        assert "already registered" in result["error"]

    async def test_register_unknown_buffer(self, bm):
        result = bm.register_buffer_update_hook("nonexistent", "hook", lambda *_: True)
        assert not result["ok"]
        assert "No buffer named" in result["error"]

    async def test_unregister_hook_success(self, bm):
        await bm.create_buffer("test")
        bm.register_buffer_update_hook("test", "myhook", lambda *_: True)
        result = bm.unregister_buffer_update_hook("test", "myhook")
        assert result["ok"]
        assert result["removed"] == "myhook"

    async def test_unregister_unknown_silent(self, bm):
        await bm.create_buffer("test")
        result = bm.unregister_buffer_update_hook("test", "unknown")
        assert not result["ok"]

    async def test_edit_buffer_hook_receives_correct_args(self, bm):
        await bm.create_buffer("test", text="A\nB\nC")
        calls = []
        def accepting_hook(buf, old_text, start, end, new_text):
            calls.append((buf, old_text, start, end, new_text))
            return True
        bm.register_buffer_update_hook("test", "h", accepting_hook)
        await bm.edit_buffer("test", "B", "X", start=1, end=2)
        assert len(calls) == 1
        buf, old_text, start, end, new_text = calls[0]
        assert old_text == "B\n"
        assert "X\n" in new_text

    async def test_drop_buffer_hook_receives_full_content_and_none(self, bm):
        await bm.create_buffer("test", text="A\nB\nC")
        calls = []
        def accepting_hook(buf, old_text, start, end, new_text):
            calls.append((buf, old_text, start, end, new_text))
            return True
        bm.register_buffer_update_hook("test", "h", accepting_hook)
        await bm.drop_buffer("test")
        assert len(calls) == 1
        buf, old_text, start, end, new_text = calls[0]
        assert "A\n" in old_text
        assert "B\n" in old_text
        assert new_text is None

    async def test_hook_accept_true_allows_write(self, bm):
        await bm.create_buffer("test")
        bm.register_buffer_update_hook("test", "h", lambda *_: True)
        result = await bm.write_buffer("test", "X\nY")
        assert result["ok"]
        content = await bm.read_buffer("test", raw=True, show_line_numbers=False)
        assert content == "X\nY"

    async def test_hook_accept_none_allows_write(self, bm):
        await bm.create_buffer("test", text="A\nB")
        bm.register_buffer_update_hook("test", "h", lambda *_: None)
        result = await bm.write_buffer("test", "X")
        assert result["ok"]

    async def test_hook_reject_blocks_write(self, bm):
        await bm.create_buffer("test", text="A\nB")
        bm.register_buffer_update_hook("test", "h", lambda *_: "rejected by hook")
        result = await bm.write_buffer("test", "X")
        assert not result["ok"]
        assert "rejected by hook" in result["error"]
        content = await bm.read_buffer("test", raw=True, show_line_numbers=False)
        assert content == "A\nB"  # unchanged

    async def test_hook_reason_string_rejected(self, bm):
        await bm.create_buffer("test", text="A\nB")
        bm.register_buffer_update_hook("test", "h", lambda *_: "custom rejection reason")
        result = await bm.write_buffer("test", "X")
        assert not result["ok"]
        assert "custom rejection reason" in result["error"]

    async def test_multiple_hooks_first_rejection_blocks(self, bm):
        await bm.create_buffer("test", text="A\nB")
        bm.register_buffer_update_hook("test", "first", lambda *_: True)
        bm.register_buffer_update_hook("test", "second", lambda *_: "rejected by hook")
        result = await bm.write_buffer("test", "X")
        assert not result["ok"]
        assert "rejected by hook" in result["error"]

    async def test_hook_receives_buffer_reference(self, bm):
        captured = []
        def spy(buf, old, start, end, new):
            captured.append(buf.modified_at)
            return True
        await bm.create_buffer("test", text="X")
        bm.register_buffer_update_hook("test", "spy", spy)
        await bm.write_buffer("test", "Y")
        assert captured[0] > 0

    async def test_hook_reject_blocks_edit(self, bm):
        await bm.create_buffer("test", text="A\nB\nC")
        bm.register_buffer_update_hook("test", "h", lambda *_: "rejected by hook")
        result = await bm.edit_buffer("test", "B", "X")
        assert not result["ok"]
        content = await bm.read_buffer("test", raw=True, show_line_numbers=False)
        assert content == "A\nB\nC"  # unchanged

    async def test_hook_reject_blocks_drop(self, bm):
        await bm.create_buffer("test", text="A\nB")
        bm.register_buffer_update_hook("test", "h", lambda *_: "rejected by hook")
        result = await bm.drop_buffer("test")
        assert not result["ok"]
        assert "rejected by hook" in result["error"]
        assert "test" in bm._buffers  # still exists

    async def test_create_buffer_overwrite_fires_drop_hooks(self, bm):
        await bm.create_buffer("test", text="keep: me")
        bm.register_buffer_update_hook("test", "nope", lambda *_: "no overwrite")
        result = await bm.create_buffer("test", text="new: thing", overwrite=True)
        assert not result["ok"]
        assert result["error"] == "no overwrite"
        existing = await bm.read_buffer("test", raw=True, show_line_numbers=False)
        assert existing == "keep: me"

    async def test_create_buffer_overwrite_succeeds_without_hooks(self, bm):
        await bm.create_buffer("test", text="old: thing")
        result = await bm.create_buffer("test", text="new: thing", overwrite=True)
        assert result["ok"]
        assert result["overwritten"]
        existing = await bm.read_buffer("test", raw=True, show_line_numbers=False)
        assert existing == "new: thing"


class TestClusterLinesToRanges:
    def test_empty_line_numbers(self):
        result = _cluster_lines_to_ranges([], 5)
        assert result == []

    def test_fewer_lines_than_clusters(self):
        result = _cluster_lines_to_ranges([1, 2, 3], 5)
        assert result == [
            {"start": 1, "end": 1, "match_count": 1},
            {"start": 2, "end": 2, "match_count": 1},
            {"start": 3, "end": 3, "match_count": 1},
        ]

    def test_clusters_without_buf_lines_does_not_crash(self):
        result = _cluster_lines_to_ranges([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
        assert len(result) == 3
        for c in result:
            assert c["start"] <= c["end"]
            assert c["match_count"] >= 1

    def test_clusters_with_buf_lines(self):
        entries = [BufferEntry(data=f"line{i}", modified_at=0.0) for i in range(10)]
        result = _cluster_lines_to_ranges([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 3, buf_lines=entries)
        assert len(result) == 3
        for c in result:
            assert c["start"] <= c["end"]
            assert c["match_count"] >= 1
