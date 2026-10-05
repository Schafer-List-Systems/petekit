"""Unit tests for StreamBufferManager, especially time-based reading and hook behavior."""

from __future__ import annotations

import asyncio
import time
import pytest

from petekit.stream.stream_buffer_manager import StreamBufferManager, StreamBufferHook, StreamBufferHookError
from petekit.text.buffer_manager import Buffer, BufferEntry


@pytest.fixture
async def sbm():
    return StreamBufferManager()


class TestStreamBufferCreateDrop:
    async def test_create_stream_requires_stream_prefix(self, sbm):
        result = await sbm.create_buffer("bad_name", stream=True)
        assert not result["ok"]
        assert "stream:" in result["error"]

    async def test_create_stream_ok(self, sbm):
        result = await sbm.create_buffer("stream:test", stream=True)
        assert result["ok"]
        assert "stream:test" in sbm.stream_buffer_configs

    async def test_create_stream_twice_blocked(self, sbm):
        await sbm.create_buffer("stream:test", stream=True)
        result = await sbm.create_buffer("stream:test", stream=True)
        assert not result["ok"]

    async def test_create_stream_overwrite(self, sbm):
        await sbm.create_buffer("stream:test", stream=True)
        result = await sbm.create_buffer("stream:test", stream=True, overwrite=True)
        assert result["ok"]

    async def test_drop_stream_cleans_up_config(self, sbm):
        await sbm.create_buffer("stream:test", stream=True)
        await sbm.drop_buffer("stream:test")
        assert "stream:test" not in sbm.stream_buffer_configs

    async def test_drop_non_stream_still_works(self, sbm):
        await sbm.create_buffer("regular")
        result = await sbm.drop_buffer("regular")
        assert result["ok"]


class TestResolveTime:
    async def test_positive_passes_through(self, sbm):
        result = sbm._resolve_time(123.456, anchor=200.0)
        assert result == 123.456

    async def test_negative_adds_to_anchor(self, sbm):
        result = sbm._resolve_time(-30.0, anchor=100.0)
        assert result == 70.0

    async def test_zero_is_absolute_not_relative(self, sbm):
        result = sbm._resolve_time(-0.0, anchor=100.0)
        assert result == -0.0


class TestStreamBufferReadBufferTimeBased:
    @pytest.fixture
    async def _sbm_with_stream(self, sbm):
        ts = [10.0, 20.0, 30.0, 40.0, 50.0]
        data = ["a", "b", "c", "d", "e"]
        entries = [BufferEntry(data=d, modified_at=t) for t, d in zip(ts, data)]
        sbm._buffers["stream:t"] = Buffer(lines=entries, created_at=ts[0], modified_at=ts[-1])
        sbm.stream_buffer_configs["stream:t"] = type("C", (), {"hooks": {}, "created_at": ts[0]})()
        return sbm

    async def test_absolute_range_inclusive(self, _sbm_with_stream):
        result = await _sbm_with_stream.read_buffer("stream:t", start=20.0, end=40.0, show_line_numbers=False)
        assert result["ok"]
        lines = result["content"].split("\n")
        timestamps_in_content = [float(l.split(")")[0][1:]) for l in lines]
        assert timestamps_in_content == [20.0, 30.0, 40.0]

    async def test_float_start_only(self, _sbm_with_stream):
        result = await _sbm_with_stream.read_buffer("stream:t", start=35.0, show_line_numbers=False)
        assert result["ok"]
        lines = result["content"].split("\n")
        assert len(lines) == 2
        timestamps = [float(l.split(")")[0][1:]) for l in lines]
        assert timestamps == [40.0, 50.0]

    async def test_float_end_only(self, _sbm_with_stream):
        result = await _sbm_with_stream.read_buffer("stream:t", end=25.0, show_line_numbers=False)
        assert result["ok"]
        lines = result["content"].split("\n")
        timestamps = [float(l.split(")")[0][1:]) for l in lines]
        assert timestamps == [10.0, 20.0]

    async def test_start_beyond_all_entries(self, _sbm_with_stream):
        result = await _sbm_with_stream.read_buffer("stream:t", start=100.0)
        assert not result["ok"]
        assert "No entries at or after" in result["error"]

    async def test_end_before_all_entries(self, _sbm_with_stream):
        result = await _sbm_with_stream.read_buffer("stream:t", end=5.0)
        assert not result["ok"]
        assert "No entries at or before" in result["error"]

    async def test_empty_buffer_error(self, _sbm_with_stream):
        _sbm_with_stream._buffers["stream:empty"] = Buffer(lines=[], created_at=0.0, modified_at=0.0)
        _sbm_with_stream.stream_buffer_configs["stream:empty"] = type("C", (), {"hooks": [], "created_at": 0.0})()
        result = await _sbm_with_stream.read_buffer("stream:empty", start=0.0)
        assert not result["ok"]
        assert "empty" in result["error"]

    async def test_non_stream_buffer_rejects_float(self, _sbm_with_stream):
        await _sbm_with_stream.create_buffer("regular", text="x")
        result = await _sbm_with_stream.read_buffer("regular", start=10.0)
        assert not result["ok"]
        assert "Only stream buffers support" in result["error"]


class TestStreamBufferReadBufferRelativeTime:
    @pytest.fixture
    async def _sbm_with_stream(self, sbm):
        ts = [10.0, 20.0, 30.0, 40.0, 50.0]
        data = ["a", "b", "c", "d", "e"]
        entries = [BufferEntry(data=d, modified_at=t) for t, d in zip(ts, data)]
        sbm._buffers["stream:t"] = Buffer(lines=entries, created_at=ts[0], modified_at=ts[-1])
        sbm.stream_buffer_configs["stream:t"] = type("C", (), {"hooks": {}, "created_at": ts[0]})()
        return sbm

    async def test_negative_start_relative_to_last_entry(self, _sbm_with_stream):
        result = await _sbm_with_stream.read_buffer("stream:t", start=-20.0, show_line_numbers=False)
        assert result["ok"]
        lines = result["content"].split("\n")
        timestamps = [float(l.split(")")[0][1:]) for l in lines]
        assert timestamps == [30.0, 40.0, 50.0]

    async def test_negative_end_relative_to_last_entry(self, _sbm_with_stream):
        result = await _sbm_with_stream.read_buffer("stream:t", end=-20.0, show_line_numbers=False)
        assert result["ok"]
        lines = result["content"].split("\n")
        timestamps = [float(l.split(")")[0][1:]) for l in lines]
        assert timestamps == [10.0, 20.0, 30.0]

    async def test_both_negative_relative_range(self, _sbm_with_stream):
        result = await _sbm_with_stream.read_buffer("stream:t", start=-30.0, end=-10.0, show_line_numbers=False)
        assert result["ok"]
        lines = result["content"].split("\n")
        timestamps = [float(l.split(")")[0][1:]) for l in lines]
        assert timestamps == [20.0, 30.0, 40.0]

    async def test_negative_zero_means_from_start(self, _sbm_with_stream):
        result = await _sbm_with_stream.read_buffer("stream:t", start=-0.0, show_line_numbers=False)
        assert result["ok"]
        lines = result["content"].split("\n")
        assert len(lines) == 5
        timestamps = [float(l.split(")")[0][1:]) for l in lines]
        assert timestamps == [10.0, 20.0, 30.0, 40.0, 50.0]

    async def test_negative_start_large_offset_captures_all(self, _sbm_with_stream):
        result = await _sbm_with_stream.read_buffer("stream:t", start=-200.0)
        assert result["ok"]
        lines = result["content"].split("\n")
        assert len(lines) == 5

    async def test_negative_end_exceeds_range(self, _sbm_with_stream):
        result = await _sbm_with_stream.read_buffer("stream:t", end=-200.0)
        assert not result["ok"]
        assert "No entries at or before" in result["error"]


class TestStreamBufferReadBufferLineBased:
    @pytest.fixture
    async def _sbm_with_stream(self, sbm):
        ts = [10.0, 20.0, 30.0, 40.0, 50.0]
        data = ["a", "b", "c", "d", "e"]
        entries = [BufferEntry(data=d, modified_at=t) for t, d in zip(ts, data)]
        sbm._buffers["stream:t"] = Buffer(lines=entries, created_at=ts[0], modified_at=ts[-1])
        sbm.stream_buffer_configs["stream:t"] = type("C", (), {"hooks": {}, "created_at": ts[0]})()
        return sbm

    async def test_read_all_with_timestamps(self, _sbm_with_stream):
        result = await _sbm_with_stream.read_buffer("stream:t", show_timestamps=True, show_line_numbers=True)
        assert result["ok"]
        lines = result["content"].split("\n")
        assert len(lines) == 5
        for line in lines:
            assert line.startswith("1: (10.0)") or ":" in line

    async def test_read_range_0_based(self, _sbm_with_stream):
        result = await _sbm_with_stream.read_buffer("stream:t", start=1, end=4, show_timestamps=True, show_line_numbers=False)
        assert result["ok"]
        lines = result["content"].split("\n")
        assert len(lines) == 3
        timestamps = [float(l.split(")")[0][1:]) for l in lines]
        assert timestamps == [20.0, 30.0, 40.0]

    async def test_negative_index_from_end(self, _sbm_with_stream):
        result = await _sbm_with_stream.read_buffer("stream:t", start=-2, end=None, show_timestamps=True, show_line_numbers=False)
        assert result["ok"]
        lines = result["content"].split("\n")
        assert len(lines) == 2
        assert lines[-1] != ""
        timestamps = [float(l.split(")")[0][1:]) for l in lines if l]
        assert timestamps == [40.0, 50.0]


class TestStreamBufferHookBehavior:
    @pytest.fixture
    async def _sbm_with_stream(self, sbm):
        sbm._buffers["stream:t"] = Buffer(lines=[], created_at=0.0, modified_at=0.0)
        sbm.stream_buffer_configs["stream:t"] = type("C", (), {"hooks": {}, "created_at": 0.0})()
        return sbm

    async def test_hook_receives_entry_and_stream_name(self, _sbm_with_stream):
        hook_called = []
        async def capture(sb, text, metadata):
            hook_called.append((sb, text))
        _sbm_with_stream.stream_buffer_configs["stream:t"].hooks["capture"] = StreamBufferHook(
            callable_=capture, priority=0
        )
        await _sbm_with_stream.write_buffer("stream:t", "hello")
        assert len(hook_called) == 1
        assert hook_called[0] == ("stream:t", "hello")

    async def test_hooks_fire_in_priority_order(self, _sbm_with_stream):
        order = []
        async def cb_low(sb, text, metadata):
            order.append("low")
        async def cb_high(sb, text, metadata):
            order.append("high")
        async def cb_mid(sb, text, metadata):
            order.append("mid")
        _sbm_with_stream.stream_buffer_configs["stream:t"].hooks = {
            "low": StreamBufferHook(callable_=cb_low, priority=10),
            "high": StreamBufferHook(callable_=cb_high, priority=100),
            "mid": StreamBufferHook(callable_=cb_mid, priority=50),
        }
        await _sbm_with_stream.write_buffer("stream:t", "x")
        assert order == ["high", "mid", "low"]

    async def test_hook_exception_does_not_propagate(self, _sbm_with_stream):
        async def bad_hook(sb, text, metadata):
            raise RuntimeError("boom")
        _sbm_with_stream.stream_buffer_configs["stream:t"].hooks["boom"] = StreamBufferHook(
            callable_=bad_hook, priority=0
        )
        caught = False
        try:
            await _sbm_with_stream.write_buffer("stream:t", "x")
        except RuntimeError:
            caught = True
        assert not caught

    async def test_hook_error_recorded(self, _sbm_with_stream):
        async def bad_hook(sb, text, metadata):
            raise ValueError("boom")
        _sbm_with_stream.stream_buffer_configs["stream:t"].hooks["boom"] = StreamBufferHook(
            callable_=bad_hook, priority=0
        )
        await _sbm_with_stream.write_buffer("stream:t", "x")
        errors = _sbm_with_stream.stream_buffer_configs["stream:t"].hooks["boom"].errors
        assert len(errors) == 1
        assert "boom" in errors[0].error


class TestListStreamBuffers:
    async def test_lists_all_streams(self, sbm):
        await sbm.create_buffer("stream:a", stream=True)
        await sbm.create_buffer("stream:b", stream=True)
        result = sbm.list_stream_buffers()
        names = [r["name"] for r in result]
        assert "stream:a" in names
        assert "stream:b" in names

    async def test_includes_hook_count(self, sbm):
        await sbm.create_buffer("stream:t", stream=True)
        async def dummy(entry, sb):
            pass
        sbm.stream_buffer_configs["stream:t"].hooks["dummy"] = StreamBufferHook(
            callable_=dummy, priority=0
        )
        result = sbm.list_stream_buffers()
        entry = next(r for r in result if r["name"] == "stream:t")
        assert entry["hook_count"] == 1

    async def test_includes_created_at(self, sbm):
        before = time.time()
        await sbm.create_buffer("stream:t", stream=True)
        after = time.time()
        result = sbm.list_stream_buffers()
        entry = next(r for r in result if r["name"] == "stream:t")
        assert entry["created_at"] >= before
        assert entry["created_at"] <= after


class TestReadBufferReturnValues:
    @pytest.fixture
    async def _sbm_with_stream(self, sbm):
        ts = [10.0, 20.0, 30.0]
        data = ["a", "b", "c"]
        entries = [BufferEntry(data=d, modified_at=t) for t, d in zip(ts, data)]
        sbm._buffers["stream:t"] = Buffer(lines=entries, created_at=ts[0], modified_at=ts[-1])
        sbm.stream_buffer_configs["stream:t"] = type("C", (), {"hooks": {}, "created_at": ts[0]})()
        return sbm

    async def test_content_return_has_ok_and_content_and_line_range(self, _sbm_with_stream):
        result = await _sbm_with_stream.read_buffer("stream:t", start=0.0, end=30.0)
        assert result["ok"]
        assert "content" in result
        assert "start" in result
        assert "end" in result

    async def test_empty_range_outside_data_is_error(self, _sbm_with_stream):
        result = await _sbm_with_stream.read_buffer("stream:t", start=5.0, end=5.0)
        assert not result["ok"]


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(standalone_mode=False, exit=False))
