"""Unit tests for StreamBufferManager, especially time-based reading and hook behavior."""

from __future__ import annotations

import asyncio
import bisect
import sys
import time
import unittest

sys.path.insert(0, "/home/frygge/projects/AIOS/peteos-kit")
sys.path.insert(0, "/home/frygge/projects/private/petekit/src/peteos/peteos")

from petekit.stream.stream_buffer_manager import StreamBufferManager, StreamBufferHook, StreamBufferHookError
from petekit.text.buffer_manager import Buffer, BufferEntry


class TestStreamBufferCreateDrop(unittest.TestCase):
    """Test create_buffer with stream=True and drop_buffer cleanup."""

    def setUp(self):
        self.sbm = StreamBufferManager()

    def test_create_stream_requires_stream_prefix(self):
        result = self.sbm.create_buffer("bad_name", stream=True)
        self.assertFalse(result["ok"])
        self.assertIn("stream:", result["error"])

    def test_create_stream_ok(self):
        result = self.sbm.create_buffer("stream:test", stream=True)
        self.assertTrue(result["ok"])
        self.assertIn("stream:test", self.sbm.stream_buffer_configs)

    def test_create_stream_twice_blocked(self):
        self.sbm.create_buffer("stream:test", stream=True)
        result = self.sbm.create_buffer("stream:test", stream=True)
        self.assertFalse(result["ok"])

    def test_create_stream_overwrite(self):
        self.sbm.create_buffer("stream:test", stream=True)
        result = self.sbm.create_buffer("stream:test", stream=True, overwrite=True)
        self.assertTrue(result["ok"])

    def test_drop_stream_cleans_up_config(self):
        self.sbm.create_buffer("stream:test", stream=True)
        self.sbm.drop_buffer("stream:test")
        self.assertNotIn("stream:test", self.sbm.stream_buffer_configs)

    def test_drop_non_stream_still_works(self):
        self.sbm.create_buffer("regular")
        result = self.sbm.drop_buffer("regular")
        self.assertTrue(result["ok"])


class TestResolveTime(unittest.TestCase):
    """Test _resolve_time for absolute vs relative timestamp resolution."""

    def setUp(self):
        self.sbm = StreamBufferManager()

    def test_positive_passes_through(self):
        result = self.sbm._resolve_time(123.456, anchor=200.0)
        self.assertEqual(result, 123.456)

    def test_negative_adds_to_anchor(self):
        result = self.sbm._resolve_time(-30.0, anchor=100.0)
        self.assertEqual(result, 70.0)

    def test_zero_is_absolute_not_relative(self):
        result = self.sbm._resolve_time(-0.0, anchor=100.0)
        self.assertEqual(result, -0.0)


class TestStreamBufferReadBufferTimeBased(unittest.TestCase):
    """Test read_buffer with float start/end for time-based reading."""

    def setUp(self):
        self.sbm = StreamBufferManager()
        self._make_stream("stream:t", ts=[10.0, 20.0, 30.0, 40.0, 50.0], data=["a", "b", "c", "d", "e"])

    def _make_stream(self, name: str, ts: list[float], data: list[str]) -> None:
        entries = [BufferEntry(data=d, timestamp=t, seen=False) for t, d in zip(ts, data)]
        self.sbm._buffers[name] = Buffer(lines=entries, created_at=ts[0], modified_at=ts[-1])
        self.sbm.stream_buffer_configs[name] = self.sbm.stream_buffer_configs.get(
            name, type("C", (), {"hooks": [], "created_at": ts[0]})()
        )

    def test_absolute_range_inclusive(self):
        result = self.sbm.read_buffer("stream:t", start=20.0, end=40.0)
        self.assertTrue(result["ok"])
        lines = result["content"].split("\n")
        timestamps_in_content = [float(l.split(")")[0][1:]) for l in lines]
        self.assertEqual(timestamps_in_content, [20.0, 30.0, 40.0])

    def test_float_start_only(self):
        result = self.sbm.read_buffer("stream:t", start=35.0)
        self.assertTrue(result["ok"])
        lines = result["content"].split("\n")
        self.assertEqual(len(lines), 2)
        timestamps = [float(l.split(")")[0][1:]) for l in lines]
        self.assertEqual(timestamps, [40.0, 50.0])

    def test_float_end_only(self):
        result = self.sbm.read_buffer("stream:t", end=25.0)
        self.assertTrue(result["ok"])
        lines = result["content"].split("\n")
        timestamps = [float(l.split(")")[0][1:]) for l in lines]
        self.assertEqual(timestamps, [10.0, 20.0])

    def test_start_beyond_all_entries(self):
        result = self.sbm.read_buffer("stream:t", start=100.0)
        self.assertFalse(result["ok"])
        self.assertIn("No entries at or after", result["error"])

    def test_end_before_all_entries(self):
        result = self.sbm.read_buffer("stream:t", end=5.0)
        self.assertFalse(result["ok"])
        self.assertIn("No entries at or before", result["error"])

    def test_empty_buffer_error(self):
        self.sbm._buffers["stream:empty"] = Buffer(lines=[], created_at=0.0, modified_at=0.0)
        self.sbm.stream_buffer_configs["stream:empty"] = type("C", (), {"hooks": [], "created_at": 0.0})()
        result = self.sbm.read_buffer("stream:empty", start=0.0)
        self.assertFalse(result["ok"])
        self.assertIn("empty", result["error"])

    def test_non_stream_buffer_rejects_float(self):
        self.sbm.create_buffer("regular", text="x")
        result = self.sbm.read_buffer("regular", start=10.0)
        self.assertFalse(result["ok"])
        self.assertIn("Only stream buffers support", result["error"])


class TestStreamBufferReadBufferRelativeTime(unittest.TestCase):
    """Test read_buffer with negative (relative) float timestamps."""

    def setUp(self):
        self.sbm = StreamBufferManager()
        self._make_stream("stream:t", ts=[10.0, 20.0, 30.0, 40.0, 50.0], data=["a", "b", "c", "d", "e"])

    def _make_stream(self, name: str, ts: list[float], data: list[str]) -> None:
        entries = [BufferEntry(data=d, timestamp=t, seen=False) for t, d in zip(ts, data)]
        self.sbm._buffers[name] = Buffer(lines=entries, created_at=ts[0], modified_at=ts[-1])
        self.sbm.stream_buffer_configs[name] = type("C", (), {"hooks": [], "created_at": ts[0]})()

    def test_negative_start_relative_to_last_entry(self):
        result = self.sbm.read_buffer("stream:t", start=-20.0)
        self.assertTrue(result["ok"])
        lines = result["content"].split("\n")
        timestamps = [float(l.split(")")[0][1:]) for l in lines]
        self.assertEqual(timestamps, [30.0, 40.0, 50.0])

    def test_negative_end_relative_to_last_entry(self):
        result = self.sbm.read_buffer("stream:t", end=-20.0)
        self.assertTrue(result["ok"])
        lines = result["content"].split("\n")
        timestamps = [float(l.split(")")[0][1:]) for l in lines]
        self.assertEqual(timestamps, [10.0, 20.0, 30.0])

    def test_both_negative_relative_range(self):
        result = self.sbm.read_buffer("stream:t", start=-30.0, end=-10.0)
        self.assertTrue(result["ok"])
        lines = result["content"].split("\n")
        timestamps = [float(l.split(")")[0][1:]) for l in lines]
        self.assertEqual(timestamps, [20.0, 30.0, 40.0])

    def test_negative_zero_means_from_start(self):
        result = self.sbm.read_buffer("stream:t", start=-0.0)
        self.assertTrue(result["ok"])
        lines = result["content"].split("\n")
        self.assertEqual(len(lines), 5)
        timestamps = [float(l.split(")")[0][1:]) for l in lines]
        self.assertEqual(timestamps, [10.0, 20.0, 30.0, 40.0, 50.0])

    def test_negative_start_large_offset_captures_all(self):
        result = self.sbm.read_buffer("stream:t", start=-200.0)
        self.assertTrue(result["ok"])
        lines = result["content"].split("\n")
        self.assertEqual(len(lines), 5)

    def test_negative_end_exceeds_range(self):
        result = self.sbm.read_buffer("stream:t", end=-200.0)
        self.assertFalse(result["ok"])
        self.assertIn("No entries at or before", result["error"])


class TestStreamBufferReadBufferLineBased(unittest.TestCase):
    """Test that line-based read_buffer still works with 0-based indexing."""

    def setUp(self):
        self.sbm = StreamBufferManager()
        self._make_stream("stream:t", ts=[10.0, 20.0, 30.0, 40.0, 50.0], data=["a", "b", "c", "d", "e"])

    def _make_stream(self, name: str, ts: list[float], data: list[str]) -> None:
        entries = [BufferEntry(data=d, timestamp=t, seen=False) for t, d in zip(ts, data)]
        self.sbm._buffers[name] = Buffer(lines=entries, created_at=ts[0], modified_at=ts[-1])
        self.sbm.stream_buffer_configs[name] = type("C", (), {"hooks": [], "created_at": ts[0]})()

    def test_read_all_with_timestamps(self):
        result = self.sbm.read_buffer("stream:t", show_timestamps=True)
        self.assertTrue(result["ok"])
        lines = result["content"].split("\n")
        self.assertEqual(len(lines), 5)
        for line in lines:
            self.assertRegex(line, r"^\(\d+\.\d+\)")

    def test_read_range_0_based(self):
        result = self.sbm.read_buffer("stream:t", start=1, end=4, show_timestamps=True)
        self.assertTrue(result["ok"])
        lines = result["content"].split("\n")
        self.assertEqual(len(lines), 3)
        timestamps = [float(l.split(")")[0][1:]) for l in lines]
        self.assertEqual(timestamps, [20.0, 30.0, 40.0])

    def test_negative_index_from_end(self):
        result = self.sbm.read_buffer("stream:t", start=-2, end=None, show_timestamps=True)
        self.assertTrue(result["ok"])
        lines = result["content"].split("\n")
        self.assertEqual(len(lines), 2)
        self.assertNotEqual(lines[-1], "")
        timestamps = [float(l.split(")")[0][1:]) for l in lines if l]
        self.assertEqual(timestamps, [40.0, 50.0])


class TestStreamBufferHookBehavior(unittest.TestCase):
    """Test that _append_stream_entry fires hooks and handles errors."""

    def setUp(self):
        self.sbm = StreamBufferManager()
        self.fired = []
        self.sbm._buffers["stream:t"] = Buffer(lines=[], created_at=0.0, modified_at=0.0)
        self.sbm.stream_buffer_configs["stream:t"] = type("C", (), {"hooks": [], "created_at": 0.0})()

    async def _append_and_collect(self, data: str) -> None:
        await self.sbm._append_stream_entry("stream:t", data)
        self.fired.append(data)

    def test_hook_receives_entry_and_stream_name(self):
        async def run():
            hook_called = []
            async def capture(entry, sb):
                hook_called.append((entry.data, sb))
            self.sbm.stream_buffer_configs["stream:t"].hooks.append(
                StreamBufferHook(callable_=capture, priority=0)
            )
            await self.sbm._append_stream_entry("stream:t", "hello")
            return hook_called

        result = asyncio.run(run())
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], ("hello", "stream:t"))

    def test_hooks_fire_in_priority_order(self):
        order = []

        async def cb_low(entry, sb):
            order.append("low")
        async def cb_high(entry, sb):
            order.append("high")
        async def cb_mid(entry, sb):
            order.append("mid")

        self.sbm.stream_buffer_configs["stream:t"].hooks = [
            StreamBufferHook(callable_=cb_low, priority=10),
            StreamBufferHook(callable_=cb_high, priority=100),
            StreamBufferHook(callable_=cb_mid, priority=50),
        ]
        asyncio.run(self.sbm._append_stream_entry("stream:t", "x"))
        self.assertEqual(order, ["high", "mid", "low"])

    def test_hook_exception_does_not_propagate(self):
        async def run():
            async def bad_hook(entry, sb):
                raise RuntimeError("boom")
            self.sbm.stream_buffer_configs["stream:t"].hooks.append(
                StreamBufferHook(callable_=bad_hook, priority=0)
            )
            try:
                await self.sbm._append_stream_entry("stream:t", "x")
                return "no error"
            except RuntimeError as e:
                return str(e)

        result = asyncio.run(run())
        self.assertEqual(result, "no error")

    def test_hook_error_recorded(self):
        async def run():
            async def bad_hook(entry, sb):
                raise ValueError("boom")
            self.sbm.stream_buffer_configs["stream:t"].hooks.append(
                StreamBufferHook(callable_=bad_hook, priority=0)
            )
            await self.sbm._append_stream_entry("stream:t", "x")
            errors = self.sbm.stream_buffer_configs["stream:t"].hooks[0].errors
            return errors

        result = asyncio.run(run())
        self.assertEqual(len(result), 1)
        self.assertIn("boom", result[0].error)


class TestListStreamBuffers(unittest.TestCase):
    """Test list_stream_buffers returns structured data."""

    def setUp(self):
        self.sbm = StreamBufferManager()

    def test_lists_all_streams(self):
        self.sbm.create_buffer("stream:a", stream=True)
        self.sbm.create_buffer("stream:b", stream=True)
        result = self.sbm.list_stream_buffers()
        names = [r["name"] for r in result]
        self.assertIn("stream:a", names)
        self.assertIn("stream:b", names)

    def test_includes_hook_count(self):
        self.sbm.create_buffer("stream:t", stream=True)
        async def dummy(entry, sb):
            pass
        self.sbm.stream_buffer_configs["stream:t"].hooks.append(
            StreamBufferHook(callable_=dummy, priority=0)
        )
        result = self.sbm.list_stream_buffers()
        entry = next(r for r in result if r["name"] == "stream:t")
        self.assertEqual(entry["hook_count"], 1)

    def test_includes_created_at(self):
        before = time.time()
        self.sbm.create_buffer("stream:t", stream=True)
        after = time.time()
        result = self.sbm.list_stream_buffers()
        entry = next(r for r in result if r["name"] == "stream:t")
        self.assertGreaterEqual(entry["created_at"], before)
        self.assertLessEqual(entry["created_at"], after)


class TestReadBufferReturnValues(unittest.TestCase):
    """Test read_buffer returns consistent dict values."""

    def setUp(self):
        self.sbm = StreamBufferManager()
        self._make_stream("stream:t", ts=[10.0, 20.0, 30.0], data=["a", "b", "c"])

    def _make_stream(self, name: str, ts: list[float], data: list[str]) -> None:
        entries = [BufferEntry(data=d, timestamp=t, seen=False) for t, d in zip(ts, data)]
        self.sbm._buffers[name] = Buffer(lines=entries, created_at=ts[0], modified_at=ts[-1])
        self.sbm.stream_buffer_configs[name] = type("C", (), {"hooks": [], "created_at": ts[0]})()

    def test_content_return_has_ok_and_content_and_line_range(self):
        result = self.sbm.read_buffer("stream:t", start=0.0, end=30.0)
        self.assertTrue(result["ok"])
        self.assertIn("content", result)
        self.assertIn("line_range", result)
        self.assertIn("lines", result)

    def test_empty_range_outside_data_is_error(self):
        result = self.sbm.read_buffer("stream:t", start=5.0, end=5.0)
        self.assertFalse(result["ok"])


if __name__ == "__main__":
    unittest.main()
