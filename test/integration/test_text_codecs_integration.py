"""Integration tests for the JSON codec make_json_codec."""

from __future__ import annotations

import pytest

from petekit import BufferManager
from petekit.utils.text_codecs import make_json_codec


@pytest.fixture
async def bm():
    return BufferManager()


class TestMakeJsonCodec:
    @pytest.fixture
    async def calls(self):
        return []

    async def test_valid_new_json_passes_to_inner(self, bm, calls):
        def _inner_accept(buf, old_json, new_json, patch):
            calls.append((buf, old_json, new_json, patch))
            return True
        await bm.create_buffer("test", text='{"a": 1}')
        bm.register_buffer_update_hook(
            "test", "jc",
            make_json_codec(_inner_accept),
        )
        result = await bm.write_buffer("test", '{"b": 2}', start=0)
        assert result["ok"]
        assert len(calls) == 1
        buf, old_json, new_json, patch = calls[0]
        assert old_json == {"a": 1}
        assert new_json == {"b": 2}

    async def test_invalid_new_json_rejected(self, bm):
        def _inner_accept(buf, old_json, new_json, patch):
            return True
        await bm.create_buffer("test", text='{"a": 1}')
        bm.register_buffer_update_hook(
            "test", "jc",
            make_json_codec(_inner_accept),
        )
        result = await bm.write_buffer("test", "not json", start=0)
        assert not result["ok"]
        assert "not valid JSON" in result["error"]

    async def test_invalid_old_json_rejected(self, bm):
        def _inner_accept(buf, old_json, new_json, patch):
            return True
        await bm.create_buffer("test", text="not json either")
        bm.register_buffer_update_hook(
            "test", "jc",
            make_json_codec(_inner_accept),
        )
        result = await bm.write_buffer("test", '{"b": 2}', start=0)
        assert not result["ok"]
        assert "not valid JSON" in result["error"]

    async def test_inner_reject_propagates(self, bm):
        def _inner_reject(buf, old_json, new_json, patch):
            return "inner rejected"
        await bm.create_buffer("test", text='{"a": 1}')
        bm.register_buffer_update_hook(
            "test", "jc",
            make_json_codec(_inner_reject),
        )
        result = await bm.write_buffer("test", '{"a": 2}', start=0)
        assert not result["ok"]
        assert result["error"] == "inner rejected"

    async def test_inner_accept_propagates(self, bm):
        def _inner_reject(buf, old_json, new_json, patch):
            return "inner rejected"
        def _inner_accept(buf, old_json, new_json, patch):
            return True
        await bm.create_buffer("test", text='{"a": 1}')
        bm.register_buffer_update_hook(
            "test", "jc",
            make_json_codec(_inner_reject),
        )
        bm.unregister_buffer_update_hook("test", "jc")
        bm.register_buffer_update_hook(
            "test", "jc",
            make_json_codec(_inner_accept),
        )
        result = await bm.write_buffer("test", '{"a": 2}', start=0)
        assert result["ok"]

    async def test_patch_contains_json_patch_operations(self, bm):
        calls = []
        def _inner_accept(buf, old_json, new_json, patch):
            calls.append((buf, old_json, new_json, patch))
            return True
        await bm.create_buffer("test", text='{"a": 1, "b": 2}')
        bm.register_buffer_update_hook(
            "test", "jc",
            make_json_codec(_inner_accept),
        )
        await bm.write_buffer("test", '{"a": 1, "c": 3}', start=0)
        _, _, _, patch = calls[0]
        ops = [p["op"] for p in patch]
        assert "remove" in ops
        assert "add" in ops

    async def test_patch_is_empty_when_identical(self, bm):
        calls = []
        def _inner_accept(buf, old_json, new_json, patch):
            calls.append((buf, old_json, new_json, patch))
            return True
        await bm.create_buffer("test", text='{"a": 1}')
        bm.register_buffer_update_hook(
            "test", "jc",
            make_json_codec(_inner_accept),
        )
        await bm.write_buffer("test", '{"a": 1}', start=0)
        _, _, _, patch = calls[0]
        assert not list(patch)

    async def test_reject_blocks_write(self, bm):
        def _inner_reject(buf, old_json, new_json, patch):
            return "inner rejected"
        await bm.create_buffer("test", text='{"a": 1}')
        bm.register_buffer_update_hook(
            "test", "jc",
            make_json_codec(_inner_reject),
        )
        result = await bm.write_buffer("test", '{"a": 2}', start=0)
        assert not result["ok"]
        content = await bm.read_buffer("test", raw=True, show_line_numbers=False)
        assert content == '{"a": 1}'
