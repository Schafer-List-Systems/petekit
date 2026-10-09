"""Integration tests for the JSON codec make_json_codec."""

from __future__ import annotations

import pytest

from petekit import BufferManager
from petekit.utils.text_codecs import make_json_codec


@pytest.fixture
async def bm():
    return BufferManager()


class TestMakeJsonCodec:
    async def test_reject_blocks_write(self, bm):
        await bm.create_buffer("test", text='{"a": 1}')
        bm.register_buffer_update_hook(
            "test", "jc",
            make_json_codec(lambda *_: "inner rejected"),
        )
        result = await bm.write_buffer("test", "not json", pos="end")
        assert not result["ok"]
        content = await bm.read_buffer("test", raw=True, show_line_numbers=False)
        assert content == '{"a": 1}'

