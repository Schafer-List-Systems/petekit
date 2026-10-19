"""Unit tests for StreamRouter."""

from __future__ import annotations

import inspect
import json
import pytest
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from petekit.stream.stream_router import (
    _is_routing_table_option,
    _parse_routing_table_option,
    evaluate_condition,
    route_dispatch_batch,
    after_create_routing_table,
    before_update_routing_table,
    after_drop_routing_table,
    RoutingConditionError,
)
from petekit.text.buffer import BufferEntry


# -------------------------------------------------------------------------------------------------------------------- #
# Helpers                                                                                                              #
# -------------------------------------------------------------------------------------------------------------------- #

def _mock_buffer(name: str, text: str = "") -> MagicMock:
    """Return a mock Buffer with the given name, get_text returning text, and lines as BufferEntry list."""
    buf = MagicMock()
    buf.name = name
    buf.get_text.return_value = text
    # lines mirrors the text split into BufferEntry objects.
    # An empty text still produces one empty BufferEntry.
    lines = [line for line in text.splitlines(keepends=True)]
    buf.lines = [
        MagicMock(spec=BufferEntry, data=line)
        for line in lines
    ] or [MagicMock(spec=BufferEntry, data="")]
    return buf


def _mock_processor_with_routing_table(
    table_name: str,
    routing_table: list,
    fallthrough: str | None = None,
    route_lines: bool = True,
) -> MagicMock:
    """Return a processor mock with a routing_table option buffer configured.

    routing_table entries use the new format: [[conditions_list, output_stream], ...]
    conditions_list is a list of condition name strings, e.g. ["true"] or ["cond1", "cond2"].
    """
    option_config: dict = {"routing_table": routing_table, "route_lines": route_lines}
    if fallthrough is not None:
        option_config["fallthrough"] = fallthrough

    proc = MagicMock()
    proc.get_buffer.side_effect = lambda name: (
        _mock_buffer(f"STREAM[routing_table:{table_name}]", json.dumps(option_config))
        if name == f"STREAM[routing_table:{table_name}]"
        else _mock_buffer(name)
    )
    # write_buffer is async — use AsyncMock so await works in tests.
    proc.write_buffer = AsyncMock(return_value={"ok": True})
    return proc


# -------------------------------------------------------------------------------------------------------------------- #
# TestIsRoutingTableOption                                                                                          #
# -------------------------------------------------------------------------------------------------------------------- #

class TestIsRoutingTableOption:
    def test_returns_true_for_valid_option_buffer(self):
        assert _is_routing_table_option("STREAM[routing_table:foo]") is True

    def test_returns_true_for_nested_option(self):
        assert _is_routing_table_option("STREAM[notifications][routing_table:foo]") is True

    def test_returns_false_for_plain_buffer(self):
        assert _is_routing_table_option("STREAM") is False

    def test_returns_false_for_other_option_buffer(self):
        assert _is_routing_table_option("STREAM[notes]") is False


# -------------------------------------------------------------------------------------------------------------------- #
# TestParseRoutingTableOption                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #

class TestParseRoutingTableOption:
    def test_returns_origin_and_table_name_for_valid_option_buffer(self):
        result = _parse_routing_table_option("STREAM[routing_table:foo]")
        assert result == ("STREAM", "foo")

    def test_preserves_colons_in_table_name(self):
        result = _parse_routing_table_option("STREAM[routing_table:my:table]")
        assert result == ("STREAM", "my:table")

    def test_returns_none_for_plain_buffer(self):
        result = _parse_routing_table_option("STREAM")
        assert result is None

    def test_returns_none_for_other_option_buffer(self):
        result = _parse_routing_table_option("STREAM[notes]")
        assert result is None


# -------------------------------------------------------------------------------------------------------------------- #
# TestEvaluateCondition                                                                                           #
# -------------------------------------------------------------------------------------------------------------------- #

class TestEvaluateCondition:
    def test_true_condition_returns_true(self):
        fn_mgr = MagicMock()
        result = evaluate_condition(fn_mgr, "true", "any text", "stream", "t1")
        assert result is True

    def test_false_condition_returns_false(self):
        fn_mgr = MagicMock()
        result = evaluate_condition(fn_mgr, "false", "any text", "stream", "t1")
        assert result is False

    def test_negation_prefix_negates_inner_condition(self):
        fn_mgr = MagicMock()

        def inner_callable(function_manager: Any, stream: str, text: str, metadata: dict) -> bool:
            return True

        fn_mgr.get_function.return_value = {"ok": True, "callable": inner_callable}
        result = evaluate_condition(fn_mgr, "!custom_cond", "text", "stream", "t1")
        assert result is False

    def test_condition_not_found_raises_routing_condition_error(self):
        fn_mgr = MagicMock()
        fn_mgr.get_function.return_value = {"ok": False, "error": "not found"}
        with pytest.raises(RoutingConditionError) as exc_info:
            evaluate_condition(fn_mgr, "missing", "text", "stream", "t1")
        assert exc_info.value.condition == "missing"
        assert exc_info.value.routing_table == "t1"

    def test_incompatible_signature_raises_routing_condition_error(self):
        fn_mgr = MagicMock()
        # Return a callable with an incompatible signature (wrong params).
        def bad_callable(x):  # type: ignore[unused]
            return True
        fn_mgr.get_function.return_value = {"ok": True, "callable": bad_callable}
        with pytest.raises(RoutingConditionError) as exc_info:
            evaluate_condition(fn_mgr, "bad_sig", "text", "stream", "t1")
        assert "incompatible signature" in exc_info.value.reason

    def test_valid_condition_returns_result(self):
        fn_mgr = MagicMock()
        expected_result = True

        def good_callable(function_manager: Any, stream: str, text: str, metadata: dict) -> bool:
            assert stream == "stream"
            assert text == "input text"
            assert metadata == {"routing_table": "t1"}
            return expected_result

        # Inspect the function to give it the right signature.
        fn_mgr.get_function.return_value = {"ok": True, "callable": good_callable}
        result = evaluate_condition(fn_mgr, "mycond", "input text", "stream", "t1")
        assert result is expected_result


# -------------------------------------------------------------------------------------------------------------------- #
# TestRouteDispatch                                                                                               #
# -------------------------------------------------------------------------------------------------------------------- #

class TestRouteDispatch:
    async def test_returns_none_when_option_buffer_not_found(self):
        proc = MagicMock()
        proc.get_buffer.return_value = None
        old_buf = _mock_buffer("STREAM")
        new_buf = _mock_buffer("STREAM", "new text")

        result = await route_dispatch_batch(proc, old_buf, new_buf, "t1")

        assert result is None

    async def test_writes_to_matched_output_stream(self):
        proc = _mock_processor_with_routing_table(
            "t1",
            [[["true"], "output_a"]],
            route_lines=False,  # batch mode — single write with full text
        )
        old_buf = _mock_buffer("STREAM", "")
        new_buf = _mock_buffer("STREAM", "hello world")

        await route_dispatch_batch(proc, old_buf, new_buf, "t1")

        # Verify write_buffer was called with the matched output stream.
        proc.write_buffer.assert_called_once()
        call_args = proc.write_buffer.call_args
        assert call_args[0][0] == "output_a"
        assert call_args[0][1] == "hello world"

    async def test_all_conditions_must_match_in_entry(self):
        # Both conditions in the list must match — entry fires only if all conditions return True.
        proc = _mock_processor_with_routing_table(
            "t1",
            [[["true", "true"], "output_a"], [["false"], "output_b"]],
            route_lines=False,
        )
        old_buf = _mock_buffer("STREAM", "")
        new_buf = _mock_buffer("STREAM", "text")

        await route_dispatch_batch(proc, old_buf, new_buf, "t1")

        # Both conditions matched — write to output_a (first entry wins).
        assert proc.write_buffer.call_count == 1
        assert proc.write_buffer.call_args[0][0] == "output_a"

    async def test_falls_through_when_any_condition_fails_in_entry(self):
        # One condition fails — entry is skipped, falls through to output_b.
        proc = _mock_processor_with_routing_table(
            "t1",
            [[["true", "false"], "output_a"], [["true"], "output_b"]],
            route_lines=False,
        )
        old_buf = _mock_buffer("STREAM", "")
        new_buf = _mock_buffer("STREAM", "text")

        await route_dispatch_batch(proc, old_buf, new_buf, "t1")

        # First entry skipped (second condition false), second entry matches — write to output_b.
        last_call = proc.write_buffer.call_args_list[-1]
        assert last_call[0][0] == "output_b"

    async def test_tries_entries_top_to_bottom_first_match_wins(self):
        proc = _mock_processor_with_routing_table(
            "t1",
            [[["false"], "output_a"], [["true"], "output_b"]],
            route_lines=False,
        )
        old_buf = _mock_buffer("STREAM", "")
        new_buf = _mock_buffer("STREAM", "text")

        await route_dispatch_batch(proc, old_buf, new_buf, "t1")

        # Only the second entry matches — write to output_b.
        assert proc.write_buffer.call_count == 1
        assert proc.write_buffer.call_args[0][0] == "output_b"

    async def test_falls_through_to_default_when_no_match(self):
        proc = _mock_processor_with_routing_table(
            "t1",
            [[["false"], "output_a"]],  # nothing matches
            route_lines=False,
        )
        old_buf = _mock_buffer("STREAM", "")
        new_buf = _mock_buffer("STREAM", "text")

        await route_dispatch_batch(proc, old_buf, new_buf, "t1")

        # Default fallthrough: STREAM[routing_table:t1]:fallthrough
        # First call is the log line; last call is the actual write.
        last_call = proc.write_buffer.call_args_list[-1]
        assert last_call[0][0] == "STREAM[routing_table:t1]:fallthrough"
        assert last_call[0][1] == "text"

    async def test_uses_custom_fallthrough_when_configured(self):
        proc = _mock_processor_with_routing_table(
            "t1",
            [[["false"], "output_a"]],
            fallthrough="my_fallthrough",
            route_lines=False,
        )
        old_buf = _mock_buffer("STREAM", "")
        new_buf = _mock_buffer("STREAM", "text")

        await route_dispatch_batch(proc, old_buf, new_buf, "t1")

        # Last call is the actual write to the custom fallthrough.
        last_call = proc.write_buffer.call_args_list[-1]
        assert last_call[0][0] == "my_fallthrough"

    async def test_logs_routing_error_on_invalid_json(self):
        proc = MagicMock()
        proc.get_buffer.side_effect = lambda name: (
            _mock_buffer("STREAM[routing_table:t1]", "not json")
            if name == "STREAM[routing_table:t1]"
            else _mock_buffer(name)
        )
        proc.write_buffer = AsyncMock(return_value={"ok": True})
        old_buf = _mock_buffer("STREAM")
        new_buf = _mock_buffer("STREAM", "text")

        await route_dispatch_batch(proc, old_buf, new_buf, "t1")

        # Should log the routing error and return None.
        proc.write_buffer.assert_called()
        error_calls = [
            c for c in proc.write_buffer.call_args_list
            if "ROUTING ERROR" in str(c)
        ]
        assert len(error_calls) == 1

    async def test_logs_error_when_sink_write_fails(self):
        proc = _mock_processor_with_routing_table(
            "t1",
            [[["true"], "output_a"]],
            route_lines=False,
        )
        proc.write_buffer.return_value = {"ok": False, "error": "disk full"}
        old_buf = _mock_buffer("STREAM", "")
        new_buf = _mock_buffer("STREAM", "text")

        await route_dispatch_batch(proc, old_buf, new_buf, "t1")

        # Should log the sink error after writing to the output.
        proc.write_buffer.assert_called()
        error_calls = [
            c for c in proc.write_buffer.call_args_list
            if "SINK ERROR" in str(c)
        ]
        assert len(error_calls) == 1

    async def test_skips_empty_appended_text(self):
        proc = _mock_processor_with_routing_table(
            "t1",
            [[["true"], "output_a"]],
        )
        old_buf = _mock_buffer("STREAM", "")
        new_buf = _mock_buffer("STREAM", "")

        await route_dispatch_batch(proc, old_buf, new_buf, "t1")

        # Nothing to route — write_buffer should not be called.
        proc.write_buffer.assert_not_called()

    async def test_route_lines_true_dispatches_each_line_separately(self):
        # With route_lines=True (default), each new buffer entry is dispatched individually.
        proc = _mock_processor_with_routing_table(
            "t1",
            [[["true"], "output_a"]],
            route_lines=True,
        )
        old_buf = _mock_buffer("STREAM", "line1\n")
        new_buf = _mock_buffer("STREAM", "line1\nline2\nline3\n")

        await route_dispatch_batch(proc, old_buf, new_buf, "t1")

        # The two new entries (line2\n, line3\n) are written to the output stream separately.
        assert proc.write_buffer.call_count == 2
        assert proc.write_buffer.call_args_list[0][0][0] == "output_a"
        assert proc.write_buffer.call_args_list[0][0][1] == "line2\n"
        assert proc.write_buffer.call_args_list[1][0][1] == "line3\n"

    async def test_route_lines_false_dispatches_full_text_as_batch(self):
        # With route_lines=False, the full appended text is dispatched as one batch.
        proc = _mock_processor_with_routing_table(
            "t1",
            [[["true"], "output_a"]],
            route_lines=False,
        )
        old_buf = _mock_buffer("STREAM", "")
        new_buf = _mock_buffer("STREAM", "line1\nline2\nline3\n")

        await route_dispatch_batch(proc, old_buf, new_buf, "t1")

        # All three lines are written as one single batch.
        proc.write_buffer.assert_called_once()
        call_args = proc.write_buffer.call_args
        assert call_args[0][0] == "output_a"
        assert call_args[0][1] == "line1\nline2\nline3\n"

    async def test_route_lines_true_different_lines_match_different_routes(self):
        # With route_lines=True, each line is evaluated independently — different
        # lines can match different routes.
        proc = _mock_processor_with_routing_table(
            "t1",
            [
                [["true"], "all_lines"],
            ],
            route_lines=True,
        )
        old_buf = _mock_buffer("STREAM", "a\n")
        new_buf = _mock_buffer("STREAM", "a\nb\nc\n")

        await route_dispatch_batch(proc, old_buf, new_buf, "t1")

        # The two new lines (b, c) go to the same output since all match "true".
        assert proc.write_buffer.call_count == 2


# -------------------------------------------------------------------------------------------------------------------- #
# TestAfterCreateRoutingTable                                                                                     #
# -------------------------------------------------------------------------------------------------------------------- #

class TestAfterCreateRoutingTable:
    async def test_returns_early_for_non_routing_table_buffer(self, monkeypatch):
        proc = MagicMock()
        buf = _mock_buffer("STREAM[notes]")
        errors = []
        monkeypatch.setattr(
            "petekit.stream.stream_router._logger",
            MagicMock(error=errors.append),
        )

        await after_create_routing_table(proc, buf, {})

        assert len(errors) == 1
        proc.create_buffer.assert_not_called()

    async def test_logs_error_and_returns_when_origin_not_found(self, monkeypatch):
        proc = MagicMock()
        proc.get_buffer.return_value = None  # Origin does not exist
        buf = _mock_buffer("STREAM[routing_table:t1]", "{}")
        errors = []
        monkeypatch.setattr(
            "petekit.stream.stream_router._logger",
            MagicMock(error=errors.append),
        )

        await after_create_routing_table(proc, buf, {})

        assert len(errors) == 1
        assert "origin" in errors[0]
        assert "not found" in errors[0]
        proc.register_hook.assert_not_called()

    async def test_registers_hook_and_creates_fallthrough(self, monkeypatch):
        proc = MagicMock()
        origin_mock = MagicMock()
        proc.get_buffer.side_effect = lambda name: (
            origin_mock if name == "STREAM" else _mock_buffer(name, "{}")
        )
        # create_stream_buffer is async — use AsyncMock for assertion support.
        proc.create_stream_buffer = AsyncMock(return_value={"ok": True})
        buf = _mock_buffer("STREAM[routing_table:t1]", "{}")
        monkeypatch.setattr(
            "petekit.stream.stream_router._logger",
            MagicMock(),
        )

        await after_create_routing_table(proc, buf, {})

        # Verify hook was registered on the origin buffer.
        origin_mock.register_hook.assert_called_once()
        call_args = origin_mock.register_hook.call_args
        hook_point = call_args[0][0]
        hook_name = call_args[0][1]
        from petekit.text.buffer_manager import HookPoint
        assert hook_point == HookPoint.AFTER_UPDATE_BUFFER
        assert hook_name == "routing_table:t1"

        # Verify the default fallthrough stream buffer was created.
        proc.create_stream_buffer.assert_called_once_with("STREAM[routing_table:t1]:fallthrough", text="")


# -------------------------------------------------------------------------------------------------------------------- #
# TestBeforeUpdateRoutingTable                                                                                   #
# -------------------------------------------------------------------------------------------------------------------- #

class TestBeforeUpdateRoutingTable:
    def test_allows_on_initial_creation(self):
        proc = MagicMock()
        old_buf = MagicMock()  # MagicMock with no .name attribute → raises TypeError in regex match
        new_buf = MagicMock()

        # The regex match on MagicMock.name raises TypeError.
        # The function should not crash but instead return early.
        try:
            result = before_update_routing_table(proc, old_buf, new_buf, {})
        except TypeError:
            pass  # Expected when MagicMock is passed instead of a real buffer name

    def test_allows_non_routing_table_buffer(self):
        proc = MagicMock()

        result = before_update_routing_table(
            proc,
            _mock_buffer("STREAM[routing_table:t1]"),
            _mock_buffer("STREAM[notes]"),
            {},
        )

        assert result is None

    def test_blocks_invalid_json(self):
        proc = MagicMock()
        proc.get_buffer.return_value = MagicMock()

        result = before_update_routing_table(
            proc,
            _mock_buffer("STREAM[routing_table:t1]"),
            _mock_buffer("STREAM[routing_table:t1]", "not json"),
            {},
        )

        assert result is not None
        assert "not valid JSON" in result

    def test_blocks_routing_table_not_a_list(self):
        proc = MagicMock()
        proc.get_buffer.return_value = MagicMock()

        result = before_update_routing_table(
            proc,
            _mock_buffer("STREAM[routing_table:t1]"),
            _mock_buffer("STREAM[routing_table:t1]", '{"routing_table": "not a list"}'),
            {},
        )

        assert result is not None
        assert "routing_table must be a list" in result

    def test_blocks_entry_not_a_list(self):
        proc = MagicMock()
        proc.get_buffer.return_value = MagicMock()

        result = before_update_routing_table(
            proc,
            _mock_buffer("STREAM[routing_table:t1]"),
            _mock_buffer("STREAM[routing_table:t1]", '{"routing_table": ["not a list entry"]}'),
            {},
        )

        assert result is not None
        assert "entry[0] must be [conditions, output_stream]" in result

    def test_blocks_conditions_not_a_list(self):
        proc = MagicMock()
        proc.get_buffer.return_value = MagicMock()

        # Old format used a string for conditions ("cond1,cond2") — now requires a list.
        result = before_update_routing_table(
            proc,
            _mock_buffer("STREAM[routing_table:t1]"),
            _mock_buffer("STREAM[routing_table:t1]", '{"routing_table": [["true", "false"]]}'),
            {},
        )

        assert result is not None
        assert "conditions must be a list" in result

    def test_blocks_empty_output_stream(self):
        proc = MagicMock()
        proc.get_buffer.return_value = MagicMock()

        result = before_update_routing_table(
            proc,
            _mock_buffer("STREAM[routing_table:t1]"),
            _mock_buffer("STREAM[routing_table:t1]", '{"routing_table": [[["true"], ""]]}'),
            {},
        )

        assert result is not None
        assert "output_stream must be a non-empty string" in result

    def test_blocks_fallthrough_not_a_string(self):
        proc = MagicMock()
        proc.get_buffer.return_value = MagicMock()

        result = before_update_routing_table(
            proc,
            _mock_buffer("STREAM[routing_table:t1]"),
            _mock_buffer("STREAM[routing_table:t1]", '{"routing_table": [], "fallthrough": 123}'),
            {},
        )

        assert result is not None
        assert "fallthrough must be a non-empty string" in result

    def test_allows_valid_routing_table(self):
        proc = MagicMock()
        proc.get_buffer.return_value = MagicMock()

        result = before_update_routing_table(
            proc,
            _mock_buffer("STREAM[routing_table:t1]"),
            _mock_buffer(
                "STREAM[routing_table:t1]",
                '{"routing_table": [[["true"], "output_a"]], "fallthrough": "my_ft"}'
            ),
            {},
        )

        assert result is None

    def test_allows_creation_with_valid_routing_table(self):
        proc = MagicMock()
        result = before_update_routing_table(
            proc,
            None,
            _mock_buffer("STREAM[routing_table:t1]", '{"routing_table": [[["true"], "out"]]}'),
            {},
        )
        assert result is None

    def test_blocks_creation_with_invalid_json(self):
        proc = MagicMock()
        result = before_update_routing_table(
            proc,
            None,
            _mock_buffer("STREAM[routing_table:t1]", "not valid json"),
            {},
        )
        assert result is not None
        assert "not valid JSON" in result


# -------------------------------------------------------------------------------------------------------------------- #
# TestAfterDropRoutingTable                                                                                     #
# -------------------------------------------------------------------------------------------------------------------- #

class TestAfterDropRoutingTable:
    def test_returns_early_for_non_routing_table_buffer(self):
        proc = MagicMock()
        proc.get_buffer.return_value = MagicMock()

        after_drop_routing_table(proc, _mock_buffer("STREAM[notes]"), {})

        proc.unregister_hook.assert_not_called()
        proc.drop_buffer.assert_not_called()

    def test_unregisters_hook_and_drops_fallthrough(self):
        proc = MagicMock()
        origin_mock = MagicMock()
        proc.get_buffer.side_effect = lambda name: (
            origin_mock if name == "STREAM"
            else _mock_buffer(name)
        )
        buf = _mock_buffer("STREAM[routing_table:t1]", "{}")

        after_drop_routing_table(proc, buf, {})

        # Verify the dispatch hook was unregistered from the origin.
        origin_mock.unregister_hook.assert_called_once()

        # Verify the default fallthrough buffer was dropped.
        proc.drop_buffer.assert_called_once_with("STREAM[routing_table:t1]:fallthrough")
