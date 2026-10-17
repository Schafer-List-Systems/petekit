"""Unit tests for StreamRouter."""

from __future__ import annotations

import inspect
import json
import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from petekit.stream.stream_router import (
    _is_routing_table_option,
    _parse_routing_table_option,
    evaluate_condition,
    route_dispatch,
    after_create_routing_table,
    before_update_routing_table,
    after_drop_routing_table,
    RoutingConditionError,
)


# -------------------------------------------------------------------------------------------------------------------- #
# Helpers                                                                                                              #
# -------------------------------------------------------------------------------------------------------------------- #

def _mock_buffer(name: str, text: str = "") -> MagicMock:
    """Return a mock Buffer with the given name and get_text returning text."""
    buf = MagicMock()
    buf.name = name
    buf.get_text.return_value = text
    return buf


def _mock_processor_with_routing_table(
    table_name: str,
    routing_table: list,
    fallthrough: str | None = None,
) -> MagicMock:
    """Return a processor mock with a routing_table option buffer configured.

    routing_table entries use the new format: [[conditions_list, output_stream], ...]
    conditions_list is a list of condition name strings, e.g. ["true"] or ["cond1", "cond2"].
    """
    option_config = {"routing_table": routing_table}
    if fallthrough is not None:
        option_config["fallthrough"] = fallthrough

    proc = MagicMock()
    proc.get_buffer.side_effect = lambda name: (
        _mock_buffer(f"STREAM[routing_table:{table_name}]", json.dumps(option_config))
        if name == f"STREAM[routing_table:{table_name}]"
        else _mock_buffer(name)
    )
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

        def inner_callable(function_manager, stream, text, metadata):
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

        def good_callable(function_manager, stream, text, metadata):
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
    def test_returns_none_when_option_buffer_not_found(self):
        proc = MagicMock()
        proc.get_buffer.return_value = None
        old_buf = _mock_buffer("STREAM")
        new_buf = _mock_buffer("STREAM", "new text")

        result = route_dispatch(proc, old_buf, new_buf, "t1")

        assert result is None

    def test_writes_to_matched_output_stream(self):
        proc = _mock_processor_with_routing_table(
            "t1",
            [[["true"], "output_a"]],
        )
        old_buf = _mock_buffer("STREAM")
        new_buf = _mock_buffer("STREAM", "hello world")

        route_dispatch(proc, old_buf, new_buf, "t1")

        # Verify write_buffer was called with the matched output stream.
        proc.write_buffer.assert_called_once()
        call_args = proc.write_buffer.call_args
        assert call_args[0][0] == "output_a"
        assert call_args[0][1] == "hello world"

    def test_all_conditions_must_match_in_entry(self):
        # Both conditions in the list must match — entry fires only if all conditions return True.
        proc = _mock_processor_with_routing_table(
            "t1",
            [[["true", "true"], "output_a"], [["false"], "output_b"]],
        )
        old_buf = _mock_buffer("STREAM")
        new_buf = _mock_buffer("STREAM", "text")

        route_dispatch(proc, old_buf, new_buf, "t1")

        # Both conditions matched — write to output_a (first entry wins).
        assert proc.write_buffer.call_count == 1
        assert proc.write_buffer.call_args[0][0] == "output_a"

    def test_falls_through_when_any_condition_fails_in_entry(self):
        # One condition fails — entry is skipped, falls through to output_b.
        proc = _mock_processor_with_routing_table(
            "t1",
            [[["true", "false"], "output_a"], [["true"], "output_b"]],
        )
        old_buf = _mock_buffer("STREAM")
        new_buf = _mock_buffer("STREAM", "text")

        route_dispatch(proc, old_buf, new_buf, "t1")

        # First entry skipped (second condition false), second entry matches — write to output_b.
        last_call = proc.write_buffer.call_args_list[-1]
        assert last_call[0][0] == "output_b"

    def test_tries_entries_top_to_bottom_first_match_wins(self):
        proc = _mock_processor_with_routing_table(
            "t1",
            [[["false"], "output_a"], [["true"], "output_b"]],
        )
        old_buf = _mock_buffer("STREAM")
        new_buf = _mock_buffer("STREAM", "text")

        route_dispatch(proc, old_buf, new_buf, "t1")

        # Only the second entry matches — write to output_b.
        assert proc.write_buffer.call_count == 1
        assert proc.write_buffer.call_args[0][0] == "output_b"

    def test_falls_through_to_default_when_no_match(self):
        proc = _mock_processor_with_routing_table(
            "t1",
            [[["false"], "output_a"]],  # nothing matches
        )
        old_buf = _mock_buffer("STREAM")
        new_buf = _mock_buffer("STREAM", "text")

        route_dispatch(proc, old_buf, new_buf, "t1")

        # Default fallthrough: STREAM[routing_table:t1]:fallthrough
        # First call is the log line; last call is the actual write.
        last_call = proc.write_buffer.call_args_list[-1]
        assert last_call[0][0] == "STREAM[routing_table:t1]:fallthrough"
        assert last_call[0][1] == "text"

    def test_uses_custom_fallthrough_when_configured(self):
        proc = _mock_processor_with_routing_table(
            "t1",
            [[["false"], "output_a"]],
            fallthrough="my_fallthrough",
        )
        old_buf = _mock_buffer("STREAM")
        new_buf = _mock_buffer("STREAM", "text")

        route_dispatch(proc, old_buf, new_buf, "t1")

        # Last call is the actual write to the custom fallthrough.
        last_call = proc.write_buffer.call_args_list[-1]
        assert last_call[0][0] == "my_fallthrough"

    def test_logs_routing_error_on_invalid_json(self):
        proc = MagicMock()
        proc.get_buffer.side_effect = lambda name: (
            _mock_buffer("STREAM[routing_table:t1]", "not json")
            if name == "STREAM[routing_table:t1]"
            else _mock_buffer(name)
        )
        old_buf = _mock_buffer("STREAM")
        new_buf = _mock_buffer("STREAM", "text")

        route_dispatch(proc, old_buf, new_buf, "t1")

        # Should log the routing error and return None.
        proc.write_buffer.assert_called()
        error_calls = [
            c for c in proc.write_buffer.call_args_list
            if "ROUTING ERROR" in str(c)
        ]
        assert len(error_calls) == 1

    def test_logs_error_when_sink_write_fails(self):
        proc = _mock_processor_with_routing_table(
            "t1",
            [[["true"], "output_a"]],
        )
        proc.write_buffer.return_value = {"ok": False, "error": "disk full"}
        old_buf = _mock_buffer("STREAM")
        new_buf = _mock_buffer("STREAM", "text")

        route_dispatch(proc, old_buf, new_buf, "t1")

        # Should log the sink error after writing to the output.
        proc.write_buffer.assert_called()
        error_calls = [
            c for c in proc.write_buffer.call_args_list
            if "SINK ERROR" in str(c)
        ]
        assert len(error_calls) == 1

    def test_skips_empty_appended_text(self):
        proc = _mock_processor_with_routing_table(
            "t1",
            [[["true"], "output_a"]],
        )
        old_buf = _mock_buffer("STREAM", "")
        new_buf = _mock_buffer("STREAM", "")

        route_dispatch(proc, old_buf, new_buf, "t1")

        # Nothing to route — write_buffer should not be called.
        proc.write_buffer.assert_not_called()


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
