"""Unit tests for regex_condition_factory pure functions."""

import asyncio
import json
import re
from typing import Any

import pytest

from petekit.stream.pattern_matchers.regex_condition_factory import (
    _guard_pattern_value,
    _make_condition,
    _validate_pattern_against_sample,
    _pattern_buffer_name,
    _before_update_condition,
    _before_drop_conditions_list,
    _before_drop_function_meta,
    _after_update_condition,
    _after_drop_condition,
    _CONDITION_LIST_BUFFER,
    _REGEX_PATTERN_BUFFER,
    RoutingConditionError,
)


# =============================================================================
# Mocks
# =============================================================================


class MockBuffer:
    def __init__(self, text: str, name: str = "system:config:regex_condition:mock") -> None:
        self._text = text
        self.name = name

    def get_text(self) -> str:
        return self._text


class MockFactory:
    def __init__(self) -> None:
        self._functions: dict[str, Any] = {}

    def get_function(self, name: str) -> dict[str, Any]:
        if name in self._functions:
            return {"ok": True, "callable": self._functions[name]}
        return {"ok": False, "error": f"function '{name}' not found"}


# =============================================================================
# _guard_pattern_value
# =============================================================================


class TestGuardPatternValue:
    def test_valid_dict_matching_sample_returns_compiled_pattern(self) -> None:
        value = {"regex": r"^hello$", "sample_line": "hello"}
        result = _guard_pattern_value("create", value)
        assert isinstance(result, re.Pattern)
        assert result.fullmatch("hello") is not None

    def test_valid_but_sample_does_not_match(self) -> None:
        value = {"regex": r"^hello$", "sample_line": "world"}
        result = _guard_pattern_value("create", value)
        assert isinstance(result, str)
        assert "sample_line does not match pattern" in result

    def test_invalid_regex_syntax(self) -> None:
        value = {"regex": r"[", "sample_line": "test"}
        result = _guard_pattern_value("create", value)
        assert isinstance(result, str)
        assert "invalid regex" in result

    def test_not_a_dict(self) -> None:
        result = _guard_pattern_value("create", "just a string")
        assert isinstance(result, str)
        assert "must supply a dict" in result

    def test_dict_missing_regex_key(self) -> None:
        result = _guard_pattern_value("create", {"sample_line": "test"})
        assert isinstance(result, str)
        assert "must supply a dict with 'regex' key" in result

    def test_dict_missing_sample_line(self) -> None:
        result = _guard_pattern_value("create", {"regex": r"^test$"})
        assert isinstance(result, str)
        assert "must supply 'sample_line'" in result


# =============================================================================
# _validate_pattern_against_sample
# =============================================================================


class TestValidatePatternAgainstSample:
    def test_compiles_and_fullmatches_returns_compiled_pattern(self) -> None:
        result = _validate_pattern_against_sample(r"^hello$", "hello")
        assert isinstance(result, re.Pattern)
        assert result.fullmatch("hello") is not None

    def test_bad_regex_syntax_returns_error(self) -> None:
        result = _validate_pattern_against_sample(r"[", "test")
        assert isinstance(result, str)
        assert "invalid regex" in result

    def test_does_not_fullmatch_returns_error(self) -> None:
        result = _validate_pattern_against_sample(r"^hello$", "world")
        assert isinstance(result, str)
        assert "sample_line does not match pattern" in result


# =============================================================================
# _pattern_buffer_name
# =============================================================================


class TestPatternBufferName:
    def test_derives_canonical_buffer_name(self) -> None:
        result = _pattern_buffer_name("my_condition")
        assert result == "system:config:regex_condition:my_condition"

    def test_sanitizes_condition_name(self) -> None:
        result = _pattern_buffer_name("my condition")
        assert " " not in result
        assert "mycondition" in result


# =============================================================================
# _before_update_condition
# =============================================================================


class TestBeforeUpdateCondition:
    def test_returns_true_when_old_buffer_is_none_and_new_is_valid(self) -> None:
        new_buf = MockBuffer(json.dumps({"p1": {"regex": r"^hello$", "sample_line": "hello"}}) + "\n")
        factory = MockFactory()
        result = asyncio.run(_before_update_condition(factory, None, new_buf, {}))
        assert result is True

    def test_returns_error_when_old_buffer_is_none_and_json_invalid(self) -> None:
        new_buf = MockBuffer("not valid json")
        factory = MockFactory()
        result = asyncio.run(_before_update_condition(factory, None, new_buf, {}))
        assert isinstance(result, str)
        assert "not valid JSON" in result

    def test_returns_error_when_old_buffer_is_none_and_regex_bad(self) -> None:
        new_buf = MockBuffer(json.dumps({"p1": {"regex": r"[", "sample_line": "test"}}) + "\n")
        factory = MockFactory()
        result = asyncio.run(_before_update_condition(factory, None, new_buf, {}))
        assert isinstance(result, str)
        assert "invalid regex" in result

    def test_returns_error_when_old_buffer_is_none_and_sample_does_not_match(self) -> None:
        new_buf = MockBuffer(json.dumps({"p1": {"regex": r"^hello$", "sample_line": "world"}}) + "\n")
        factory = MockFactory()
        result = asyncio.run(_before_update_condition(factory, None, new_buf, {}))
        assert isinstance(result, str)
        assert "sample_line does not match pattern" in result

    def test_returns_true_when_pattern_added_and_valid(self) -> None:
        old_buf = MockBuffer(json.dumps({}) + "\n")
        new_buf = MockBuffer(json.dumps({"p1": {"regex": r"^hello$", "sample_line": "hello"}}) + "\n")
        factory = MockFactory()
        result = asyncio.run(_before_update_condition(factory, old_buf, new_buf, {}))
        assert result is True

    def test_returns_error_when_touched_pattern_invalid(self) -> None:
        old_buf = MockBuffer(json.dumps({"p1": {"regex": r"^hello$", "sample_line": "hello"}}) + "\n")
        new_buf = MockBuffer(json.dumps({"p1": {"regex": r"^hello$", "sample_line": "world"}}) + "\n")
        factory = MockFactory()
        result = asyncio.run(_before_update_condition(factory, old_buf, new_buf, {}))
        assert isinstance(result, str)
        assert "sample_line does not match pattern" in result

    def test_returns_true_when_all_touched_patterns_valid(self) -> None:
        old_buf = MockBuffer(json.dumps({"p1": {"regex": r"^hello$", "sample_line": "hello"}}) + "\n")
        new_buf = MockBuffer(json.dumps({"p1": {"regex": r"^hello$", "sample_line": "hello"}, "p2": {"regex": r"^world$", "sample_line": "world"}}) + "\n")
        factory = MockFactory()
        result = asyncio.run(_before_update_condition(factory, old_buf, new_buf, {}))
        assert result is True


# =============================================================================
# _before_drop_conditions_list
# =============================================================================


class TestBeforeDropConditionsList:
    def test_returns_error_string_blocks_drop(self) -> None:
        buf = MockBuffer("")
        result = _before_drop_conditions_list(MockFactory(), buf, {})
        assert isinstance(result, str)
        assert "protected" in result.lower() or "cannot be dropped" in result.lower()


# =============================================================================
# _before_drop_function_meta
# =============================================================================


class TestBeforeDropFunctionMeta:
    def test_returns_error_string_blocks_drop(self) -> None:
        buf = MockBuffer("system:function:status:my_condition")
        result = _before_drop_function_meta(MockFactory(), buf, {})
        assert isinstance(result, str)
        assert "managed by RegexConditionManager" in result or "cannot be dropped" in result.lower()


# =============================================================================
# Integration test mocks
# =============================================================================


class MockFactoryIntegration:
    """Factory mock for integration tests with _read_buffer and write_buffer support."""

    def __init__(
        self,
        buffers: dict[str, str] | None = None,
        functions: dict[str, Any] | None = None,
    ) -> None:
        self._buffers = buffers or {}
        self._functions = functions or {}
        self._write_calls: list[tuple[str, str]] = []

    def _read_buffer(self, name: str) -> dict[str, Any]:
        if name in self._buffers:
            return {"ok": True, "content": self._buffers[name]}
        return {"ok": False, "error": f"buffer '{name}' not found"}

    async def write_buffer(self, name: str, text: str, pos: str = "end") -> dict[str, Any]:
        if name in self._buffers:
            self._buffers[name] = text
        else:
            self._buffers[name] = text
        self._write_calls.append((name, text))
        return {"ok": True}

    async def create_buffer(
        self,
        name: str,
        text: str = "",
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        self._buffers[name] = text
        return {"ok": True, "name": name}

    def get_buffer(self, name: str) -> MockBuffer | None:
        if name in self._buffers:
            return MockBuffer(self._buffers[name], name)
        return None


# =============================================================================
# _make_condition integration tests
# =============================================================================


class TestMakeCondition:
    def test_closure_returns_true_when_pattern_matches(self) -> None:
        def read_buffer(name: str) -> dict[str, Any]:
            return {
                "ok": True,
                "content": json.dumps({"p1": {"regex": r"^hello$", "sample_line": "hello"}}) + "\n",
            }

        cond_fn = _make_condition(
            f"{_REGEX_PATTERN_BUFFER}:my_cond",
            read_buffer,
            "my_cond",
        )
        result = cond_fn(None, "stream", "hello", {})
        assert result is True

    def test_closure_returns_false_when_pattern_does_not_match(self) -> None:
        def read_buffer(name: str) -> dict[str, Any]:
            return {
                "ok": True,
                "content": json.dumps({"p1": {"regex": r"^hello$", "sample_line": "hello"}}) + "\n",
            }

        cond_fn = _make_condition(
            f"{_REGEX_PATTERN_BUFFER}:my_cond",
            read_buffer,
            "my_cond",
        )
        result = cond_fn(None, "stream", "world", {})
        assert result is False

    def test_closure_raises_when_buffer_read_fails(self) -> None:
        def read_buffer(name: str) -> dict[str, Any]:
            return {"ok": False, "error": "buffer not found"}

        cond_fn = _make_condition(
            f"{_REGEX_PATTERN_BUFFER}:my_cond",
            read_buffer,
            "my_cond",
        )
        with pytest.raises(RoutingConditionError) as exc_info:
            cond_fn(None, "stream", "hello", {})
        assert "pattern buffer could not be read" in str(exc_info.value)

    def test_closure_raises_when_buffer_json_invalid(self) -> None:
        def read_buffer(name: str) -> dict[str, Any]:
            return {"ok": True, "content": "not valid json\n"}

        cond_fn = _make_condition(
            f"{_REGEX_PATTERN_BUFFER}:my_cond",
            read_buffer,
            "my_cond",
        )
        with pytest.raises(RoutingConditionError) as exc_info:
            cond_fn(None, "stream", "hello", {})
        assert "not valid JSON" in str(exc_info.value)

    def test_closure_returns_true_when_any_pattern_matches(self) -> None:
        def read_buffer(name: str) -> dict[str, Any]:
            return {
                "ok": True,
                "content": json.dumps({
                    "p1": {"regex": r"^hello$", "sample_line": "hello"},
                    "p2": {"regex": r"^world$", "sample_line": "world"},
                })
                + "\n",
            }

        cond_fn = _make_condition(
            f"{_REGEX_PATTERN_BUFFER}:my_cond",
            read_buffer,
            "my_cond",
        )
        assert cond_fn(None, "stream", "world", {}) is True
        assert cond_fn(None, "stream", "hello", {}) is True
        assert cond_fn(None, "stream", "nope", {}) is False


# =============================================================================
# _after_update_condition integration tests
# =============================================================================


class TestAfterUpdateCondition:
    def test_syncs_pattern_count_to_conditions_list(self) -> None:
        factory = MockFactoryIntegration(
            buffers={
                _CONDITION_LIST_BUFFER: json.dumps({
                    "conditions": {"my_cond": {"name": "my_cond", "pattern_count": 1}}
                })
                + "\n",
            }
        )
        new_buf = MockBuffer(
            json.dumps({
                "p1": {"regex": r"^hello$", "sample_line": "hello"},
                "p2": {"regex": r"^world$", "sample_line": "world"},
            }) + "\n",
            name=f"{_REGEX_PATTERN_BUFFER}:my_cond",
        )
        asyncio.run(_after_update_condition(factory, new_buf, new_buf, {}))

        # Read the conditions list and check pattern_count was updated.
        content = factory._buffers[_CONDITION_LIST_BUFFER]
        data = json.loads(content)
        assert data["conditions"]["my_cond"]["pattern_count"] == 2

    def test_returns_true_when_condition_not_in_catalog(self) -> None:
        factory = MockFactoryIntegration(
            buffers={
                _CONDITION_LIST_BUFFER: json.dumps({"conditions": {}}) + "\n",
            }
        )
        new_buf = MockBuffer(
            json.dumps({"p1": {"regex": r"^hello$", "sample_line": "hello"}}) + "\n",
            name=f"{_REGEX_PATTERN_BUFFER}:unknown_cond",
        )
        result = asyncio.run(_after_update_condition(factory, new_buf, new_buf, {}))
        assert result is True


# =============================================================================
# _after_drop_condition integration tests
# =============================================================================


class TestAfterDropCondition:
    def test_removes_condition_entry_from_conditions_list(self) -> None:
        factory = MockFactoryIntegration(
            buffers={
                _CONDITION_LIST_BUFFER: json.dumps({
                    "conditions": {
                        "my_cond": {"name": "my_cond", "pattern_count": 1}
                    }
                })
                + "\n",
            }
        )
        dropped_buf = MockBuffer(
            json.dumps({}) + "\n",
            name=f"{_REGEX_PATTERN_BUFFER}:my_cond",
        )
        asyncio.run(_after_drop_condition(factory, dropped_buf, {}))

        content = factory._buffers[_CONDITION_LIST_BUFFER]
        data = json.loads(content)
        assert "my_cond" not in data["conditions"]

    def test_does_nothing_when_condition_not_in_catalog(self) -> None:
        factory = MockFactoryIntegration(
            buffers={
                _CONDITION_LIST_BUFFER: json.dumps({"conditions": {}}) + "\n",
            }
        )
        dropped_buf = MockBuffer(
            json.dumps({}) + "\n",
            name=f"{_REGEX_PATTERN_BUFFER}:unknown_cond",
        )
        result = asyncio.run(_after_drop_condition(factory, dropped_buf, {}))
        assert result is True


# =============================================================================
# BEFORE_CREATE_BUFFER lambda bridge tests
# =============================================================================


class TestBeforeCreateBufferBridge:
    def test_bridge_calls_before_update_condition_with_none_for_old_buffer(self) -> None:
        # The lambda registered for BEFORE_CREATE_BUFFER calls
        # _before_update_condition(factory, None, buffer, args).
        # Test it by constructing the bridge lambda directly.
        from petekit.stream.pattern_matchers.regex_condition_factory import (
            _before_update_condition,
        )
        bridge = lambda factory, buffer, args: _before_update_condition(factory, None, buffer, args)

        valid_buffer = MockBuffer(
            json.dumps({"p1": {"regex": r"^hello$", "sample_line": "hello"}}) + "\n",
        )
        factory = MockFactory()
        result = asyncio.run(bridge(factory, valid_buffer, {}))
        assert result is True

    def test_bridge_rejects_invalid_pattern_at_create_time(self) -> None:
        from petekit.stream.pattern_matchers.regex_condition_factory import (
            _before_update_condition,
        )
        bridge = lambda factory, buffer, args: _before_update_condition(factory, None, buffer, args)

        bad_buffer = MockBuffer(
            json.dumps({"p1": {"regex": r"^hello$", "sample_line": "world"}}) + "\n",
        )
        factory = MockFactory()
        result = asyncio.run(bridge(factory, bad_buffer, {}))
        assert isinstance(result, str)
        assert "sample_line does not match pattern" in result
