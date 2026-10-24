"""Unit tests for OptionBufferManager."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from petekit.text.option_buffer_manager import (
    _origin_name_from_option_buffer_name,
    _option_type_from_option_buffer_name,
    _reject_creation_if_origin_missing,
    _register_origin_cascade_drop,
)


class TestOriginNameFromOption:
    def test_returns_origin_for_valid_option_buffer(self):
        result = _origin_name_from_option_buffer_name("STREAM[notifications]")
        assert result == "STREAM"

    def test_returns_origin_with_special_characters(self):
        result = _origin_name_from_option_buffer_name("system:log[notes]")
        assert result == "system:log"

    def test_returns_origin_with_nested_brackets(self):
        result = _origin_name_from_option_buffer_name("STREAM[notifications][more]")
        assert result == "STREAM[notifications]"

    def test_returns_none_for_plain_buffer(self):
        result = _origin_name_from_option_buffer_name("STREAM")
        assert result is None

    def test_returns_none_for_empty_brackets(self):
        result = _origin_name_from_option_buffer_name("STREAM[]")
        # Empty brackets [] do not match the option buffer pattern (requires 1+ chars inside brackets).
        assert result is None


class TestOptionNameFromOption:
    def test_returns_option_name_for_valid_option_buffer(self):
        result = _option_type_from_option_buffer_name("STREAM[notifications]")
        assert result == "notifications"

    def test_returns_option_name_with_special_characters(self):
        result = _option_type_from_option_buffer_name("system:log[notes]")
        assert result == "notes"

    def test_returns_option_name_with_nested_brackets(self):
        result = _option_type_from_option_buffer_name("STREAM[notifications][more]")
        assert result == "more"

    def test_returns_none_for_plain_buffer(self):
        result = _option_type_from_option_buffer_name("STREAM")
        assert result is None

    def test_returns_none_for_empty_brackets(self):
        result = _option_type_from_option_buffer_name("STREAM[]")
        # Empty brackets [] do not match the option buffer pattern — returns None.
        assert result is None


class TestRejectCreationIfOriginMissing:
    def test_allows_creation_of_plain_buffer(self):
        buffer_manager = MagicMock()
        buffer = MagicMock()
        buffer.name = "STREAM"
        result = _reject_creation_if_origin_missing(buffer_manager, buffer, {})
        assert result is None  # None means allowed

    def test_blocks_creation_when_origin_does_not_exist(self):
        buffer_manager = MagicMock()
        buffer_manager.get_buffer.return_value = None
        buffer = MagicMock()
        buffer.name = "STREAM[notifications]"

        result = _reject_creation_if_origin_missing(buffer_manager, buffer, {})

        assert result is not None
        assert "origin buffer 'STREAM' does not exist" in result

    def test_allows_creation_when_origin_exists(self):
        buffer_manager = MagicMock()
        buffer_manager.get_buffer.return_value = MagicMock()  # Origin exists
        buffer = MagicMock()
        buffer.name = "STREAM[notifications]"

        result = _reject_creation_if_origin_missing(buffer_manager, buffer, {})

        assert result is None  # None means allowed

    def test_allows_chain_option_when_intermediate_option_exists(self):
        # STREAM[notifications] is the origin; STREAM[notifications][more] is the chain option.
        # When the pre-create guard fires for STREAM[notifications][more],
        # the origin STREAM[notifications] already exists in _buffers, so creation is allowed.
        buffer_manager = MagicMock()
        buffer_manager.get_buffer.return_value = MagicMock()  # STREAM[notifications] exists
        buffer = MagicMock()
        buffer.name = "STREAM[notifications][more]"

        result = _reject_creation_if_origin_missing(buffer_manager, buffer, {})

        # The origin "STREAM[notifications]" IS in _buffers (as an option buffer).
        # Therefore creation of STREAM[notifications][more] is allowed.
        assert result is None

    def test_blocks_chain_option_when_intermediate_option_missing(self):
        # STREAM[notifications][more] creation is proposed but STREAM[notifications] is gone.
        buffer_manager = MagicMock()
        buffer_manager.get_buffer.return_value = None  # Origin doesn't exist
        buffer = MagicMock()
        buffer.name = "STREAM[notifications][more]"

        result = _reject_creation_if_origin_missing(buffer_manager, buffer, {})

        assert result is not None
        assert "origin buffer 'STREAM[notifications]' does not exist" in result


class TestRegisterOriginCascadeDrop:
    def test_logs_error_when_buffer_name_not_option_buffer(self):
        buffer_manager = MagicMock()
        buffer = MagicMock()
        buffer.name = "STREAM"  # Not an option buffer

        errors = []
        def capture_error(msg):
            errors.append(msg)
        import petekit.text.option_buffer_manager as obm
        obm._logger.error = capture_error

        _register_origin_cascade_drop(buffer_manager, buffer, {})

        assert len(errors) == 1
        assert "does not match the option buffer pattern" in errors[0]

    def test_logs_error_when_origin_buffer_not_found(self):
        buffer_manager = MagicMock()
        buffer_manager.get_buffer.return_value = None
        buffer = MagicMock()
        buffer.name = "STREAM[notifications]"

        errors = []
        def capture_error(msg):
            errors.append(msg)
        import petekit.text.option_buffer_manager as obm
        obm._logger.error = capture_error

        _register_origin_cascade_drop(buffer_manager, buffer, {})

        assert len(errors) == 1
        assert "origin 'STREAM' not found" in errors[0]

    def test_registers_cascade_drop_hook_on_origin(self):
        buffer_manager = MagicMock()
        origin_mock = MagicMock()
        buffer_manager.get_buffer.return_value = origin_mock
        buffer = MagicMock()
        buffer.name = "STREAM[notifications]"

        _register_origin_cascade_drop(buffer_manager, buffer, {})

        # Verify register_hook was called on the origin buffer.
        origin_mock.register_hook.assert_called_once()
        call_args = origin_mock.register_hook.call_args
        hook_point = call_args[0][0]
        hook_name = call_args[0][1]
        hook_fn = call_args[0][2]

        # Check the hook point is AFTER_DROP_BUFFER.
        from petekit.text.buffer_manager import HookPoint
        assert hook_point == HookPoint.AFTER_DROP_BUFFER

        # Check the hook name embeds the option buffer name.
        assert "STREAM[notifications]" in hook_name

        # Invoke the registered hook — it should drop the option buffer.
        buffer_manager.reset_mock()
        buffer_manager.get_buffer.side_effect = lambda name: (
            MagicMock() if name == "STREAM[notifications]" else None
        )
        hook_fn(buffer_manager, origin_mock, {})
        buffer_manager.drop_buffer.assert_called_once_with("STREAM[notifications]")

    def test_does_not_drop_when_option_buffer_already_gone(self):
        buffer_manager = MagicMock()
        origin_mock = MagicMock()
        buffer = MagicMock()
        buffer.name = "STREAM[notifications]"

        # Origin buffer "STREAM" is found, but the option buffer is already gone.
        # The side_effect makes get_buffer("STREAM[notifications]") return None (already dropped).
        def get_buffer_mock(name):
            if name == "STREAM":
                return origin_mock
            return None  # Option already gone
        buffer_manager.get_buffer.side_effect = get_buffer_mock

        _register_origin_cascade_drop(buffer_manager, buffer, {})

        registered_hook_fn = origin_mock.register_hook.call_args[0][2]
        buffer_manager.drop_buffer.reset_mock()
        registered_hook_fn(buffer_manager, origin_mock, {})

        # drop_buffer should not have been called since the option is already gone.
        buffer_manager.drop_buffer.assert_not_called()

    def test_cascade_is_one_hop_not_recursive(self):
        # Verify that chain options (STREAM[notifications][more]) only cascade one level.
        # When STREAM[notifications] is dropped, STREAM[notifications][more] IS dropped.
        # But when STREAM is dropped, STREAM[notifications][more] is NOT dropped
        # (STREAM has no hook for STREAM[notifications][more] — only for STREAM[notifications]).
        buffer_manager = MagicMock()
        stream_mock = MagicMock()
        notifications_mock = MagicMock()
        # Only STREAM[notifications] is an origin buffer of another option.
        # STREAM[notifications][more] has its own origin (STREAM[notifications]) but
        # there is no hook on STREAM that drops STREAM[notifications][more].
        buffer_manager.get_buffer.side_effect = lambda name: (
            stream_mock if name == "STREAM"
            else notifications_mock if name == "STREAM[notifications]"
            else None
        )

        # Simulate drop of STREAM[notifications], which should trigger cascade drop.
        buffer_manager.drop_buffer.reset_mock()
        notifications_mock.register_hook.reset_mock()

        # First register the cascade hook on STREAM (simulated via after_create).
        buffer_for_cascade = MagicMock()
        buffer_for_cascade.name = "STREAM[notifications]"
        _register_origin_cascade_drop(buffer_manager, buffer_for_cascade, {})

        # Now trigger the origin drop — STREAM is being dropped.
        drop_hook_fn = stream_mock.register_hook.call_args[0][2]
        buffer_for_cascade.name = "STREAM"  # Simulate the origin is being dropped
        drop_hook_fn(buffer_manager, stream_mock, {})

        # STREAM[notifications] should be dropped.
        buffer_manager.drop_buffer.assert_called()
        # But STREAM[notifications][more] is NOT in the call list
        # (the hook only drops the direct option, not nested ones).
        call_args_list = buffer_manager.drop_buffer.call_args_list
        dropped_names = [c[0][0] for c in call_args_list]
        assert "STREAM[notifications]" in dropped_names
        assert "STREAM[notifications][more]" not in dropped_names

