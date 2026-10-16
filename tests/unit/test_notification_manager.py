"""Unit tests for NotificationManager."""

from __future__ import annotations

import asyncio
import json
import pytest
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from petekit.stream.notification_manager import (
    _notifications_option_name,
    _validate_notifications_config,
    after_create_notification_option,
    before_update_notification_option,
    after_drop_notification_option,
    notify_buffer,
    _most_recent_persistent_thread,
)


# -------------------------------------------------------------------------------------------------------------------- #
# Helpers                                                                                                              #
# -------------------------------------------------------------------------------------------------------------------- #

def _make_notification_option(text: str = "{}") -> MagicMock:
    """Return a mock option buffer that returns `text` from get_text()."""
    opt = MagicMock()
    opt.name = "STREAM[notifications]"
    opt.get_text.return_value = text
    return opt


def _make_processor_with_option(
    option_text: str = "{}",
    extra_buffers: dict[str, MagicMock] | None = None,
    notify_agent: AsyncMock | None = None,
) -> MagicMock:
    """Return a processor mock with get_buffer configured for STREAM and STREAM[notifications]."""
    proc = MagicMock()
    # Default to None for unknown buffers (not MagicMock) so None-checks in code work.
    _extra = extra_buffers or {}
    proc.get_buffer.side_effect = lambda name: (
        _make_notification_option(option_text) if name == "STREAM[notifications]"
        else _extra.get(name, None)
    )
    if notify_agent is not None:
        proc.notify_agent = notify_agent
    proc._oap_json = {"thread_store": {}}
    proc._oap_session_store = {}
    return proc


# -------------------------------------------------------------------------------------------------------------------- #
# TestNotificationsOptionName                                                                                        #
# -------------------------------------------------------------------------------------------------------------------- #

class TestNotificationsOptionName:
    def test_returns_buffer_name_with_bracket_suffix(self):
        result = _notifications_option_name("ORIGIN")
        assert result == "ORIGIN[notifications]"

    def test_returns_buffer_name_preserves_any_name(self):
        result = _notifications_option_name("system:log:stream_processor")
        assert result == "system:log:stream_processor[notifications]"


# -------------------------------------------------------------------------------------------------------------------- #
# TestValidateNotificationsConfig                                                                                   #
# -------------------------------------------------------------------------------------------------------------------- #

class TestValidateNotificationsConfig:
    def test_valid_empty_config_returns_none(self):
        result = _validate_notifications_config("ORIGIN", {})
        assert result is None

    def test_valid_full_config_returns_none(self):
        config = {
            "batch_size": 5,
            "interval_secs": 120.0,
            "notify_on_empty": True,
            "persistent_thread_id": "thread-abc-123",
        }
        result = _validate_notifications_config("ORIGIN", config)
        assert result is None

    def test_valid_omitted_optional_fields_returns_none(self):
        config = {"batch_size": 1}
        result = _validate_notifications_config("ORIGIN", config)
        assert result is None

    def test_persistent_thread_id_null_is_valid(self):
        config = {"persistent_thread_id": None}
        result = _validate_notifications_config("ORIGIN", config)
        assert result is None

    def test_invalid_config_not_a_dict_returns_error(self):
        result = _validate_notifications_config("ORIGIN", "not a dict")
        assert result is not None
        assert "JSON object" in result

    def test_invalid_batch_size_string_returns_error(self):
        result = _validate_notifications_config("ORIGIN", {"batch_size": "three"})
        assert result is not None
        assert "batch_size" in result
        assert "not a valid int" in result

    def test_invalid_batch_size_zero_returns_error(self):
        result = _validate_notifications_config("ORIGIN", {"batch_size": 0})
        assert result is not None
        assert "batch_size" in result
        assert "must be at least 1" in result

    def test_invalid_batch_size_negative_returns_error(self):
        result = _validate_notifications_config("ORIGIN", {"batch_size": -5})
        assert result is not None
        assert "must be at least 1" in result

    def test_invalid_interval_secs_string_returns_error(self):
        result = _validate_notifications_config("ORIGIN", {"interval_secs": "sixty"})
        assert result is not None
        assert "interval_secs" in result
        assert "not a valid float" in result

    def test_invalid_interval_below_min_returns_error(self):
        result = _validate_notifications_config("ORIGIN", {"interval_secs": 5.0})
        assert result is not None
        assert "interval_secs" in result
        assert "must be between" in result

    def test_invalid_interval_above_max_returns_error(self):
        result = _validate_notifications_config("ORIGIN", {"interval_secs": 9999.0})
        assert result is not None
        assert "interval_secs" in result
        assert "must be between" in result

    def test_invalid_notify_on_empty_string_returns_error(self):
        result = _validate_notifications_config("ORIGIN", {"notify_on_empty": "yes"})
        assert result is not None
        assert "notify_on_empty" in result
        assert "must be a bool" in result

    def test_invalid_persistent_thread_id_int_returns_error(self):
        result = _validate_notifications_config("ORIGIN", {"persistent_thread_id": 42})
        assert result is not None
        assert "persistent_thread_id" in result
        assert "must be a string" in result


# -------------------------------------------------------------------------------------------------------------------- #
# TestAfterCreateNotificationOption                                                                                 #
# -------------------------------------------------------------------------------------------------------------------- #

class TestAfterCreateNotificationOption:
    async def test_logs_error_for_malformed_option_buffer_name(self, monkeypatch):
        processor = MagicMock()
        processor._stream_buffer_notifications_loop = {}
        option_buffer = MagicMock()
        option_buffer.name = "no-suffix-here"  # Does not end with [notifications]

        errors = []
        monkeypatch.setattr(
            "petekit.stream.notification_manager._logger",
            MagicMock(error=errors.append),
        )

        await after_create_notification_option(processor, option_buffer, {})

        # Should log and return without starting a timer.
        assert len(errors) == 1
        assert "[after_create_notification_option]" in errors[0]
        assert len(processor._stream_buffer_notifications_loop) == 0

    async def test_logs_error_for_empty_origin_buffer_name(self, monkeypatch):
        processor = MagicMock()
        processor._stream_buffer_notifications_loop = {}
        option_buffer = MagicMock()
        option_buffer.name = "[notifications]"  # Strips to empty string

        errors = []
        monkeypatch.setattr(
            "petekit.stream.notification_manager._logger",
            MagicMock(error=errors.append),
        )

        await after_create_notification_option(processor, option_buffer, {})

        assert len(errors) == 1
        assert len(processor._stream_buffer_notifications_loop) == 0

    async def test_starts_timer_and_stores_task(self, monkeypatch):
        processor = MagicMock()
        processor._stream_buffer_notifications_loop = {}
        option_buffer = MagicMock()
        option_buffer.name = "STREAM[notifications]"

        # Patch create_task to create a real task, start it (so it's tracked by the
        # event loop), then immediately cancel it — avoids the unawaited-coroutine
        # warning that results from just calling create_task on a real coroutine.
        real_create_task = asyncio.create_task
        started_tasks: list = []

        def mock_create_task(coro, **kw):
            task = real_create_task(coro)
            started_tasks.append(task)
            task.cancel()
            return task

        monkeypatch.setattr(asyncio, "create_task", mock_create_task)

        await after_create_notification_option(processor, option_buffer, {})

        # Verify a task was stored in the loop dict keyed by the origin buffer name.
        assert "STREAM" in processor._stream_buffer_notifications_loop
        assert len(started_tasks) == 1


# -------------------------------------------------------------------------------------------------------------------- #
# TestBeforeUpdateNotificationOption                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #

class TestBeforeUpdateNotificationOption:
    async def test_blocks_malformed_json_returns_error(self):
        processor = MagicMock()
        processor.get_buffer.return_value = MagicMock()
        old_option_buffer = MagicMock()
        new_option_buffer = MagicMock()
        new_option_buffer.name = "STREAM[notifications]"
        new_option_buffer.get_text.return_value = "not valid json at all"

        result = await before_update_notification_option(processor, old_option_buffer, new_option_buffer, {})

        assert result is not None
        assert "not valid JSON" in result

    async def test_blocks_invalid_config_returns_error(self):
        processor = MagicMock()
        processor.get_buffer.return_value = MagicMock()
        old_option_buffer = MagicMock()
        new_option_buffer = MagicMock()
        new_option_buffer.name = "STREAM[notifications]"
        new_option_buffer.get_text.return_value = json.dumps({"interval_secs": 5.0})

        result = await before_update_notification_option(processor, old_option_buffer, new_option_buffer, {})

        assert result is not None
        assert "must be between" in result

    async def test_allows_valid_config_returns_true(self):
        processor = MagicMock()
        processor.get_buffer.return_value = MagicMock()
        old_option_buffer = MagicMock()
        new_option_buffer = MagicMock()
        new_option_buffer.name = "STREAM[notifications]"
        new_option_buffer.get_text.return_value = json.dumps({
            "batch_size": 3,
            "interval_secs": 60.0,
            "notify_on_empty": False,
        })

        result = await before_update_notification_option(processor, old_option_buffer, new_option_buffer, {})

        assert result is True

    async def test_blocks_empty_origin_buffer_name_returns_error(self):
        processor = MagicMock()
        old_option_buffer = MagicMock()
        new_option_buffer = MagicMock()
        new_option_buffer.name = "[notifications]"  # Strips to empty

        result = await before_update_notification_option(processor, old_option_buffer, new_option_buffer, {})

        assert result is not None
        assert "cannot parse origin buffer name" in result

    async def test_skips_non_matching_option_buffers_returns_none(self):
        processor = MagicMock()
        old_option_buffer = MagicMock()
        new_option_buffer = MagicMock()
        new_option_buffer.name = "STREAM[routing_table:foo]"

        result = await before_update_notification_option(processor, old_option_buffer, new_option_buffer, {})

        assert result is None  # Allow non-matching buffers through without blocking


# -------------------------------------------------------------------------------------------------------------------- #
# TestAfterDropNotificationOption                                                                                  #
# -------------------------------------------------------------------------------------------------------------------- #

class TestAfterDropNotificationOption:
    async def test_cancels_and_removes_task_from_loop(self):
        processor = MagicMock()
        mock_task = MagicMock()
        processor._stream_buffer_notifications_loop = {"STREAM": mock_task}
        option_buffer = MagicMock()
        option_buffer.name = "STREAM[notifications]"

        await after_drop_notification_option(processor, option_buffer, {})

        mock_task.cancel.assert_called_once()
        assert "STREAM" not in processor._stream_buffer_notifications_loop

    async def test_no_error_when_no_task_for_buffer(self):
        processor = MagicMock()
        processor._stream_buffer_notifications_loop = {}
        option_buffer = MagicMock()
        option_buffer.name = "STREAM[notifications]"

        # Should not raise — pop returns None, cancel is never called.
        await after_drop_notification_option(processor, option_buffer, {})

        assert len(processor._stream_buffer_notifications_loop) == 0


# -------------------------------------------------------------------------------------------------------------------- #
# TestNotifyBuffer                                                                                                  #
# -------------------------------------------------------------------------------------------------------------------- #

class TestNotifyBuffer:
    async def test_returns_early_when_option_buffer_not_found(self, monkeypatch):
        processor = MagicMock()
        processor.get_buffer.return_value = None

        errors = []
        monkeypatch.setattr(
            "petekit.stream.notification_manager._logger",
            MagicMock(error=errors.append),
        )

        await notify_buffer(processor, "STREAM")

        assert len(errors) == 1
        assert "notifications option not found" in errors[0]

    async def test_returns_early_when_origin_buffer_not_found(self, monkeypatch):
        # The notifications option exists, but the origin STREAM buffer is missing.
        processor = _make_processor_with_option(
            option_text="{}",
            extra_buffers={
                # STREAM is intentionally absent — only option buffer is present.
            },
        )
        # Directly assign to the module namespace so _logger.error(...) in the code
        # calls our capture list instead of the real logger.
        errors = []
        monkeypatch.setattr(
            "petekit.stream.notification_manager._logger",
            MagicMock(error=errors.append),
        )

        await notify_buffer(processor, "STREAM")

        assert len(errors) == 1
        assert "stream buffer" in errors[0]
        assert "not found" in errors[0]

    async def test_returns_early_when_batch_threshold_not_met(self):
        # All lines already have agent_at set — nothing unseen.
        proc = _make_processor_with_option(
            option_text=json.dumps({"batch_size": 1}),
            extra_buffers={
                "STREAM": MagicMock(
                    lines=[
                        MagicMock(agent_at=123.0),
                        MagicMock(agent_at=124.0),
                    ]
                )
            },
            notify_agent=AsyncMock(),
        )

        await notify_buffer(proc, "STREAM")

        # notify_agent should not have been called — batch threshold not met.
        proc.notify_agent.assert_not_called()

    async def test_fires_notify_agent_when_batch_threshold_met(self):
        proc = _make_processor_with_option(
            option_text=json.dumps({"batch_size": 1, "persistent_thread_id": "thread-1"}),
            extra_buffers={
                "STREAM": MagicMock(
                    lines=[
                        MagicMock(agent_at=None, modified_at=100.0),
                        MagicMock(agent_at=None, modified_at=101.0),
                    ]
                )
            },
            notify_agent=AsyncMock(),
        )

        await notify_buffer(proc, "STREAM")

        proc.notify_agent.assert_called_once()
        call_kwargs = proc.notify_agent.call_args[1]
        assert call_kwargs["persistent_thread_id"] == "thread-1"

    async def test_falls_back_to_most_recent_thread_when_not_in_config(self, monkeypatch):
        proc = _make_processor_with_option(
            option_text=json.dumps({"batch_size": 1}),  # No persistent_thread_id in config
            extra_buffers={
                "STREAM": MagicMock(
                    lines=[MagicMock(agent_at=None, modified_at=100.0)]
                )
            },
            notify_agent=AsyncMock(),
        )

        monkeypatch.setattr(
            "petekit.stream.notification_manager._most_recent_persistent_thread",
            lambda obj: "most-recent-thread-id",
        )

        await notify_buffer(proc, "STREAM")

        proc.notify_agent.assert_called_once()
        call_kwargs = proc.notify_agent.call_args[1]
        assert call_kwargs["persistent_thread_id"] == "most-recent-thread-id"


# -------------------------------------------------------------------------------------------------------------------- #
# TestMostRecentPersistentThread                                                                                   #
# -------------------------------------------------------------------------------------------------------------------- #

class TestMostRecentPersistentThread:
    def test_returns_none_when_thread_store_empty(self):
        obj = MagicMock()
        obj._oap_json = {"thread_store": {}}
        obj._oap_session_store = {}

        result = _most_recent_persistent_thread(obj)

        assert result is None

    def test_returns_none_when_no_sessions(self):
        obj = MagicMock()
        obj._oap_json = {"thread_store": {"t1": "s1"}}
        obj._oap_session_store = {}  # No sessions in store

        result = _most_recent_persistent_thread(obj)

        assert result is None
