from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from .buffer_manager import (
    Buffer,
    BufferManager,
    HookPoint,
)

_logger = logging.getLogger("option_buffer_manager")

_OPTION_BUFFER_PATTERN = re.compile(r"^(.+)\[(.+)\]$")
_OPTION_BUFFER_DOC_BUFFER = "doc:option_buffers"
_OPTION_BUFFER_DOC = """\
## Option Buffers

Option buffers are subordinate buffers that exist only alongside an origin buffer.
Their names carry a trailing [...]: ORIGIN[option].

## Invariants

- An option buffer may only be created when its origin buffer already exists.
- Dropping an origin buffer automatically drops all its option buffers.

## System Buffers

system:status:buffers  — live listing of all buffers (including option buffers)
"""


def _origin_name_from_option_buffer_name(name: str) -> str | None:
    """Return the origin buffer name if name is an option buffer (has [...]), else None."""
    match = _OPTION_BUFFER_PATTERN.fullmatch(name)
    if not match:
        return None
    return match.group(1)


def _option_type_from_option_buffer_name(name: str) -> str | None:
    """Return the option type (the content inside [...]) if name is an option buffer, else None."""
    match = _OPTION_BUFFER_PATTERN.fullmatch(name)
    if not match:
        return None
    return match.group(2)


def _reject_creation_if_origin_missing(
    buffer_manager: BufferManager,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Pre-create guard: reject option buffer creation if the origin does not exist."""
    origin_name = _origin_name_from_option_buffer_name(buffer.name)
    if origin_name is None:
        return None  # Not an option buffer — allow creation to proceed.

    # Reject if the origin buffer does not exist in the manager.
    if buffer_manager.get_buffer(origin_name) is None:
        return (
            f"Cannot create option buffer '{buffer.name}': "
            f"origin buffer '{origin_name}' does not exist."
        )
    return None  # Origin exists — allow creation.


def _register_origin_cascade_drop(
    buffer_manager: BufferManager,
    buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Post-create hook: register a cascade-drop hook on the origin buffer so the option is dropped when the origin is dropped."""
    origin_name = _origin_name_from_option_buffer_name(buffer.name)
    if origin_name is None:
        _logger.error(f"[_register_origin_cascade_drop] buffer '{buffer.name}' does not match the option buffer pattern — hook should not have been invoked")
        return None  # Not an option buffer — nothing to register.

    # Resolve the origin buffer — it must exist since the pre-create guard passed.
    origin = buffer_manager.get_buffer(origin_name)
    if origin is None:
        _logger.error(f"[_register_origin_cascade_drop] origin '{origin_name}' not found for option buffer '{buffer.name}'")
        return None

    # Capture the full option buffer name in the closure — this is what gets dropped.
    option_name = buffer.name

    async def _cascade_drop_option(
        hook_buffer_manager: BufferManager,
        origin_buffer: Buffer,
        args: dict[str, Any],
    ) -> None:
        """Drop the option buffer when the origin buffer is dropped."""
        if hook_buffer_manager.get_buffer(option_name) is not None:
            await hook_buffer_manager.drop_buffer(option_name)

    # Register the cascade-drop hook on the origin buffer's AFTER_DROP_BUFFER hook point.
    origin.register_hook(
        HookPoint.AFTER_DROP_BUFFER,
        f"option_cascade_{option_name}",
        _cascade_drop_option,
    )


class OptionBufferManager(BufferManager):
    """Option buffers (named ORIGIN[option]) always require an existing origin buffer, and are dropped automatically when the origin is dropped. See doc:option_buffers for details."""

    def __init__(self, **kwargs: Any) -> None:
        # Initialize the base buffer manager.
        super().__init__(**kwargs)

        # Seed the schema documentation buffer.
        self._create_buffer(_OPTION_BUFFER_DOC_BUFFER, text=_OPTION_BUFFER_DOC)

        # Register the pattern hooks that enforce the option-buffer/origin invariant.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "option_buffer_requires_origin",
            r"^.+\[.+\]$",
            _reject_creation_if_origin_missing,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "option_buffer_cascade_drop",
            r"^.+\[.+\]$",
            _register_origin_cascade_drop,
        )
