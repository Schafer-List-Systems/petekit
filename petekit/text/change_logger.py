from __future__ import annotations

import logging
from typing import Any

from peteos.oap.agentic_object import AgenticObject


from petekit.text.buffer_manager import Buffer, HookPoint
from petekit.text.option_buffer_manager import OptionBufferManager
from petekit.stream.stream_buffer_manager import StreamBufferManager


_logger = logging.getLogger("changelog")


# -------------------------------------------------------------------------------------------------------------------- #
# Constants                                                                                                          #
# -------------------------------------------------------------------------------------------------------------------- #


_CHANGELOG_DOC_BUFFER = "doc:changelog"

_CHANGELOG_DOC = """\
## ChangeLogger

Track changes to a buffer by creating a [changelog] option buffer on it.
Every update to the buffer appends a one-line log entry to the configured
log buffer.

### Enable Change Logging

Create an option buffer named `<buffer>[changelog]` on the buffer to log.
The option buffer must hold a JSON object:

```json
{"log_buffer": "<buffer>.changelog", "no_ops": false}
```

### Configuration

- `log_buffer`  --  required; the name of the buffer to append log entries to.
- `no_ops`      --  optional, defaults to false; if true, log even when old and new content are identical.

After the [changelog] option buffer is created, every update to the
origin buffer appends a one-line log entry to the log buffer.

### Log Entry Shape

```
<buffer> updated +3 -1 ~2
```

`+N` = number of lines added, `-M` = number of lines removed,
`~C` = number of lines changed. When old and new content are identical
and `no_ops` is false, no log entry is written.

### Stop Change Logging

Drop the `<buffer>[changelog]` option buffer to remove the hook and stop
logging.
"""


# -------------------------------------------------------------------------------------------------------------------- #
# Utils                                                                                                             #
# -------------------------------------------------------------------------------------------------------------------- #


def _origin_name_from_changelog_option(option_buffer_name: str) -> str | None:
    """Derive the origin buffer name from a [changelog] option buffer name.

    Splits on the last '[changelog]' suffix and returns everything before it.
    Returns None if the buffer name does not end with '[changelog]'.
    """
    if not option_buffer_name.endswith("[changelog]"):
        return None
    return option_buffer_name.removesuffix("[changelog]")


def _get_log_buffer_name(changelog_buffer: Buffer) -> str | None:
    """Read the log buffer name from the changelog config.

    Returns the log buffer name on success, or None if absent or invalid.
    """
    text = changelog_buffer.get_text()
    if not text or not text.strip():
        return None

    try:
        import json
        config = json.loads(text)
    except json.JSONDecodeError:
        return None

    return config.get("log_buffer") or None


# -------------------------------------------------------------------------------------------------------------------- #
# Buffer hooks                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #


def _before_create_changelog(
    bm: ChangeLogger,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Guard the [changelog] option buffer against invalid JSON.

    The buffer must be a JSON object with a "log_buffer" key.
    """
    # Reject empty buffer content.
    text = new_buffer.get_text()
    if not text or not text.strip():
        return "changelog option buffer is empty  --  a 'log_buffer' field is required"

    # Reject non-JSON content.
    try:
        import json
        config = json.loads(text)
    except json.JSONDecodeError:
        return "changelog buffer is not valid JSON"

    # Reject non-dict content.
    if not isinstance(config, dict):
        return "changelog buffer must be a JSON object"

    # Reject a missing or empty log_buffer field.
    if not config.get("log_buffer"):
        return "changelog buffer requires a 'log_buffer' field"

    # Reject non-boolean no_ops field.
    if "no_ops" in config and not isinstance(config["no_ops"], bool):
        return "no_ops must be a boolean"

    return None


async def _after_create_changelog(
    changelog: ChangeLogger,
    option_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after a [changelog] option buffer is created.

    Registers the AFTER_UPDATE hook on the origin buffer so every update
    appends a log entry to the configured target.
    """
    # Derive the origin buffer name from the option buffer name.
    origin = _origin_name_from_changelog_option(option_buffer.name)
    if origin is None:
        _logger.error(f"[ChangeLogger] cannot derive origin from '{option_buffer.name}'")
        return

    # Guard that the origin buffer exists so the hook can be registered.
    origin_buffer = changelog.get_buffer(origin)
    if origin_buffer is None:
        _logger.error(f"[ChangeLogger] origin buffer '{origin}' does not exist")
        return

    # Register the AFTER_UPDATE hook on the origin buffer so every update
    # appends a log entry to the target configured in the changelog option buffer.
    origin_buffer.register_hook(
        HookPoint.AFTER_UPDATE_BUFFER,
        "changelog_update_origin",
        _after_update_origin,
    )


async def _after_update_origin(
    changelog: ChangeLogger,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after the origin buffer is updated.

    Reads the changelog config, extracts the target buffer name, and appends
    a log entry to it.
    """
    # Read the changelog option buffer to get the target buffer name.
    changelog_option = changelog.get_buffer(f"{new_buffer.name}[changelog]")
    if changelog_option is None:
        _logger.error(f"[ChangeLogger] changelog option buffer for '{new_buffer.name}' is gone  --  discarding log entry")
        return

    log_buffer_name = _get_log_buffer_name(changelog_option)
    if log_buffer_name is None:
        _logger.error(f"[ChangeLogger] log_buffer name is absent or invalid in changelog for '{new_buffer.name}'  --  discarding log entry")
        return

    # Guard that the old and new content are not identical, unless no_ops is True.
    import json
    config = json.loads(changelog_option.get_text())
    old_text = (old_buffer.get_text() or "") if old_buffer else ""
    new_text = new_buffer.get_text() or ""
    if old_text == new_text and not config.get("no_ops", False):
        return

    # Create the log entry as a plain text line with delta counts.
    import difflib
    matcher = difflib.SequenceMatcher(None, old_text.splitlines(), new_text.splitlines())
    added = sum(
        j2 - j1 for tag, _, _, j1, j2 in matcher.get_opcodes()
        if tag == "insert"
    )
    removed = sum(
        i2 - i1 for tag, i1, i2, _, _ in matcher.get_opcodes()
        if tag == "delete"
    )
    changed = sum(
        min(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag == "replace"
    )
    log_entry = f"{new_buffer.name} updated +{added} -{removed} ~{changed}\n"

    # Ensure the log buffer exists before writing to it.
    if changelog.get_buffer(log_buffer_name) is None:
        await changelog.create_stream_buffer(log_buffer_name)

    # Write the log entry to the log buffer.
    await changelog.write_buffer(log_buffer_name, text=log_entry)


# -------------------------------------------------------------------------------------------------------------------- #
# ChangeLogger                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #


class ChangeLogger(StreamBufferManager, OptionBufferManager, AgenticObject):
    """I append log entries to a target buffer whenever the origin buffer changes.

    The "doc:changelog" buffer holds the details.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Seed the doc buffer with the module documentation.
        self._create_buffer(_CHANGELOG_DOC_BUFFER, text=_CHANGELOG_DOC)

        # Register the pattern hook for [changelog] option buffers.
        # On AFTER_CREATE: register the AFTER_UPDATE origin hook and seed the changelog config.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "changelog_created",
            r"^.+\[changelog\]$",
            _after_create_changelog,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "changelog_requires_json",
            r"^.+\[changelog\]$",
            _before_create_changelog,
            apply_to_existing=True,
        )
