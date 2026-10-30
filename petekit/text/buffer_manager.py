from __future__ import annotations
import asyncio
import difflib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from typing import Literal

from .buffer import (
    HookPoint,
    BeforeCreateBuffer,
    AfterCreateBuffer,
    BeforeUpdateBuffer,
    AfterUpdateBuffer,
    BeforeDropBuffer,
    AfterBufferDropped,
    AnyHook,
    BufferEntry,
    Buffer,
    _cluster_lines_to_ranges,
    _resolve_line_range,
    _resolve_line_number,
)


from peteos.oap.agentic_object import AgenticObject
from peteos import tool, sandbox
from peteos.utils import get_logger
from peteos.utils.watchdog import Watchdog
from peteos.utils.watchdog import FileSystemEvent

_logger = get_logger(__name__)


# --------------------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------------------


# Managed buffer names — system:status: namespace and diff namespace
_STATUS_BUFFER = "system:status:buffers"
_DIFF_BUFFER_PREFIX = "diff:"

_NO_BUFFER_ = object()


# --------------------------------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------------------------------


@dataclass
class ReadBufferResult:
    """Result of _read_buffer. Signals which branch was taken and carries data for hint construction."""
    kind: Literal["content", "skip", "bucket", "error"]
    content: str | None = None
    start: int | None = None
    end: int | None = None
    total_chars: int = 0
    line_count: int = 0
    skip_lines: list[tuple[int, int]] | None = None  # (0-based line index, char count)
    bucket_info: list[tuple[int, int, int]] | None = None  # (start, end, chars)
    error: str | None = None


@dataclass
class GrepResult:
    """Result of Buffer.grep and _grep_buffer. Carries raw match data for downstream processing."""
    ok: bool
    matches: list[int] | None = None
    start: int | None = None
    end: int | None = None
    count: int = 0
    error: str | None = None


@dataclass
class PatternHook:
    """A single pattern-based hook registered on the registry.

    buffer_pattern is a regex; a buffer name matches when re.fullmatch(pattern, buffer_name) is true.
    callable is a hook of type AnyHook — callable signature depends on its hook point.
    """
    name: str
    buffer_pattern: str
    compiled_pattern: re.Pattern
    callable: AnyHook


@dataclass
class HookPointRegistry:
    """Registry of pattern-based hooks for one hook point.

    hooks_by_name maps a unique hook name to its PatternHook.
    """
    hooks_by_name: dict[str, PatternHook] = field(default_factory=dict)

    def _sort_hooks_for(self, buffer_name: str) -> list[PatternHook]:
        """Return the hooks whose buffer_pattern matches the given buffer name.

        Filters to hooks whose pattern fully matches the buffer name, then sorts
        alphanumerically — ordering is not semantically meaningful; use explicit
        priority fields when order matters.
        """
        # Filter to hooks whose pattern matches the buffer name in its entirety.
        matching_hooks = [
            hook for hook in self.hooks_by_name.values()
            if re.fullmatch(hook.compiled_pattern, buffer_name) is not None
        ]

        # Sort by buffer_pattern alphanumerically — deterministic tie-break only.
        matching_hooks.sort(key=lambda h: h.buffer_pattern)

        return matching_hooks


###############################################################################
# Buffers-buffer maintenance hooks
# Keep the system:status:buffers JSON listing consistent after any buffer mutation.
###############################################################################

async def _refresh_buffers_buffer_on_mutation(
    buffer_manager,
    old_buffer: Buffer | None,
    new_buffer: Buffer | None,
    args: dict[str, Any],
) -> None:
    """Keep the system:status:buffers registry current after any buffer structural change.

    Incremental update: patches the JSON listing in-place rather than rebuilding
    from scratch. Avoids the extra create_buffer call and its attendant hook storms.

    On AFTER_CREATE_BUFFER:  old_buffer is None, new_buffer is the freshly created buffer.
    On AFTER_DROP_BUFFER:    old_buffer is the dropped buffer, new_buffer is None.
    On AFTER_UPDATE_BUFFER:  old_buffer is pre-mutation (with prior line count),
                              new_buffer is post-mutation (with current line count).

    The handler never recurses: if the buffer being mutated is system:status:buffers
    itself, the refresh is a no-op to prevent the listing refresh from triggering itself.
    """
    # Guard: never recurse on the registry buffer itself.
    if (new_buffer is not None and new_buffer.name == _STATUS_BUFFER) or \
       (old_buffer is not None and old_buffer.name == _STATUS_BUFFER):
        return

    # Pull the current registry buffer — created at init so it must exist.
    status_buffer = buffer_manager.get_buffer(_STATUS_BUFFER)
    if status_buffer is None:
        import logging
        logging.error(
            f"[buffers_buffer] system:status:buffers buffer not found in manager "
            f"after mutation hook — expected it to exist since it is created at init. "
            f"The status buffer may have been dropped or corrupted."
        )
        return

    # Parse the current JSON listing. An empty file is a valid start state.
    current_text = status_buffer.get_text()
    try:
        records = json.loads(current_text) if current_text.strip() else []
    except json.JSONDecodeError:
        records = []

    # Build a lookup dict for O(1) mutations without linear scan.
    records_by_name = {record["name"]: record for record in records}

    # Apply the incremental mutation based on which hook point fired.
    if new_buffer is not None:
        # AFTER_CREATE_BUFFER or AFTER_UPDATE_BUFFER — add or overwrite the entry.
        records_by_name[new_buffer.name] = {
            "name": new_buffer.name,
            "lines": len(new_buffer.lines),
        }

    elif old_buffer is not None:
        # AFTER_DROP_BUFFER — remove the dropped buffer's entry.
        records_by_name.pop(old_buffer.name, None)

    else:
        # Neither buffer changed — nothing to do.
        return

    # Sort for deterministic output and serialize back to text.
    sorted_records = sorted(records_by_name.values(), key=lambda r: r["name"])
    new_text = json.dumps(sorted_records, indent=2)

    # Write the updated listing to the status buffer. The AFTER_UPDATE_BUFFER hook on the
    # status buffer fires, but the recursion guard at the top of this handler returns
    # early since both new_buffer and old_buffer are the status buffer.
    result = await buffer_manager.create_buffer(_STATUS_BUFFER, text=new_text, overwrite=True)
    if not result.get("ok"):
        raise RuntimeError(
            f"failed to refresh buffer registry: "
            f"create_buffer failed: {result.get('error', 'unknown')}"
        )


def _diff_buffer_name(a: str, b: str) -> str:
    # Derive the canonical diff buffer name from two buffer identifiers.
    return f"{_DIFF_BUFFER_PREFIX}{a}→{b}"


def _sanitize_buffer_name(name: str) -> str:
    """Remove ASCII whitespace from a buffer name segment. Use this for any name derived from agent input."""
    return "".join(c for c in name if c not in " \t\n\r")


# --------------------------------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------------------------------

def _manifest_path_from_buffer_path(buffer_path: Path) -> Path:
    """Given a buffer file path, return the path to its corresponding manifest (.json) file.

    The manifest uses a dot-prefix naming convention so it appears as a hidden file
    alongside the named buffer file on disk.
    """
    return buffer_path.with_name("." + buffer_path.name + ".json")


def _is_hidden_file(buffer_path: Path) -> bool:
    """Return True when the file name is hidden on Unix (starts with a dot).

    This is the naming convention for buffer manifest files on disk.
    """
    return buffer_path.name.startswith(".")


def _buffer_name_from_event_path(event_src_path: str, persist_dir: Path) -> str:
    """Return the buffer name extracted from a watchdog event src_path.

    Strips the persistence directory prefix from the full file path and returns
    the relative remainder as the buffer name. For example:
        event_src_path = '/persist_dir/foo'  -> 'foo'
        event_src_path = '/persist_dir/sub/bar' -> 'sub/bar'
    """
    return str(Path(event_src_path).relative_to(persist_dir))


def _try_read_manifest(buffer_path: Path) -> dict | None:
    """Return the manifest dict when the file is a valid buffer manifest, or None.

    A valid manifest has a name starting with dot and ending with .json and
    contains both created_at and modified_at.
    """
    # Build the manifest path from the buffer path and check it exists.
    manifest_path = _manifest_path_from_buffer_path(buffer_path)
    if not manifest_path.exists():
        return None

    # Validate the naming convention -- must start with dot and end with .json.
    manifest_name = manifest_path.name
    if not (manifest_name.startswith(".") and manifest_name.endswith(".json")):
        return None

    # Parse the manifest JSON -- return None when the file is not valid JSON.
    import json
    try:
        content = json.loads(manifest_path.read_text())
    except Exception:
        return None

    # Confirm the manifest carries the required timestamp fields.
    if not isinstance(content, dict):
        return None
    if "created_at" not in content or "modified_at" not in content:
        return None

    return content


def _buffer_timestamps_from_disk(buffer_path: Path) -> dict[str, float]:
    """Return the buffer timestamps read from a file on disk.

    On Linux, st_ctime is the inode change time which is set to now on file creation,
    so it serves as the creation time proxy. On macOS, st_birth is the true creation time.
    """
    import os
    stat = buffer_path.stat()
    return {
        "created_at": getattr(stat, "st_birth", None) or stat.st_ctime,
        "modified_at": stat.st_mtime,
    }


# --------------------------------------------------------------------------------------------------
# Buffer persistence
# --------------------------------------------------------------------------------------------------

def _get_persistence_dir(bm: BufferManager) -> Path:
    """Return the BufferManager persistence directory inside the buffer manager's instance dir."""
    return bm.instance_dir / "buffer_manager"


async def _restore_create_buffers(self: BufferManager) -> None:
    """Restore hook: re-creates buffers from persisted state after instance restoration.

    Lists the buffer manager's persistence directory for buffer JSON manifests,
    validates each, and re-creates the buffer with its original timestamps.
    """
    # Ensure the persistence directory exists — nothing to restore if it was never persisted.
    dir = _get_persistence_dir(self)
    if not dir.exists():
        return

    # Find all hidden JSON manifest files, each representing a single persisted buffer.
    manifest_files = sorted(dir.glob("**/*.json"))
    valid_manifests: list[tuple[float, Path]] = []

    # Collect only valid buffer manifests and sort by creation time — oldest first.
    for manifest_path in manifest_files:
        # Parse the JSON manifest; skip on any read/parse error.
        import json
        try:
            meta = json.loads(manifest_path.read_text())
        except Exception:
            continue

        # Guard: skip files that do not carry the buffer kind marker.
        if not isinstance(meta, dict) or meta.get("kind") != "buffer":
            continue

        # Extract buffer identity and timestamps from the manifest.
        name = meta.get("name")
        created_at = meta.get("created_at")
        if not name or not created_at:
            continue

        # Record this manifest as valid for the restore pass.
        valid_manifests.append((created_at, manifest_path))

    # Restore in creation-time order — oldest buffers first.
    valid_manifests.sort(key=lambda x: x[0])

    # Restore each buffer from its manifest — oldest first.
    for _, manifest_path in valid_manifests:
        # Re-parse the manifest and extract all required fields for restoration.
        import json
        try:
            meta = json.loads(manifest_path.read_text())
        except Exception:
            continue
        name = meta.get("name")
        created_at = meta.get("created_at")
        modified_at = meta.get("modified_at")

        # Guard: skip if the content file is absent — nothing to restore.
        content_path = dir / name
        if not content_path.exists():
            continue

        # Load the raw buffer content from the corresponding content file.
        text = content_path.read_text()

        # Re-create the buffer with its original timestamps.
        metadata = {"created_at": created_at, "modified_at": modified_at or created_at}
        try:
            await _create_buffer_implementation(self, name, text, False, metadata)
        except Exception as e:
            # Best-effort: non-critical restore — log and continue.
            import logging
            logging.warning(f"[_restore_create_buffers] failed to restore buffer '{name}': {e}")

        # Reconcile: the status buffer was created with an empty listing. Populate it now
        # with an entry for every buffer currently in the manager, including itself.
        # This is not recursive — we write directly via create_buffer, not through a hook.
        records = [{"name": name, "lines": len(buffer)} for name, buffer in self._buffers.items()]
        status_text = json.dumps(records, indent=2)
        result = self._create_buffer(_STATUS_BUFFER, text=status_text)
        if not result.get("ok"):
            raise RuntimeError(
                f"failed to seed status buffer: {result.get('error', 'unknown')}"
            )

    # Register a watch for created events so disk file creation syncs to a memory buffer.
    async def _dispatch_created(event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        await _handle_watchdog_created(self, event)

    # Set up a recursive watch on the persistence directory for created file events.
    await self._watchdog.watch(
        name="buffer_file_created",
        path=_get_persistence_dir(self),
        callback=_dispatch_created,
        events={"created"},
        recursive=True,
    )

    # Register a watch for modified events so disk file changes sync to the memory buffer.
    async def _dispatch_modified(event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        await _handle_watchdog_modified(self, event)

    # Set up a recursive watch on the persistence directory for modified file events.
    await self._watchdog.watch(
        name="buffer_file_modified",
        path=_get_persistence_dir(self),
        callback=_dispatch_modified,
        events={"modified"},
        recursive=True,
    )

    # Register a watch for deleted events so disk file removal drops the memory buffer.
    async def _dispatch_deleted(event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        await _handle_watchdog_deleted(self, event)

    # Set up a recursive watch on the persistence directory for deleted file events.
    await self._watchdog.watch(
        name="buffer_file_deleted",
        path=_get_persistence_dir(self),
        callback=_dispatch_deleted,
        events={"deleted"},
        recursive=True,
    )


def _persist_buffer(bm: BufferManager, buffer: Buffer) -> None:
    """Write buffer content and metadata to disk for persistence.

    Stores the raw line data in a file and structural metadata in a .json file.
    The meta file is written first so that a watchdog watching the content file
    sees a complete pair on disk — not a half-written state that could cause a
    false create event.
    """
    # Ensure the persistence directory for this buffer manager exists.
    dir = _get_persistence_dir(bm)
    dir.mkdir(parents=True, exist_ok=True)

    # Build the paths for content and metadata files.
    buffer_path = dir / buffer.name
    meta_path = buffer_path.with_name("." + buffer_path.name + ".json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    # Write metadata first — a watchdog seeing only the content file is a
    # no-op since the meta (the authoritative state marker) is not yet present.
    import json
    meta = {
        "kind": "buffer",
        "name": buffer.name,
        "line_count": len(buffer.lines),
        "created_at": buffer.created_at,
        "modified_at": buffer.modified_at,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    # Write content second — the meta file already exists, so a watchdog
    # create event for this file is ignored.
    buff_path = dir / f"{buffer.name}"
    buff_path.parent.mkdir(parents=True, exist_ok=True)
    buff_path.write_text(buffer.get_text())


def _drop_persisted_buffer(bm: BufferManager, buffer_name: str) -> None:
    """Remove persisted content and metadata for the named buffer.

    The meta file is deleted first so that a watchdog watching the content file
    sees the meta gone first — a delete event for the content file alone is
    ignored since the authoritative state marker is already gone.
    """
    dir = _get_persistence_dir(bm)

    # Delete meta first — a watchdog seeing only the content file deleted is
    # a no-op since the meta (the authoritative state marker) is not present.
    (dir / f".{buffer_name}.json").unlink(missing_ok=True)

    # Delete content second — the meta is already gone, so this delete event
    # is ignored by the watchdog.
    (dir / f"{buffer_name}").unlink(missing_ok=True)


def _make_persist_buffer_hook(bm: BufferManager) -> AfterUpdateBuffer:
    """Factory that returns a post-update hook that persists the buffer on every modification.

    Closes over bm so the hook can call _persist_buffer with the correct BufferManager.
    """
    def _persist_buffer_update_hook(
        buffer_manager: BufferManager,
        old_buffer: Buffer,
        new_buffer: Buffer,
        args: dict[str, Any],
    ) -> None:
        # Persist the buffer to disk — called after every successful mutation.
        _persist_buffer(buffer_manager, new_buffer)
    return _persist_buffer_update_hook


# Disk-originated buffer events -- handlers are filled in during later steps.
async def _handle_watchdog_created(bm: BufferManager, event: FileSystemEvent) -> None:
    """Recreate a buffer in memory when a new file appears on disk."""
    # Extract the file path from the event -- this is the buffer file that appeared.
    buffer_path = Path(event.src_path)

    # Skip manifest files -- they are a side-effect of persist_buffer, not a
    # separate buffer event to act on.
    if _is_hidden_file(buffer_path):
        return

    # Extract the buffer name from the file path -- the file name is the buffer name.
    buffer_name = _buffer_name_from_event_path(event.src_path, _get_persistence_dir(bm))

    # Skip when the buffer already exists in memory -- no need to recreate it.
    if buffer_name in bm._buffers:
        return

    # Skip when a manifest already exists -- this file was likely recreated after
    # a failed create and is not a fresh disk-originated event.
    manifest = _try_read_manifest(buffer_path)
    if manifest is not None:
        return

    # Read the file content and timestamps from disk -- the file content becomes
    # the buffer text, the stat timestamps become the buffer created_at/modified_at.
    file_text = buffer_path.read_text()
    metadata = _buffer_timestamps_from_disk(buffer_path)

    # Create the buffer in memory, passing the disk timestamps through.
    result = await _create_buffer_implementation(
        bm,
        buffer_name,
        file_text,
        False,
        metadata,
    )

    # Report the error when the create was rejected -- the manifest must not be
    # written so that a future fix to the file can succeed.
    if isinstance(result, dict) and not result.get("ok", False):
        _logger.error(
            f"[_handle_watchdog_created] failed to create buffer '{buffer_name}': "
            f"{result.get('error', 'unknown')}"
        )
        return


async def _handle_watchdog_modified(bm: BufferManager, event: FileSystemEvent) -> None:
    """Update a buffer in memory when its file changes on disk."""
    # Extract the file path from the event -- this is the buffer file that changed.
    buffer_path = Path(event.src_path)

    # Skip manifest files -- they are a side-effect of persist_buffer, not a
    # separate buffer event to act on.
    if _is_hidden_file(buffer_path):
        return

    # Extract the buffer name from the file path -- use the path relative to the
    # persistence directory so that subdirectory paths become part of the buffer name.
    buffer_name = _buffer_name_from_event_path(event.src_path, _get_persistence_dir(bm))

    # Route to the create handler when no manifest exists -- a prior create may
    # have been rejected, leaving the file on disk but no buffer in memory.
    manifest = _try_read_manifest(buffer_path)
    if manifest is None:
        await _handle_watchdog_created(bm, event)
        return

    # Read the new content from the disk file -- this is what the buffer will be
    # updated to.
    disk_text = buffer_path.read_text()

    # Get the current buffer text from memory -- this is the original state that
    # the patch will be applied to.
    current_buffer = bm.get_buffer(buffer_name)
    current_text = current_buffer.get_text() if current_buffer else ""

    # Skip when the buffer in memory already matches the file on disk -- the disk
    # event was likely triggered by our own persist write, not an external change.
    if current_text == disk_text:
        return

    # Build a unified diff from the current (old) text to the disk (new) text.
    # The diff format is understood by patch_buffer so it can be applied reliably.
    import difflib
    diff = difflib.unified_diff(
        current_text.splitlines(keepends=True),
        disk_text.splitlines(keepends=True),
        fromfile=buffer_name,
        tofile=buffer_name,
    )
    patch_text = "".join(diff)

    # Apply the patch to the buffer -- the BEFORE_UPDATE_BUFFER hooks run as normal.
    # A rejection means the buffer state is protected and the disk content is stale.
    patch_result = await bm.patch_buffer(buffer_name, patch_text)

    # Report the error when the patch was rejected -- the disk file is now out of
    # sync with memory. No further action is taken; the file must be corrected on
    # disk and the event will fire again.
    if isinstance(patch_result, dict) and not patch_result.get("ok", False):
        _logger.error(
            f"[_handle_watchdog_modified] patch rejected for '{buffer_name}': "
            f"{patch_result.get('error', 'unknown')}"
        )


async def _handle_watchdog_deleted(bm: BufferManager, event: FileSystemEvent) -> None:
    """Drop a buffer from memory when its file is removed from disk."""
    # Extract the file path from the event -- this is the buffer file that disappeared.
    buffer_path = Path(event.src_path)

    # Skip manifest files -- they are a side-effect of persist_buffer, not a
    # separate buffer event to act on.
    if _is_hidden_file(buffer_path):
        return

    # Extract the buffer name from the file path -- use the path relative to the
    # persistence directory so that subdirectory paths become part of the buffer name.
    buffer_name = _buffer_name_from_event_path(event.src_path, _get_persistence_dir(bm))

    # Skip when no manifest exists -- the buffer may have already been dropped or
    # this is not a buffer file at all.
    manifest = _try_read_manifest(buffer_path)
    if manifest is None:
        return

    # Attempt to drop the buffer from memory -- the disk sync flag in metadata tells
    # any pre-drop hooks that this is a disk-originated drop.
    drop_args = {"_op": "drop_buffer", "name": buffer_name}
    drop_result = await bm.drop_buffer(buffer_name)

    # When the drop was rejected, restore the buffer file so the buffer on disk and
    # in memory stay in sync. Re-create the file with the content from the manifest.
    if isinstance(drop_result, dict) and not drop_result.get("ok", False):
        _logger.error(
            f"[_handle_watchdog_deleted] pre-drop hook rejected '{buffer_name}': "
            f"{drop_result.get('error', 'unknown')} -- restoring file on disk"
        )
        restored_text = buffer_path.read_text()
        buffer_path.write_text(restored_text)


# --------------------------------------------------------------------------------------------------
# Buffer hooks
# --------------------------------------------------------------------------------------------------


def _initialize_buffer_hooks(
    bm: BufferManager,
    buffer: Buffer,
    hook_point: HookPoint,
) -> None:
    """Initialize the buffer's _hooks from the registry for the given hook point.

    Queries the HookPointRegistry for all pattern-matching hooks and copies each into
    buffer._hooks under the same name. Mirroring is additive — existing hooks on the
    buffer are not overwritten; the registry is not modified.

    This runs after the buffer is in bm._buffers so that the buffer is eligible for
    update/drop hooks. After mirroring, all hook points on this buffer fire uniformly
    through buffer._fire_hooks — no more registry lookups for this buffer.
    """
    if hook_point not in bm._buffer_hooks:
        return
    registry = bm._buffer_hooks[hook_point]

    # Find all pattern-matching hooks for this buffer name.
    matching = registry._sort_hooks_for(buffer.name)
    if hook_point not in buffer._hooks:
        buffer._hooks[hook_point] = {}

    # Copy each matching hook into the buffer — additive, does not overwrite existing.
    for hook in matching:
        if hook.name not in buffer._hooks[hook_point]:
            buffer._hooks[hook_point][hook.name] = hook.callable


async def _create_buffer_implementation(
    bm: BufferManager,
    name: str,
    text: str | None = None,
    overwrite: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a buffer with the given name, text, and timestamps.

    metadata must contain created_at and modified_at. Always called with metadata
    from the tool wrapper or from the restore hook.
    """
    existed = name in bm._buffers

    # Reject new creation if the name is already taken and overwrite is not requested.
    if existed and not overwrite:
        return {
            "ok": False,
            "error": f"Buffer '{name}' already exists. Use overwrite=True to replace it.",
        }

    # Compute timestamps from metadata when provided, otherwise use now for both.
    created_at = metadata.get("created_at", time.time()) if metadata else time.time()
    modified_at = metadata.get("modified_at", created_at) if metadata else created_at

    # Gate the creation or overwrite through pre-creation hooks; a rejection blocks the operation.
    create_args = {
        "_op": "create_buffer",
        "name": name,
        "text": text,
        "overwrite": overwrite,
        "metadata": metadata,
    }

    # Compute the proposed new lines from the provided text.
    new_lines = [
        BufferEntry(data=line, modified_at=modified_at, object_at=modified_at)
        for line in (text or "").split("\n")
    ]

    # For existing buffers: behave as an update — gate through pre-update hooks, modify in place.
    if existed:
        buf = bm._buffers[name]

        # Build the proposed new state as an alias of the buffer with new content.
        # The pre-update hook sees what the buffer will become and can accept or reject.
        buffer_alias = buf.alias()
        buffer_alias.lines = new_lines

        # Gate the overwrite through pre-update hooks; a rejection blocks the replacement.
        update_args = {"_op": "create_buffer", "name": name, "text": text, "overwrite": overwrite}
        pre_result = await buf._fire_hooks(bm, buffer_alias, HookPoint.BEFORE_UPDATE_BUFFER, update_args)
        if pre_result is not True and pre_result is not None:
            return {"ok": False, "error": str(pre_result)}

        # Overwrite the buffer's content in place — keep created_at, update modified_at and lines.
        buf.modified_at = modified_at
        buf.lines, buffer_alias.lines = buffer_alias.lines, buf.lines
        create_result = {"ok": True, "lines": len(buf.lines)}

        # Fire post-update hooks now that the overwrite is committed.
        await buffer_alias._fire_hooks(bm, buf, HookPoint.AFTER_UPDATE_BUFFER, update_args)

    # For new buffers: behave as a creation — gate through pre-create hooks, then create the buffer.
    else:
        # Build a working buffer for pre-create hooks to inspect.
        buf = Buffer(name=name, lines=new_lines, created_at=created_at, modified_at=created_at)

        # Mirror all pattern-based hook points into the buffer before creation hooks fire.
        _initialize_buffer_hooks(bm, buf, HookPoint.BEFORE_CREATE_BUFFER)
        _initialize_buffer_hooks(bm, buf, HookPoint.AFTER_CREATE_BUFFER)
        _initialize_buffer_hooks(bm, buf, HookPoint.BEFORE_UPDATE_BUFFER)
        _initialize_buffer_hooks(bm, buf, HookPoint.AFTER_UPDATE_BUFFER)
        _initialize_buffer_hooks(bm, buf, HookPoint.BEFORE_DROP_BUFFER)
        _initialize_buffer_hooks(bm, buf, HookPoint.AFTER_DROP_BUFFER)
        _initialize_buffer_hooks(bm, buf, HookPoint.BEFORE_READ_BUFFER)
        _initialize_buffer_hooks(bm, buf, HookPoint.AFTER_READ_BUFFER)

        # Fire pre-create hooks — a rejection blocks the creation.
        pre_result = await buf._fire_hooks(bm, None, HookPoint.BEFORE_CREATE_BUFFER, create_args)
        if pre_result is not True and pre_result is not None:
            return {"ok": False, "error": str(pre_result)}

        # Add the buffer to the manager's dict.
        # Add the buffer to the manager's dict.
        bm._buffers[name] = buf

        # Create and register the persist hook on the buffer.
        persist_hook = _make_persist_buffer_hook(bm)
        buf.register_hook(HookPoint.AFTER_UPDATE_BUFFER, "persist_buffer", persist_hook)

        # Fire persist immediately for the fresh buffer.
        persist_hook(bm, buf, buf, create_args)

        # Commit and report success with the line count.
        create_result = {"ok": True, "lines": len(buf.lines)}

        # Fire post-creation hooks — a rejection is logged but does not block the committed buffer.
        post_result = await buf._fire_hooks(bm, None, HookPoint.AFTER_CREATE_BUFFER, create_args)
        if post_result is not True and post_result is not None:
            post_result = str(post_result)

    # Signal whether this was a fresh creation or an overwrite, and how many lines were written.
    return {"ok": True, "created": not existed, "overwritten": existed, "lines": create_result.get("lines", 0)}


def _lines_to_skip(segment: list[str], start_offset: int, max_chars: int) -> tuple[list[tuple[int, int]], int]:
    """Find the minimum set of longest lines whose removal makes the segment fit under max_chars.

    Sorts lines longest-first, then greedily removes them until the remaining total fits.
    Returns a list of (1-based line index, char_count) to skip, and the char count of
    what remains.
    """
    # Index each line with its absolute position and sort longest-first.
    lines_with_idx = [(start_offset + i, l) for i, l in enumerate(segment)]
    by_len = sorted(lines_with_idx, key=lambda x: len(x[1]), reverse=True)
    seg_total = sum(len(l) for l in segment) + len(segment)

    # Greedily skip longest lines until the segment fits under the threshold.
    skip = []
    skip_total_chars = 0
    for idx, line in by_len:
        if seg_total - skip_total_chars <= max_chars:
            break
        skip.append((idx, len(line)))
        skip_total_chars += len(line) + 1

    return (skip, seg_total - skip_total_chars)


def _make_buckets(segment: list[str], num_buckets: int, start_offset: int) -> list[tuple[int, int, int]]:
    """Divide a segment into num_buckets equal-size buckets.

    Each bucket is described by its start line index, end line index, and total char
    count. All indices are absolute (1-based relative to the full buffer).
    """
    n = len(segment)
    if n == 0:
        return []
    bucket_size = n // num_buckets

    # Partition the segment into equal-sized buckets, the last bucket absorbing any remainder.
    buckets = []
    for b in range(num_buckets):
        start = b * bucket_size
        end = n if b == num_buckets - 1 else start + bucket_size
        lines = segment[start:end]
        chars = sum(len(l) + 1 for l in lines)
        buckets.append((start_offset + start, start_offset + end - 1, chars))
    return buckets


# --------------------------------------------------------------------------------------------------
# Main class
# --------------------------------------------------------------------------------------------------


class BufferManager(AgenticObject):
    """I am a buffer manager. I hold multiple named buffers, each a list of lines in memory.
    - I can `create`, insert/append (`write`), search (`grep`), `read` ranges from, and edit (`edit` and `patch`) any named buffer.
    - Read the "system:status:buffers" buffer to see what exists. Drop unused buffers when it becomes messy!
      Read it to get a JSON array of {name, lines} for each buffer.
    - Line indices are 0-based, just like Python array indexing.
      Example: buf[0] is the first line, buf[-1] is the last line, buf[0:5] is the first 5 lines.
    - Ranges use [start, end) semantics: start is included, end is excluded. Omit end to read to the end of the buffer.
    - Use (?i) at the start of a grep pattern for case-insensitive matching.
    - All timestamps are rounded to 6 decimals.
    - **Be efficient!** Read only as much as you need instead of the entire buffer!
    - **Be careful!** Read before you write! Don't delete managed buffers.
    """

    _MAX_CHUNK_CHARS = 8000
    _NUM_CLUSTERS = 5

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # Named buffers — the primary store for all managed buffers.
        self._buffers: dict[str, Buffer] = {}

        # Pattern-based hook registries keyed by hook point.
        # Each registry holds creation hooks to be mirrored to a buffer on first creation.
        self._buffer_hooks: dict[HookPoint, HookPointRegistry] = {
            HookPoint.BEFORE_CREATE_BUFFER: HookPointRegistry(),
            HookPoint.AFTER_CREATE_BUFFER: HookPointRegistry(),
            HookPoint.BEFORE_UPDATE_BUFFER: HookPointRegistry(),
            HookPoint.AFTER_UPDATE_BUFFER: HookPointRegistry(),
            HookPoint.BEFORE_DROP_BUFFER: HookPointRegistry(),
            HookPoint.AFTER_DROP_BUFFER: HookPointRegistry(),
            HookPoint.BEFORE_READ_BUFFER: HookPointRegistry(),
            HookPoint.AFTER_READ_BUFFER: HookPointRegistry(),
        }

        # Bootstrap the system status buffer — this must exist before any agent runs.
        self._create_buffer(_STATUS_BUFFER, text="[]")

        # Register the buffers-buffer maintenance hooks as pattern hooks so every buffer
        # created from this point forward automatically participates in keeping the
        # registry current. apply_to_existing=True handles the status buffer right now.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "maintain_buffers_buffer",
            ".*",
            lambda buffer_manager, buffer, args: _refresh_buffers_buffer_on_mutation(
                buffer_manager, None, buffer, args
            ),
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_UPDATE_BUFFER,
            "maintain_buffers_buffer",
            ".*",
            _refresh_buffers_buffer_on_mutation,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            "maintain_buffers_buffer",
            ".*",
            lambda buffer_manager, buffer, args: _refresh_buffers_buffer_on_mutation(
                buffer_manager, buffer, None, args
            ),
            apply_to_existing=True,
        )

        # Register the restore hook so persisted buffers survive instance restarts.
        self._register_restore_hook("create_buffers", _restore_create_buffers)

        # Own a Watchdog that watches the persistence directory for disk-originated
        # buffer events so the BufferManager can sync disk changes back into memory.
        # Named with the BufferManager's identity so logs are traceable.
        self._watchdog = Watchdog(name=f"BufferManager/{hex(id(self))}")

    def get_buffer(self, name: str) -> Buffer | None:
        """Protected buffer lookup — returns the Buffer object or None if not found.

        Not agent-facing. Used by subclasses and internal code that needs direct
        buffer access without the dict-wrapping overhead of a tool response.
        """
        return self._buffers.get(name)

    def _create_buffer(self, name: str, text: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Protected buffer creation for initialization only.
        Always overwrites. Does not fire hooks.
        Used by __init__ and constructors — not for agent-facing operations.
        """
        if metadata is None:
            metadata = {}

        # Validate the buffer name contains no ASCII whitespace; unsanitized names corrupt storage and lookups.
        sanitized = _sanitize_buffer_name(name)
        if sanitized != name:
            return {
                "ok": False,
                "error": (
                    f"Buffer name '{name}' contains illegal whitespace characters. "
                    f"Buffer names must be continuous — no spaces, tabs, or newlines. "
                    f"Example of a valid name: '{sanitized}'"
                ),
            }

        created_at = metadata.get("created_at", time.time())
        modified_at = metadata.get("modified_at", created_at)
        buf = Buffer(name=name, lines=[], created_at=created_at, modified_at=modified_at)
        self._buffers[name] = buf
        lines = 0

        if text:
            buf.lines = [
                BufferEntry(
                    data=line,
                    modified_at=created_at,
                    object_at=created_at
                )
                for line in text.split("\n")
            ]
            lines = len(buf.lines)

        # Register the persist hook so every subsequent modification is written to disk.
        persist_buffer_hook = _make_persist_buffer_hook(self)
        buf.register_hook(HookPoint.AFTER_UPDATE_BUFFER, "persist_buffer", persist_buffer_hook)

        # Persist the freshly created buffer immediately.
        persist_buffer_hook(self, buf, buf, {"_op": "create_buffer", "name": name, "text": text})

        return {"ok": True, "lines": lines}

    @tool
    async def create_buffer(self, name: str, text: str | None = None, overwrite: bool = False) -> dict[str, Any]:
        """Create a new buffer with the given name.
        Optionally provide initial text.
        Use overwrite=True to replace an existing buffer.
        """
        ts = time.time()
        metadata = {"created_at": ts, "modified_at": ts}
        return await _create_buffer_implementation(self, name, text, overwrite, metadata)

    @tool
    async def copy_buffer(self, source_name: str, target_name: str, overwrite: bool = False) -> dict[str, Any]:
        """Copy a buffer to a new name.
        Copies all lines by value. Timestamps of individual lines are preserved.
        Set overwrite=True to replace an existing target buffer.
        """
        if source_name not in self._buffers:
            return {"ok": False, "error": f"No buffer named '{source_name}'."}
        if target_name in self._buffers and not overwrite:
            return {"ok": False, "error": f"Buffer '{target_name}' already exists. Use overwrite=True to replace it."}
        src = self._buffers[source_name]
        src_text = src.get_text()
        result = await self.create_buffer(target_name, text=src_text, overwrite=overwrite)
        if not result.get("ok"):
            return result
        return {"ok": True, "target": target_name, "source": source_name, "lines": result.get("lines", 0)}

    @tool
    async def write_buffer(self, name: str, text: str, pos: int | str = 'end') -> dict[str, Any]:
        """Write into an existing text buffer.
        Default: APPEND
        A trailing newline is always appended, i.e., an empty input creates a blank line in the buffer.
        Set pos to an integer (0-based line index) to insert at that position.
        Negative indices are relative to the end (Python array style).
        """
        # Guard: reject unknown buffer names to keep write operations consistent.
        if name not in self._buffers:
            return {"ok": False, "error": f"No buffer named '{name}'. Use create_buffer first."}
        buf = self._buffers[name]

        # Resolve the line number range and validate it before any content is read.
        total = len(buf.lines)
        resolved = _resolve_line_number(total, pos)
        if isinstance(resolved, dict):
            return resolved
        pos = resolved

        # Produce the alias with the written content; propagate error dicts directly.
        write_result = buf.inserted_buffer(text, pos)
        if not write_result.get("ok"):
            return write_result
        alias = write_result["buffer"]
        count = write_result["count"]

        # Gate the write through pre-update hooks — rejection blocks the insertion.
        write_args = {"_op": "write_buffer", "name": name, "text": text, "pos": pos, "start": pos, "end": pos}
        hook_result = await buf._fire_hooks(
            self,
            alias,
            HookPoint.BEFORE_UPDATE_BUFFER,
            write_args,
        )
        if hook_result is not True and hook_result is not None:
            return {"ok": False, "error": str(hook_result)}

        # Swap the buffer contents in place — preserving the object reference for any callers holding it.
        buf.lines, alias.lines = alias.lines, buf.lines
        buf.modified_at, alias.modified_at = alias.modified_at, buf.modified_at

        # Fire post-update hooks: `alias` is the buffer that was updated,
        # `buf` now holds the new state and is passed as `new_buffer`.
        await alias._fire_hooks(
            self,
            buf,
            HookPoint.AFTER_UPDATE_BUFFER,
            write_args,
        )

        return {"ok": True, "lines_written": count, "total_lines": len(buf.lines)}

    @tool
    async def drop_buffer(self, name: str) -> dict[str, Any]:
        """Drop (delete) a buffer by name.
        The buffer and its content are discarded.
        """
        # Guard: reject unknown buffer names to keep drop operations consistent.
        if name not in self._buffers:
            return {"ok": False, "error": f"No buffer named '{name}'."}
        buf = self._buffers[name]

        # Guard: the system listing buffer may not be dropped.
        if name == _STATUS_BUFFER:
            return {"ok": False, "error": f"Cannot drop the '{_STATUS_BUFFER}' buffer."}

        # Capture content before deletion — needed by BEFORE_DROP_BUFFER and AFTER_DROP_BUFFER.
        old_text = buf.get_text()

        # Fire pre-drop hooks — a rejection blocks the drop.
        drop_args = {"_op": "drop_buffer", "name": name, "start": 0, "end": len(buf.lines), "old_text": old_text}
        drop_guard = await buf._fire_hooks(
            self,
            None,
            HookPoint.BEFORE_DROP_BUFFER,
            drop_args,
        )
        if drop_guard is not True and drop_guard is not None:
            return {"ok": False, "error": str(drop_guard)}

        # Delete the buffer from the manager — it is now invisible to all code.
        del self._buffers[name]

        # Fire AFTER_DROP_BUFFER post-drop hooks — buffer is gone from the manager.
        # The buffer object itself is still alive for the duration of this call.
        await buf._fire_hooks(self, None, HookPoint.AFTER_DROP_BUFFER, drop_args)

        # Remove the persisted files for this buffer now that the drop has succeeded.
        _drop_persisted_buffer(self, name)

        return {"ok": True, "dropped": name}

    def register_buffer_pattern_hook(
        self,
        hook_point: HookPoint,
        hook_name: str,
        pattern: str,
        hook: (
            BeforeCreateBuffer
            | BeforeUpdateBuffer
            | BeforeDropBuffer
            | AfterCreateBuffer
            | AfterUpdateBuffer
            | AfterBufferDropped
        ),
        overwrite_existing: bool = False,
        apply_to_existing: bool = False,
    ) -> dict[str, Any]:
        """Register a hook by pattern on the BufferManager's HookPointRegistry.

        The pattern is always treated as a regex and matched with re.fullmatch against
        each buffer name. When a buffer matching the pattern is created, the hook is
        copied into that buffer's _hooks via _initialize_buffer_hooks. The pattern
        registry is never consulted at hook-fire time — only at buffer creation.

        If apply_to_existing is True, the hook is also immediately registered on all
        buffers that already exist and match the pattern. If overwrite_existing is
        also True, existing hooks with the same name are replaced rather than rejected.

        Rejects if hook_name already exists under this hook point in the registry.
        """
        # Reject if the hook point is not a known hook point with a registry.
        if hook_point not in self._buffer_hooks:
            return {
                "ok": False,
                "error": f"no registry for '{hook_point.value}' — hook point not recognised.",
            }
        registry = self._buffer_hooks[hook_point]

        # Check for duplicate hook names under this hook point in the registry.
        if hook_name in registry.hooks_by_name:
            if overwrite_existing:
                # Overwrite is requested — remove the existing entry and reregister.
                del registry.hooks_by_name[hook_name]
            else:
                return {
                    "ok": False,
                    "error": f"hook '{hook_name}' already registered under '{hook_point.value}'. Unregister first, or use overwrite_existing.",
                }

        # Compile and validate the pattern now — fail fast rather than silently never matching.
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            return {"ok": False, "error": f"Invalid buffer regex '{pattern}': {e}."}

        # Store the pattern hook in the registry for future buffer creation events.
        registry.hooks_by_name[hook_name] = PatternHook(
            name=hook_name,
            buffer_pattern=pattern,
            compiled_pattern=compiled,
            callable=hook,
        )

        # Build the success return dict, extending it only when apply_to_existing populates extra data.
        result = {
            "ok": True,
            "hook_point": hook_point.value,
            "buffer_pattern": pattern,
            "hook": hook_name,
        }

        # If apply_to_existing is True, register the hook directly on all buffers that already exist and match the pattern.
        # Track how many matched and how many were actually registered.
        if apply_to_existing:
            matched_buffers = 0
            registered_on_buffers = 0
            for buf in self._buffers.values():
                if re.fullmatch(compiled, buf.name) is not None:
                    matched_buffers += 1
                    reg_result = buf.register_hook(hook_point, hook_name, hook, overwrite_existing=overwrite_existing)
                    if reg_result["ok"]:
                        registered_on_buffers += 1
            result["matched_buffers"] = matched_buffers
            result["registered_on_buffers"] = registered_on_buffers

        return result

    def unregister_buffer_pattern_hook(
        self,
        hook_point: HookPoint,
        hook_name: str,
        pattern: str,
        remove_from_existing: bool = False,
    ) -> dict[str, Any]:
        """Remove a named hook from the BufferManager's HookPointRegistry.

        If remove_from_existing is True, the hook is also removed from all buffers
        that currently exist and match the pattern. Buffets that do not have the
        hook are silently skipped.

        Rejects if the hook point is not recognised or the hook name is not found.
        """
        # Reject if the hook point is not a known hook point with a registry.
        if hook_point not in self._buffer_hooks:
            return {
                "ok": False,
                "error": f"no registry for '{hook_point.value}' — hook point not recognised.",
            }
        registry = self._buffer_hooks[hook_point]

        # Initialise the base result dict — extra keys added only when remove_from_existing applies.
        result = {
            "ok": True,
            "hook_point": hook_point.value,
            "pattern": pattern,
            "removed": hook_name,
        }

        # If remove_from_existing is True, unregister from all buffers that match the pattern.
        if remove_from_existing:
            # Initialise counters to track how many buffers matched and how many actually had the hook removed.
            matched_buffers = 0
            removed_from_buffers = 0

            # Compile the pattern once for fast per-buffer matching.
            compiled = re.compile(pattern)
            
            # Iterate all buffers, unregistering from each one that matches the pattern.
            for buf in self._buffers.values():
                # Skip buffers whose name does not match the pattern.
                if re.fullmatch(compiled, buf.name) is not None:
                    matched_buffers += 1
                    unreg_result = buf.unregister_hook(hook_point, hook_name)
                    # Only count successful removals — a buffer may not have the hook registered.
                    if unreg_result["ok"]:
                        removed_from_buffers += 1

            # Add the counters and flags to the result.
            result["from_buffers"] = removed_from_buffers > 0
            result["from_registry"] = False
            result["matched_buffers"] = matched_buffers
            result["removed_from_buffers"] = removed_from_buffers

        # Reject if the hook name is not registered under this hook point.
        if hook_name not in registry.hooks_by_name:
            if remove_from_existing and removed_from_buffers > 0:
                return result
            return {
                "ok": False,
                "error": f"no hook '{hook_name}' registered under '{hook_point.value}'.",
            }

        # Remove the hook from the registry.
        del registry.hooks_by_name[hook_name]

        # Complete the result dict — for full registry removal, from_buffers reflects the buffer-level outcome.
        # Only add the extra keys when remove_from_existing was set.
        if remove_from_existing:
            result["from_buffers"] = removed_from_buffers > 0
            result["from_registry"] = True

        return result

    @sandbox
    def _grep_buffer(self, name: str, pattern: str, start: int = 0, end: int | str = "end") -> dict[str, Any]:
        """Search a buffer for a regex pattern.
        Searches the full buffer by default, or a [start:end) range if provided.
        Returns a dict with ok/error or ok/matches.
        Use (?i) at the start for case-insensitive matching.
        """
        # Look up the named buffer and delegate grep to Buffer.grep, returning an error if the buffer is missing.
        if name not in self._buffers:
            return {"ok": False, "error": f"No buffer named '{name}'."}
        buf = self._buffers[name]
        result = buf.grep(pattern=pattern, start=start, end=end)
        if not result.ok:
            return {"ok": False, "error": result.error}
        return {
            "ok": True,
            "matches": result.matches,
            "start": result.start,
            "end": result.end,
            "count": result.count,
        }

    @tool
    async def grep_buffer(self, name: str, pattern: str, start: int = 0, end: int | str = "end") -> dict[str, Any]:
        """Search a buffer for a regex pattern.
        Searches the full buffer by default, or a [start:end) range if provided.
        Returns a dict with ok/error or ok/matches.
        Use (?i) at the start for case-insensitive matching.
        """
        # Guard: reject unknown buffer names to keep grep operations consistent.
        if name not in self._buffers:
            return {"ok": False, "error": f"No buffer named '{name}'."}
        buf = self._buffers[name]

        # Fire BEFORE_READ_BUFFER hooks — a rejection blocks the grep.
        read_args: dict[str, Any] = {"_op": "grep_buffer", "name": name, "pattern": pattern, "start": start, "end": end}
        pre_result = await buf._fire_hooks(self, None, HookPoint.BEFORE_READ_BUFFER, read_args)
        if pre_result is not True and pre_result is not None:
            return {"ok": False, "error": str(pre_result)}

        # Fire AFTER_READ_BUFFER hooks with an isolated buffer copy.
        # The hook can mutate the copy via set_text, and the subsequent grep reads the modified content.
        buf_copy = buf.clone()
        await buf_copy._fire_hooks(self, None, HookPoint.AFTER_READ_BUFFER, read_args)

        # Perform the grep on the buffer copy — uses content the AFTER hook may have modified.
        grep_result = buf_copy.grep(pattern=pattern, start=start, end=end)
        if not grep_result.ok:
            return {"ok": False, "error": grep_result.error}

        # Cluster the match indices to keep the agent context bounded when results are numerous.
        clusters = _cluster_lines_to_ranges(grep_result.matches, self._NUM_CLUSTERS)
        return {"ok": True, "matches": clusters, "count": grep_result.count}

    @sandbox
    def _read_buffer(self, name: str, start: int = 0, end: int | str = "end", show_timestamps: bool = False, show_line_numbers: bool = False) -> dict[str, Any]:
        """Read a range of lines from a buffer.
        Omit start to read from the beginning; omit end to read to the last line.
        Set show_timestamps=True to prefix each line with its unix timestamp.
        Set show_line_numbers=True (default) to prefix each line with its 0-based line index.
        Returns a dict with ok/error or ok/content on success.
        Set raw=True to get the raw string instead of a dict — errors always return dict."""
        # Look up the named buffer and delegate retrieval to Buffer.read, returning an error if the buffer is missing.
        if name not in self._buffers:
            return {"ok": False, "error": f"Error: no buffer named '{name}' found."}
        buf = self._buffers[name]
        result = buf.read(start=start, end=end, show_timestamps=show_timestamps, show_line_numbers=show_line_numbers)
        if result.kind == "error":
            return {"ok": False, "error": result.error}

        # Surface the full untruncated content with resolved range metadata.
        skipped_tail = len(buf.lines) - result.end
        resp = {
            "ok": True,
            "content": result.content,
            "start": result.start,
            "end": result.end,
            "total_chars": result.total_chars,
            "line_count": result.line_count,
        }
        if skipped_tail:
            resp["skipped_tail"] = skipped_tail
        return resp

    @tool
    async def read_buffer(
        self,
        name: str,
        start: int = 0,
        end: int | str = "end",
        show_timestamps: bool = False,
        show_line_numbers: bool = False,
        raw: bool = False,
    ) -> dict[str, Any] | str:
        """Read a range of lines from a buffer.
        Omit start to read from the beginning; omit end to read to the last line.
        Set show_timestamps=True to prefix each line with its unix timestamp.
        Set show_line_numbers=True (default) to prefix each line with its 0-based line index.
        Returns a dict with ok/error or ok/content on success.
        Set raw=True to get the raw string instead of a dict — errors always return dict."""
        # Guard: reject unknown buffer names to keep read operations consistent.
        if name not in self._buffers:
            return {"ok": False, "error": f"No buffer named '{name}'."}
        buf = self._buffers[name]

        # Fire BEFORE_READ_BUFFER hooks — a rejection blocks the read.
        read_args: dict[str, Any] = {"_op": "read_buffer", "name": name, "start": start, "end": end, "show_timestamps": show_timestamps, "show_line_numbers": show_line_numbers}
        pre_result = await buf._fire_hooks(self, None, HookPoint.BEFORE_READ_BUFFER, read_args)
        if pre_result is not True and pre_result is not None:
            return {"ok": False, "error": str(pre_result)}

        # Fire AFTER_READ_BUFFER hooks with an isolated buffer copy.
        # The hook populates the copy via set_text with the email content.
        buf_copy = buf.clone()
        await buf_copy._fire_hooks(self, None, HookPoint.AFTER_READ_BUFFER, read_args)

        # Read from the buffer copy — gets content the AFTER hook populated, or untouched buffer.
        read_result = buf_copy.read(start=start, end=end, show_timestamps=show_timestamps, show_line_numbers=show_line_numbers)
        if read_result.kind == "error":
            return {"ok": False, "error": read_result.error}
        result = {
            "ok": True,
            "content": read_result.content,
            "start": read_result.start,
            "end": read_result.end,
            "total_chars": read_result.total_chars,
            "line_count": read_result.line_count,
        }
        total_chars = result["total_chars"]

        # Within-size path: mark entries as seen by the agent and return content directly.
        if total_chars <= BufferManager._MAX_CHUNK_CHARS:
            now = time.time()
            buf = self._buffers[name]
            for entry in buf.lines[result["start"]:result["end"]]:
                entry.agent_at = now
            if raw:
                return result["content"]
            resp = {"ok": True, "content": result["content"], "start": result["start"], "end": result["end"], "lines": result["line_count"]}
            if result.get("skipped_tail"):
                resp["skipped_tail"] = result["skipped_tail"]
            return resp

        # Split the formatted content into lines for heavy-line detection; this segment reflects the
        # pre-formatted text (with line numbers and timestamps as applicable) and its total char
        # count is the authoritative measure used by the skip and bucket heuristics.
        segment = result["content"].split("\n")
        if result["content"].endswith("\n"):
            segment = segment[:-1]

        # Over-size path: check whether excluding the heaviest lines would bring the segment under the threshold.
        to_skip, _ = _lines_to_skip(segment, result["start"], BufferManager._MAX_CHUNK_CHARS)
        if to_skip and len(to_skip) <= 10:
            skip_lines = to_skip
            skip_msg = ", ".join(f"line {idx}({cl} chars)" for idx, cl in skip_lines)
            return {
                "ok": False,
                "error": (
                    f"Range [{result['start']}:{result['end']}) is {total_chars} chars ({result['line_count']} lines). "
                    f"Heaviest lines ({len(skip_lines)} totaling {sum(c for _, c in skip_lines)} chars): {skip_msg}. "
                    f"Consider re-reading with a narrower range to avoid them."
                ),
            }

        # Over-size path: report bucket distribution and prompt the caller to narrow the range.
        buckets = _make_buckets(segment, 10, result["start"])
        bucket_msgs = ", ".join(f"{sa}-{en}:{c}" for sa, en, c in buckets)
        return {
            "ok": False,
            "error": (
                f"Range [{result['start']}:{result['end']}) is {total_chars} chars ({result['line_count']} lines) and exceeds the {BufferManager._MAX_CHUNK_CHARS}-char threshold. "
                f"Reduce the read range to stay under the limit. "
                f"Bucket distribution (start-end:chars): {bucket_msgs}."
            ),
        }

    @tool
    async def edit_buffer(self, name: str, old_string: str, new_string: str, start: int = 0, end: int | str = "end", replace_all: bool = False) -> dict[str, Any]:
        """Replace old_string with new_string in the buffer content within a [start:end) range.
        Both old_string and new_string can span multiple lines.
        To remove an entire line, remember to add the trailing new-line character '\n' at the end of `old_string`, i.e. `old_string="old line\n"`. Otherwise, you keep a blank line.
        Editing fails when old_string is found more than once. Set replace_all=True to replace ALL occurrences.
        """
        # Validate the named buffer exists.
        if name not in self._buffers:
            return {"ok": False, "error": f"No buffer named '{name}'. Use read_buffer on \"{_STATUS_BUFFER}\" to see available buffers."}
        if not old_string:
            return {"ok": False, "error": "old_string must not be empty."}
        buf = self._buffers[name]
        total = len(buf.lines)

        # Resolve start/end to absolute 0-based indices; return an error dict if out of range.
        resolved = _resolve_line_range(total, start, end)
        if isinstance(resolved, dict):
            return resolved
        lo, hi = resolved

        # Guard: empty range — agent requested a range that resolves to no lines.
        if lo == hi:
            return {"ok": False, "error": f"The range you specified is empty (start=end={lo}). Nothing was replaced."}

        # Produce the alias with the edited content; propagate error dicts directly.
        edit_result = buf.edited_buffer(old_string, new_string, start, end, replace_all)
        if not edit_result.get("ok"):
            return edit_result
        alias = edit_result["buffer"]
        count = edit_result["count"]

        # Build hook args with the full edit context for downstream hooks to inspect.
        edit_args = {
            "_op": "edit_buffer",
            "name": name,
            "old_string": old_string,
            "new_string": new_string,
            "start": lo,
            "end": hi,
            "replace_all": replace_all,
        }

        # Gate the edit through pre-update hooks — rejection blocks the replacement.
        # The pre-hook receives `alias` as `new_buffer` so it can inspect the proposed state.
        hook_result = await buf._fire_hooks(
            self,
            alias,
            HookPoint.BEFORE_UPDATE_BUFFER,
            edit_args,
        )
        if hook_result is not True and hook_result is not None:
            return {"ok": False, "error": str(hook_result)}

        # Swap the buffer contents in place — preserving the object reference for any callers holding it.
        buf.lines, alias.lines = alias.lines, buf.lines
        buf.modified_at, alias.modified_at = alias.modified_at, buf.modified_at

        # Fire post-update hooks with roles swapped: `alias` is the buffer that was updated,
        # `buf` now holds the new state and is passed as `new_buffer` for post-hooks to inspect.
        await alias._fire_hooks(
            self,
            buf,
            HookPoint.AFTER_UPDATE_BUFFER,
            edit_args,
        )

        return {"ok": True, "count": count, "old_string": old_string}

    @tool
    async def patch_buffer(self, name: str, patch_text: str) -> dict[str, Any]:
        """Apply a unified diff to the buffer content.

        The patch_text must be a valid unified diff. Lines starting with '-' are removed;
        lines starting with '+' are added; context lines (prefixed with space) are used for
        alignment. The buffer content is used as the "original" state that the patch is applied to.

        If the patch cannot be applied (e.g. context lines no longer match), returns an error
        with the failed hunk context so the caller can inspect and retry.
        """
        # Validate the named buffer exists.
        if name not in self._buffers:
            return {"ok": False, "error": f"No buffer named '{name}'. Use read_buffer on \"{_STATUS_BUFFER}\" to see available buffers."}

        buf = self._buffers[name]

        # Produce the alias with the patched content; propagate error dicts directly.
        patch_result = buf.patched_buffer(patch_text)
        if not patch_result.get("ok"):
            return patch_result
        alias = patch_result["buffer"]
        count = patch_result["count"]

        # Gate the patch through pre-update hooks — rejection blocks the patch.
        patch_args = {"_op": "patch_buffer", "name": name, "patch_text": patch_text}
        hook_result = await buf._fire_hooks(
            self,
            alias,
            HookPoint.BEFORE_UPDATE_BUFFER,
            patch_args,
        )
        if hook_result is not True and hook_result is not None:
            return {"ok": False, "error": str(hook_result)}

        # Swap the buffer contents in place — preserving the object reference for any callers holding it.
        buf.lines, alias.lines = alias.lines, buf.lines
        buf.modified_at, alias.modified_at = alias.modified_at, buf.modified_at

        # Fire post-update hooks: `alias` is the buffer that was updated,
        # `buf` now holds the new state and is passed as `new_buffer`.
        await alias._fire_hooks(
            self,
            buf,
            HookPoint.AFTER_UPDATE_BUFFER,
            patch_args,
        )

        return {"ok": True, "count": count, "old_string": ""}

    @tool
    async def diff_buffers(self, a: str, b: str, overwrite: bool = False) -> dict[str, Any]:
        """Diff two buffers line-by-line using a unified diff.
        Stores the result in a target buffer named diff:a→b.
        Use overwrite=True to overwrite an existing diff buffer.
        """
        # Validate both buffer names exist before proceeding.
        if a not in self._buffers:
            return {"ok": False, "error": f"No buffer named '{a}'."}
        if b not in self._buffers:
            return {"ok": False, "error": f"No buffer named '{b}'."}

        # Extract line data for the diff computation.
        buf_a = [e.data for e in self._buffers[a].lines]
        buf_b = [e.data for e in self._buffers[b].lines]
        if buf_a == buf_b:
            return {"ok": True, "identical": True, "a": a, "b": b}

        # Guard against overwriting an existing diff buffer.
        diff_name = _diff_buffer_name(a, b)
        if diff_name in self._buffers and not overwrite:
            return {
                "ok": False,
                "error": (
                    f"Target buffer '{diff_name}' already exists. "
                    f"Use overwrite=True to replace it, or drop it first with drop_buffer."
                ),
            }

        # Generate the unified diff and persist it as a new named buffer.
        import difflib
        diff_lines = list(difflib.unified_diff(
            buf_a, buf_b,
            fromfile=f"buffer:{a}", tofile=f"buffer:{b}",
            lineterm="",
        ))
        diff_text = "\n".join(diff_lines) + "\n"
        create_result = await self.create_buffer(diff_name, text=diff_text, overwrite=True)
        if not create_result.get("ok"):
            return create_result
        return {"ok": True, "buffer": diff_name, "lines": create_result.get("lines", 0)}
