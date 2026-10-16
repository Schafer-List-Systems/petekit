from __future__ import annotations
import asyncio
import difflib
import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Literal


class HookPoint(Enum):
    """Hook points at which buffer lifecycle events can be intercepted."""
    BEFORE_CREATE_BUFFER = "before_create_buffer"
    AFTER_CREATE_BUFFER = "after_create_buffer"
    BEFORE_UPDATE_BUFFER = "before_update_buffer"
    AFTER_UPDATE_BUFFER = "after_update_buffer"
    BEFORE_DROP_BUFFER = "before_drop_buffer"
    AFTER_DROP_BUFFER = "after_drop_buffer"


# Pre-create: fires before buffer is added to the dict. Rejection blocks creation.
BeforeCreateBuffer = Callable[["BufferManager", "Buffer", dict[str, Any]], bool | str | None]

# Post-create: fires after buffer is in the dict. Informative only — returns None.
AfterCreateBuffer = Callable[["BufferManager", "Buffer", dict[str, Any]], None]

# Pre-update: fires before mutation is committed. Rejection blocks the update.
BeforeUpdateBuffer = Callable[["BufferManager", "Buffer", "Buffer", dict[str, Any]], bool | str | None]

# Post-update: fires after mutation is committed. Informative only — returns None.
AfterUpdateBuffer = Callable[["BufferManager", "Buffer", "Buffer", dict[str, Any]], None]

# Pre-drop: fires before buffer is removed from the dict. Rejection blocks the drop.
BeforeDropBuffer = Callable[["BufferManager", "Buffer", dict[str, Any]], bool | str | None]

# Post-drop: fires after buffer is removed. Informative only — returns None.
AfterBufferDropped = Callable[["BufferManager", "Buffer", dict[str, Any]], None]

# AnyHook is the union of all six hook type aliases — use this where a parameter
# accepts any hook regardless of hook point.
AnyHook = (
    BeforeCreateBuffer
    | BeforeUpdateBuffer
    | BeforeDropBuffer
    | AfterCreateBuffer
    | AfterUpdateBuffer
    | AfterBufferDropped
)


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


from peteos.oap.agentic_object import AgenticObject
from peteos import tool, sandbox


# Managed buffer names — system:status: namespace and diff namespace
_STATUS_BUFFER = "system:status:buffers"
_DIFF_BUFFER_PREFIX = "diff:"


def _diff_buffer_name(a: str, b: str) -> str:
    # Derive the canonical diff buffer name from two buffer identifiers.
    return f"{_DIFF_BUFFER_PREFIX}{a}→{b}"


def _sanitize_buffer_name(name: str) -> str:
    """Remove ASCII whitespace from a buffer name segment. Use this for any name derived from agent input."""
    return "".join(c for c in name if c not in " \t\n\r")


def _get_persistence_dir(bm: BufferManager) -> Path:
    """Return the BufferManager persistence directory inside the buffer manager's instance dir."""
    return bm.instance_dir / "buffer_manager"


def _persist_buffer(bm: BufferManager, buffer: Buffer) -> None:
    """Write buffer content and metadata to disk for persistence.

    Stores the raw line data in a file and structural metadata in a .json file.
    """
    # Ensure the persistence directory for this buffer manager exists.
    dir = _get_persistence_dir(bm)
    dir.mkdir(parents=True, exist_ok=True)

    # Raw content file — one line per line, preserving exact buffer text.
    buff_path = dir / f"{buffer.name}"
    buff_path.parent.mkdir(parents=True, exist_ok=True)
    buff_path.write_text(buffer.get_text())

    # Metadata JSON — name, line count, timestamps.
    import json
    meta = {
        "kind": "buffer",
        "name": buffer.name,
        "line_count": len(buffer.lines),
        "created_at": buffer.created_at,
        "modified_at": buffer.modified_at,
    }
    (dir / f".{buffer.name}.json").write_text(json.dumps(meta, indent=2))


def _drop_persisted_buffer(bm: BufferManager, buffer_name: str) -> None:
    """Remove persisted content and metadata for the named buffer."""
    dir = _get_persistence_dir(bm)

    # Remove content and metadata files; missing_ok=True handles the already-gone case.
    (dir / f"{buffer_name}").unlink(missing_ok=True)
    (dir / f".{buffer_name}.json").unlink(missing_ok=True)


def _make_persist_buffer_hook(bm: BufferManager) -> AfterUpdateBuffer:
    """Factory that returns a post-update hook that persists the buffer on every modification.

    Closes over bm so the hook can call _persist_buffer with the correct BufferManager.
    """
    def _persist_buffer_update_hook(
        buffer_manager: BufferManager,
        buffer: Buffer,
        new_buffer: Buffer,
        args: dict[str, Any],
    ) -> None:
        # Persist the buffer to disk — called after every successful mutation.
        _persist_buffer(buffer_manager, buffer)
    return _persist_buffer_update_hook


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
    create_args = {"_op": "create_buffer", "name": name, "text": text, "overwrite": overwrite}

    # Compute the proposed new lines from the provided text.
    new_lines = [
        BufferEntry(data=line, modified_at=modified_at, object_at=modified_at)
        for line in (text or "").splitlines()
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

    # Refresh the system buffer listing so the new buffer is visible to the agent.
    if name != _STATUS_BUFFER:
        refresh_result = await bm._refresh_buffers_buffer()
        if not refresh_result.get("ok"):
            raise RuntimeError(f"failed to refresh buffers list: {refresh_result.get('error')}")

    # Signal whether this was a fresh creation or an overwrite, and how many lines were written.
    return {"ok": True, "created": not existed, "overwritten": existed, "lines": create_result.get("lines", 0)}


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
    manifest_files = sorted(dir.glob(".*.json"))
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


def _resolve_line_number(total: int, pos: int  | str | None, arg: str = 'pos') -> dict[str, Any] | int:
    # coerce to int
    if pos is None:
        pos = total
    elif pos == "end" or isinstance(pos, str):
        if pos == "end":
            pos = total
        else:
            try:
                pos = int(pos)
            except ValueError:
                return {"ok": False, "error": f"{arg} must be an integer or 'end', got '{pos}'."}

    # check for valid rance
    if pos < -total or pos > total:
        return {"ok": False, "error": f"{arg}={pos} is out of range. Valid range: -total to total (i.e. -{total} to {total})."}

    # python logic for negative line numbers
    if pos < 0:
        pos = total + pos

    # return the resolve line number
    return pos


def _resolve_line_range(total: int, start: int | str | None, end: int | str | None) -> dict[str, Any] | tuple[int, int]:
    # resolve start line number
    start = _resolve_line_number(total, start, 'start')
    if isinstance(start, dict):
        return start

    # resolve end line number
    end = _resolve_line_number(total, end, 'end')
    if isinstance(end, dict):
        return end

    # check for correct order of start and end
    if start > end:
        return {"ok": False, "error": f"Start line number and end line number resolve to {start} > {end}, which is not allowed."}

    return (start, end)


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


_NO_BUFFER_ = object()


def _cluster_lines_to_ranges(
    line_numbers: list[int],
    num_clusters: int,
    buf_lines: list[BufferEntry] | None = _NO_BUFFER_,
) -> list[dict[str, Any]]:
    """Cluster matching line numbers into num_clusters ranges by agglomerative neighbor merging.

    Each line starts as its own cluster. Repeatedly merges the pair of neighboring
    clusters with the smallest content-based distance (sum of both cluster char
    counts plus the char count of the lines between them). Stops when num_clusters
    remain. Returns a list of dicts with start, end, match_count keys.
    """
    if not line_numbers:
        return []
    if len(line_numbers) <= num_clusters:
        return [{"start": ln, "end": ln, "match_count": 1} for ln in line_numbers]

    # Bootstrap clusters, using buffer line lengths when a buffer is available.
    if buf_lines is _NO_BUFFER_:
        clusters = [{'s': ln, 'e': ln, 'v': 0, 'n': 1} for ln in line_numbers]
    else:
        clusters = [{'s': ln, 'e': ln, 'v': len(buf_lines[ln].data) + 1, 'n': 1} for ln in line_numbers]

    # Sum the char counts of two adjacent clusters plus the gap between them.
    def span_chars(start: int, end: int) -> int:
        if buf_lines is _NO_BUFFER_:
            return 0
        return sum(len(buf_lines[i].data) + 1 for i in range(start, end + 1))

    def neighbor_distance(i: int) -> int:
        a = clusters[i]
        b = clusters[i + 1]
        between = span_chars(a['e'] + 1, b['s'])
        return a['v'] + b['v'] + between

    # Repeatedly merge the closest neighboring clusters until the target cluster count is reached.
    while len(clusters) > num_clusters:
        min_i = 0
        min_d = neighbor_distance(0)
        for i in range(1, len(clusters) - 1):
            d = neighbor_distance(i)
            if d < min_d:
                min_d = d
                min_i = i
        a = clusters[min_i]
        b = clusters[min_i + 1]
        merged = {
            's': a['s'],
            'e': b['e'],
            'v': span_chars(a['s'], b['e']),
            'n': a['n'] + b['n'],
        }
        clusters = clusters[:min_i] + [merged] + clusters[min_i + 2:]

    return [{"start": c["s"], "end": c["e"], "match_count": c["n"]} for c in clusters]


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
class BufferEntry:
    """A single line in a buffer with three temporal markers for observability.

    modified_at — when the entry was created or last changed.
    object_at   — when the entry was last touched by source code (read, not written).
    agent_at    — when the entry was injected into an agent chat context; None means unseen by agent.
    """
    data: str
    modified_at: float
    object_at: float | None = None
    agent_at: float | None = None


@dataclass
class Buffer:
    """A buffer holding a list of line entries with creation and modification timestamps."""
    name: str
    lines: list[BufferEntry]
    created_at: float
    modified_at: float
    _hooks: dict[
        HookPoint,
        dict[str, AnyHook],
    ] = field(default_factory=lambda: {
        HookPoint.BEFORE_CREATE_BUFFER: {},
        HookPoint.AFTER_CREATE_BUFFER: {},
        HookPoint.BEFORE_UPDATE_BUFFER: {},
        HookPoint.AFTER_UPDATE_BUFFER: {},
        HookPoint.BEFORE_DROP_BUFFER: {},
        HookPoint.AFTER_DROP_BUFFER: {},
    })

    def alias(self) -> Buffer:
        """Return an alias of this buffer sharing the same lines and hooks.

        No new list or dict is created — the alias points to the identical objects.
        Modifying the alias's lines list modifies the original's lines, and the
        same applies to the hooks dict.
        """
        return Buffer(
            name=self.name,
            lines=self.lines,
            created_at=self.created_at,
            modified_at=self.modified_at,
            _hooks=self._hooks,
        )

    def copy(self) -> Buffer:
        """Return a buffer that is fully independent of the original.

        The lines list is deep-copied — no BufferEntry or its data is shared with the original.
        The hooks dict is also deep-copied — modifying a hook in the copy does not affect the original.
        """
        import copy

        return Buffer(
            name=self.name,
            lines=copy.deepcopy(self.lines),
            created_at=self.created_at,
            modified_at=self.modified_at,
            _hooks=copy.deepcopy(self._hooks),
        )

    def edited_buffer(
        self,
        old_string: str,
        new_string: str,
        start: int = 0,
        end: int | str = "end",
        replace_all: bool = False,
    ) -> dict[str, Any]:
        """Return a result dict with the edited buffer or an error.

        Returns {"ok": True, "buffer": Buffer, "old_text": str, "new_text": str, "count": int}
        if successful. Returns {"ok": False, "error": str} on any error.
        The buffer in the ok case shares this buffer's hooks dict — use copy() if
        independence is needed.
        """
        if not old_string:
            return {"ok": False, "error": "old_string must not be empty."}
        total = len(self.lines)

        # Resolve start/end to absolute 0-based indices; return error on out-of-range.
        resolved = _resolve_line_range(total, start, end)
        if isinstance(resolved, dict):
            return resolved
        lo, hi = resolved

        # Guard: empty range.
        if lo == hi:
            return {"ok": False, "error": f"The range you specified is empty (start=end={lo}). Nothing was replaced."}

        # Extract the target segment text and collect its entries.
        segment_entries = self.lines[lo:hi]
        segment_text = "\n".join(e.data for e in segment_entries) + "\n"

        # Find every occurrence of old_string within the segment.
        all_ranges: list[tuple[int, int]] = []
        pos = 0
        while True:
            idx = segment_text.find(old_string, pos)
            if idx == -1:
                break
            all_ranges.append((idx, idx + len(old_string)))
            pos = idx + 1

        # Guard: no matches found in the segment.
        if len(all_ranges) == 0:
            return {"ok": False, "error": f"Search pattern '{old_string}' not found in the range [{lo}:{hi})."}

        # Guard: multiple matches found but replace_all is False; report clusters and decline.
        if not replace_all and len(all_ranges) > 1:
            line_numbers = [lo + i for i, e in enumerate(segment_entries) if old_string in e.data]
            clusters = _cluster_lines_to_ranges(line_numbers, 5, self.lines)
            cluster_msgs = ", ".join(f"lines {c['start']}–{c['end']} ({c['match_count']} occurrence(s))" for c in clusters)
            return {
                "ok": False,
                "error": (
                    f"Search pattern '{old_string}' found multiple times ({segment_text.count(old_string)} total) at: {cluster_msgs}. "
                    "Set replace_all=True to replace all occurrences."
                ),
            }

        # Build new_content by applying all replacements in reverse order.
        new_content = segment_text
        for r_start, r_end in reversed(all_ranges):
            new_content = new_content[:r_start] + new_string + new_content[r_end:]

        # Split the replaced text into lines and prepare timestamp bookkeeping.
        new_lines = new_content.splitlines()
        now = time.time()
        result_entries: list[BufferEntry] = []
        used_old_indices: set[int] = set()

        # Identify which original line indices fall inside any replacement range.
        sorted_ranges = sorted(all_ranges)
        replaced_line_indices: set[int] = set()
        for r_start, r_end in sorted_ranges:
            first_newline = segment_text[:r_start].count('\n')
            last_newline = segment_text[:r_end].count('\n') - 1
            for li in range(first_newline, last_newline + 1):
                replaced_line_indices.add(lo + li)

        old_entries = self.lines
        old_timestamps = [e.modified_at for e in old_entries]

        # Walk each new line, tracking its position relative to original content so the
        # correct modified_at can be assigned: replaced lines get the current now;
        # untouched lines that match the original content at the same index keep theirs.
        char_offset = 0
        for new_i, line_text in enumerate(new_lines):
            char_pos_new = sum(len(new_lines[j]) + 1 for j in range(new_i)) if new_i > 0 else 0

            in_replaced = False
            effective_offset = 0
            matched_range: tuple[int, int] | None = None
            for r_start, r_end in sorted_ranges:
                adj_start = r_start + effective_offset
                adj_end = r_end + effective_offset
                if char_pos_new < adj_start:
                    break
                if adj_start <= char_pos_new < adj_end:
                    in_replaced = True
                    matched_range = (r_start, r_end)
                    effective_offset += len(new_string) - (r_end - r_start)
                    break
                effective_offset += len(new_string) - (r_end - r_start)

            # Re-check replaced status after offset adjustment.
            if in_replaced and matched_range is not None:
                r_start, r_end = matched_range
                net_change = len(new_string) - (r_end - r_start)
                if net_change < 0:
                    remapped_pos = char_pos_new - net_change
                else:
                    remapped_pos = char_pos_new - net_change
                if not (r_start <= remapped_pos < r_end):
                    in_replaced = False

            # Assign modified_at: replaced lines get now; untouched preserved lines keep
            # their original modified_at; novel content also gets now.
            if in_replaced:
                result_entries.append(BufferEntry(data=line_text, modified_at=now, object_at=now))
            else:
                mapped_pos = char_pos_new - effective_offset
                if mapped_pos <= 0:
                    mapped_line_idx = 0
                else:
                    mapped_line_idx = segment_text[:mapped_pos].count('\n')
                abs_mapped = lo + mapped_line_idx
                if (
                    abs_mapped < len(old_entries)
                    and old_entries[abs_mapped].data == line_text
                    and abs_mapped not in used_old_indices
                    and abs_mapped not in replaced_line_indices
                ):
                    result_entries.append(
                        BufferEntry(data=line_text, modified_at=old_timestamps[abs_mapped], object_at=now)
                    )
                    used_old_indices.add(abs_mapped)
                else:
                    result_entries.append(BufferEntry(data=line_text, modified_at=now, object_at=now))

        # Return an alias of this buffer with the edited lines and the current timestamp.
        # Also return old_text, new_text, and the replacement count for the caller.
        aliased = self.alias()
        aliased.lines = self.lines[:lo] + result_entries + self.lines[hi:]
        aliased.modified_at = now

        return {
            "ok": True,
            "buffer": aliased,
            "count": len(all_ranges),
        }

    def written_buffer(self, text: str, pos: int) -> dict[str, Any]:
        """Return a result dict with the buffer after text is inserted at pos, or an error.

        Returns {"ok": True, "buffer": Buffer, "count": int} if the insertion is valid.
        Returns {"ok": False, "error": str} if the position is out of range.
        The returned buffer shares this buffer's hooks dict — use copy() if independence is needed.
        """
        if pos < 0 or pos > len(self.lines):
            return {"ok": False, "error": f"Position {pos} is out of range (0–{len(self.lines)})."}

        # Build the new entries from the provided text — a trailing newline is always appended.
        now = time.time()
        new_entries = [
            BufferEntry(data=line, modified_at=now, object_at=now)
            for line in (text + "\n").splitlines()
        ]

        # Return an alias of this buffer with the new entries inserted at pos.
        aliased = self.alias()
        aliased.lines = self.lines[:pos] + new_entries + self.lines[pos:]
        aliased.modified_at = now

        return {
            "ok": True,
            "buffer": aliased,
            "count": len(new_entries),
        }

    def patched_buffer(self, patch_text: str) -> dict[str, Any]:
        """Return a result dict with the buffer after a unified diff is applied, or an error.

        Returns {"ok": True, "buffer": Buffer, "count": int} on success.
        Returns {"ok": False, "error": str} if the patch cannot be parsed or applied.
        The returned buffer shares this buffer's hooks dict — use copy() if independence is needed.
        """
        # Import paatch; surface a clear error if it is not installed.
        try:
            import paatch
        except ImportError:
            return {"ok": False, "error": "paatch library not installed. Install it with: pip install paatch"}

        # Parse the provided diff, returning a parse error if the format is unrecognizable.
        if not isinstance(patch_text, bytes):
            patch_bytes = patch_text.encode("utf-8")
        else:
            patch_bytes = patch_text
        ps = paatch.fromstring(patch_bytes)
        if not ps:
            return {"ok": False, "error": "Could not parse patch: the unified diff format could not be recognized."}

        # Apply the patch in-memory using patch_stream.
        buf_text = self.get_text()
        from io import BytesIO
        hunks = ps.items[0].hunks
        try:
            src = BytesIO(buf_text.encode("utf-8"))
            patched_bytes = b"".join(ps.patch_stream(src, hunks))
        except Exception as e:
            return {"ok": False, "error": f"Could not apply patch: {e}."}

        # Build the new buffer from the patched text.
        new_text = patched_bytes.decode("utf-8")
        new_lines = new_text.splitlines()
        now = time.time()
        result_entries = [
            BufferEntry(data=line, modified_at=now, object_at=now) for line in new_lines
        ]

        # Construct the result alias with the patched lines and the new timestamp.
        aliased = self.alias()
        aliased.lines = result_entries
        aliased.modified_at = now

        return {
            "ok": True,
            "buffer": aliased,
            "count": len(hunks),
        }

    async def _fire_pre_op_hooks(
        self,
        hooks: list[tuple[str, BeforeCreateBuffer | BeforeUpdateBuffer | BeforeDropBuffer]],
        hook_args: list[Any],
    ) -> bool | str:
        """Fire pre-operation hooks in order; return the first rejection.

        Update hooks receive (bm, buffer, new_buffer, args). Create and drop hooks
        receive (bm, buffer, args). If a hook returns a non-None/non-True value, that
        is a rejection and is returned immediately — the operation is blocked.
        If a hook is async and raises, the exception propagates directly to the caller.
        """
        # Fire each hook in order; stop and return the first rejection.
        for hook_name, hook in hooks:
            # Await the hook if it is a coroutine; otherwise invoke it synchronously.
            result = hook(*hook_args)
            if asyncio.iscoroutine(result):
                result = await result

            # Rejection — a non-None, non-True return means the hook denied the operation.
            # Return it immediately so the operation is blocked without firing remaining hooks.
            if result is not None and result is not True:
                return result
        return True

    async def _fire_post_op_hooks(
        self,
        hooks: list[tuple[str, AfterCreateBuffer | AfterUpdateBuffer | AfterBufferDropped]],
        hook_args: list[Any],
    ) -> None:
        """Fire post-operation hooks in order; catch and log any raises, continue to the next.

        Update hooks receive (bm, buffer, new_buffer, args). Create and drop hooks
        receive (bm, buffer, args). Return values are ignored — post-op hooks are
        side-effect only and their return value has no meaning to the caller.
        """
        import logging

        # Fire each hook in order; if it raises, log the error and move on.
        for hook_name, hook in hooks:
            try:
                result = hook(*hook_args)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logging.error(f"[hook] post-operation hook '{hook_name}' on '{self.name}' raised: {e}")

    async def _fire_hooks(
        self,
        buffer_manager: BufferManager,
        new_buffer: Buffer | None,
        hook_point: HookPoint,
        args: dict[str, Any],
    ) -> bool | str:
        """Fire all hooks for the given hook point.

        Pre-operation hooks (BEFORE_CREATE_BUFFER, BEFORE_UPDATE_BUFFER, BEFORE_DROP_BUFFER) block on first
        rejection. Post-operation hooks (AFTER_CREATE_BUFFER, AFTER_UPDATE_BUFFER, AFTER_DROP_BUFFER)
        fire all hooks and log any rejections.
        """
        # Return early if no hooks are registered for this hook point.
        if hook_point not in self._hooks or not self._hooks[hook_point]:
            return True

        # Pre-operation hooks block on first rejection; post-operation hooks fire all and log.
        hook_items = list(self._hooks[hook_point].items())
        pre_op_point = hook_point in (
            HookPoint.BEFORE_CREATE_BUFFER,
            HookPoint.BEFORE_UPDATE_BUFFER,
            HookPoint.BEFORE_DROP_BUFFER,
        )

        # Build the positional args list — update hooks get 4 args, create/drop get 3.
        if new_buffer is not None:
            hook_args = [buffer_manager, self, new_buffer, args]
        else:
            hook_args = [buffer_manager, self, args]

        # Fire pre-op hooks — block on first rejection.
        if pre_op_point:
            return await self._fire_pre_op_hooks(hook_items, hook_args)

        # Fire post-op hooks — log all rejections and continue.
        await self._fire_post_op_hooks(hook_items, hook_args)
        return True

    def register_hook(
        self,
        hook_point: HookPoint,
        hook_name: str,
        hook: AnyHook,
        overwrite_existing: bool = False,
    ) -> dict[str, Any]:
        """Register a hook on this buffer for the given hook point.

        Stores the hook directly in buffer._hooks under the hook point.
        If overwrite_existing is True and the hook already exists, replaces it.
        If overwrite_existing is False and the hook already exists, rejects.
        """
        # Handle the case where the hook already exists under this buffer and hook point.
        if hook_name in self._hooks[hook_point]:
            if overwrite_existing:
                # Overwrite is requested — remove the existing hook and reregister.
                del self._hooks[hook_point][hook_name]
            else:
                return {
                    "ok": False,
                    "error": (
                        f"hook '{hook_name}' already registered on '{self.name}' "
                        f"at '{hook_point.value}'. Unregister first, or use overwrite_existing."
                    ),
                }

        # Store the hook by name on this buffer.
        self._hooks[hook_point][hook_name] = hook

        return {
            "ok": True,
            "hook_point": hook_point.value,
            "buffer": self.name,
            "hook": hook_name,
        }

    def unregister_hook(
        self,
        hook_point: HookPoint,
        hook_name: str,
    ) -> dict[str, Any]:
        """Remove a named hook from this buffer and hook point.

        Rejects if the hook point or hook name is not found on this buffer.
        """
        if hook_name not in self._hooks[hook_point]:
            return {
                "ok": False,
                "error": (
                    f"no hook '{hook_name}' on '{self.name}' "
                    f"at '{hook_point.value}'."
                ),
            }

        # Remove the named hook from this buffer and hook point.
        del self._hooks[hook_point][hook_name]

        # Confirm the removal with the hook name that was removed.
        return {
            "ok": True,
            "hook_point": hook_point.value,
            "buffer": self.name,
            "removed": hook_name,
        }

    def get_text(self, start: int = 0, end: int | None = None) -> str:
        """Return the buffer content between start and end.

        Returns the selected slice as a single string with entries joined by newlines
        and a trailing newline at the end.

        Args:
            start: 0-based start index (inclusive). Negative values are relative to the end.
            end: 0-based end index (exclusive), or None (default) to go to the last line.
            Negative values are relative to the end.

        Returns:
            The selected slice as a single string with a trailing newline.
            Returns an empty string with no newline if the slice is empty.
        """
        # Empty buffer — nothing to return.
        if not self.lines:
            return ""

        # Slice the lines and join them, preserving the trailing newline.
        return "\n".join(entry.data for entry in self.lines[start:end]) + "\n"

    def __len__(self) -> int:
        """Return the number of lines in the buffer."""
        return len(self.lines)

    def read(
        self,
        start: int = 0,
        end: int | str = "end",
        show_timestamps: bool = False,
        show_line_numbers: bool = True,
    ) -> ReadBufferResult:
        # Resolve the requested range to absolute 0-based indices, returning an error if out of bounds.
        total = len(self.lines)
        resolved = _resolve_line_range(total, start, end)
        if isinstance(resolved, dict):
            return ReadBufferResult(kind="error", error=resolved["error"])
        start, end = resolved
        if start > end:
            return ReadBufferResult(kind="error", error=f"Error: start ({start}) > end ({end}). Buffer has {total} lines.")
        if start >= total:
            return ReadBufferResult(kind="error", error=f"Error: start ({start}) is at or beyond buffer length ({total} lines).")

        # Format each line with optional line numbers and timestamps.
        segment = []
        for i, entry in enumerate(self.lines[start:end], start=start):
            if show_timestamps and show_line_numbers:
                segment.append(f"{i}: ({entry.modified_at:.6f}) {entry.data}")
            elif show_timestamps:
                segment.append(f"({entry.modified_at:.6f}) {entry.data}")
            elif show_line_numbers:
                segment.append(f"{i}: {entry.data}")
            else:
                segment.append(entry.data)

        # Compute total char count and return the content result.
        total_chars = sum(len(l) for l in segment) + len(segment)
        return ReadBufferResult(
            kind="content",
            content="\n".join(segment),
            start=start,
            end=end,
            total_chars=total_chars,
            line_count=len(segment),
        )

    def grep(self, pattern: str, start: int = 0, end: int | None = None) -> GrepResult:
        # Resolve the requested range to absolute 0-based indices, returning an error if out of bounds.
        total = len(self.lines)
        resolved = _resolve_line_range(total, start, end)
        if isinstance(resolved, dict):
            return GrepResult(ok=False, error=resolved["error"])
        start, end = resolved
        if start > end:
            return GrepResult(ok=False, error=f"start ({start}) > end ({end}).")

        # Compile the regex pattern and collect line indices where it matches.
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            return GrepResult(ok=False, error=f"Invalid regex pattern '{pattern}': {e}.")
        matches = []
        for i, entry in enumerate(self.lines[start:end], start=start):
            if compiled.search(entry.data):
                matches.append(i)

        # Return the unclustered raw match list for the caller to further process.
        return GrepResult(ok=True, matches=matches, start=start, end=end, count=len(matches))


class BufferManager(AgenticObject):
    """You are a buffer manager. You hold multiple named buffers, each a list of lines in memory.
    - You can `create`, insert/append (`write`), search (`grep`), `read` ranges from, and edit (`edit` and `patch`) any named buffer.
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
        }

        # Bootstrap the system status buffer — this must exist before any agent runs.
        self._create_buffer(_STATUS_BUFFER, text="[]")

        # Register the restore hook so persisted buffers survive instance restarts.
        self._register_restore_hook("create_buffers", _restore_create_buffers)

    async def _refresh_buffers_buffer(self) -> dict[str, Any]:
        records = [{"name": name, "lines": len(buf.lines)} for name, buf in self._buffers.items()]
        text = json.dumps(records, indent=2)
        return await self.create_buffer(_STATUS_BUFFER, text=text, overwrite=True)

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
            buf.lines = [BufferEntry(data=line, modified_at=created_at, object_at=created_at) for line in text.splitlines()]
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
        write_result = buf.written_buffer(text, pos)
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

        # Refresh the buffer registry to keep the system listing current.
        if name != _STATUS_BUFFER:
            refresh_result = await self._refresh_buffers_buffer()
            if not refresh_result.get("ok"):
                # TODO(design): buffer already modified — rolling back would require restoring prior
                # content. Until a rollback strategy is defined, raising the error leaves the
                # BufferManager in an inconsistent state (content changed, listing stale).
                raise RuntimeError(f"failed to refresh buffers list: {refresh_result.get('error')}")
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

        refresh_result = await self._refresh_buffers_buffer()
        if not refresh_result.get("ok"):
            # TODO(design): buffer already deleted from self._buffers — rolling back would require
            # restoring from a snapshot. Until a rollback strategy is defined, raising the error
            # leaves BufferManager in an inconsistent state (buffer gone, listing stale).
            raise RuntimeError(f"failed to refresh buffers list: {refresh_result.get('error')}")

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
        # Delegate retrieval to the sandboxed helper, which surfaces raw unclustered matches.
        result = self._grep_buffer(name, pattern, start, end)
        if not result.get("ok"):
            return result

        # Cluster the match indices to keep the agent context bounded when results are numerous.
        clusters = _cluster_lines_to_ranges(result["matches"], self._NUM_CLUSTERS)
        return {"ok": True, "matches": clusters, "count": result["count"]}

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
        # Retrieve the full untruncated buffer content.
        result = self._read_buffer(name, start, end, show_timestamps, show_line_numbers)
        if not result.get("ok"):
            return result
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
            "start": start,
            "end": end,
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
