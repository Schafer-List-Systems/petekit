from __future__ import annotations

import asyncio
import inspect
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Literal


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


class Buffer:
    """A buffer holding a list of line entries with creation and modification timestamps.

    The last line is the in-progress line — it has no trailing newline in the file.
    The lines list must never be empty. An empty buffer has one BufferEntry(data="") —
    a single in-progress empty line.
    """

    name: str
    lines: list[BufferEntry]
    created_at: float
    modified_at: float
    _hooks: dict[HookPoint, dict[str, AnyHook]]

    def __init__(
        self,
        name: str,
        lines: list[BufferEntry] | None = None,
        created_at: float | None = None,
        modified_at: float | None = None,
        _hooks: dict[HookPoint, dict[str, AnyHook]] | None = None,
    ) -> None:
        self.name = name
        self.created_at = created_at if created_at is not None else time.time()
        self.modified_at = modified_at if modified_at is not None else self.created_at
        self.lines = lines if lines else [BufferEntry(data="", modified_at=self.modified_at)]
        self._hooks = _hooks if _hooks is not None else {
            HookPoint.BEFORE_CREATE_BUFFER: {},
            HookPoint.AFTER_CREATE_BUFFER: {},
            HookPoint.BEFORE_UPDATE_BUFFER: {},
            HookPoint.AFTER_UPDATE_BUFFER: {},
            HookPoint.BEFORE_DROP_BUFFER: {},
            HookPoint.AFTER_DROP_BUFFER: {},
        }

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

    def get_text(self, start: int = 0, end: int | None = None) -> str:
        """Return the buffer content between start and end.

        The buffer's last line is the "in-progress" line — it has no trailing newline
        in the file. When the range includes the last line of the buffer (end >=
        len(lines)), the result has no trailing newline either. A trailing newline is
        added only when the result ends before the buffer's last line — the range
        is clearly bounded and the final newline can be added without ambiguity.

        Args:
            start: 0-based start index (inclusive). Negative values are relative to the end.
            end: 0-based end index (exclusive), or None (default) to go to the last line.
            Negative values are relative to the end.

        Returns:
            The selected slice as a single string joined by newlines, with a
            trailing newline only when the range ends before the buffer's last line.
            Returns an empty string with no newline if the slice is empty.
        """
        # Empty buffer — nothing to return.
        if not self.lines:
            return ""

        # Resolve end to an absolute index, capping at the actual last line.
        # The last line has no trailing newline in the file, so it is excluded from
        # the natural range when end is None.
        if end is None:
            resolved_end = len(self.lines)
        else:
            resolved_end = min(end, len(self.lines))

        # Slice the lines and join them. A trailing newline is added only when
        # the result is a proper suffix — if the range ends before the buffer's
        # last line, the final newline is unambiguous and is appended.
        # If the range reaches or caps to the buffer's last line, no newline is
        # added — that line is the in-progress line with no file-level newline.
        return "\n".join(entry.data for entry in self.lines[start:resolved_end]) + ("\n" if end is not None and end < len(self.lines) else "")

    def set_text(self, text: str, modified_at: float | None = None) -> None:
        """Replace all buffer lines with entries parsed from text.

        The text is split on newlines using split() to preserve the trailing
        empty entry if text ends with a newline. The modified_at timestamp is
        applied to all new entries. Uses the current buffer's modified_at if
        not provided.
        """
        # Resolve the timestamp to apply to all new entries.
        ts = modified_at if modified_at is not None else self.modified_at

        # Split preserving trailing empty entry, then build BufferEntry objects.
        lines = text.split("\n")
        self.lines = [
            BufferEntry(data=line, modified_at=ts, object_at=ts)
            for line in lines
        ]
        self.modified_at = ts

    def inserted_buffer(self, new_text: str, pos: int) -> dict[str, Any]:
        """Return a result dict with the buffer after text is inserted at pos, or an error.

        Inserting text at a line index means: the buffer's text is split at the start
        of the target line, text is concatenated in between, and the suffix is
        appended. The last line of the inserted text that has no trailing newline
        merges into the first line of the suffix.

        Appending (pos >= len(lines)): the first line of text is appended to the
        buffer's last line. Remaining lines of text become new entries. The last
        line of text with no trailing newline merges into the (now pushed) last
        buffer line.

        Returns {"ok": True, "buffer": Buffer, "count": int} if the insertion is valid.
        Returns {"ok": False, "error": str} if the position is out of range.
        The returned buffer shares this buffer's hooks dict — use copy() if
        independence is needed.
        """
        # Guard: reject out-of-bounds positions.
        if pos < 0 or pos > len(self.lines):
            return {"ok": False, "error": f"Position {pos} is out of range (0–{len(self.lines)})."}
        
        # Capture the operation timestamp and parse the incoming text.
        now = time.time()
        new_text_lines = new_text.split("\n")

        # Create alias for mutation — all branches share the same aliased buffer.
        aliased = self.alias()
        aliased.modified_at = now

        # Create new BufferEntry objects for all lines of the new text.
        new_entries = [
            BufferEntry(data=line, modified_at=now, object_at=now)
            for line in new_text_lines
        ]

        # If pos >= len(lines), then we append.
        if pos >= len(self.lines):
            # Prepend the text of the buffer's last entry to the text of the first new buffer entry.
            new_entries[0].data = aliased.lines[-1].data + new_entries[0].data

            # Set the new list of entries.
            aliased.lines = aliased.lines[:-1] + new_entries

        # Otherise, if pos < len(self.lines), then we insert.
        else:
            # Append the buffer's line[pos] to the last line of the new_entries.
            new_entries[-1].data += aliased.lines[pos].data

            # Set the new list of entries.
            aliased.lines = aliased.lines[:pos] + new_entries + aliased.lines[pos + 1:]

        return {"ok": True, "buffer": aliased, "count": len(new_text_lines)}

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

        # Extract the target segment text using get_text and keep the entries for clustering.
        segment_entries = self.lines[lo:hi]
        segment_text = self.get_text(lo, hi)

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
        new_lines = new_content.split("\n")
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
        new_lines = new_text.split("\n")
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

        # If this is an UPDATE hook with a 3-arg signature, wrap it to accept 4 args
        # so callers can pass old_buffer uniformly. The wrapped version discards it.
        if hook_point in (HookPoint.BEFORE_UPDATE_BUFFER, HookPoint.AFTER_UPDATE_BUFFER):
            param_count = len(inspect.signature(hook).parameters)
            if param_count == 3:
                original = hook
                def wrapped(bm, old, new, args):
                    return original(bm, new, args)
                hook = wrapped

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
            content="\n".join(segment) + ("\n" if end < total else ""),
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

        # Walk each line and test the pattern against it.
        matches = []
        for i, entry in enumerate(self.lines[start:end], start=start):
            # Add trailing newline for non-last lines so that patterns containing \n can match them.
            if compiled.search(entry.data + ("\n" if i < len(self.lines) - 1 else "")):
                matches.append(i)

        # Return the unclustered raw match list for the caller to further process.
        return GrepResult(ok=True, matches=matches, start=start, end=end, count=len(matches))

