from __future__ import annotations
import asyncio
import difflib
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any
from typing import Callable
from typing import Literal


UpdateHook = Callable[["Buffer", str | None, int, int, str | None], bool | str | None]

from peteos.oap.agentic_object import AgenticObject
from peteos import tool, sandbox
from petekit.utils.text_formatters import format_dict_list_for_buffer


def _resolve_line_range(total: int, start: int | None, end: int | None) -> dict[str, Any] | tuple[int, int]:
    if start is None:
        start = total
    if end is None:
        end = total
    if start < -total or start > total:
        return {"ok": False, "error": f"start={start} is out of range. Valid range: -total to total (i.e. -{total} to {total})."}
    if end < -total or end > total:
        return {"ok": False, "error": f"end={end} is out of range. Valid range: -total to total (i.e. -{total} to {total})."}
    if start < 0:
        start = total + start
    if end < 0:
        end = total + end
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
) -> list[tuple[int, int, int]]:
    """Cluster matching line numbers into num_clusters ranges by agglomerative neighbor merging.

    Each line starts as its own cluster. Repeatedly merges the pair of neighboring
    clusters with the smallest content-based distance (sum of both cluster char
    counts plus the char count of the lines between them). Stops when num_clusters
    remain. Returns a list of (cluster_start, cluster_end, match_count).
    """
    if not line_numbers:
        return []
    if len(line_numbers) <= num_clusters:
        return [(ln, ln, 1) for ln in line_numbers]

    # Bootstrap clusters, using buffer line lengths when a buffer is available.
    if buf_lines is _NO_BUFFER_:
        clusters = [{'s': ln, 'e': ln, 'v': 0, 'n': 1} for ln in line_numbers]
    else:
        clusters = [{'s': ln, 'e': ln, 'v': len(buf_lines[ln].data) + 1, 'n': 1} for ln in line_numbers]

    # Sum the char counts of two adjacent clusters plus the gap between them.
    def span_chars(start: int, end: int) -> int:
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

    return [(c['s'], c['e'], c['n']) for c in clusters]


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
    lines: list[BufferEntry]
    created_at: float
    modified_at: float
    _update_hooks: dict[str, UpdateHook] = field(default_factory=dict)

    async def _fire_update_hooks(self, old_text: str | None, start: int, end: int, new_text: str | None) -> bool | str:
        # Guard: return early if no hooks are registered.
        if not self._update_hooks:
            return True

        # Fire each hook in registration order; stop and return the first rejection.
        for hook_name, hook in self._update_hooks.items():
            result = hook(self, old_text, start, end, new_text)
            if asyncio.iscoroutine(result):
                result = await result
            if result is not None and result is not True:
                return result

        return True

    def read(
        self,
        start: int = 0,
        end: int | None = None,
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
    - You can create, write, search, read ranges from, and edit any named buffer.
    - Read the "system:list:buffers" buffer to see what exists. Drop unused buffers when it becomes messy!
    - Line indices are 0-based, just like Python array indexing.
      Example: buf[0] is the first line, buf[-1] is the last line, buf[0:5] is the first 5 lines.
    - Ranges use [start, end) semantics: start is included, end is excluded. Omit end to read to the end of the buffer.
    - Use (?i) at the start of a grep pattern for case-insensitive matching.
    - Always prefer read_buffer with start/end over reading entire buffers when working with large content.
    - All timestamps are rounded to 6 decimals.
      Read it to get a JSON array of {name, lines} for each buffer.
    """

    _MAX_CHUNK_CHARS = 8000
    _NUM_CLUSTERS = 5

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._buffers: dict[str, Buffer] = {}
        loop = asyncio.get_running_loop()
        loop.create_task(self._refresh_buffers_buffer())

    async def _refresh_buffers_buffer(self) -> dict[str, Any]:
        records = [{"name": name, "lines": len(buf.lines)} for name, buf in self._buffers.items()]
        text = format_dict_list_for_buffer(records)
        return await self.create_buffer("system:list:buffers", text=text, overwrite=True)

    def _create_buffer(self, name: str, text: str | None = None) -> dict[str, Any]:
        """Protected buffer creation for initialization only.
        Always overwrites. Does not fire hooks.
        Used by __init__ and constructors — not for agent-facing operations.
        """
        ts = time.time()
        self._buffers[name] = Buffer(lines=[], created_at=ts, modified_at=ts)
        lines = 0

        if text:
            self._buffers[name].lines = [BufferEntry(data=line, modified_at=ts, object_at=ts) for line in text.splitlines()]
            lines = len(self._buffers[name].lines)

        return {"ok": True, "lines": lines}

    @tool
    async def create_buffer(self, name: str, text: str | None = None, overwrite: bool = False) -> dict[str, Any]:
        """Create a new buffer with the given name.
        Optionally provide initial text.
        Use overwrite=True to replace an existing buffer.
        """
        # Reject new creation if the name is already taken and overwrite is not requested.
        if name in self._buffers and not overwrite:
            return {"ok": False, "error": f"Buffer '{name}' already exists. Use overwrite=True to replace it."}

        ts = time.time()
        existed = name in self._buffers

        # Gate overwrite through the existing buffer's drop hooks; a rejection blocks the replacement.
        if existed:
            buf = self._buffers[name]
            old_text = "\n".join(e.data for e in buf.lines) + "\n"
            hook_result = await buf._fire_update_hooks(old_text, 0, len(buf.lines), None)
            if hook_result is not True and hook_result is not None:
                return {"ok": False, "error": str(hook_result)}

        # Create a fresh Buffer with no lines and the current timestamp; it starts with zero hooks.
        # Delegated to protected creation — hooks are not fired here.
        create_result = self._create_buffer(name, text)

        # Refresh the system buffer listing so the new buffer is visible to the agent.
        if name != "system:list:buffers":
            refresh_result = await self._refresh_buffers_buffer()
            if not refresh_result.get("ok"):
                # TODO(design): buffer already created and registered — rolling back would require
                # deleting from self._buffers. Until a rollback strategy is defined, raising the
                # error leaves BufferManager in an inconsistent state (buffer exists, listing stale).
                raise RuntimeError(f"failed to refresh buffers list: {refresh_result.get('error')}")

        # Signal whether this was a fresh creation or an overwrite, and how many lines were written.
        return {"ok": True, "created": not existed, "overwritten": existed, "lines": create_result.get("lines", 0)}

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
        now = time.time()
        new_entries = [BufferEntry(data=e.data, modified_at=e.modified_at, object_at=now) for e in src.lines]
        self._buffers[target_name] = Buffer(lines=new_entries, created_at=now, modified_at=now)
        if target_name != "system:list:buffers":
            refresh_result = await self._refresh_buffers_buffer()
            if not refresh_result.get("ok"):
                # TODO(design): buffer already created and registered — rolling back would require
                # deleting from self._buffers. Until a rollback strategy is defined, raising the
                # error leaves BufferManager in an inconsistent state (buffer exists, listing stale).
                raise RuntimeError(f"failed to refresh buffers list: {refresh_result.get('error')}")
        return {"ok": True, "target": target_name, "source": source_name, "lines": len(new_entries)}

    @tool
    async def write_buffer(self, name: str, text: str, start: int | None = None, end: int | None = None) -> dict[str, Any]:
        """Write to a text buffer.
        Overwrite the range [start, end) of an existing buffer. Default: APPEND
        A trailing newline is always appended, so a blank line in the input
        creates a blank line in the buffer. Omitting start means start=END.
        Omitting end means end=END.
        INSERT at line N: pass start=N and end=N.
        APPEND: Omit both start and end.
        OVERWRITE the whole buffer: pass start=0.
        REPLACE lines M to N: pass start=M and end=N."""
        # Guard: reject unknown buffer names to keep write operations consistent.
        if name not in self._buffers:
            return {"ok": False, "error": f"No buffer named '{name}'. Use create_buffer first."}
        buf = self._buffers[name]

        # Resolve the requested range and validate it before any content is read.
        total = len(buf.lines)
        resolved = _resolve_line_range(total, start, end)
        if isinstance(resolved, dict):
            return resolved
        start, end = resolved
        if start > end:
            return {"ok": False, "error": f"start ({start}) > end ({end})."}

        # Capture old buffer content and compute new content, then fire update hooks.
        # The hook sees the exact slice that will be replaced and the proposed replacement.
        old_text = "\n".join(buf.lines[i].data for i in range(start, min(end, total))) + "\n"
        new_text = text + "\n"
        hook_result = await buf._fire_update_hooks(old_text, start, end, new_text)
        if hook_result is not True and hook_result is not None:
            return {"ok": False, "error": str(hook_result)}

        # Commit the mutation to the buffer and stamp the modification time.
        now = time.time()
        new_entries = [BufferEntry(data=line, modified_at=now, object_at=now) for line in (text + "\n").splitlines()]
        buf.lines[start:end] = new_entries
        buf.modified_at = time.time()

        # Refresh the buffer registry to keep the system listing current.
        if name != "system:list:buffers":
            refresh_result = await self._refresh_buffers_buffer()
            if not refresh_result.get("ok"):
                # TODO(design): buffer already modified — rolling back would require restoring prior
                # content. Until a rollback strategy is defined, raising the error leaves the
                # BufferManager in an inconsistent state (content changed, listing stale).
                raise RuntimeError(f"failed to refresh buffers list: {refresh_result.get('error')}")
        return {"ok": True, "lines_written": len(new_entries), "total_lines": len(buf.lines)}

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
        if name == "system:list:buffers":
            return {"ok": False, "error": "Cannot drop the 'system:list:buffers' buffer."}

        # Fire update hooks with the full buffer content and None for new_text (signals drop).
        # Rejection means the drop is denied and the buffer must not be deleted.
        old_text = "\n".join(e.data for e in buf.lines) + "\n"
        hook_result = await buf._fire_update_hooks(old_text, 0, len(buf.lines), None)
        if hook_result is not True and hook_result is not None:
            return {"ok": False, "error": str(hook_result)}

        # Delete the buffer and refresh the system listing.
        del self._buffers[name]
        refresh_result = await self._refresh_buffers_buffer()
        if not refresh_result.get("ok"):
            # TODO(design): buffer already deleted from self._buffers — rolling back would require
            # restoring from a snapshot. Until a rollback strategy is defined, raising the error
            # leaves BufferManager in an inconsistent state (buffer gone, listing stale).
            raise RuntimeError(f"failed to refresh buffers list: {refresh_result.get('error')}")
        return {"ok": True, "dropped": name}

    def register_buffer_update_hook(self, buffer_name: str, hook_name: str, hook: UpdateHook) -> dict[str, Any]:
        """Register an update hook on a buffer by name. The hook is called before any mutation
        with the old buffer content and the proposed new content. Return True/None to accept,
        or a string reason to reject. Multiple hooks per buffer are supported; the first
        rejection blocks the operation. old_text=None signals buffer creation,
        new_text=None signals buffer drop."""

        # Guard: reject unknown buffers to keep hook registration consistent.
        if buffer_name not in self._buffers:
            return {"ok": False, "error": f"No buffer named '{buffer_name}'."}
        buf = self._buffers[buffer_name]

        # Guard: reject overwrites to keep hook registration consistent; use unregister first.
        if hook_name in buf._update_hooks:
            return {"ok": False, "error": f"hook '{hook_name}' already registered on '{buffer_name}'. Unregister first."}
        buf._update_hooks[hook_name] = hook

        return {"ok": True, "buffer": buffer_name, "hook": hook_name, "hook_count": len(buf._update_hooks)}

    def unregister_buffer_update_hook(self, buffer_name: str, hook_name: str) -> dict[str, Any]:
        """Remove a named update hook from a buffer. Silently succeeds if the hook or buffer
        does not exist — the caller does not need to know whether the hook was registered."""

        # Guard: reject unknown buffers to keep hook registration consistent.
        if buffer_name not in self._buffers:
            return {"ok": False, "error": f"No buffer named '{buffer_name}'."}
        buf = self._buffers[buffer_name]

        # Silently succeed if the named hook is not registered on this buffer.
        if hook_name not in buf._update_hooks:
            return {"ok": False, "error": f"no hook named '{hook_name}' on '{buffer_name}'"}
        del buf._update_hooks[hook_name]

        return {"ok": True, "buffer": buffer_name, "removed": hook_name}

    @sandbox
    def _grep_buffer(self, name: str, pattern: str, start: int = 0, end: int | None = None) -> dict[str, Any]:
        """Search a buffer for a regex pattern.
        Searches the full buffer by default, or a [start, end) range if provided.
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
    async def grep_buffer(self, name: str, pattern: str, start: int = 0, end: int | None = None) -> dict[str, Any]:
        """Search a buffer for a regex pattern.
        Searches the full buffer by default, or a [start, end) range if provided.
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
    def _read_buffer(self, name: str, start: int = 0, end: int | None = None, show_timestamps: bool = False, show_line_numbers: bool = False) -> dict[str, Any]:
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
        return {
            "ok": True,
            "content": result.content,
            "start": result.start,
            "end": result.end,
            "total_chars": result.total_chars,
            "line_count": result.line_count,
        }

    @tool
    async def read_buffer(
        self,
        name: str,
        start: int = 0,
        end: int | None = None,
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
            return {"ok": True, "content": result["content"], "start": result["start"], "end": result["end"], "lines": result["line_count"]}

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
                    f"Range [{result['start']}–{result['end']}] is {total_chars} chars ({result['line_count']} lines). "
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
                f"Range [{result['start']}–{result['end']}] is {total_chars} chars ({result['line_count']} lines) and exceeds the {BufferManager._MAX_CHUNK_CHARS}-char threshold. "
                f"Reduce the read range to stay under the limit. "
                f"Bucket distribution (start-end:chars): {bucket_msgs}."
            ),
        }

    @tool
    async def edit_buffer(self, name: str, old_string: str, new_string: str, start: int = 0, end: int | None = None, replace_all: bool = False) -> dict[str, Any]:
        """Replace old_string with new_string in the buffer content within a [start, end) range.
        Both old_string and new_string can span multiple lines.
        Editing fails when old_string is found more than once. Set replace_all=True to replace ALL occurrences.
        """
        # Validate the named buffer exists.
        if name not in self._buffers:
            return {"ok": False, "error": f"No buffer named '{name}'. Use read_buffer on \"system:list:buffers\" to see available buffers."}
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

        # Extract the target segment text and collect its entries.
        segment_entries = buf.lines[lo:hi]
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
            return {"ok": False, "error": f"Search pattern '{old_string}' not found in the range [{lo}, {end})."}

        # Guard: multiple matches found but replace_all is False — report clusters and decline.
        if not replace_all and len(all_ranges) > 1:
            line_numbers = [lo + i for i, e in enumerate(segment_entries) if old_string in e.data]
            clusters = _cluster_lines_to_ranges(line_numbers, 5, self._buffers[name].lines)
            cluster_msgs = ", ".join(f"lines {s}–{e} ({n} occurrence(s))" for s, e, n in clusters)
            return {
                "ok": False,
                "error": (
                    f"Search pattern '{old_string}' found multiple times ({segment_text.count(old_string)} total) at: {cluster_msgs}. "
                    "Set replace_all=True to replace all occurrences."
                ),
            }

        # Narrow char_ranges: all occurrences if replace_all else the first occurrence only.
        char_ranges = all_ranges if replace_all else [all_ranges[0]]

        # Build new_content by applying all replacements in reverse order.
        # NOTE: this block is duplicated below at lines 369–371 — the same construction
        # runs again unconditionally before the result is committed.
        new_content = segment_text
        for r_start, r_end in reversed(char_ranges):
            new_content = new_content[:r_start] + new_string + new_content[r_end:]

        # Split the replaced text into lines and prepare timestamp bookkeeping.
        new_lines = new_content.splitlines()
        now = time.time()
        result_entries: list[BufferEntry] = []
        used_old_indices: set[int] = set()

        # Identify which original line indices fall inside any replacement range.
        sorted_ranges = sorted(char_ranges)
        replaced_line_indices: set[int] = set()
        for r_start, r_end in sorted_ranges:
            first_newline = segment_text[:r_start].count('\n')
            last_newline = segment_text[:r_end].count('\n') - 1
            for li in range(first_newline, last_newline + 1):
                replaced_line_indices.add(lo + li)

        old_entries = buf.lines
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

        # Fire update hooks with the old segment and the computed new content.
        # Rejection means the edit is denied and the buffer must not be modified.
        hook_result = await buf._fire_update_hooks(segment_text, lo, hi, new_content)
        if hook_result is not True and hook_result is not None:
            return {"ok": False, "error": str(hook_result)}

        # Commit the new line entries back into the buffer at the target range.
        buf.lines[lo:hi] = result_entries
        buf.modified_at = now
        return {"ok": True, "count": len(char_ranges), "old_string": old_string}

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
        diff_name = f"diff:{a}→{b}"
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
