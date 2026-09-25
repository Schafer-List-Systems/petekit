from __future__ import annotations
import difflib
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any
from typing import Literal

from peteos.oap.agentic_object import AgenticObject
from peteos import tool
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


@dataclass
class ReadBufferResult:
    """Result of _read_buffer. Signals which branch was taken and carries data for hint construction."""
    kind: Literal["content", "skip", "bucket", "error"]
    content: str | None = None
    line_range: tuple[int, int] | None = None  # (0-based start, 0-based end, [start, end) semantics)
    total_chars: int = 0
    line_count: int = 0
    skip_lines: list[tuple[int, int]] | None = None  # (0-based line index, char count)
    bucket_info: list[tuple[int, int, int]] | None = None  # (start, end, chars)
    error: str | None = None


@dataclass
class BufferEntry:
    """A single line in a buffer with timestamp and seen flag."""
    data: str
    timestamp: float
    seen: bool = False


@dataclass
class Buffer:
    """A buffer holding a list of line entries with creation and modification timestamps."""
    lines: list[BufferEntry]
    created_at: float
    modified_at: float


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
        refresh_result = self._refresh_buffers_buffer()
        if not refresh_result.get("ok"):
            raise RuntimeError(f"failed to refresh buffers list: {refresh_result.get('error')}")

    def _refresh_buffers_buffer(self) -> dict[str, Any]:
        records = [{"name": name, "lines": len(buf.lines)} for name, buf in self._buffers.items()]
        text = format_dict_list_for_buffer(records)
        return self.create_buffer("system:list:buffers", text=text, overwrite=True)

    @tool
    def create_buffer(self, name: str, text: str | None = None, overwrite: bool = False) -> dict[str, Any]:
        """Create a new buffer with the given name.
        Optionally provide initial text.
        Use overwrite=True to replace an existing buffer.
        """
        if name in self._buffers and not overwrite:
            return {"ok": False, "error": f"Buffer '{name}' already exists. Use overwrite=True to replace it."}
        ts = time.time()
        existed = name in self._buffers
        self._buffers[name] = Buffer(lines=[], created_at=ts, modified_at=ts)
        lines = 0
        if text:
            self._buffers[name].lines = [BufferEntry(data=line, timestamp=ts, seen=True) for line in text.splitlines()]
            lines = len(self._buffers[name].lines)
        if name != "system:list:buffers":
            refresh_result = self._refresh_buffers_buffer()
            if not refresh_result.get("ok"):
                # TODO(design): buffer already created and registered — rolling back would require
                # deleting from self._buffers. Until a rollback strategy is defined, raising the
                # error leaves BufferManager in an inconsistent state (buffer exists, listing stale).
                raise RuntimeError(f"failed to refresh buffers list: {refresh_result.get('error')}")
        return {"ok": True, "created": not existed, "overwritten": existed, "lines": lines}

    @tool
    def copy_buffer(self, source_name: str, target_name: str, overwrite: bool = False) -> dict[str, Any]:
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
        new_entries = [BufferEntry(data=e.data, timestamp=e.timestamp, seen=e.seen) for e in src.lines]
        self._buffers[target_name] = Buffer(lines=new_entries, created_at=now, modified_at=now)
        if target_name != "system:list:buffers":
            refresh_result = self._refresh_buffers_buffer()
            if not refresh_result.get("ok"):
                # TODO(design): buffer already created and registered — rolling back would require
                # deleting from self._buffers. Until a rollback strategy is defined, raising the
                # error leaves BufferManager in an inconsistent state (buffer exists, listing stale).
                raise RuntimeError(f"failed to refresh buffers list: {refresh_result.get('error')}")
        return {"ok": True, "target": target_name, "source": source_name, "lines": len(new_entries)}

    @tool
    def write_buffer(self, name: str, text: str, start: int | None = None, end: int | None = None) -> dict[str, Any]:
        """Overwrite the text in the range [start, end) of an existing buffer.
        A trailing newline is always appended, so a blank line in the input
        creates a blank line in the buffer. Omitting start means start=END.
        Omitting end means end=END.
        INSERT at line N: pass start=N and end=N.
        APPEND: Omit both start and end.
        OVERWRITE the whole buffer: pass start=0.
        REPLACE lines M to N: pass start=M and end=N."""
        if name not in self._buffers:
            return {"ok": False, "error": f"No buffer named '{name}'. Use create_buffer first."}
        buf = self._buffers[name]
        total = len(buf.lines)
        resolved = _resolve_line_range(total, start, end)
        if isinstance(resolved, dict):
            return resolved
        start, end = resolved
        if start > end:
            return {"ok": False, "error": f"start ({start}) > end ({end})."}
        new_entries = [BufferEntry(data=line, timestamp=time.time(), seen=True) for line in (text + "\n").splitlines()]
        buf.lines[start:end] = new_entries
        buf.modified_at = time.time()
        if name != "system:list:buffers":
            refresh_result = self._refresh_buffers_buffer()
            if not refresh_result.get("ok"):
                # TODO(design): buffer already modified — rolling back would require restoring prior
                # content. Until a rollback strategy is defined, raising the error leaves the
                # BufferManager in an inconsistent state (content changed, listing stale).
                raise RuntimeError(f"failed to refresh buffers list: {refresh_result.get('error')}")
        return {"ok": True, "lines_written": len(new_entries), "total_lines": len(buf.lines)}

    @tool
    def drop_buffer(self, name: str) -> dict[str, Any]:
        """Drop (delete) a buffer by name.
        The buffer and its content are discarded.
        """
        if name not in self._buffers:
            return {"ok": False, "error": f"No buffer named '{name}'."}
        if name == "system:list:buffers":
            return {"ok": False, "error": "Cannot drop the 'system:list:buffers' buffer."}
        del self._buffers[name]
        refresh_result = self._refresh_buffers_buffer()
        if not refresh_result.get("ok"):
            # TODO(design): buffer already deleted from self._buffers — rolling back would require
            # restoring from a snapshot. Until a rollback strategy is defined, raising the error
            # leaves BufferManager in an inconsistent state (buffer gone, listing stale).
            raise RuntimeError(f"failed to refresh buffers list: {refresh_result.get('error')}")
        return {"ok": True, "dropped": name}

    @tool
    def grep_buffer(self, name: str, pattern: str, start: int = 0, end: int | None = None) -> dict[str, Any]:
        """Search a buffer for a regex pattern.
        Searches the full buffer by default, or a [start, end) range if provided.
        Returns a dict with ok/error or ok/matches.
        Use (?i) at the start for case-insensitive matching.
        """
        if name not in self._buffers:
            return {"ok": False, "error": f"No buffer named '{name}'."}
        total = len(self._buffers[name].lines)
        resolved = _resolve_line_range(total, start, end)
        if isinstance(resolved, dict):
            return resolved
        start, end = resolved
        if start > end:
            return {"ok": False, "error": f"start ({start}) > end ({end})."}
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            return {"ok": False, "error": f"Invalid regex pattern '{pattern}': {e}."}
        matches = []
        buf = self._buffers[name]
        for i, entry in enumerate(buf.lines[start:end], start=start):
            if compiled.search(entry.data):
                matches.append(i)
        clusters = self._cluster_lines_to_ranges(name, matches, self._NUM_CLUSTERS)
        return {"ok": True, "matches": clusters, "count": len(clusters)}

    def _read_buffer(self, name: str, start: int = 0, end: int | None = None, show_timestamps: bool = False, show_line_numbers: bool = True) -> ReadBufferResult:
        """Internal read. Returns ReadBufferResult with kind and branch data. Uses 0-based indices with [start, end) semantics."""
        if name not in self._buffers:
            return ReadBufferResult(kind="error", error=f"Error: no buffer named '{name}'. Use create_buffer first.")
        buf = self._buffers[name]
        total = len(buf.lines)

        resolved = _resolve_line_range(total, start, end)
        if isinstance(resolved, dict):
            return ReadBufferResult(kind="error", error=resolved["error"])
        start, end = resolved
        if start > end:
            return ReadBufferResult(kind="error", error=f"Error: start ({start}) > end ({end}). Buffer has {total} lines.")
        if start >= total:
            return ReadBufferResult(kind="error", error=f"Error: start ({start}) is at or beyond buffer length ({total} lines).")

        segment = []
        for i, entry in enumerate(buf.lines[start:end], start=start):
            if show_timestamps and show_line_numbers:
                segment.append(f"{i}: ({entry.timestamp:.6f}) {entry.data}")
            elif show_timestamps:
                segment.append(f"({entry.timestamp:.6f}) {entry.data}")
            elif show_line_numbers:
                segment.append(f"{i}: {entry.data}")
            else:
                segment.append(entry.data)
        total_chars = sum(len(l) for l in segment) + len(segment)

        if total_chars <= BufferManager._MAX_CHUNK_CHARS:
            for entry in buf.lines[start:end]:
                entry.seen = True
            return ReadBufferResult(
                kind="content",
                content="\n".join(segment),
                line_range=(start, end),
                total_chars=total_chars,
                line_count=len(segment),
            )

        to_skip, _ = self._lines_to_skip(segment, start)
        if to_skip and len(to_skip) <= 10:
            return ReadBufferResult(
                kind="skip",
                line_range=(start, end),
                total_chars=total_chars,
                line_count=len(segment),
                skip_lines=to_skip,
            )

        buckets = self._make_buckets(segment, 10, start)
        return ReadBufferResult(
            kind="bucket",
            line_range=(start, end),
            total_chars=total_chars,
            line_count=len(segment),
            bucket_info=buckets,
        )

    @tool
    def read_buffer(
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
        result = self._read_buffer(name, start, end, show_timestamps, show_line_numbers)
        if result.kind == "error":
            return {"ok": False, "error": result.error}
        if result.kind == "content":
            if raw:
                return result.content
            return {"ok": True, "content": result.content, "line_range": result.line_range, "lines": result.line_count}
        if result.kind == "skip":
            skip_lines = result.skip_lines
            skip_msg = ", ".join(f"line {idx}({cl} chars)" for idx, cl in skip_lines)
            return {
                "ok": False,
                "error": (
                    f"Range [{result.line_range[0]}–{result.line_range[1]}] is {result.total_chars} chars ({result.line_count} lines). "
                    f"Heaviest lines ({len(skip_lines)} totaling {sum(c for _, c in skip_lines)} chars): {skip_msg}. "
                    f"Consider re-reading with a narrower range to avoid them."
                ),
            }
        # kind == "bucket"
        result_bucket = result.bucket_info  # type: ignore
        bucket_msgs = ", ".join(f"{sa}-{en}:{c}" for sa, en, c in result_bucket)
        return {
            "ok": False,
            "error": (
                f"Range [{result.line_range[0]}–{result.line_range[1]}] is {result.total_chars} chars ({result.line_count} lines) and exceeds the {BufferManager._MAX_CHUNK_CHARS}-char threshold. "
                f"Reduce the read range to stay under the limit. "
                f"Bucket distribution (start-end:chars): {bucket_msgs}."
            ),
        }

    @tool
    def edit_buffer(self, name: str, old_string: str, new_string: str, start: int = 0, end: int | None = None, replace_all: bool = False) -> dict[str, Any]:
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
            clusters = self._cluster_lines_to_ranges(name, line_numbers, 5)
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
        old_timestamps = [e.timestamp for e in old_entries]

        # Walk each new line, tracking its position relative to original content so the
        # correct timestamp can be assigned: replaced lines get the current timestamp;
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

            # Assign timestamp: replaced lines get now; untouched preserved lines keep
            # their original timestamp; novel content also gets now.
            if in_replaced:
                result_entries.append(BufferEntry(data=line_text, timestamp=now, seen=True))
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
                        BufferEntry(data=line_text, timestamp=old_timestamps[abs_mapped], seen=True)
                    )
                    used_old_indices.add(abs_mapped)
                else:
                    result_entries.append(BufferEntry(data=line_text, timestamp=now, seen=True))

        # Commit the new line entries back into the buffer at the target range.
        buf.lines[lo:hi] = result_entries
        buf.modified_at = now
        return {"ok": True, "count": len(char_ranges), "old_string": old_string}

    @tool
    def diff_buffers(self, a: str, b: str, overwrite: bool = False) -> dict[str, Any]:
        """Diff two buffers line-by-line using a unified diff.
        Stores the result in a target buffer named diff:a→b.
        Use overwrite=True to overwrite an existing diff buffer.
        """
        if a not in self._buffers:
            return {"ok": False, "error": f"No buffer named '{a}'."}
        if b not in self._buffers:
            return {"ok": False, "error": f"No buffer named '{b}'."}
        buf_a = [e.data for e in self._buffers[a].lines]
        buf_b = [e.data for e in self._buffers[b].lines]
        if buf_a == buf_b:
            return {"ok": True, "identical": True, "a": a, "b": b}
        diff_name = f"diff:{a}→{b}"
        if diff_name in self._buffers and not overwrite:
            return {
                "ok": False,
                "error": (
                    f"Target buffer '{diff_name}' already exists. "
                    f"Use overwrite=True to replace it, or drop it first with drop_buffer."
                ),
            }
        import difflib
        diff_lines = list(difflib.unified_diff(
            buf_a, buf_b,
            fromfile=f"buffer:{a}", tofile=f"buffer:{b}",
            lineterm="",
        ))
        now = time.time()
        self._buffers[diff_name] = Buffer(lines=[BufferEntry(data=line, timestamp=now, seen=True) for line in diff_lines], created_at=now, modified_at=now)
        return {"ok": True, "buffer": diff_name, "lines": len(diff_lines)}

    def _lines_to_skip(self, segment: list[str], start_offset: int) -> tuple[list[tuple[int, int]], int]:
        """Find the minimum set of longest lines whose removal makes the segment fit under MAX.

        Args:
            segment: list of line strings
            start_offset: 1-based index of the first line in the segment

        Returns:
            (list of (1-based line index, char_count) to skip, sorted by line index,
             total chars of remaining lines including their newlines)
        """
        lines_with_idx = [(start_offset + i, l) for i, l in enumerate(segment)]
        by_len = sorted(lines_with_idx, key=lambda x: len(x[1]), reverse=True)
        seg_total = sum(len(l) for l in segment) + len(segment)
        skip = []
        skip_total_chars = 0
        for idx, line in by_len:
            # if we do not exceed the threshold, we can stop removing / skipping lines
            if seg_total - skip_total_chars <= BufferManager._MAX_CHUNK_CHARS:
                break

            skip.append((idx, len(line)))
            skip_total_chars += len(line) + 1
        return (skip, seg_total - skip_total_chars)

    def _make_buckets(self, segment: list[str], num_buckets: int, start_offset: int) -> list[tuple[int, int, int]]:
        """Divide segment into num_buckets buckets. Returns list of (start, end, char_count) with absolute line numbers."""
        n = len(segment)
        if n == 0:
            return []
        bucket_size = n // num_buckets
        buckets = []
        for b in range(num_buckets):
            start = b * bucket_size
            if b == num_buckets - 1:
                end = n
            else:
                end = start + bucket_size
            lines = segment[start:end]
            chars = sum(len(l) + 1 for l in lines)
            buckets.append((start_offset + start, start_offset + end - 1, chars))
        return buckets

    def _cluster_lines_to_ranges(self, name: str, line_numbers: list[int], num_clusters: int) -> list[tuple[int, int, int]]:
        """Cluster matching line numbers into ranges by repeatedly merging closest neighbors.

        Agglomerative clustering: each line starts as its own cluster with count 1.
        Repeatedly merge the pair of neighboring clusters with the smallest
        content-based distance (sum of both cluster char counts plus char count
        of lines between them). Stops when num_clusters remain.
        Returns list of (cluster_start, cluster_end, match_count).
        """
        if not line_numbers or name not in self._buffers:
            return []
        if len(line_numbers) <= num_clusters:
            return [(ln, ln, 1) for ln in line_numbers]

        buf_lines = self._buffers[name].lines

        def span_chars(start: int, end: int) -> int:
            return sum(len(buf_lines[i].data) + 1 for i in range(start, end + 1))

        clusters = [
            {'s': ln, 'e': ln, 'v': len(buf_lines[ln].data) + 1, 'n': 1}
            for ln in line_numbers
        ]

        def neighbor_distance(i: int) -> int:
            a = clusters[i]
            b = clusters[i + 1]
            between = span_chars(a['e'] + 1, b['s'])
            return a['v'] + b['v'] + between

        # TODO: Replace O(n^2) linear scan with O(n log n) min-heap.
        #   Maintain a priority queue of (distance, index) for adjacent cluster pairs.
        #   After each merge, re-insert the two new neighbor distances.
        #   This mirrors mesh simplification (e.g. QSlim): each merge updates only
        #   the affected local region, not the entire list.
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
                'n': a['n'] + b['n']
            }
            clusters = clusters[:min_i] + [merged] + clusters[min_i + 2:]

        return [(c['s'], c['e'], c['n']) for c in clusters]

