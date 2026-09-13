from __future__ import annotations
import difflib
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from peteos.oap.agentic_object import AgenticObject
from peteos.oap.decorators import tool


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
    - Use list_buffers to see what exists.
    - Use (?i) at the start of a grep pattern for case-insensitive matching.
    - Always prefer read_buffer with start/end over reading entire buffers when working with large content.
    """

    _MAX_CHUNK_CHARS = 8000
    _NUM_CLUSTERS = 5

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._buffers: dict[str, Buffer] = {}

    @tool(description="List all existing buffers by name, showing line count for each.")
    def list_buffers(self) -> dict[str, int]:
        """List all buffers and their line counts."""
        return {name: len(buf.lines) for name, buf in self._buffers.items()}

    def _create_buffer(self, name: str, text: str | None = None, modified_at: float | None = None, overwrite: bool = False) -> int | None:
        """Internal buffer creation. Returns number of lines stored, or None if buffer exists and overwrite=False."""
        ts = modified_at if modified_at is not None else time.time()
        if name in self._buffers:
            if not overwrite:
                return None
        self._buffers[name] = Buffer(lines=[], created_at=ts, modified_at=ts)
        if text is None:
            return 0
        self._buffers[name].lines = [BufferEntry(data=line, timestamp=ts, seen=True) for line in text.splitlines()]
        return len(self._buffers[name].lines)

    @tool(description="Create a new buffer with the given name. Optionally provide initial text. Use overwrite=True to replace an existing buffer.")
    def create_buffer(self, name: str, text: str | None = None, overwrite: bool = False) -> str:
        """Create a new named buffer, optionally populated with text."""
        existed = name in self._buffers
        result = self._create_buffer(name, text, overwrite=overwrite)
        if result is None:
            return f"Error: a buffer named '{name}' already exists. Use overwrite=True to replace it."
        if existed:
            return f"Buffer '{name}' overwritten with {result} lines."
        return f"Buffer '{name}' created ({result} lines)."

    @tool(description="Copy a buffer to a new name. Copies all lines by value. Timestamps of individual lines are preserved. Set overwrite=True to replace an existing target buffer.")
    def copy_buffer(self, source_name: str, target_name: str, overwrite: bool = False) -> str:
        """Copy a buffer by value to a new name."""
        if source_name not in self._buffers:
            return f"Error: no buffer named '{source_name}'."
        if target_name in self._buffers and not overwrite:
            return f"Error: buffer '{target_name}' already exists. Use overwrite=True to replace it."
        src = self._buffers[source_name]
        now = time.time()
        new_entries = [BufferEntry(data=e.data, timestamp=e.timestamp, seen=e.seen) for e in src.lines]
        self._buffers[target_name] = Buffer(lines=new_entries, created_at=now, modified_at=now)
        return f"Buffer '{target_name}' copied from '{source_name}' ({len(new_entries)} lines)."

    @tool(description="Fill a buffer with text. Each line in the text becomes one line in the buffer. Replaces existing content. The buffer must already exist.")
    def write_buffer(self, name: str, text: str) -> str:
        """Write text into a named buffer, replacing its content."""
        if name not in self._buffers:
            return f"Error: no buffer named '{name}'. Use create_buffer first."
        result = self._create_buffer(name, text=text, overwrite=True)
        return f"Wrote {result} lines to buffer '{name}'."

    @tool(description="Drop (delete) a buffer by name. The buffer and its content are discarded.")
    def drop_buffer(self, name: str) -> str:
        """Delete a named buffer."""
        if name not in self._buffers:
            return f"Error: no buffer named '{name}'."
        del self._buffers[name]
        return f"Buffer '{name}' dropped."

    @tool(description="Search a buffer for a regex pattern. Returns a list of (start, end, match_count) ranges. Use (?i) at the start for case-insensitive matching.")
    def grep_buffer(self, name: str, pattern: str) -> list[tuple[int, int, int]]:
        """Search buffer for pattern, return clustered ranges with match counts."""
        if name not in self._buffers:
            return []
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{pattern}': {e}")
        matches = []
        buf = self._buffers[name]
        for i, entry in enumerate(buf.lines, start=1):
            if compiled.search(entry.data):
                matches.append(i)
        return self._cluster_lines_to_ranges(name, matches, self._NUM_CLUSTERS)

    @tool(description="Read a range of lines from a buffer. start and end are 1-based and inclusive. Set show_timestamps=True to prefix each line with its unix timestamp.")
    def read_buffer(self, name: str, start: int | None = None, end: int | None = None, show_timestamps: bool = False) -> str:
        """Read buffer content with size guard-rails. Returns a hint if the range is too large."""
        if name not in self._buffers:
            return f"Error: no buffer named '{name}'. Use create_buffer first."
        buf = self._buffers[name]
        total = len(buf.lines)

        if start is not None and end is not None:
            if start > end:
                return f"Error: start ({start}) > end ({end}). Buffer has {total} lines."
        if start is not None and start > total:
            return f"Error: start ({start}) is beyond buffer length ({total} lines)."
        if end is not None and end < 1:
            return f"Error: end ({end}) must be at least 1."

        s = max(0, (start - 1) if start is not None else 0)
        e = min(total, end if end is not None else total)
        if show_timestamps:
            segment = [f"({entry.timestamp:.6f}) {entry.data}" for entry in buf.lines[s:e]]
        else:
            segment = [entry.data for entry in buf.lines[s:e]]
        total_chars = sum(len(l) for l in segment) + len(segment)

        if total_chars <= BufferManager._MAX_CHUNK_CHARS:
            for entry in buf.lines[s:e]:
                entry.seen = True
            return "\n".join(segment)

        to_skip, rest = self._lines_to_skip(segment, s + 1)
        if to_skip and len(to_skip) <= 10:
            skip_msg = ", ".join(f"line {idx}({cl} chars)" for idx, cl in to_skip)
            return (
                f"Range [{s+1}–{e}] is {total_chars} chars ({len(segment)} lines). "
                f"Heaviest lines ({len(to_skip)} totaling {sum(c for _, c in to_skip)} chars): {skip_msg}. "
                f"Consider re-reading with a narrower range to avoid them."
            )

        buckets = self._make_buckets(segment, 10, s + 1)
        bucket_msgs = ", ".join(f"{sa}-{en}:{c}" for sa, en, c in buckets)
        return (
            f"Range [{s+1}–{e}] is {total_chars} chars ({len(segment)} lines) and exceeds the {BufferManager._MAX_CHUNK_CHARS}-char threshold. "
            f"Reduce the read range to stay under the limit. "
            f"Bucket distribution (start-end:chars): {bucket_msgs}."
        )

    @tool(description="Replace old_string with new_string in the buffer content. Both old_string and new_string can span multiple lines. Use replace_all to replace all occurrences (default False — errors if old_string appears more than once).")
    def edit_buffer(self, name: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
        """Replace text content with new text, rebuild lines array."""
        if name not in self._buffers:
            return f"Error: no buffer named '{name}'. Use create_buffer first."
        buf = self._buffers[name]
        old_entries = buf.lines
        old_timestamps = [e.timestamp for e in old_entries]
        original_content = "\n".join(entry.data for entry in old_entries)

        all_ranges: list[tuple[int, int]] = []
        pos = 0
        while True:
            idx = original_content.find(old_string, pos)
            if idx == -1:
                break
            all_ranges.append((idx, idx + len(old_string)))
            pos = idx + 1

        if len(all_ranges) == 0:
            return f"Error: '{old_string}' not found in buffer."

        if not replace_all and len(all_ranges) > 1:
            line_numbers = [i + 1 for i, entry in enumerate(buf.lines) if old_string in entry.data]
            clusters = self._cluster_lines_to_ranges(name, line_numbers, 5)
            cluster_msgs = ", ".join(f"lines {s}–{e} ({n} occurrence(s))" for s, e, n in clusters)
            return (
                f"Error: '{old_string}' found multiple times ({original_content.count(old_string)} total) at: {cluster_msgs}. "
                "Set replace_all=True to replace all occurrences."
            )
        char_ranges = all_ranges if replace_all else [all_ranges[0]]

        new_content = original_content
        for start, end in reversed(char_ranges):
            new_content = new_content[:start] + new_string + new_content[end:]

        if not replace_all and len(char_ranges) > 1:
            line_numbers = [i + 1 for i, entry in enumerate(buf.lines) if old_string in entry.data]
            clusters = self._cluster_lines_to_ranges(name, line_numbers, 5)
            cluster_msgs = ", ".join(f"lines {s}–{e} ({n} occurrence(s))" for s, e, n in clusters)
            return (
                f"Error: '{old_string}' found multiple times ({original_content.count(old_string)} total) at: {cluster_msgs}. "
                "Set replace_all=True to replace all occurrences."
            )
        if len(char_ranges) == 0:
            return f"Error: '{old_string}' not found in buffer."

        new_content = original_content
        for start, end in reversed(char_ranges):
            new_content = new_content[:start] + new_string + new_content[end:]

        new_lines = new_content.splitlines()
        now = time.time()
        result_entries: list[BufferEntry] = []
        used_old_indices: set[int] = set()

        sorted_ranges = sorted(char_ranges)

        replaced_line_indices: set[int] = set()
        for r_start, r_end in sorted_ranges:
            first_newline = original_content[:r_start].count('\n')
            last_newline = original_content[:r_end].count('\n') - 1
            for li in range(first_newline, last_newline + 1):
                replaced_line_indices.add(li)

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

            if in_replaced and matched_range is not None:
                r_start, r_end = matched_range
                net_change = len(new_string) - (r_end - r_start)
                if net_change < 0:
                    remapped_pos = char_pos_new - net_change
                else:
                    remapped_pos = char_pos_new - net_change
                if not (r_start <= remapped_pos < r_end):
                    in_replaced = False

            if in_replaced:
                result_entries.append(BufferEntry(data=line_text, timestamp=now, seen=True))
            else:
                mapped_pos = char_pos_new - effective_offset
                if mapped_pos <= 0:
                    mapped_line_idx = 0
                else:
                    mapped_line_idx = original_content[:mapped_pos].count('\n')
                if (
                    new_i < len(old_entries)
                    and old_entries[new_i].data == line_text
                    and new_i not in used_old_indices
                    and new_i not in replaced_line_indices
                ):
                    result_entries.append(
                        BufferEntry(data=line_text, timestamp=old_timestamps[new_i], seen=True)
                    )
                    used_old_indices.add(new_i)
                elif (
                    mapped_line_idx < len(old_entries)
                    and old_entries[mapped_line_idx].data == line_text
                    and mapped_line_idx not in used_old_indices
                    and mapped_line_idx not in replaced_line_indices
                ):
                    result_entries.append(
                        BufferEntry(data=line_text, timestamp=old_timestamps[mapped_line_idx], seen=True)
                    )
                    used_old_indices.add(mapped_line_idx)
                else:
                    result_entries.append(BufferEntry(data=line_text, timestamp=now, seen=True))

        buf.lines = result_entries
        buf.modified_at = now
        count = len(char_ranges)
        return f"Replaced {count} occurrence(s) of '{old_string}'."

    @tool(description="Diff two buffers line-by-line using a unified diff. Stores the result in a target buffer named diff:a→b. Use overwrite=True to overwrite an existing diff buffer.")
    def diff_buffers(self, a: str, b: str, overwrite: bool = False) -> str:
        """Diff two buffers and store the result in a diff buffer."""
        if a not in self._buffers:
            return f"Error: no buffer named '{a}'."
        if b not in self._buffers:
            return f"Error: no buffer named '{b}'."
        buf_a = [e.data for e in self._buffers[a].lines]
        buf_b = [e.data for e in self._buffers[b].lines]
        if buf_a == buf_b:
            return f"No differences — buffers '{a}' and '{b}' are identical."
        diff_name = f"diff:{a}→{b}"
        if diff_name in self._buffers and not overwrite:
            return (
                f"Target Buffer '{diff_name}' already exists. "
                f"To overwrite it, call diff_buffers again with overwrite=True. "
                f"Alternatively, drop it first with drop_buffer. "
                f"Make sure not to overwrite or drop data that you still need."
            )
        import difflib
        diff_lines = list(difflib.unified_diff(
            buf_a, buf_b,
            fromfile=f"buffer:{a}", tofile=f"buffer:{b}",
            lineterm="",
        ))
        now = time.time()
        self._buffers[diff_name] = Buffer(lines=[BufferEntry(data=line, timestamp=now, seen=True) for line in diff_lines], created_at=now, modified_at=now)
        return f"Diff written to buffer '{diff_name}' ({len(diff_lines)} lines). Use read_buffer to access it."

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
            return sum(len(buf_lines[i - 1]) + 1 for i in range(start, end + 1))

        clusters = [
            {'s': ln, 'e': ln, 'v': len(buf_lines[ln - 1]) + 1, 'n': 1}
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
