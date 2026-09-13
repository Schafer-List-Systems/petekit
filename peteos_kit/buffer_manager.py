from __future__ import annotations
import difflib
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from peteos.oap.agentic_object import AgenticObject
from peteos.oap.decorators import tool


@dataclass
class Buffer:
    """A buffer holding text as a list of lines with timestamps."""
    lines: list[str]
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

    @tool(description="Create a new empty buffer with the given name. Fails if a buffer with that name already exists.")
    def create_buffer(self, name: str) -> str:
        """Create a new empty named buffer."""
        if name in self._buffers:
            return f"Error: a buffer named '{name}' already exists."
        now = time.time()
        self._buffers[name] = Buffer(lines=[], created_at=now, modified_at=now)
        return f"Buffer '{name}' created (empty)."

    @tool(description="Fill a buffer with text. Each line in the text becomes one line in the buffer. Replaces existing content. The buffer must already exist.")
    def write_buffer(self, name: str, text: str) -> str:
        """Write text into a named buffer, replacing its content."""
        if name not in self._buffers:
            return f"Error: no buffer named '{name}'. Use create_buffer first."
        now = time.time()
        self._buffers[name].lines = text.splitlines()
        self._buffers[name].modified_at = now
        return f"Wrote {len(self._buffers[name].lines)} lines to buffer '{name}'."

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
        for i, line in enumerate(buf.lines, start=1):
            if compiled.search(line):
                matches.append(i)
        return self._cluster_lines_to_ranges(name, matches, self._NUM_CLUSTERS)

    @tool(description="Read a range of lines from a buffer. start and end are 1-based and inclusive.")
    def read_buffer(self, name: str, start: int | None = None, end: int | None = None) -> str:
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
        segment = buf.lines[s:e]
        total_chars = sum(len(l) for l in segment) + len(segment)

        if total_chars <= BufferManager._MAX_CHUNK_CHARS:
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

    @tool(description="Replace all occurrences of `old` with `new` across all lines in a buffer. old is interpreted as a regex. Use (?i) at the start of old for case-insensitive replacement.")
    def edit_buffer(self, name: str, old: str, new: str) -> str:
        """Replace occurrences of old with new across all lines in a buffer."""
        if name not in self._buffers:
            return f"Error: no buffer named '{name}'. Use create_buffer first."
        try:
            compiled = re.compile(old)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern '{old}': {e}")
        buf = self._buffers[name]
        count = 0
        new_lines: list[str] = []
        for line in buf.lines:
            new_line, n = compiled.subn(new, line)
            new_lines.append(new_line)
            count += n
        buf.lines = new_lines
        buf.modified_at = time.time()
        return f"Replaced {count} occurrence(s) in buffer '{name}'."

    @tool(description="Diff two buffers line-by-line using a unified diff. Stores the result in a target buffer named diff:a→b. Use overwrite=True to overwrite an existing diff buffer.")
    def diff_buffers(self, a: str, b: str, overwrite: bool = False) -> str:
        """Diff two buffers and store the result in a diff buffer."""
        if a not in self._buffers:
            return f"Error: no buffer named '{a}'."
        if b not in self._buffers:
            return f"Error: no buffer named '{b}'."
        buf_a = self._buffers[a].lines
        buf_b = self._buffers[b].lines
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
        self._buffers[diff_name] = Buffer(lines=diff_lines, created_at=now, modified_at=now)
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
