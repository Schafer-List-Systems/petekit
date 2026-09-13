"""TextEditor agentic object — file editing backed by a multi-file buffer manager."""

from __future__ import annotations

import difflib
import os

from peteos.oap.agentic_object import AgenticObject
from peteos.oap.decorators import tool

from .buffer_manager import Buffer, BufferManager


class TextEditor(BufferManager, AgenticObject):
    """You are a text editor. You load files into memory, edit them, and write them back.

    Workflow:
      1. load_file(file_path) — load a file into a buffer (keyed by its path)
      2. read_buffer(file_path, start=N, end=M) — read a line range
      3. edit_buffer(file_path, old, new) — make in-memory changes
      4. store_file(file_path) — write back to disk (refuses if file was modified externally)
    """

    @tool(description="Load a file from disk into a buffer. The buffer is keyed by its file path.")
    def load_file(self, file_path: str, overwrite_internal_buffer: bool = False) -> str:
        """Load a file. Creates it if it does not exist. Sets modified_at from the file's mtime."""
        abs_path = os.path.abspath(file_path)
        if not os.path.isfile(abs_path):
            with open(abs_path, "w", encoding="utf-8") as f:
                pass
            mtime = os.path.getmtime(abs_path)
            count = self._create_buffer(abs_path, text="", modified_at=mtime)
            return f"Created empty buffer '{abs_path}' (0 lines)."
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()
            mtime = os.path.getmtime(abs_path)
            count = self._create_buffer(abs_path, text=content, modified_at=mtime, overwrite=overwrite_internal_buffer)
            if count is None:
                return f"Error: buffer '{abs_path}' already exists. Use overwrite_internal_buffer=True to replace it."
            return f"Loaded buffer '{abs_path}' ({count} lines)."
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    @tool(description="Write the buffer back to its file on disk. Fails if the file was modified externally since load.")
    def store_file(self, file_path: str) -> str:
        """Store buffer to disk, checking mtime guard rail."""
        if file_path not in self._buffers:
            return f"Error: no buffer for '{file_path}'. Use load_file first."
        abs_path = os.path.abspath(file_path)
        buf = self._buffers[abs_path]
        try:
            current_mtime = os.path.getmtime(abs_path)
        except FileNotFoundError:
            return f"Error: file '{abs_path}' no longer exists."
        if current_mtime != buf.modified_at:
            return (
                f"Error: file '{abs_path}' was modified externally. "
                f"Your loaded mtime: {buf.modified_at}, current mtime: {current_mtime}. "
                "Use load_file to refresh before storing."
            )
        try:
            text = "\n".join(buf.lines)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(text)
            new_mtime = os.path.getmtime(abs_path)
            buf.modified_at = new_mtime
            return f"Stored {len(buf.lines)} lines to {abs_path}."
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    @tool(description="Compare the buffer against the file on disk. Stores the unified diff in a buffer named diff:<file>. Use overwrite=True to overwrite an existing diff buffer.")
    def diff_file(self, file_path: str, overwrite: bool = False) -> str:
        """Diff buffer against disk, store result in a diff buffer."""
        if file_path not in self._buffers:
            return f"Error: no buffer for '{file_path}'. Use load_file first."
        abs_path = os.path.abspath(file_path)
        if not os.path.isfile(abs_path):
            return f"Error: file '{abs_path}' no longer exists."
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                disk_content = f.read()
            disk_lines = disk_content.splitlines()
        except Exception as e:
            return f"Error reading file: {type(e).__name__}: {e}"
        buf = self._buffers[abs_path]
        if buf.lines == disk_lines:
            return "No differences — buffer matches the file on disk."
        diff_name = f"diff:{abs_path}"
        if diff_name in self._buffers and not overwrite:
            return (
                f"Buffer '{diff_name}' already exists. "
                f"To overwrite it, call diff_file again with overwrite=True. "
                f"Alternatively, drop it first with drop_buffer (but only if you no longer need its content)."
            )
        import difflib
        diff_lines = list(difflib.unified_diff(
            disk_lines, buf.lines,
            fromfile=abs_path, tofile=abs_path,
            lineterm="",
        ))
        import time as time_mod
        now = time_mod.time()
        diff_text = "\n".join(diff_lines)
        self._create_buffer(diff_name, text=diff_text, modified_at=now)
        return f"Diff written to buffer '{diff_name}' ({len(diff_lines)} lines). Use read_buffer to access it."
