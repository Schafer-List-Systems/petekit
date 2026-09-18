"""TextEditor agentic object — file editing backed by a multi-file buffer manager."""

from __future__ import annotations

import difflib
from typing import Any
from ..utils.three_merge import merge
import os

from dataclasses import dataclass

from peteos.oap.agentic_object import AgenticObject
from peteos.oap.decorators import tool

from .buffer_manager import BufferEntry, BufferManager


@dataclass
class ExpectedFileData:
    """The file state the agent expects — mtime and content lines when loaded or last stored."""
    mtime: float
    content: list[str]


class TextEditor(BufferManager, AgenticObject):
    """You are a text editor. You load files into memory, edit them, and write them back.

    Workflow:
      1. load_text(file_path) — load a file into a buffer (keyed by its path)
      2. read_buffer(...) — read a line range
      3. edit_buffer(...) — make in-memory changes
      4. store_text(buffer_name, file_path) — write back to disk (refuses if file was modified externally)
    """

    def __init__(self) -> None:
        super().__init__()
        self.expected_file_state: dict[str, ExpectedFileData] = {}

    @tool(description="Load a file from disk into a buffer named 'file:<abs_path>'.")
    def load_text(self, file_path: str, overwrite_internal_buffer: bool = False) -> dict[str, Any]:
        """Load a file. Creates it if it does not exist. Sets modified_at from the file's mtime."""
        abs_path = os.path.abspath(file_path)
        if not os.path.isfile(abs_path):
            return {"ok": False, "error": f"File '{abs_path}' does not exist. Cannot load a non-existent file."}
        try:
            with open(abs_path, "r", encoding="utf-8") as f:
                content = f.read()
            mtime = os.path.getmtime(abs_path)
            self.expected_file_state[abs_path] = ExpectedFileData(mtime=mtime, content=content.splitlines())
            key = f"file:{abs_path}"
            count = self._create_buffer(key, text=content, modified_at=mtime, overwrite=overwrite_internal_buffer)
            if count is None:
                return {"ok": False, "error": f"Buffer 'file:{abs_path}' already exists. Use overwrite_internal_buffer=True to replace it."}
            return {"ok": True, "buffer": key, "lines": count}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    @tool(description="Write a buffer to a file on disk. If merge_changes=True, attempts a 3-way merge when the file was modified externally.")
    def store_text(self, buffer_name: str, file_path: str, merge_changes: bool = False) -> dict[str, Any]:
        """Store buffer to disk, checking mtime guard rail via expected_file_state."""
        if buffer_name not in self._buffers:
            return {"ok": False, "error": f"No buffer named '{buffer_name}'. Use load_text first."}
        buf = self._buffers[buffer_name]
        abs_path = os.path.abspath(file_path)
        if os.path.isfile(abs_path):
            if abs_path not in self.expected_file_state:
                return {"ok": False, "error": f"No expected_file_state for '{abs_path}'. Use load_text first."}
            try:
                current_mtime = os.path.getmtime(abs_path)
            except FileNotFoundError:
                pass
            else:
                if current_mtime > self.expected_file_state[abs_path].mtime:
                    if not merge_changes:
                        return {
                            "ok": False,
                            "error": (
                                f"File '{abs_path}' was modified externally. "
                                f"Your loaded mtime: {self.expected_file_state[abs_path].mtime}, current mtime: {current_mtime}. "
                                "Use diff_text to see the external changes, or pass merge_changes=True to store_text to attempt a 3-way merge."
                            ),
                        }
                    try:
                        with open(abs_path, "r", encoding="utf-8") as f:
                            disk_content = f.read()
                        disk_lines = disk_content.splitlines()
                    except Exception:
                        pass
                    base = self.expected_file_state[abs_path].content
                    my_lines = [entry.data for entry in buf.lines]
                    merged_text, had_conflicts = merge("\n".join(disk_lines), "\n".join(my_lines), "\n".join(base))
                    try:
                        with open(abs_path, "w", encoding="utf-8") as f:
                            f.write(merged_text)
                        new_mtime = os.path.getmtime(abs_path)
                        buf.modified_at = new_mtime
                        self.expected_file_state[abs_path] = ExpectedFileData(mtime=new_mtime, content=merged_text.splitlines())
                        merged_lines = merged_text.splitlines()
                        buf.lines = [BufferEntry(data=line, timestamp=new_mtime, seen=True) for line in merged_lines]
                    except Exception as e:
                        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
                    if not had_conflicts:
                        return {"ok": True, "file": abs_path, "merged": True, "conflicts": False, "lines": len(merged_lines)}
                    return {
                        "ok": True,
                        "file": abs_path,
                        "merged": True,
                        "conflicts": True,
                        "lines": len(merged_lines),
                        "error": "Unresolved conflicts detected. Use diff_text to locate and resolve them.",
                    }
        try:
            text = "\n".join(entry.data for entry in buf.lines)
            with open(abs_path, "w", encoding="utf-8") as f:
                f.write(text)
            new_mtime = os.path.getmtime(abs_path)
            buf.modified_at = new_mtime
            buf_content = [entry.data for entry in buf.lines]
            self.expected_file_state[abs_path] = ExpectedFileData(mtime=new_mtime, content=buf_content)
            return {"ok": True, "file": abs_path, "lines": len(buf.lines)}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    @tool
    def diff_text(self, file_path: str, scope: str = "external", overwrite: bool = False) -> dict[str, Any]:
        """
        Compare buffer and file states based on scope and store the unified diff in a buffer.

        The diff is computed between two of these three states:
          - buffer: the current in-memory buffer content
          - expected: the file content when it was loaded or last stored (your base)
          - disk: the actual file on disk right now

        scope selects which two are compared:
          - mine: expected vs buffer — what edits you made since loading the file.
          - external [DEFAULT]: expected vs disk — what changed externally since you loaded the file.
          - buffer_disk: buffer vs disk — what would change if you would overwrite external changes.

        The unified diff is stored in a buffer named 'diff:<abs_path>'.
        Use overwrite=True to overwrite an existing diff buffer.
        """
        abs_path = os.path.abspath(file_path)
        key = f"file:{abs_path}"
        if key not in self._buffers:
            return {"ok": False, "error": f"no buffer named 'file:{abs_path}'. Use load_text first."}
        if scope != "buffer_disk":
            if abs_path not in self.expected_file_state:
                return {"ok": False, "error": f"no expected_file_state for '{abs_path}'. Use load_text first."}
        if scope != "mine":
            if not os.path.isfile(abs_path):
                return {"ok": False, "error": f"file '{abs_path}' no longer exists."}
        buf = self._buffers[key]
        buf_data_lines = [entry.data for entry in buf.lines]
        base_lines = self.expected_file_state[abs_path].content if abs_path in self.expected_file_state else []
        disk_lines: list[str] = []
        if scope != "mine":
            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    disk_content = f.read()
                disk_lines = disk_content.splitlines()
            except Exception as e:
                return {"ok": False, "error": f"reading file: {type(e).__name__}: {e}"}
        if scope == "mine":
            left_lines = base_lines
            right_lines = buf_data_lines
            scope_hint = f"[scope=mine: expected vs buffer — your edits from the original]"
        elif scope == "external":
            left_lines = base_lines
            right_lines = disk_lines
            scope_hint = f"[scope=external: expected vs disk — external changes since load]"
        else:
            left_lines = buf_data_lines
            right_lines = disk_lines
            scope_hint = f"[scope=buffer_disk: buffer vs disk — current divergence]"
        if left_lines == right_lines:
            return {"ok": True, "identical": True, "scope": scope, "scope_hint": scope_hint}
        diff_name = f"diff:{abs_path}"
        if diff_name in self._buffers and not overwrite:
            return {
                "ok": False,
                "error": f"buffer '{diff_name}' already exists",
                "diff_buffer": diff_name,
                "hint": "call diff_text again with overwrite=True, or drop it first with drop_buffer",
            }
        import difflib
        diff_lines = list(difflib.unified_diff(
            left_lines, right_lines,
            fromfile=abs_path, tofile=abs_path,
            lineterm="",
        ))
        import time as time_mod
        now = time_mod.time()
        diff_text = "\n".join(diff_lines)
        self._create_buffer(diff_name, text=diff_text, modified_at=now)
        return {"ok": True, "diff_buffer": diff_name, "lines": len(diff_lines)}
