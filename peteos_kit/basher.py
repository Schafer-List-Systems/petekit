"""Basher — unrestricted bash execution agentic object."""

from __future__ import annotations

import subprocess
from pathlib import Path

from peteos.oap.agentic_object import AgenticObject
from peteos.oap.decorators import tool

from .buffer import Buffer, BufferManager


class Basher(BufferManager, AgenticObject):
    """You are a bash agent with access to the filesystem and shell commands.

    Commands are executed in the current working directory. Output from
    bash commands is stored in buffers and must be retrieved with read_buffer.

    Use put_file to stage a file into a buffer, then write it to disk with
    bash_exec (echo from the buffer). Use get_file to read file contents back.

    The agent has no built-in restrictions — it can access the full filesystem
    and run any command the user permits. Be mindful of destructive operations.
    """

    def __init__(self) -> None:
        super().__init__()
        self._bash_buffer_counter: int = 0

    def _next_buffer_name(self) -> str:
        self._bash_buffer_counter += 1
        return f"bash:{self._bash_buffer_counter}"

    @tool
    def bash_exec(self, command: str, timeout: int = 30) -> str:
        """Execute a bash command.

        Output is stored in a buffer. Use read_buffer to access the result.

        Args:
            command: The bash command to execute.
            timeout: Max seconds before the command is killed.

        Returns:
            A hint with buffer name, line count, and exit code.
        """
        if not command.strip():
            return "Error: Empty command."

        try:
            import time as time_mod
            result = subprocess.run(
                ["/bin/sh", "-c", command],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            buffer_name = self._next_buffer_name()
            output_lines = []
            if result.stdout:
                output_lines.extend(result.stdout.splitlines())
            if result.stderr:
                output_lines.append(f"[stderr] {result.stderr}")
            now = time_mod.time()
            self._buffers[buffer_name] = Buffer(
                lines=output_lines,
                created_at=now,
                modified_at=now,
            )
            return (
                f"Output stored in buffer '{buffer_name}' ({len(output_lines)} lines, "
                f"exit_code={result.returncode}). "
                f"Use read_buffer to access the content."
            )
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds."
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    @tool
    def put_file(self, buffer_name: str, file_name: str) -> str:
        """Stage a file into the workspace from a buffer.

        Args:
            buffer_name: The name of the buffer containing the file content.
            file_name: The filename to write to disk.

        Returns:
            Confirmation message.
        """
        if buffer_name not in self._buffers:
            return f"Error: buffer '{buffer_name}' not found. Stage the content first."
        filepath = Path(file_name).resolve()
        try:
            filepath.write_text("\n".join(self._buffers[buffer_name].lines))
            return f"OK: File '{file_name}' written."
        except Exception as e:
            return f"Error: Failed to write '{file_name}': {e}"

    @tool
    def get_file(self, file_name: str) -> str:
        """Read the content of a file from disk into a buffer.

        Args:
            file_name: The filename to read.

        Returns:
            The file contents as a string.
        """
        filepath = Path(file_name).resolve()
        if not filepath.is_file():
            return f"Error: File '{file_name}' not found."
        try:
            return filepath.read_text()
        except Exception as e:
            return f"Error: Failed to read '{file_name}': {e}"
