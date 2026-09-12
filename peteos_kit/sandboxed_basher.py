"""SandboxedBasher — locked-down bash agentic object inheriting from Basher."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from peteos.oap.agentic_object import AgenticObject
from peteos.oap.decorators import tool
from peteos.utils import get_logger

from .buffer import Buffer
from .basher import Basher

_logger = get_logger(__name__)

# Safe POSIX utilities available in the sandbox.
# No network tools (curl, wget, nc), no shells (bash -c via env vars), no
# process control (kill, top), no system introspection (ps, whoami).
_SAFE_COMMANDS: set[str] = {
    "cat", "echo", "grep", "egrep", "sed", "awk", "sort", "uniq", "wc",
    "head", "tail", "cut", "tr", "mkdir", "cp", "mv", "rm", "ln", "chmod",
    "touch", "find", "basename", "dirname", "test", "date", "sleep",
    "tee", "xargs", "shuf", "paste", "join", "diff", "comm", "uniq",
    "base64", "md5sum", "sha256sum", "stat", "du", "file", "ls",
}

# Strictly minimal POSIX environment: no PATH leaks, no secrets, no user config.
_MINIMAL_ENV: dict[str, str] = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/tmp/bash_workspace_home",
    "LANG": "C",
    "LC_ALL": "C",
    "TERM": "dumb",
    "TMPDIR": "/tmp",
}


class SandboxedBasher(Basher, AgenticObject):
    """You are a secure bash workspace for executing shell commands.

    You can run bash commands on files placed in your workspace. You have
    NO access to the rest of the filesystem beyond your workspace directory.
    The environment is minimal: no network tools, no secrets, no user data.

    Your workspace is created fresh for each invocation and destroyed
    afterward. Files must be provided to you — you cannot access anything
    outside your workspace directory.

    Available commands: cat, echo, grep, sed, awk, sort, uniq, wc, head,
    tail, cut, tr, mkdir, cp, mv, rm, ln, chmod, touch, find, basename,
    dirname, test, date, sleep, tee, xargs, shuf, paste, join, diff, comm,
    base64, md5sum, sha256sum, stat, du, file, ls.

    No network tools (curl, wget, nc) or shell escapes are available.
    """

    def __init__(self) -> None:
        super().__init__()
        self._workspace_dir: Path | None = None

    @property
    def workspace_dir(self) -> Path | None:
        """The current workspace directory, or None if not active."""
        return self._workspace_dir

    def _setup_workspace(self) -> Path:
        """Create or return the workspace directory.

        Uses an existing workspace if available, otherwise creates a new
        temporary directory.
        """
        if self._workspace_dir is None or not self._workspace_dir.is_dir():
            self._workspace_dir = Path(tempfile.mkdtemp(prefix="bash_workspace_"))
            (self._workspace_dir / "home").mkdir(exist_ok=True)
            os.environ["HOME"] = str(self._workspace_dir / "home")
        return self._workspace_dir

    def _teardown_workspace(self) -> None:
        """Remove the workspace directory."""
        if self._workspace_dir and self._workspace_dir.is_dir():
            import shutil
            shutil.rmtree(self._workspace_dir, ignore_errors=True)
        self._workspace_dir = None

    @property
    def _env(self) -> dict[str, str]:
        """Build the sandbox environment with the workspace HOME."""
        env = {**_MINIMAL_ENV}
        if self._workspace_dir:
            env["HOME"] = str(self._workspace_dir / "home")
        for key in ("TERM", "LANG", "LC_ALL"):
            val = os.environ.get(key)
            if val:
                env[key] = val
        return env

    def _is_safe_command(self, command: str) -> bool:
        """Check if the command (first word) is in the allowlist."""
        first = command.strip().split()[0] if command.strip() else ""
        if first == "/bin/sh" or first == "sh":
            return True
        base = os.path.basename(first)
        return base in _SAFE_COMMANDS

    @tool
    def bash_exec(self, command: str, timeout: int = 30) -> str:
        """Execute a bash command in the sandboxed workspace.

        Output is stored in a buffer. Use read_buffer to access the result.

        Args:
            command: The bash command to execute.
            timeout: Max seconds before the command is killed.

        Returns:
            A hint with buffer name, line count, and exit code.
        """
        if not command.strip():
            return "Error: Empty command."

        if ".." in command:
            return "Error: Path traversal is not allowed."

        _blocked_chars = ("`", "$(")
        for bad in _blocked_chars:
            if bad in command:
                return f"Error: The character or sequence '{bad}' is not allowed."

        if not self._is_safe_command(command):
            base = os.path.basename(command.strip().split()[0])
            return f"Error: Command '{base}' is not allowed in the sandbox."

        ws = self._setup_workspace()
        _logger.debug("[bash_exec] workspace=%s, command=%r", ws, command)

        try:
            import time as time_mod
            result = subprocess.run(
                ["/bin/sh", "-c", command],
                cwd=ws,
                env=self._env,
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
            file_name: The filename to create in the workspace.

        Returns:
            Confirmation message.
        """
        if buffer_name not in self._buffers:
            return f"Error: buffer '{buffer_name}' not found. Stage the content first."
        ws = self._setup_workspace()
        filepath = (ws / file_name).resolve()
        try:
            filepath.relative_to(ws.resolve())
        except ValueError:
            return f"Error: File name '{file_name}' is not allowed."
        try:
            filepath.write_text("\n".join(self._buffers[buffer_name].lines))
            return f"OK: File '{file_name}' staged in workspace."
        except Exception as e:
            return f"Error: Failed to stage '{file_name}': {e}"

    @tool
    def get_file(self, file_name: str) -> str:
        """Read the content of a file from the workspace.

        Args:
            file_name: The filename to read from the workspace.

        Returns:
            The file contents as a string.
        """
        ws = self._setup_workspace()
        filepath = (ws / file_name).resolve()
        try:
            filepath.relative_to(ws.resolve())
        except ValueError:
            return f"Error: File name '{file_name}' is not allowed."
        if not filepath.is_file():
            return f"Error: File '{file_name}' not found in workspace."
        try:
            return filepath.read_text()
        except Exception as e:
            return f"Error: Failed to read '{file_name}': {e}"
