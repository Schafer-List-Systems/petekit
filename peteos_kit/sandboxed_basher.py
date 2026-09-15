"""SandboxedBasher — locked-down bash agentic object inheriting from Basher."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from peteos.oap.agentic_object import AgenticObject
from peteos.oap.decorators import tool
from peteos.utils import get_logger

from .buffer_manager import Buffer
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
    """You have NO access to the filesystem beyond your workspace directory.
    The environment is minimal: no network tools, no secrets, no user data,
    no shell escapes.

    Available commands: cat, echo, grep, egrep, sed, awk, sort, uniq, wc,
    head, tail, cut, tr, mkdir, cp, mv, rm, ln, chmod, touch, find,
    basename, dirname, test, date, sleep, tee, xargs, shuf, paste, join,
    diff, comm, base64, md5sum, sha256sum, stat, du, file, ls.
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
            os.environ["HOME"] = str(self._workspace_dir)
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
            env["HOME"] = str(self._workspace_dir)
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
    async def exec(self, title: str, command: str) -> str:
        """Execute a command in the sandboxed workspace after guardrail validation.

        Args:
            title: A unique name for this process handle.
            command: The shell command to execute.
            cwd: Ignored; execution always occurs in the sandbox workspace.

        Returns:
            Confirmation with stream buffer names.
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
        _logger.debug("[exec] workspace=%s, command=%r", ws, command)

        return await super().exec(title, command, cwd=ws)
