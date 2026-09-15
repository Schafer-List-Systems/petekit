"""Basher — process execution agentic object with stream buffer support."""

from __future__ import annotations

import asyncio
import subprocess
import time
from dataclasses import dataclass

from peteos.oap.agentic_object import AgenticObject
from peteos.oap.decorators import tool

from .stream_buffer_manager import StreamBufferManager


@dataclass
class BashHandle:
    """Holds the execution context for a tracked process."""
    process_id: int
    title: str
    command: str
    cwd: str | None = None
    process: asyncio.subprocess.Process | None = None
    task: asyncio.Task | None = None
    stdin_buffer: str = ""
    stdout_buffer: str = ""
    stderr_buffer: str = ""
    closed: bool = False
    exit_code: int | None = None


class Basher(StreamBufferManager, AgenticObject):
    """You are a bash agent with access to the filesystem and shell commands.

    Commands are executed in the current working directory. Output from
    bash commands is stored in stream buffers and must be retrieved with
    read_buffer or read_stream_buffer.

    Use exec to run a command in the background with live stream buffers.
    Use bash_exec for one-shot synchronous command execution.
    Use put_file to stage a file into a buffer, then write it to disk.
    Use get_file to read file contents back.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._processes: dict[int, BashHandle] = {}
        self._bash_buffer_counter: int = 0

    def _next_buffer_name(self) -> str:
        self._bash_buffer_counter += 1
        return f"bash:{self._bash_buffer_counter}"

    def _next_process_id(self) -> int:
        self._bash_buffer_counter += 1
        return self._bash_buffer_counter

    @tool
    async def exec(self, title: str, command: str, cwd: str | None = None) -> str:
        """Start a command in the background, creating live in/out stream buffers.

        Args:
            name: A unique name for this process handle.
            command: The shell command to execute.
            cwd: Optional working directory.

        Returns:
            Confirmation with stream buffer names.
        """
        if not command.strip():
            raise ValueError("Empty command.")

        now = time.time()
        process_id = self._next_process_id()
        stdin_buffer = f"stream:bash:stdin:{process_id}"
        stdout_buffer = f"stream:bash:stdout:{process_id}"
        stderr_buffer = f"stream:bash:stderr:{process_id}"
        self._create_stream(stdin_buffer)
        self._create_stream(stdout_buffer)
        self._create_stream(stderr_buffer)

        try:
            process = await asyncio.create_subprocess_exec(
                "/bin/sh", "-c", command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
            )
        except Exception as e:
            self._drop_stream(stdin_buffer)
            self._drop_stream(stdout_buffer)
            self._drop_stream(stderr_buffer)
            raise RuntimeError(f"Failed to start process: {e}") from e

        handle = BashHandle(
            process_id=process_id,
            title=title,
            command=command,
            cwd=cwd,
            process=process,
            stdin_buffer=stdin_buffer,
            stdout_buffer=stdout_buffer,
            stderr_buffer=stderr_buffer,
        )
        self._processes[process_id] = handle
        handle.task = asyncio.create_task(self._bash_read_loop(process_id, handle))
        return (
            f"Started process [{process_id}] '{title}': {command}. "
            f"Streams created at {now}. "
            f"stdin='{stdin_buffer}', stdout='{stdout_buffer}', stderr='{stderr_buffer}'. "
            f"Use list_processes to track, terminate to stop."
        )

    @tool
    def list_processes(self) -> list[dict]:
        """List all running processes with their details."""
        return [
            {
                "process_id": process_id,
                "title": h.title,
                "command": h.command,
                "cwd": h.cwd,
                "running": not h.closed,
                "exit_code": h.exit_code,
            }
            for process_id, h in sorted(self._processes.items())
        ]

    @tool
    async def bash_send(self, process_id: int, text: str = "", flush: bool = False, trailing_newline: bool = False) -> str:
        """Send text to a running process's stdin. Pass flush=True to drain the write buffer without closing. Pass trailing_newline=True to append a trailing newline (as if pressing Enter)."""
        if process_id not in self._processes:
            raise KeyError(f"No process with id '{process_id}'.")
        handle = self._processes[process_id]
        if handle.closed:
            raise ConnectionError(f"Process '{handle.title}' is not running (exit_code={handle.exit_code}).")
        if handle.process is None or handle.process.stdin is None:
            raise ConnectionError(f"Process '{handle.title}' has no stdin.")
        try:
            if text:
                data = text.encode("utf-8")
                handle.process.stdin.write(data)
                if trailing_newline:
                    handle.process.stdin.write(b"\n")
            if flush:
                await handle.process.stdin.drain()
        except Exception as e:
            raise ConnectionError(f"Send error: {e}")
        for line in text.splitlines():
            await self._append_stream_entry(handle.stdin_buffer, line)
        return f"Sent {len(text)} chars to process '{handle.title}'."

    @tool
    async def terminate(self, process_id: int, drop_buffers: bool = True) -> str:
        """Terminate a running process. Pass drop_buffers=False to keep streams for analysis."""
        if process_id not in self._processes:
            raise KeyError(f"No process with id '{process_id}'.")
        handle = self._processes.pop(process_id)
        if handle.task:
            handle.task.cancel()
        if handle.process and handle.process.returncode is None:
            handle.process.terminate()
            try:
                await asyncio.wait_for(handle.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                handle.process.kill()
            except asyncio.CancelledError:
                handle.process.kill()
        handle.closed = True
        handle.exit_code = handle.process.returncode if handle.process else None
        if drop_buffers:
            self._drop_stream(handle.stdin_buffer)
            self._drop_stream(handle.stdout_buffer)
            self._drop_stream(handle.stderr_buffer)
            return (
                f"Terminated process [{handle.process_id}] '{handle.title}': {handle.command} "
                f"(exit_code={handle.exit_code}). Dropped buffers."
            )
        return (
            f"Terminated process [{handle.process_id}] '{handle.title}': {handle.command} "
            f"(exit_code={handle.exit_code}). "
            f"Buffers '{handle.stdin_buffer}', '{handle.stdout_buffer}', '{handle.stderr_buffer}' remain. "
            f"Use drop_buffer to remove them."
        )

    async def _bash_read_loop(self, process_id: int, handle: BashHandle) -> None:
        """Continuously read stdout and stderr from the process using concurrent tasks."""
        stdout = handle.process.stdout
        stderr = handle.process.stderr

        async def read_stdout() -> None:
            if stdout is None:
                return
            try:
                while True:
                    line_bytes = await stdout.readline()
                    if not line_bytes:
                        break
                    line = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
                    await self._append_stream_entry(handle.stdout_buffer, line)
            except asyncio.CancelledError:
                pass

        async def read_stderr() -> None:
            if stderr is None:
                return
            try:
                while True:
                    line_bytes = await stderr.readline()
                    if not line_bytes:
                        break
                    line = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
                    await self._append_stream_entry(handle.stderr_buffer, line)
            except asyncio.CancelledError:
                pass

        out_task = asyncio.create_task(read_stdout())
        err_task = asyncio.create_task(read_stderr())
        try:
            await asyncio.gather(out_task, err_task)
        except asyncio.CancelledError:
            out_task.cancel()
            err_task.cancel()
            raise
        finally:
            handle.closed = True
            if handle.process:
                handle.exit_code = handle.process.returncode
                try:
                    await handle.process.wait()
                except asyncio.CancelledError:
                    pass
            else:
                handle.exit_code = None
