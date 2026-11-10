"""Basher — process execution agentic object with stream buffer support."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Any

from peteos.oap.agentic_object import AgenticObject
from peteos import sandbox, tool
from .stream_buffer_manager import StreamBufferManager
from ..text.buffer_manager import Buffer, BufferManager, HookPoint
from ..text.option_buffer_manager import _origin_name_from_option_buffer_name


_BASH_PROCESSES_BUFFER = "system:status:bash"
_BASH_MANAGER_DOC_BUFFER = "doc:bash"

_logger = logging.getLogger("basher")

# Pattern matching agent-created bash buffers (not system buffers).
_BASH_PROCESS_PATTERN = r"^bash:[^:\[\]]+$"

# New-flow: arbitrary script buffer with [bash] option buffer trigger.
_BASH_SUFFIX = "[bash]"
_BASH_OPTION_BUFFER_PATTERN = rf".+\[{_BASH_SUFFIX[1:-1]}\]$"

_BASH_MANAGER_DOC = f"""\
## Bash Process Manager

You can run bash commands and interact with their input and output as stream buffers.
The system manages process lifecycle — you work directly with the buffers.

### Start a process (two buffers)

**Step 1** — Create a script buffer with your bash code. The name is arbitrary:

```
buffer: my-script.sh
content: cd /tmp && ls -la
```

**Step 2** — Create the `[bash]` option buffer to start the process:

```
buffer: my-script.sh[bash]
content: {{}}
```

The system creates four stream buffers automatically:
- `my-script.sh:stdin` — write to this to send input to the process
- `my-script.sh:stdout` — read from this to see the process output
- `my-script.sh:stderr` — read from this to see error output
- `my-script.sh:status` — current status as JSON

The process starts as soon as the `[bash]` option buffer is created.
The origin buffer (`my-script.sh`) is left intact while the process runs.

### Send input to a running process

Write to the `<origin>:stdin` buffer. Each line you write is forwarded to the
process's standard input. To simulate pressing Enter, write your input with a
trailing newline.

### Read output from a running process

Read from `<origin>:stdout` for normal output, or `<origin>:stderr` for
error output. Both are stream buffers — new output appears here as the process runs.

### Check if a process is still running

Read the `<origin>:status` buffer. It contains JSON:

```json
{{
    "pid": 42,
    "title": "my-script.sh",
    "running": true,
    "exit_code": null
}}
```

When the process exits, `running` becomes `false` and `exit_code` is set.

### Stop a process

**Option 1** — Drop the option buffer `<origin>[bash]`. The process is terminated
and all four associated buffers (stdin, stdout, stderr, status) are removed.
The origin buffer is left intact.

**Option 2** — Drop the origin buffer `<origin>`. The process is terminated and all
buffers are cleaned up. The option buffer is also removed automatically.

A process cannot be stopped while it is in use.

### Monitor all running processes

Read the `system:status:bash` buffer. It lists every running bash process as JSON
with their process IDs, titles, commands, and working directories.

### Requirements

- The script buffer can have any name (e.g. `deploy.sh`, `build.sh`)
- The `[bash]` option buffer must be created after the script buffer exists
- Commands run in the current working directory
- Only one process can run per script buffer at a time
"""


# -------------------------------------------------------------------------------------------------------------------- #
# Data Classes                                                                                                         #
# -------------------------------------------------------------------------------------------------------------------- #


@dataclass
class BashHandle:
    """Holds the execution context for a tracked process."""
    title: str
    command: str
    origin: str = ""  # the script buffer that triggered this process
    cwd: str | None = None
    process: asyncio.subprocess.Process | None = None
    task: asyncio.Task | None = None
    stdin_buffer: str = ""
    stdout_buffer: str = ""
    stderr_buffer: str = ""
    status_buffer: str = ""
    closed: bool = False
    exit_code: int | None = None


# -------------------------------------------------------------------------------------------------------------------- #
# Utils                                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #


def _bash_title_from_name(buffer_name: str) -> str:
    """Parse the title from a bash:{title} buffer name.

    Strips the "bash:" prefix and returns the title. The caller is responsible
    for ensuring the buffer name matches the _BASH_PROCESS_PATTERN.
    """
    return buffer_name[len("bash:") :]


async def _bash_send_hook(basher, title: str, text: str = "", flush: bool = True, trailing_newline: bool = False) -> str:
    """Send text to a running process's stdin. Pass flush=True to drain the write buffer without closing. Pass trailing_newline=True to append a trailing newline (as if pressing Enter)."""
    if title not in basher.bash_processes:
        raise KeyError(f"No process with title '{title}'.")
    handle = basher.bash_processes[title]

    # Guard: reject sends to closed or exited processes.
    if handle.closed:
        raise ConnectionError(f"Process '{handle.title}' is not running (exit_code={handle.exit_code}).")
    if handle.process is None or handle.process.stdin is None:
        raise ConnectionError(f"Process '{handle.title}' has no stdin.")

    # Write: encode text and write to the subprocess stdin; optionally append a trailing newline.
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

    return f"Sent {len(text)} chars to process '{handle.title}'."


async def _bash_read_loop(basher, handle: BashHandle) -> None:
    """Continuously read stdout and stderr from the process using concurrent tasks."""
    stdout = handle.process.stdout
    stderr = handle.process.stderr

    # Stream: spin up two async line readers for stdout and stderr, writing each line to the respective buffer.
    async def read_stdout() -> None:
        # Guard: no pipe means nothing to stream.
        if stdout is None:
            return
        try:
            # Drain: read lines until EOF, writing each to the stdout buffer.
            while True:
                line_bytes = await stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace")
                await basher.write_buffer(handle.stdout_buffer, line)
        except asyncio.CancelledError:
            # Silently absorb cancellation — the caller drives cleanup via gather.
            pass

    async def read_stderr() -> None:
        # Guard: no pipe means nothing to_bash_send_hook stream.
        if stderr is None:
            return
        try:
            # Drain: read lines until EOF, writing each to the stderr buffer.
            while True:
                line_bytes = await stderr.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace")
                await basher.write_buffer(handle.stderr_buffer, line)
        except asyncio.CancelledError:
            # Silently absorb cancellation — the caller drives cleanup via gather.
            pass

    # Wait: drive both stream readers to completion or cancellation, then clean up.
    out_task = asyncio.create_task(read_stdout())
    err_task = asyncio.create_task(read_stderr())
    try:
        # Wait: drive both stream readers to completion.
        await asyncio.gather(out_task, err_task)

    except asyncio.CancelledError:
        # Propagate: cancel both reader tasks before re-raising.
        out_task.cancel()
        err_task.cancel()
        raise

    finally:
        # Clean up: mark the handle closed, reap the subprocess, refresh the processes listing.
        handle.closed = True
        if handle.process:
            handle.exit_code = handle.process.returncode
            try:
                await handle.process.wait()
            except asyncio.CancelledError:
                pass

        else:
            handle.exit_code = None
        
        await _refresh_bash_processes_buffer(basher)


async def _refresh_bash_processes_buffer(basher) -> None:
    records = basher.list_bash_processes()
    import json
    text = json.dumps(records, indent=2)
    await basher.create_buffer(_BASH_PROCESSES_BUFFER, text=text, overwrite=True)


# -------------------------------------------------------------------------------------------------------------------- #
# Hooks                                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #


async def before_create_bash_buffer(
    basher: Basher,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Guard: validate the proposed bash buffer before creation.

    Rejects empty command buffers. Returns None to allow creation, or an error
    string to block it. All side effects (process spawning, buffer creation) happen
    in the AFTER_CREATE_BUFFER hook.
    """
    # Guard: reject buffers with empty command content.
    content = buffer.get_text().strip()
    if not content:
        return "Empty bash buffer — cannot execute an empty command."

    # Guard: syntax-check the script without executing it.
    # Bash -n exits non-zero and prints the parse error on failure.
    proc = await asyncio.create_subprocess_exec(
        "/bin/bash", "-n", "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr_bytes = await proc.communicate(input=content.encode("utf-8"))

    # Interpret the result: non-zero exit means parse error.
    if proc.returncode != 0:
        stderr = stderr_bytes.decode("utf-8", errors="replace").splitlines()[0]
        return f"Bash syntax error: {stderr}"
    return None


async def after_create_bash_buffer(
    basher: Basher,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Spawn a bash process when a bash:{title} buffer is created.

    Reads the command from the buffer text, forks the subprocess, creates the
    stdin/stdout/stderr/status buffers, and starts the read loop.
    """
    # Parse the title from the buffer name (e.g. "bash:my-script" → "my-script").
    title = _bash_title_from_name(buffer.name)

    # Read the command to execute from the buffer's text.
    command = buffer.get_text().strip()

    # Guard: do not spawn if a process for this title is already running.
    if title in basher.bash_processes:
        return f"A process for '{title}' is already running."

    # Provision: derive the four stream buffer names from the buffer name.
    stdin_name = f"{buffer.name}:stdin"
    stdout_name = f"{buffer.name}:stdout"
    stderr_name = f"{buffer.name}:stderr"
    status_name = f"{buffer.name}:status"
    await basher.create_stream_buffer(stdin_name)
    await basher.create_stream_buffer(stdout_name)
    await basher.create_stream_buffer(stderr_name)

    # Register: attach a hook on stdin so writes to that buffer forward text to the process.
    stdin_buf = basher.get_buffer(stdin_name)
    if stdin_buf is None:
        await basher.drop_buffer(stdin_name)
        await basher.drop_buffer(stdout_name)
        await basher.drop_buffer(stderr_name)
        return f"stdin buffer '{stdin_name}' not found after creation"

    # Register: attach a hook on stdin so writes to that buffer forward text to the process.
    async def _bash_forward(
        buffer_manager: BufferManager,
        old_buffer: Buffer,
        new_buffer: Buffer,
        inner_args: dict[str, Any],
    ) -> str | None:
        # Extract only the newly appended text: use the old line count as the start index.
        appended_text = new_buffer.get_text(start=len(old_buffer) - 1)
        return await _bash_send_hook(basher, title, appended_text)

    stdin_buf.register_hook(HookPoint.AFTER_UPDATE_BUFFER, "bash_send", _bash_forward)

    # Launch: fork the subprocess with connected stdin/stdout/stderr pipes.
    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=None,
        )
    except Exception as e:
        return f"Failed to start process: {e}"

    # Register: store the handle, start the read loop, and refresh the processes listing.
    handle = BashHandle(
        title=title,
        command=command,
        cwd=None,
        process=process,
        stdin_buffer=stdin_name,
        stdout_buffer=stdout_name,
        stderr_buffer=stderr_name,
        status_buffer=status_name,
    )
    basher.bash_processes[title] = handle
    handle.task = asyncio.create_task(_bash_read_loop(basher, handle))

    # Seed the status buffer and refresh the system process list.
    status_content = (
        f'{{"pid": {process.pid}, "title": {title!r}, '
        f'"command": {command!r}, "running": true, "exit_code": null}}\n'
    )
    await basher.create_buffer(status_name, text=status_content)
    await _refresh_bash_processes_buffer(basher)
    return None


async def after_drop_bash_buffer(
    basher: Basher,
    buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Terminate the bash process and drop its stream buffers when bash:{title} is dropped."""
    # Parse the title from the buffer name.
    title = _bash_title_from_name(buffer.name)

    # Guard: no active process for this title.
    # This means the buffer was created but the process was never registered —
    # the after_create hook must have failed partway through.
    if title not in basher.bash_processes:
        import logging
        logging.error(f"[after_drop_bash_buffer] no process for '{buffer.name}' — after_create must have failed partway through.")
        return
    handle = basher.bash_processes.pop(title)

    # Stop: cancel the read loop task and terminate the subprocess.
    if handle.task:
        handle.task.cancel()
    if handle.process and handle.process.returncode is None:
        handle.process.terminate()

        # Wait for the process to exit, then kill it if it times out.
        try:
            await asyncio.wait_for(handle.process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            # Force-kill the process after the graceful timeout.
            handle.process.kill()
        except asyncio.CancelledError:
            # Force-kill the process when cancelled.
            handle.process.kill()
    handle.closed = True
    handle.exit_code = handle.process.returncode if handle.process else None

    # Drop the four associated stream buffers.
    for buf_name in (
        handle.stdin_buffer,
        handle.stdout_buffer,
        handle.stderr_buffer,
        handle.status_buffer,
    ):
        if buf_name:
            await basher.drop_buffer(buf_name)

    # Refresh the system process list.
    await _refresh_bash_processes_buffer(basher)


# -------------------------------------------------------------------------------------------------------------------- #
# Option Buffer Flow -- [bash] suffix                                                                                                   #
# -------------------------------------------------------------------------------------------------------------------- #


async def _before_create_bash_option(
    basher: Basher,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """BEFORE_CREATE_BUFFER hook for [bash] option buffers -- validate the origin buffer and config."""
    # Parse the JSON config from the option buffer.
    try:
        config = json.loads(buffer.get_text())
    except json.JSONDecodeError as e:
        return f"[_before_create_bash_option] option buffer is not valid JSON: {e}"

    # Derive the origin buffer name from the option buffer name.
    origin_name = _origin_name_from_option_buffer_name(buffer.name)
    if not origin_name:
        return f"[_before_create_bash_option] cannot derive origin from '{buffer.name}'"

    # The origin buffer must exist and hold a non-empty script.
    origin_buffer = basher.get_buffer(origin_name)
    if origin_buffer is None:
        return f"[_before_create_bash_option] origin buffer '{origin_name}' not found"

    # Reject when the origin buffer holds an empty script.
    script = origin_buffer.get_text()
    if not script.strip():
        return f"[_before_create_bash_option] script in '{origin_name}' is empty"

    # Syntax-check the script without executing it -- read-only, no side effects.
    proc = await asyncio.create_subprocess_exec(
        "/bin/bash", "-n", "-",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr_bytes = await proc.communicate(input=script.encode("utf-8"))

    # Reject when the script has a syntax error.
    if proc.returncode != 0:
        stderr = stderr_bytes.decode("utf-8", errors="replace").splitlines()[0]
        return f"[_before_create_bash_option] bash syntax error: {stderr}"

    return None


async def _after_create_bash_option(
    basher: Basher,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """AFTER_CREATE_BUFFER hook for [bash] option buffers -- spawn the process and wire hooks."""
    # Derive the origin buffer name from the option buffer name.
    origin_name = _origin_name_from_option_buffer_name(buffer.name)
    if not origin_name:
        _logger.error(f"[_after_create_bash_option] cannot derive origin from '{buffer.name}'")
        return None

    # Guard: do not spawn if a process for this origin is already running.
    if origin_name in basher.bash_processes:
        return f"[_after_create_bash_option] process for '{origin_name}' is already running"

    # Read the script from the origin buffer.
    origin_buffer = basher.get_buffer(origin_name)
    if origin_buffer is None:
        _logger.error(f"[_after_create_bash_option] origin buffer '{origin_name}' not found")
        return None

    # Extract the script and derive the title from the origin buffer name.
    script = origin_buffer.get_text().strip()
    title = origin_name

    # Derive the four sub-buffer names from the origin buffer name.
    stdin_name = f"{origin_name}:stdin"
    stdout_name = f"{origin_name}:stdout"
    stderr_name = f"{origin_name}:stderr"
    status_name = f"{origin_name}:status"

    # Create the three stream buffers for stdout, stderr, and stdin.
    await basher.create_stream_buffer(stdin_name)
    await basher.create_stream_buffer(stdout_name)
    await basher.create_stream_buffer(stderr_name)

    # Get the stdin buffer to attach the forward hook.
    stdin_buf = basher.get_buffer(stdin_name)
    if stdin_buf is None:
        await basher.drop_buffer(stdin_name)
        await basher.drop_buffer(stdout_name)
        await basher.drop_buffer(stderr_name)
        return f"[_after_create_bash_option] stdin buffer '{stdin_name}' not found after creation"

    # Register: attach AFTER_UPDATE_BUFFER on stdin to forward writes to the process.
    async def _bash_forward(
        buffer_manager: BufferManager,
        old_buffer: Buffer,
        new_buffer: Buffer,
        inner_args: dict[str, Any],
    ) -> str | None:
        appended_text = new_buffer.get_text(start=len(old_buffer) - 1)
        return await _bash_send_hook(basher, title, appended_text)

    # Register AFTER_UPDATE_BUFFER on stdin to forward writes to the process.
    stdin_buf.register_hook(HookPoint.AFTER_UPDATE_BUFFER, "bash_send", _bash_forward)

    # Launch the subprocess with connected stdin/stdout/stderr pipes.
    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.create_subprocess_exec(
            "/bin/sh",
            "-c",
            script,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=None,
        )
    except Exception as e:
        return f"[_after_create_bash_option] failed to start process: {e}"

    # Register: store the handle, start the read loop, and refresh the processes listing.
    handle = BashHandle(
        title=title,
        command=script,
        origin=origin_name,
        process=process,
        stdin_buffer=stdin_name,
        stdout_buffer=stdout_name,
        stderr_buffer=stderr_name,
        status_buffer=status_name,
    )
    basher.bash_processes[origin_name] = handle
    handle.task = asyncio.create_task(_bash_read_loop(basher, handle))

    # Seed the status buffer and refresh the system process list.
    status_content = (
        f'{{"pid": {process.pid}, "title": "{title}", '
        f'"running": true, "exit_code": null}}\n'
    )
    await basher.create_buffer(status_name, text=status_content)
    await _refresh_bash_processes_buffer(basher)

    # Confirm successful registration.
    _logger.info(f"[>>] process '{title}' started, stdin/stdout/stderr/status buffers created")


async def _before_drop_bash_option(
    basher: Basher,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """BEFORE_DROP_BUFFER hook for [bash] option buffers -- allow the drop."""
    return None


async def _after_drop_bash_option(
    basher: Basher,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """AFTER_DROP_BUFFER hook for [bash] option buffers -- terminate the process and clean up sub-buffers."""
    # Derive the origin buffer name from the option buffer name.
    origin_name = _origin_name_from_option_buffer_name(buffer.name)
    if not origin_name:
        _logger.error(f"[_after_drop_bash_option] cannot derive origin from '{buffer.name}'")
        return None

    # Look up the handle for this origin.
    if origin_name not in basher.bash_processes:
        _logger.warning(f"[_after_drop_bash_option] no process for '{origin_name}'")
        return None

    # Remove the handle from the processes dictionary.
    handle = basher.bash_processes.pop(origin_name)

    # Stop: cancel the read loop task and terminate the subprocess.
    if handle.task:
        handle.task.cancel()
    if handle.process and handle.process.returncode is None:
        handle.process.terminate()

        # Wait for the process to exit, then kill it if it times out.
        try:
            await asyncio.wait_for(handle.process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            # Force-kill the process after the graceful timeout.
            handle.process.kill()
        except asyncio.CancelledError:
            # Force-kill the process when cancelled.
            handle.process.kill()
    handle.closed = True
    handle.exit_code = handle.process.returncode if handle.process else None

    # Drop the four associated sub-buffers.
    for buf_name in (
        handle.stdin_buffer,
        handle.stdout_buffer,
        handle.stderr_buffer,
        handle.status_buffer,
    ):
        if buf_name:
            await basher.drop_buffer(buf_name)

    # Refresh the system process list.
    await _refresh_bash_processes_buffer(basher)

    # Confirm the cleanup is complete.
    _logger.info(f"[>>] process '{origin_name}' terminated, sub-buffers removed")


async def _before_drop_command(
    basher: Basher,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """BEFORE_DROP_BUFFER hook on the script buffer -- allow the drop."""
    return None


async def _after_drop_command(
    basher: Basher,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """AFTER_DROP_BUFFER hook on the script buffer -- clean up when the command buffer is dropped directly."""
    # Look up the handle for this origin.
    origin_name = buffer.name
    if origin_name not in basher.bash_processes:
        return None

    # Remove the handle from the processes dictionary.
    handle = basher.bash_processes.pop(origin_name)

    # Stop: cancel the read loop task and terminate the subprocess.
    if handle.task:
        handle.task.cancel()
    if handle.process and handle.process.returncode is None:
        handle.process.terminate()

        # Wait for the process to exit, then kill it if it times out.
        try:
            await asyncio.wait_for(handle.process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            # Force-kill the process after the graceful timeout.
            handle.process.kill()
        except asyncio.CancelledError:
            # Force-kill the process when cancelled.
            handle.process.kill()
    handle.closed = True
    handle.exit_code = handle.process.returncode if handle.process else None

    # Derive the option buffer name and drop it.
    option_name = origin_name + _BASH_SUFFIX
    option_exists = basher.get_buffer(option_name)
    if option_exists is not None:
        await basher.drop_buffer(option_name)

    # Drop the four associated sub-buffers.
    for buf_name in (
        handle.stdin_buffer,
        handle.stdout_buffer,
        handle.stderr_buffer,
        handle.status_buffer,
    ):
        if buf_name:
            await basher.drop_buffer(buf_name)

    # Refresh the system process list.
    await _refresh_bash_processes_buffer(basher)

    # Confirm the cleanup is complete.
    _logger.info(f"[>>] process '{origin_name}' cleaned up after command buffer drop")


# -------------------------------------------------------------------------------------------------------------------- #
# Basher                                                                                                               #
# -------------------------------------------------------------------------------------------------------------------- #


class Basher(StreamBufferManager, AgenticObject):
    """I can run bash commands or scripts and interact with their input and output as stream buffers.
    "doc:bash" holds details.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bash_processes: dict[str, BashHandle] = {}  # keyed by title (from buffer name)
        self._create_buffer(_BASH_PROCESSES_BUFFER, "[]\n")
        self._create_buffer(_BASH_MANAGER_DOC_BUFFER, text=_BASH_MANAGER_DOC)

        # Register buffer lifecycle hooks for bash:{title} buffers.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "validate_bash_buffer",
            _BASH_PROCESS_PATTERN,
            before_create_bash_buffer,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "spawn_bash_process",
            _BASH_PROCESS_PATTERN,
            after_create_bash_buffer,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            "teardown_bash_process",
            _BASH_PROCESS_PATTERN,
            after_drop_bash_buffer,
        )

        # Validate the origin buffer and config before creating the [bash] option buffer.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "bash_option:before_create",
            _BASH_OPTION_BUFFER_PATTERN,
            _before_create_bash_option,
            apply_to_existing=True,
        )

        # Block the option buffer drop when the process has active refs.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_DROP_BUFFER,
            "bash_option:before_drop",
            _BASH_OPTION_BUFFER_PATTERN,
            _before_drop_bash_option,
            apply_to_existing=True,
        )

        # Terminate the process and remove sub-buffers after the option buffer is dropped.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            "bash_option:after_drop",
            _BASH_OPTION_BUFFER_PATTERN,
            _after_drop_bash_option,
            apply_to_existing=True,
        )

        # Spawn the process, create sub-buffers, and wire the stdin forward hook.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "bash_option:after_create",
            _BASH_OPTION_BUFFER_PATTERN,
            _after_create_bash_option,
            apply_to_existing=True,
        )

    @sandbox
    def list_bash_processes(self) -> list[dict]:
        """List all running processes with their details."""
        return [
            {
                "pid": h.process.pid if h.process else None,
                "title": title,
                "command": h.command,
                "cwd": h.cwd,
                "running": not h.closed,
                "exit_code": h.exit_code,
            }
            for title, h in sorted(self.bash_processes.items())
        ]
