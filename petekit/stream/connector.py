import asyncio
import time
from dataclasses import dataclass
from typing import Any

from peteos import AgenticObject, sandbox, tool
from .stream_buffer_manager import StreamBufferManager, format_dict_list_for_buffer
from ..utils.text_formatters import _sanitize_title


@dataclass
class ConnectionHandle:
    """Holds the connection details and handles for a named connection."""
    host: str
    port: int
    ssl: bool = False
    reader: asyncio.StreamReader | None = None
    writer: asyncio.StreamWriter | None = None
    task: asyncio.Task | None = None
    in_buffer: str = ""
    out_buffer: str = ""
    closed: bool = False
    close_reason: str = ""


class Connector(StreamBufferManager, AgenticObject):
    """You manage long standing TCP/IP sockets. As usual: communication via stream buffers.
    - Use connect and disconnect to open / close a connection.
    - The buffer "system:status:network:connections" is always up to date with all current connections."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.connections: dict[str, ConnectionHandle] = {}
        self._create_buffer("system:status:network:connections", "[]\n")

    async def _refresh_connections_buffer(self) -> None:
        records = self.list_connections()
        text = format_dict_list_for_buffer(records)
        await self.create_buffer("system:status:network:connections", text=text, overwrite=True)

    @tool
    async def connect(self, name: str, host: str, port: int, ssl: bool = False) -> dict[str, Any]:
        """Establish a connection to a host on the given port."""
        if name in self.connections:
            return {"ok": False, "error": f"Connection '{name}' already exists."}

        # Guard: reject duplicate connection names before allocating resources.
        # Prepare: name the stream buffers and teardown any stale ones from a prior session.
        now = time.time()
        in_buffer = f"stream:in:{_sanitize_title(name)}"
        out_buffer = f"stream:out:{_sanitize_title(name)}"
        for buf in (in_buffer, out_buffer):
            try:
                await self.drop_buffer(buf)
            except KeyError:
                pass

        # Provision: create the two stream buffers; on failure roll back the first.
        await self.create_buffer(in_buffer, stream=True)
        try:
            await self.create_buffer(out_buffer, stream=True)
        except KeyError:
            await self.drop_buffer(in_buffer)
            raise

        # Register: attach a hook on the out buffer so writes to it forward text to the socket.
        hook_result = self._set_stream_on_append_hook(
            out_buffer,
            name="connection_send",
            hook=lambda s, t, m: self._connection_send_hook(name, t),
        )
        if not hook_result.get("ok"):
            await self.drop_buffer(in_buffer)
            await self.drop_buffer(out_buffer)
            raise ConnectionError(f"failed to register connection_send hook: {hook_result.get('error')}")

        # Connect: open the TCP socket.
        try:
            reader, writer = await asyncio.open_connection(host, port, ssl=ssl)
        except Exception as e:
            await self.drop_buffer(in_buffer)
            await self.drop_buffer(out_buffer)
            raise

        # Register: store the handle, schedule the read loop, and refresh the connections listing.
        handle = ConnectionHandle(
            host=host, port=port, ssl=ssl,
            reader=reader, writer=writer,
            in_buffer=in_buffer,
            out_buffer=out_buffer,
        )
        self.connections[name] = handle
        handle.task = asyncio.create_task(self._read_loop(name, handle))
        await self._refresh_connections_buffer()

        message = f"Connected to {host}:{port} (ssl={ssl}) as '{name}'. Streams created at {now}. Receiving data into '{in_buffer}', sending data into '{out_buffer}'."
        return {"ok": True, "name": name, "in_buffer": in_buffer, "out_buffer": out_buffer, "message": message}

    @sandbox
    def list_connections(self) -> list[dict]:
        """List all connections with their details."""
        return [
            {
                "name": name,
                "host": h.host,
                "port": h.port,
                "ssl": h.ssl,
                "closed": h.closed,
            }
            for name, h in self.connections.items()
        ]

    @tool
    async def disconnect(self, name: str, drop_buffers: bool = True) -> dict[str, Any]:
        """Close the named connection. Pass drop_buffers=True (default) to also drop stream buffers."""
        if name not in self.connections:
            return {"ok": False, "error": f"No connection named '{name}'."}
        handle = self.connections.pop(name)

        # Stop: cancel the read loop task and close the socket.
        if not handle.closed:
            handle.task.cancel()
            handle.writer.close()
        reason = f"({handle.close_reason})" if handle.closed else ""

        # Refresh the connections listing and optionally drop the stream buffers.
        await self._refresh_connections_buffer()
        if drop_buffers:
            await self.drop_buffer(handle.in_buffer)
            await self.drop_buffer(handle.out_buffer)
            message = f"Disconnected '{name}'. {reason}\nCleaned up stream buffers."
            return {"ok": True, "name": name, "dropped_buffers": True, "message": message}

        message = f"Disconnected '{name}'. {reason}\nStream buffers '{handle.in_buffer}', '{handle.out_buffer}' still exist for analysis. — drop them with drop_buffer before reconnecting."
        return {"ok": True, "name": name, "dropped_buffers": False, "message": message}

    async def _connection_send_hook(self, name: str, text: str, flush: bool = False, fix_crlf: bool = False) -> str:
        """Send text over the named connection. Pass flush=True to drain the write buffer without sending new data. Pass fix_crlf=True to replace LF (\\n) with CRLF (\\r\\n) as required by protocols such as HTTP."""
        if name not in self.connections:
            raise KeyError(f"No connection named '{name}'.")
        handle = self.connections[name]

        # Guard: reject sends on already-closed connections.
        if handle.closed:
            await self.disconnect(name, cleanup=True)
            raise ConnectionError(f"Connection '{name}' is closed. {handle.close_reason}")

        # Send: encode text and write to the socket; optionally fix CRLF and drain.
        try:
            if text:
                if fix_crlf:
                    # CRLF fix: replace bare LF with CRLF as required by HTTP and similar protocols.
                    text = text.replace("\n", "\r\n")
                data = text.encode("utf-8")
                handle.writer.write(data)
            if flush:
                # Drain: flush the write buffer to ensure data is actually sent over the wire.
                await handle.writer.drain()
        except Exception as e:
            hint = await self.disconnect(name, cleanup=False)
            raise ConnectionError(f"Send error ({e}) — connection closed. {hint}")

        if text and flush:
            return f"Sent {len(text)} chars to '{name}'. "
        if flush:
            return f"Flushing write buffer for '{name}'."
        return f"Sending {len(text)} chars to '{name}'."

    async def _read_loop(self, name: str, handle: ConnectionHandle) -> None:
        """Continuously read lines from the connection."""
        try:
            while True:
                line_bytes = await handle.reader.readline()
                if not line_bytes:
                    # EOF — remote closed, mark closed and exit
                    handle.closed = True
                    handle.close_reason = "Remote closed."
                    handle.task.cancel()
                    handle.writer.close()
                    break
                line = line_bytes.decode("utf-8", errors="replace").rstrip("\n")
                await self.write_buffer(handle.in_buffer, line)
        except asyncio.CancelledError:
            pass
