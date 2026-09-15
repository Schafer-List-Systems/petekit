import asyncio
import time
from dataclasses import dataclass

from peteos import AgenticObject, tool
from .stream_buffer_manager import StreamBufferManager


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
    """You manage network connections."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.connections: dict[str, ConnectionHandle] = {}

    @tool
    async def connect(self, name: str, host: str, port: int, ssl: bool = False) -> str:
        """Establish a connection to a host on the given port."""
        if name in self.connections:
            raise KeyError(f"Connection '{name}' already exists.")

        # Prepare stream buffers for incoming and outgoing data
        now = time.time()
        in_buffer = f"stream:in:{name}"
        out_buffer = f"stream:out:{name}"
        for buf in (in_buffer, out_buffer):
            try:
                self._drop_stream(buf)
            except KeyError:
                pass
        self._create_stream(in_buffer)
        try:
            self._create_stream(out_buffer)
        except KeyError:
            self._drop_stream(in_buffer)
            raise

        try:
            reader, writer = await asyncio.open_connection(host, port, ssl=ssl)
        except Exception as e:
            # Connection failed — roll back stream buffers before propagating
            self._drop_stream(in_buffer)
            self._drop_stream(out_buffer)
            raise

        handle = ConnectionHandle(
            host=host, port=port, ssl=ssl,
            reader=reader, writer=writer,
            in_buffer=in_buffer,
            out_buffer=out_buffer,
        )
        self.connections[name] = handle
        handle.task = asyncio.create_task(self._read_loop(name, handle))
        return (
            f"Connected to {host}:{port} (ssl={ssl}) as '{name}'. "
            f"Streams created at {now}. "
            f"Receiving data into '{in_buffer}', sending data into '{out_buffer}'."
        )

    @tool
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
    async def disconnect(self, name: str, drop_buffers: bool = True) -> str:
        """Close the named connection. Pass drop_buffers=True (default) to also drop stream buffers."""
        if name not in self.connections:
            raise KeyError(f"No connection named '{name}'.")
        handle = self.connections.pop(name)
        if not handle.closed:
            handle.task.cancel()
            handle.writer.close()
        reason = f"({handle.close_reason})" if handle.closed else ""
        if drop_buffers:
            self._drop_stream(handle.in_buffer)
            self._drop_stream(handle.out_buffer)
            return (
                f"Disconnected '{name}'. {reason}\n"
                f"Cleaned up stream buffers."
            )
        return (
            f"Disconnected '{name}'. {reason}\n"
            f"Stream buffers '{handle.in_buffer}', '{handle.out_buffer}' still exist for analysis. "
            f"— drop them with drop_buffer before reconnecting."
        )

    @tool
    async def send(self, name: str, text: str, flush: bool = False, fix_crlf: bool = False) -> str:
        """Send text over the named connection. Pass flush=True to drain the write buffer without sending new data. Pass fix_crlf=True to replace LF (\\n) with CRLF (\\r\\n) as required by protocols such as HTTP."""
        if name not in self.connections:
            raise KeyError(f"No connection named '{name}'.")
        handle = self.connections[name]
        if handle.closed:
            await self.disconnect(name, cleanup=True)
            raise ConnectionError(f"Connection '{name}' is closed. {handle.close_reason}")
        try:
            if text:
                if fix_crlf:
                    text = text.replace("\n", "\r\n")
                data = text.encode("utf-8")
                handle.writer.write(data)
            if flush:
                await handle.writer.drain()
        except Exception as e:
            hint = await self.disconnect(name, cleanup=False)
            raise ConnectionError(f"Send error ({e}) — connection closed. {hint}")
        if text:
            for line in text.splitlines():
                await self._append_stream_entry(handle.out_buffer, line)
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
                line = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
                await self._append_stream_entry(handle.in_buffer, line)
        except asyncio.CancelledError:
            pass
