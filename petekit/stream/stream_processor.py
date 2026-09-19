"""StreamProcessor - routes stream data based on condition evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field

from peteos import AgenticObject, sandbox
from .stream_buffer_manager import StreamBufferManager


class ConditionNotFound(Exception):
    def __init__(self, condition: str) -> None:
        self.condition = condition
        super().__init__(f"condition not found: {condition}")


@dataclass
class RoutingTable:
    input_stream: str
    conditions: list[tuple[str, str]] = field(default_factory=list)


def _routing_table_key(name: str) -> str:
    return f"system:routing_table:{name}"


def _format_routing_entry(condition_list: str, output_stream: str) -> str:
    return f"{condition_list} {output_stream}\n"


class StreamProcessor(StreamBufferManager, AgenticObject):
    """You are a stream processor managing multiple routing tables.
    - The set of routing tables is listed in the "system:list:routing_tables" buffer.
    - Each routing table is stored in an individual buffer containing rules [<condition_list>,<output_stream>]
    - Rules fire top-to-bottom; first match wins and stops evaluation.
    - Condition lists are comma-separated; each sub-condition must be true (AND semantics).
    - Prefix a sub-condition with "!" to negate it.
    - The condition name "true" always matches (good for catch-all / fallback rule).
    - Condition names are sandbox-decorated methods with signature (stream: str, text: str, metadata: dict) -> bool.
      Discover available conditions in docs:reflect:sandbox (hardcoded) and docs:reflect:dynamic (runtime); use define_function to add your own.
    - Control the stream processor by writing to "stream:processor:control":
      new <routing_table> <input_stream>  — create routing table
      drop <routing_table>              — delete routing table
      add <routing_table> <condition_list> <output_stream>  — add rule
      del <routing_table> <condition_list> <output_stream>  — remove rule
    - Read feedback and streaming errors (command results, FALLTHROUGH) from "stream:processor:feedback".
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._routing_tables: dict[str, RoutingTable] = {}
        result = self.create_buffer("stream:processor:control", stream=True)
        if not result.get("ok"):
            raise RuntimeError(f"failed to create stream:processor:control: {result.get('error')}")
        result = self.create_buffer("stream:processor:feedback", stream=True)
        if not result.get("ok"):
            self.drop_buffer("stream:processor:control")
            raise RuntimeError(f"failed to create stream:processor:feedback: {result.get('error')}")
        hook_result = self._set_stream_on_append_hook(
            "stream:processor:control",
            name="system_processor",
            hook=lambda stream, text, metadata: self._system_processor_hook(stream, text, metadata),
        )
        if not hook_result.get("ok"):
            self.drop_buffer("stream:processor:feedback")
            self.drop_buffer("stream:processor:control")
            raise RuntimeError(f"failed to register system_processor hook: {hook_result.get('error')}")
        refresh_result = self._refresh_routing_tables_buffer()
        if not refresh_result.get("ok"):
            self._set_stream_on_append_hook("stream:processor:control", name="system_processor", hook=None)
            self.drop_buffer("stream:processor:feedback")
            self.drop_buffer("stream:processor:control")
            raise RuntimeError(f"failed to refresh routing tables buffer: {refresh_result.get('error')}")

    def _refresh_routing_tables_buffer(self) -> dict[str, Any]:
        """Refresh the system:list:routing_tables buffer."""
        from petekit.utils.text_formatters import format_dict_list_for_buffer
        return self.create_buffer(
            "system:list:routing_tables",
            text=format_dict_list_for_buffer(self._list_routing_tables()),
            overwrite=True
        )

    @sandbox
    def _list_routing_tables(self) -> list[dict]:
        """Return all routing tables with name, input_stream, and condition count."""
        return [
            {"name": name, "input_stream": rt.input_stream, "conditions": len(rt.conditions)}
            for name, rt in self._routing_tables.items()
        ]

    async def _system_processor_hook(self, stream: str, text: str, metadata: dict) -> None:
        """Handle a command written to the control stream.

        Expected text format (space-separated):
          new <routing_table> <input_stream>
          drop <routing_table>
          add <routing_table> <condition_list> <output_stream>
          del <routing_table> <condition_list> <output_stream>

        Multiple commands supported — one per line.
        """
        for line in text.splitlines():
            if not line.strip():
                continue
            parts = line.split()
            cmd = parts[0]
            result: dict[str, Any] | None = None
            if cmd == "new" and len(parts) >= 3:
                routing_table, input_stream = parts[1], parts[2]
                result = self._routing_table_new(routing_table, input_stream)
            elif cmd == "drop" and len(parts) >= 2:
                routing_table = parts[1]
                result = self._routing_table_drop(routing_table)
            elif cmd == "add" and len(parts) >= 4:
                routing_table, condition_list, output_stream = parts[1], parts[2], " ".join(parts[3:])
                result = self._routing_table_add(routing_table, condition_list, output_stream)
            elif cmd == "del" and len(parts) >= 4:
                routing_table, condition_list, output_stream = parts[1], parts[2], " ".join(parts[3:])
                result = self._routing_table_del(routing_table, condition_list, output_stream)
            else:
                fb_result = await self.write_buffer("stream:processor:feedback", f"unknown or malformed command: {line}")
                if not fb_result.get("ok"):
                    raise RuntimeError(f"failed to write to feedback buffer: {fb_result.get('error')}")
                continue

            if not result.get("ok"):
                raise RuntimeError(f"command '{cmd}' failed: {result.get('error')}")
            fb_result = await self.write_buffer("stream:processor:feedback", str(result))
            if not fb_result.get("ok"):
                raise RuntimeError(f"failed to write to feedback buffer: {fb_result.get('error')}")

    @sandbox
    def _routing_table_new(self, routing_table: str, input_stream: str) -> dict[str, Any]:
        """Create a routing table with a fixed input stream."""
        table_name = _routing_table_key(routing_table)
        if routing_table in self._routing_tables or table_name in self._buffers:
            return {"ok": False, "error": f"routing table '{routing_table}' already exists"}
        create_result = self.create_buffer(table_name, stream=True)
        if not create_result.get("ok"):
            raise RuntimeError(f"failed to create routing table buffer '{table_name}': {create_result.get('error')}")
        self._routing_tables[routing_table] = RoutingTable(input_stream=input_stream)
        hook_result = self._set_stream_on_append_hook(
            input_stream,
            name=f"routing_table:{routing_table}",
            hook=lambda stream_buffer, text, metadata: self._routing_table_dispatch(routing_table, text, metadata),
        )
        if not hook_result.get("ok"):
            del self._routing_tables[routing_table]
            self.drop_buffer(table_name)
            raise RuntimeError(f"failed to hook input stream '{input_stream}': {hook_result.get('error')}")
        refresh_result = self._refresh_routing_tables_buffer()
        if not refresh_result.get("ok"):
            self._set_stream_on_append_hook(input_stream, name=f"routing_table:{routing_table}", hook=None)
            del self._routing_tables[routing_table]
            self.drop_buffer(table_name)
            raise RuntimeError(f"failed to refresh routing tables buffer: {refresh_result.get('error')}")
        return {"ok": True, "table": routing_table, "input_stream": input_stream}

    @sandbox
    def _routing_table_drop(self, routing_table: str) -> dict[str, Any]:
        """Delete a routing table and remove its input stream hook."""
        table_name = _routing_table_key(routing_table)
        if routing_table not in self._routing_tables:
            return {"ok": False, "error": f"routing table '{routing_table}' not found"}
        input_stream = self._routing_tables[routing_table].input_stream
        unhook_result = self._set_stream_on_append_hook(input_stream, name=f"routing_table:{routing_table}", hook=None)
        if not unhook_result.get("ok"):
            return {"ok": False, "error": f"failed to unhook '{input_stream}': {unhook_result.get('error')}"}
        del self._routing_tables[routing_table]
        if table_name in self._buffers:
            self.drop_buffer(table_name)
        refresh_result = self._refresh_routing_tables_buffer()
        if not refresh_result.get("ok"):
            raise RuntimeError(f"failed to refresh routing tables buffer: {refresh_result.get('error')}")
        return {"ok": True, "dropped": routing_table}

    @sandbox
    def _routing_table_add(self, routing_table: str, condition_list: str, output_stream: str) -> dict[str, Any]:
        """Append a condition->output entry to a routing table."""
        table_name = _routing_table_key(routing_table)
        if routing_table not in self._routing_tables or table_name not in self._buffers:
            return {"ok": False, "error": f"routing table '{routing_table}' not found. Use 'new' first."}
        entry = _format_routing_entry(condition_list, output_stream)
        create_result = self.create_buffer(table_name, text=entry, overwrite=False)
        if not create_result.get("ok"):
            raise RuntimeError(f"failed to append to routing table buffer: {create_result.get('error')}")
        self._routing_tables[routing_table].conditions.append((condition_list, output_stream))
        refresh_result = self._refresh_routing_tables_buffer()
        if not refresh_result.get("ok"):
            raise RuntimeError(f"failed to refresh routing tables buffer: {refresh_result.get('error')}")
        return {"ok": True, "table": routing_table, "condition_list": condition_list, "output": output_stream}

    @sandbox
    def _routing_table_del(self, routing_table: str, condition_list: str, output_stream: str) -> dict[str, Any]:
        """Remove the first matching condition->output entry from a routing table."""
        table_name = _routing_table_key(routing_table)
        if routing_table not in self._routing_tables:
            return {"ok": False, "error": f"routing table '{routing_table}' not found"}
        table = self._routing_tables[routing_table]
        for i, (cn, os) in enumerate(table.conditions):
            if cn == condition_list and os == output_stream:
                del table.conditions[i]
                entry = _format_routing_entry(condition_list, output_stream)
                self.edit_buffer(table_name, old_string=entry, new_string="")
                return {"ok": True, "removed": f"{condition_list} {output_stream}"}
        return {"ok": False, "error": f"no matching entry in '{routing_table}'"}

    async def _routing_table_dispatch(self, routing_table: str, text: str, metadata: dict | None = None) -> None:
        """Private hook: evaluate incoming data against the routing table conditions.
        First matching condition writes the data to its output stream. No match = reported as FALLTHROUGH.
        """
        if routing_table not in self._routing_tables:
            fb_result = await self.write_buffer("stream:processor:feedback", f"ERROR: routing_table '{routing_table}' not found")
            if not fb_result.get("ok"):
                raise RuntimeError(f"failed to write ERROR to feedback: {fb_result.get('error')}")
            return

        table = self._routing_tables[routing_table]
        ts = metadata.get("timestamp", 0) if metadata else 0
        for condition_list, output_stream in table.conditions:
            all_match = True
            for sub_condition in condition_list.split(","):
                sc = sub_condition.strip()
                try:
                    matched = await self._evaluate_condition(sc, text, metadata)
                except ConditionNotFound as e:
                    fb_result = await self.write_buffer("stream:processor:feedback", f"FALLTHROUGH source={table.input_stream} ts={ts} reason={e}")
                    if not fb_result.get("ok"):
                        raise RuntimeError(f"failed to write FALLTHROUGH to feedback: {fb_result.get('error')}")
                    return
                if not matched:
                    all_match = False
                    break
            if all_match:
                await self.write_buffer(output_stream, text)
                return
        fb_result = await self.write_buffer("stream:processor:feedback", f"FALLTHROUGH source={table.input_stream} ts={ts}")
        if not fb_result.get("ok"):
            raise RuntimeError(f"failed to write FALLTHROUGH to feedback: {fb_result.get('error')}")

    async def _evaluate_condition(self, condition_name: str, text: str, metadata: dict | None = None) -> bool:
        if condition_name.startswith("!"):
            inner = condition_name[1:]
            return not await self._evaluate_condition(inner, text, metadata)

        if condition_name.lower() == "true":
            return True

        import inspect
        members = self._gather_sandbox_members()
        if condition_name not in members:
            raise ConditionNotFound(condition_name)

        method = members[condition_name]
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        if params[:3] == ["stream", "text", "metadata"] and sig.return_annotation in (bool, inspect.Parameter.empty):
            return await method(stream=condition_name, text=text, metadata=metadata or {})

        raise ConditionNotFound(condition_name)
