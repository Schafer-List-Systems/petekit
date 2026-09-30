"""StreamProcessor - routes stream data based on condition evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time

from peteos import AgenticObject, sandbox
from peteos.persona.agent import Agent
from .stream_buffer_manager import StreamBufferManager
from ..function.function_manager import FunctionManager
import asyncio


_MIN_NOTIFICATION_INTERVAL = 20.0
_MAX_NOTIFICATION_INTERVAL = 600.0


# HACK: no clean API exists to get ptid by recency — only session dirs on disk carry mtime.
# Feature request for PeteOS core: expose a method or property on Agent or AgenticObject
# that returns persistent threads ordered by last activity, with their timestamps.
# Until then, this workaround resolves _oap_thread_store entries against Agent.agent_base
# and uses filesystem mtime as a proxy for recency.
def _most_recent_persistent_thread(obj: AgenticObject) -> str | None:
    """Return the persistent_thread_id whose session directory has the newest mtime.

    Reads _oap_thread_store to get ptid -> session_uuid pairs, resolves each
    session directory under Agent.agent_base, and returns the ptid whose path
    has the highest filesystem modification time. Returns None if no sessions
    exist or the agent base is not set.
    """
    # Guard: require a populated thread store and a valid agent base directory.
    thread_store = getattr(obj, "_oap_thread_store", None)
    if not thread_store:
        return None
    agent_base = obj.agent.agent_base
    if not agent_base:
        return None

    # Resolve the base path under which per-role session directories live.
    role_name = obj._oap_role.name
    base_path = Path(agent_base) / role_name

    # Scan each session, tracking whichever has the most recent filesystem mtime.
    best_ptid: str | None = None
    best_mtime = -1.0
    for ptid, session_uuid in thread_store.items():
        session_path = base_path / session_uuid
        if session_path.exists():
            try:
                mtime = session_path.stat().st_mtime
                if mtime > best_mtime:
                    best_mtime = mtime
                    best_ptid = ptid
            except OSError:
                pass

    # The ptid of the session with the highest filesystem mtime, or None if no sessions exist.
    return best_ptid


def _routing_table_key(name: str) -> str:
    return f"system:routing_table:{name}"


def _format_routing_entry(condition_list: str, output_stream: str) -> str:
    return f"{condition_list} {output_stream}\n"


def _start_stream_notification_timer(obj: StreamProcessor, loop: asyncio.AbstractEventLoop) -> asyncio.Task:
    """Create and start the async notification timer loop. Returns the task."""

    async def _notification_timer_loop() -> None:
        while True:
            await asyncio.sleep(_MIN_NOTIFICATION_INTERVAL)
            await obj._stream_notification_timer_fire()

    task = loop.create_task(_notification_timer_loop())
    return task


class RoutingConditionError(Exception):
    def __init__(self, condition: str, routing_table: str | None = None, reason: str | None = None) -> None:
        self.condition = condition
        self.routing_table = routing_table
        self.reason = reason
        parts = [f"condition '{condition}'"]
        if routing_table:
            parts.append(f"routing_table='{routing_table}'")
        if reason:
            parts.append(f"reason={reason}")
        super().__init__(", ".join(parts))


@dataclass
class RoutingTable:
    input_stream: str
    conditions: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class _NotificationConfig:
    """Per-stream notification configuration."""
    batch_size: int = 1
    interval_secs: float = _MIN_NOTIFICATION_INTERVAL
    persistent_thread_id: str | None = None
    _last_fired: float = field(default=0.0, init=False, repr=False)
    immutable: bool = False
    notify_on_empty: bool = False


class StreamProcessor(FunctionManager, StreamBufferManager, AgenticObject):
    """You are a stream processor managing multiple routing tables.
    You also have a notification system for stream updates — it sends you automatic [NOTIFICATION] chat messages disguised as user messages.
    While stream buffer hooks allow you to execute functions automatically, the notification system allows you to get notified (chat messages) automatically.
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
      enable_notification <stream>  — enable notification on a stream
      disable_notification <stream>  — disable notification on a stream
      configure_notification <stream> [batch_size=<N>] [interval_secs=<T>] [notify_on_empty=<bool>]  — configure notification
    - The feedback stream "stream:processor:feedback" is always observed (immutable, cannot be disabled).
    - Read feedback and streaming errors (command results, FALLTHROUGH) from "stream:processor:feedback".
    """

    def __init__(self, **kwargs) -> None:
        # Initialize instance variables.
        super().__init__(**kwargs)

        # Require a running event loop for async notification setup.
        loop = asyncio.get_running_loop()

        self._routing_tables: dict[str, RoutingTable] = {}
        self._stream_notification_configs: dict[str, _NotificationConfig] = {}
        self._stream_notification_timer: asyncio.Task | None = None

        # Create the control buffer for routing table commands (sync, no hooks at construction).
        result = self._create_buffer("stream:processor:control", stream=True)
        if not result.get("ok"):
            raise RuntimeError(f"failed to create stream:processor:control: {result.get('error')}")

        # Create the feedback buffer for routing results and errors (sync, no hooks at construction).
        result = self._create_buffer("stream:processor:feedback", stream=True)
        if not result.get("ok"):
            raise RuntimeError(f"failed to create stream:processor:feedback: {result.get('error')}")

        # Register the system processor hook to handle control stream commands.
        hook_result = self._set_stream_on_append_hook(
            "stream:processor:control",
            name="system_processor",
            hook=lambda stream, text, metadata: self._system_processor_hook(stream, text, metadata),
        )
        if not hook_result.get("ok"):
            raise RuntimeError(f"failed to register system_processor hook: {hook_result.get('error')}")

        # Seed the routing tables listing and notification configs listing (sync, no hooks at construction).
        result = self._create_buffer("system:list:routing_tables", text="[]")
        if not result.get("ok"):
            raise RuntimeError(f"failed to create routing tables buffer: {result.get('error')}")

        result = self._create_buffer("system:list:notification_configs", text="[]")
        if not result.get("ok"):
            raise RuntimeError(f"failed to create notification configs buffer: {result.get('error')}")

        # Start the global notification timer (sync — uses loop.create_task internally).
        self._stream_notification_timer = _start_stream_notification_timer(self, loop)

        # Register the immutable feedback notification (fire-and-forget).
        task = loop.create_task(self._enable_notification("stream:processor:feedback", immutable=True))
        task.add_done_callback(
            lambda t: (
                asyncio.get_running_loop().call_soon_threadsafe(
                    lambda: self.write_buffer("stream:processor:feedback", f"NOTIFICATION INIT FAILED: {t.exception()}")
                )
                if t.exception() else None
            )
        )

    async def _refresh_routing_tables_buffer(self) -> dict[str, Any]:
        """Refresh the system:list:routing_tables buffer."""
        from petekit.utils.text_formatters import format_dict_list_for_buffer
        return await self.create_buffer(
            "system:list:routing_tables",
            text=format_dict_list_for_buffer(self._list_routing_tables()),
            overwrite=True
        )

    @sandbox
    def _list_routing_tables(self) -> list[dict]:
        """Return all routing tables with name, buffer, input_stream, and condition count."""
        return [
            {"name": name, "buffer": _routing_table_key(name), "input_stream": rt.input_stream, "conditions": len(rt.conditions)}
            for name, rt in self._routing_tables.items()
        ]

    async def _configure_notification(self, args: list[str]) -> dict[str, Any]:
        """Configure notification for any stream: configure_notification <stream> [batch_size=<N>] [interval_secs=<T>] [notify_on_empty=<bool>].
        The persistent_thread_id is auto-selected to the most recent thread (hack — see _most_recent_persistent_thread).
        The feedback stream is always configurable. Arbitrary streams must be enabled first with enable_notification."""
        # Reject if no arguments given.
        if not args:
            return {"ok": False, "error": "usage: configure_notification <stream> [batch_size=<N>] [interval_secs=<T>] [notify_on_empty=<bool>]"}

        # Reject if stream not registered.
        stream = args[0]
        if stream not in self._stream_notification_configs:
            return {"ok": False, "error": f"notification not enabled for '{stream}'. Use 'enable_notification' first."}
        cfg = self._stream_notification_configs[stream]

        # Parse remaining key=value arguments.
        batch_size: int | None = None
        interval_secs: float | None = None
        notify_on_empty: bool | None = None
        for arg in args[1:]:
            if "=" in arg:
                key, val = arg.split("=", 1)
                if key == "batch_size":
                    try:
                        batch_size = max(1, int(val))
                    except ValueError:
                        return {"ok": False, "error": f"invalid batch_size: {val}"}
                elif key == "interval_secs":
                    try:
                        interval_secs = max(_MIN_NOTIFICATION_INTERVAL, min(_MAX_NOTIFICATION_INTERVAL, float(val)))
                    except ValueError:
                        return {"ok": False, "error": f"invalid interval_secs: {val}"}
                elif key == "notify_on_empty":
                    notify_on_empty = val.lower() in ("true", "1", "yes")

        # Apply parsed values to config.
        if batch_size is not None:
            cfg.batch_size = batch_size
        if interval_secs is not None:
            cfg.interval_secs = interval_secs
        if notify_on_empty is not None:
            cfg.notify_on_empty = notify_on_empty

        refresh_result = await self._refresh_notification_configs_buffer()
        if not refresh_result.get("ok"):
            return {"ok": False, "error": f"failed to refresh notification configs buffer: {refresh_result.get('error')}"}

        return {"ok": True, "stream": stream, "batch_size": cfg.batch_size, "interval_secs": cfg.interval_secs, "notify_on_empty": cfg.notify_on_empty}

    async def _enable_notification(self, stream: str, immutable: bool = False) -> dict[str, Any]:
        """Enable notification on a stream. immutable=True prevents disable_notification from removing it."""
        # First reject if already registered.
        if stream in self._stream_notification_configs:
            return {"ok": False, "error": f"notification already enabled for '{stream}'"}

        # Reject if stream doesn't exist.
        if stream not in self._buffers:
            return {"ok": False, "error": f"stream '{stream}' does not exist"}

        # The GIL means we safely register the hook before storing config — if hook
        # registration fails we return error without any cleanup needed.
        # TODO(investigate): _stream_notification_hook_async is a thin wrapper around _notify_if_ready.
        # Could call _notify_if_ready directly here and remove the indirection.
        hook_result = self._set_stream_on_append_hook(
            stream,
            name=f"notification:{stream}",
            hook=lambda stream, text, metadata: self._stream_notification_hook_async(stream, text, metadata),
        )
        if not hook_result.get("ok"):
            return {"ok": False, "error": f"failed to register hook: {hook_result.get('error')}"}

        # Config is stored last so on failure we return without modifying any state.
        cfg = _NotificationConfig(immutable=immutable)
        self._stream_notification_configs[stream] = cfg

        # HACK: auto-select most recent persistent thread — see _most_recent_persistent_thread
        try:
            cfg.persistent_thread_id = _most_recent_persistent_thread(self)
        except Exception as e:
            fb_result = await self.write_buffer("stream:processor:feedback", f"PERSISTENT THREAD DETECTION FAILED: {e}")
            cfg.persistent_thread_id = None

        refresh_result = await self._refresh_notification_configs_buffer()
        if not refresh_result.get("ok"):
            return {"ok": False, "error": f"failed to refresh notification configs buffer: {refresh_result.get('error')}"}

        return {"ok": True, "stream": stream, "batch_size": cfg.batch_size, "interval_secs": cfg.interval_secs, "notify_on_empty": cfg.notify_on_empty}

    async def _disable_notification(self, stream: str) -> dict[str, Any]:
        """Disable notification on an arbitrary stream (fails for immutable streams like feedback)."""
        # Reject if not registered.
        if stream not in self._stream_notification_configs:
            return {"ok": False, "error": f"notification not enabled for '{stream}'"}

        # Reject immutable streams — they cannot be disabled.
        cfg = self._stream_notification_configs[stream]
        if cfg.immutable:
            return {"ok": False, "error": f"notification on '{stream}' is immutable and cannot be disabled"}

        # Unhook removes the append callback so the stream stops notifying.
        unhook_result = self._set_stream_on_append_hook(
            stream,
            name=f"notification:{stream}",
            hook=None,
        )
        if not unhook_result.get("ok"):
            return {"ok": False, "error": f"failed to unhook: {unhook_result.get('error')}"}

        # Config is removed last so on unhook failure we return without modifying state.
        del self._stream_notification_configs[stream]

        refresh_result = await self._refresh_notification_configs_buffer()
        if not refresh_result.get("ok"):
            return {"ok": False, "error": f"failed to refresh notification configs buffer: {refresh_result.get('error')}"}

        return {"ok": True, "stream": stream}

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
            try:
                if cmd == "new" and len(parts) >= 3:
                    routing_table, input_stream = parts[1], parts[2]
                    result = await self._routing_table_new(routing_table, input_stream)
                elif cmd == "drop" and len(parts) >= 2:
                    routing_table = parts[1]
                    result = await self._routing_table_drop(routing_table)
                elif cmd == "add" and len(parts) >= 4:
                    routing_table, condition_list, output_stream = parts[1], parts[2], " ".join(parts[3:])
                    result = await self._routing_table_add(routing_table, condition_list, output_stream)
                elif cmd == "del" and len(parts) >= 4:
                    routing_table, condition_list, output_stream = parts[1], parts[2], " ".join(parts[3:])
                    result = await self._routing_table_del(routing_table, condition_list, output_stream)
                elif cmd == "configure_notification" and len(parts) >= 2:
                    result = await self._configure_notification(parts[1:])
                elif cmd == "enable_notification" and len(parts) >= 2:
                    stream_name = parts[1]
                    result = await self._enable_notification(stream_name)
                elif cmd == "disable_notification" and len(parts) >= 2:
                    stream_name = parts[1]
                    result = await self._disable_notification(stream_name)
                else:
                    fb_result = await self.write_buffer("stream:processor:feedback", f"unknown or malformed command: {line}")
                    if not fb_result.get("ok"):
                        raise RuntimeError(f"failed to write to feedback buffer: {fb_result.get('error')}")
                    continue
            except Exception as e:
                fb_result = await self.write_buffer("stream:processor:feedback", f"SYSTEM ERROR: {e}")
                if not fb_result.get("ok"):
                    raise RuntimeError(f"failed to write system error to feedback buffer: {fb_result.get('error')}") from e
                raise

            if not result.get("ok"):
                raise RuntimeError(f"command '{cmd}' failed: {result.get('error')}")
            fb_result = await self.write_buffer("stream:processor:feedback", str(result))
            if not fb_result.get("ok"):
                raise RuntimeError(f"failed to write to feedback buffer: {fb_result.get('error')}")

    async def _routing_table_new(self, routing_table: str, input_stream: str) -> dict[str, Any]:
        """Create a routing table with a fixed input stream."""
        # Guard: reject duplicate routing table names.
        table_name = _routing_table_key(routing_table)
        if routing_table in self._routing_tables or table_name in self._buffers:
            return {"ok": False, "error": f"routing table '{routing_table}' already exists"}

        # Create the routing table buffer to hold the rules.
        create_result = await self.create_buffer(table_name)
        if not create_result.get("ok"):
            raise RuntimeError(f"failed to create routing table buffer '{table_name}': {create_result.get('error')}")

        # Register the routing table and attach its dispatch hook to the input stream.
        self._routing_tables[routing_table] = RoutingTable(input_stream=input_stream)
        hook_result = self._set_stream_on_append_hook(
            input_stream,
            name=f"routing_table:{routing_table}",
            hook=lambda stream, text, metadata: self._routing_table_dispatch(routing_table, text, metadata),
        )
        if not hook_result.get("ok"):
            # Roll back: unregister the table and drop its buffer.
            del self._routing_tables[routing_table]
            await self.drop_buffer(table_name)
            raise RuntimeError(f"failed to hook input stream '{input_stream}': {hook_result.get('error')}")

        # Refresh the routing tables listing and propagate any failure.
        refresh_result = await self._refresh_routing_tables_buffer()
        if not refresh_result.get("ok"):
            self._set_stream_on_append_hook(input_stream, name=f"routing_table:{routing_table}", hook=None)
            del self._routing_tables[routing_table]
            await self.drop_buffer(table_name)
            raise RuntimeError(f"failed to refresh routing tables buffer: {refresh_result.get('error')}")

        return {"ok": True, "table": routing_table, "input_stream": input_stream}

    async def _routing_table_drop(self, routing_table: str) -> dict[str, Any]:
        """Delete a routing table and remove its input stream hook."""
        # Guard: the table must exist before anything is mutated.
        table_name = _routing_table_key(routing_table)
        if routing_table not in self._routing_tables:
            return {"ok": False, "error": f"routing table '{routing_table}' not found"}
        input_stream = self._routing_tables[routing_table].input_stream

        # Unhook removes the dispatch callback so the stream stops routing.
        unhook_result = self._set_stream_on_append_hook(input_stream, name=f"routing_table:{routing_table}", hook=None)
        if not unhook_result.get("ok"):
            return {"ok": False, "error": f"failed to unhook '{input_stream}': {unhook_result.get('error')}"}

        # Mutate: remove the table and drop its buffer.
        del self._routing_tables[routing_table]
        if table_name in self._buffers:
            await self.drop_buffer(table_name)

        # Refresh the routing tables listing and propagate any failure.
        refresh_result = await self._refresh_routing_tables_buffer()
        if not refresh_result.get("ok"):
            raise RuntimeError(f"failed to refresh routing tables buffer: {refresh_result.get('error')}")

        return {"ok": True, "dropped": routing_table}

    async def _routing_table_add(self, routing_table: str, condition_list: str, output_stream: str) -> dict[str, Any]:
        """Append a condition->output entry to a routing table."""
        # Guard: the table must already exist.
        table_name = _routing_table_key(routing_table)
        if routing_table not in self._routing_tables or table_name not in self._buffers:
            return {"ok": False, "error": f"routing table '{routing_table}' not found. Use 'new' first."}

        # Format the entry and append it to the table buffer.
        entry = _format_routing_entry(condition_list, output_stream)
        append_result = await self.write_buffer(table_name, entry)
        if not append_result.get("ok"):
            raise RuntimeError(f"failed to append to routing table buffer: {append_result.get('error')}")

        # Update in-memory state and refresh the routing tables listing.
        self._routing_tables[routing_table].conditions.append((condition_list, output_stream))
        refresh_result = await self._refresh_routing_tables_buffer()
        if not refresh_result.get("ok"):
            raise RuntimeError(f"failed to refresh routing tables buffer: {refresh_result.get('error')}")

        return {"ok": True, "table": routing_table, "condition_list": condition_list, "output": output_stream}

    async def _routing_table_del(self, routing_table: str, condition_list: str, output_stream: str) -> dict[str, Any]:
        """Remove the first matching condition->output entry from a routing table."""
        # Guard: the table must exist.
        table_name = _routing_table_key(routing_table)
        if routing_table not in self._routing_tables:
            return {"ok": False, "error": f"routing table '{routing_table}' not found"}
        table = self._routing_tables[routing_table]

        # Locate and remove the first matching entry.
        for i, (cn, os) in enumerate(table.conditions):
            if cn == condition_list and os == output_stream:
                del table.conditions[i]
                entry = _format_routing_entry(condition_list, output_stream)
                edit_result = await self.edit_buffer(table_name, old_string=entry, new_string="")
                if not edit_result.get("ok"):
                    error_msg = edit_result.get("error", "")
                    if "not found" in error_msg:
                        return edit_result
                    raise RuntimeError(f"failed to edit routing table buffer: {error_msg}")
                return {"ok": True, "removed": f"{condition_list} {output_stream}"}
        return {"ok": False, "error": f"no matching entry in '{routing_table}'"}

    async def _stream_notification_timer_fire(self) -> None:
        """Notify all streams that are ready (called by the async timer loop)."""
        # HACK: feedback stream always uses the most recent persistent thread (refreshed on every tick).
        # This must run on every tick to keep the config current.
        fb_cfg = self._stream_notification_configs.get("stream:processor:feedback")
        if fb_cfg is not None:
            try:
                fb_cfg.persistent_thread_id = _most_recent_persistent_thread(self)
            except Exception:
                pass
            await self._refresh_notification_configs_buffer()

        # Notify each configured stream that is ready to deliver accumulated batched messages.
        for stream in list(self._stream_notification_configs.keys()):
            await self._notify_if_ready(stream)

    async def _stream_notification_hook_async(self, stream: str, text: str, metadata: dict) -> None:
        """Entry point from stream append hook — delegates to central _notify_if_ready."""
        await self._notify_if_ready(stream)

    async def _notify_if_ready(self, stream: str) -> None:
        """Central notification logic. Checks batch and timer conditions; fires if ready.

        Called from the async timer loop and from a stream append hook (both async).
        """
        # Reject if stream not registered.
        cfg = self._stream_notification_configs.get(stream)
        if cfg is None:
            return

        # Reject if buffer not found.
        buf = self._buffers.get(stream)
        if buf is None:
            return

        # Gather unseen entries and current time for condition checks.
        # Unseen = entries not yet injected into agent chat context (agent_at is None).
        now = time.time()
        unseen = [e for e in buf.lines if e.agent_at is None]

        # Determine if notification should fire: batch threshold met OR interval elapsed since last fire.
        fire_due_to_batch = bool(unseen) and len(unseen) >= cfg.batch_size
        fire_due_to_interval = cfg.interval_secs is not None and cfg._last_fired > 0 and (now - cfg._last_fired >= cfg.interval_secs) and (bool(unseen) or cfg.notify_on_empty)

        # Record fire time.
        cfg._last_fired = now

        if not fire_due_to_batch and not fire_due_to_interval:
            return

        # Build notification prompt from unseen entries or last known entry.
        if unseen:
            ts_start = unseen[0].modified_at
            ts_end = unseen[-1].modified_at
            prompt = (
                f"[NOTIFICATION] Stream '{stream}' has {len(unseen)} new entries "
                f"between {ts_start:.2f} and {ts_end:.2f}. "
            )
        else:
            last_ts = 0.0
            if stream in self._buffers and self._buffers[stream].lines:
                last_ts = self._buffers[stream].lines[-1].modified_at
            prompt = (
                f"[NOTIFICATION] Stream '{stream}' has 0 new entries. "
                f"Last entry at {last_ts:.2f}. "
            )

        # Mark entries as seen by the agent (stamped before injection).
        for entry in unseen:
            entry.agent_at = now

        # Invoke agent and report errors to feedback buffer.
        from peteos import Error
        result = await self.invoke_agent(prompt, persistent_thread_id=cfg.persistent_thread_id)
        if isinstance(result, Error):
            fb_result = await self.write_buffer("stream:processor:feedback", f"NOTIFICATION ERROR: {result.message}")
            if not fb_result.get("ok"):
                pass

    async def _refresh_notification_configs_buffer(self) -> dict[str, Any]:
        """Refresh the system:list:notification_configs buffer."""
        from petekit.utils.text_formatters import format_dict_list_for_buffer
        return await self.create_buffer(
            "system:list:notification_configs",
            text=format_dict_list_for_buffer(self.list_notification_configs()),
            overwrite=True
        )

    @sandbox
    def list_notification_configs(self) -> list[dict]:
        """List all notification configurations for discoverability."""
        return [
            {
                "stream": stream,
                "batch_size": cfg.batch_size,
                "interval_secs": cfg.interval_secs,
                "persistent_thread_id": cfg.persistent_thread_id,
                "immutable": cfg.immutable,
                "notify_on_empty": cfg.notify_on_empty,
            }
            for stream, cfg in self._stream_notification_configs.items()
        ]

    async def _routing_table_dispatch(self, routing_table: str, text: str, metadata: dict | None = None) -> None:
        """Private hook: evaluate incoming data against the routing table conditions.
        First matching condition writes the data to its output stream. No match = reported as FALLTHROUGH.
        """
        # Guard: reject if the routing table does not exist.
        if routing_table not in self._routing_tables:
            fb_result = await self.write_buffer("stream:processor:feedback", f"ERROR: routing_table '{routing_table}' not found")
            if not fb_result.get("ok"):
                raise RuntimeError(f"failed to write ERROR to feedback: {fb_result.get('error')}")
            return

        # Resolve the table and annotate metadata with the routing table name for condition visibility.
        table = self._routing_tables[routing_table]
        # TODO(timestamp-missing): ts below is always 0 — metadata has no timestamp.
        # Real timestamp lives on BufferEntry.modified_at in BufferManager.write_buffer (text/buffer_manager.py:335).
        # Fix: StreamBufferManager.write_buffer (stream/stream_buffer_manager.py:233) should pass the entry's modified_at into the hook.
        ts = metadata.get("timestamp", 0) if metadata else 0
        metadata = {**(metadata or {}), "routing_table": routing_table}

        # Evaluate condition lists top-to-bottom; first list where all sub-conditions match wins.
        for condition_list, output_stream in table.conditions:
            all_match = True
            for sub_condition in condition_list.split(","):
                sc = sub_condition.strip()
                try:
                    matched = await self._evaluate_condition(sc, text, metadata)
                except RoutingConditionError as e:
                    fb_result = await self.write_buffer("stream:processor:feedback", f"FALLTHROUGH source={table.input_stream} ts={ts} reason={e}")
                    if not fb_result.get("ok"):
                        raise RuntimeError(f"failed to write FALLTHROUGH to feedback: {fb_result.get('error')}")
                    return
                except Exception as e:
                    fb_result = await self.write_buffer("stream:processor:feedback", f"CONDITION ERROR condition='{sc}' text='{text[:40]}' error='{e}'")
                    if not fb_result.get("ok"):
                        raise RuntimeError(f"failed to write CONDITION ERROR to feedback: {fb_result.get('error')}")
                    return
                if not matched:
                    all_match = False
                    break

            # All sub-conditions matched: write to the output stream and stop evaluation.
            if all_match:
                sink_result = await self.write_buffer(output_stream, text)
                if not sink_result.get("ok"):
                    fb_result = await self.write_buffer("stream:processor:feedback", f"SINK ERROR: failed to write to '{output_stream}': {sink_result.get('error')}")
                    if not fb_result.get("ok"):
                        raise RuntimeError(f"failed to write SINK ERROR to feedback: {fb_result.get('error')}")
                    return
                return

        # No condition list matched: report FALLTHROUGH to the feedback stream.
        fb_result = await self.write_buffer("stream:processor:feedback", f"FALLTHROUGH source={table.input_stream} ts={ts}")
        if not fb_result.get("ok"):
            raise RuntimeError(f"failed to write FALLTHROUGH to feedback: {fb_result.get('error')}")

    async def _evaluate_condition(self, condition_name: str, text: str, metadata: dict | None = None) -> bool:
        """Evaluate a single named condition against the given text and routing metadata.
        Conditions are sandbox-decorated methods with signature (stream, text, metadata) -> bool
        or (self, stream, text, metadata) -> bool. Prefix the name with "!" to negate the result.
        The built-in names "true" and "false" return their respective boolean values.
        Raises RoutingConditionError if the name is not registered as a @sandbox decorated method
        or if the method signature is incompatible with (stream, text, metadata) -> bool.
        or if the signature does not match the required form.
        """
        # Handle negation prefix: strip "!" and return the negated result of the inner condition.
        if condition_name.startswith("!"):
            inner = condition_name[1:]
            return not await self._evaluate_condition(inner, text, metadata)

        # Handle the built-in "true" and "false" conditions.
        if condition_name.lower() == "true":
            return True
        if condition_name.lower() == "false":
            return False

        # Locate the condition — first in sandbox methods, then in hidden functions.
        import inspect
        members = self._gather_sandbox_members()
        if condition_name in members:
            method = members[condition_name]
        else:
            fn_result = self.get_function(condition_name)
            if not fn_result.get("ok"):
                raise RoutingConditionError(
                    condition=condition_name,
                    routing_table=(metadata or {}).get("routing_table"),
                    reason=f"not found in sandbox members or function manager: {fn_result.get('error')}",
                )
            method = fn_result["callable"]
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())

        # Dispatch using the 3-arg or 4-arg signature form; validate return type is bool.
        _is_bool_return = (
            sig.return_annotation is bool
            or sig.return_annotation is inspect.Parameter.empty
            or sig.return_annotation == "bool"
        )
        if params[:3] == ["stream", "text", "metadata"] and _is_bool_return:
            result = method(stream=condition_name, text=text, metadata=metadata or {})
        elif params[:4] == ["self", "stream", "text", "metadata"] and _is_bool_return:
            result = method(self, stream=condition_name, text=text, metadata=metadata or {})
        else:
            raise RoutingConditionError(
                condition=condition_name,
                routing_table=(metadata or {}).get("routing_table"),
                reason=f"incompatible signature: {params}",
            )

        # Await the result if the condition is a coroutine.
        if asyncio.iscoroutine(result):
            result = await result

        return result
