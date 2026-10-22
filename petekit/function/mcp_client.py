"""MCP client — MCP server connection management and remote function registration."""

from __future__ import annotations

import json
import logging

from dataclasses import dataclass
from typing import Any, Callable

from mcp import Client
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters

from peteos.oap.agentic_object import AgenticObject

from petekit.function.function_caller import FunctionCaller
from petekit.text.buffer_manager import Buffer, HookPoint


# -------------------------------------------------------------------------------------------------------------------- #
# Constants                                                                                                          #
# -------------------------------------------------------------------------------------------------------------------- #


_MCP_CLIENT_SCOPE = "mcp:"
_MCP_CLIENT_DOC_BUFFER = "doc:mcp_client"
_MCP_CLIENT_STATUS_SUFFIX = ":status"
_MCP_CLIENT_TOOLS_SUFFIX = ":tools/"
_MCP_CLIENT_RESOURCES_SUFFIX = ":resources/"
_MCP_CLIENT_PROMPTS_SUFFIX = ":prompts/"
_MCP_CLIENT_DOC = f"""\
## MCP Client

Connect to MCP servers and use their tools as if they were local functions.

### Connection Configuration

Create one buffer per MCP connection at `mcp:<name>:config`.
There are two ways to configure a connection — URL mode or command mode.

#### URL mode — HTTP/SSE or WebSocket

```json
{{
  "url": "http://localhost:3000",
  "active": false
}}
```

- `url` --  required; the full URL to the MCP server (http, https, or ws)
- `active` --  write `true` to connect; write `false` to disconnect

#### Command mode — stdio (subprocess)

```json
{{
  "command": "python",
  "args": ["-m", "my_mcp_server"],
  "active": false
}}
```

- `command` --  required; the executable to run
- `args` --  required; the command-line arguments passed to the executable
- `active` --  write `true` to connect; write `false` to disconnect

### Status

When a connection is active, a status buffer is created at
`mcp:<name>:status`. It is updated automatically and lists everything
available through the connection:

```json
{{
  "status": "connected",
  "tools": ["search_books", "lookup_book"],
  "resources": ["catalog::genres"],
  "prompts": ["recommend"]
}}
```

You can read this buffer to discover what functions, resources, and prompts
are available. Do not write to this buffer manually — it is
automatically maintained.

### Tools

Each tool from the MCP server is registered as a local callable.
Functions are registered under their full buffer name:
`mcp:<name>:tools/<tool_name>`.

Read the status buffer to get the list of available tool short names, then
read each tool's detail buffer at `mcp:<name>:tools/<tool_name>` for the full
function name to pass to `call_function`.

Call it using the full buffer name. For example:
`call_function("mcp:playwright:tools/browser_close", {{}})`.

If the connection is not active when you call a tool, you get an error.
Reconnect by setting `active` to `true` in the config buffer.

### Resources

Resources are read-only data objects exposed by the MCP server.
Read the status buffer to learn which resources are available.
Use `call_function` to fetch a resource, or read its detail buffer directly.

### Prompts

Prompts are pre-built prompt templates with arguments.
Read the status buffer to learn which prompts are available.
Render a prompt by calling it with the required arguments.

### Disconnect

Set `active` to `false` in `mcp:<name>:config` to close the connection.
The config and status buffers remain; the status buffer is updated to
`"status": "disconnected"`.

### Reconnect

Set `active` to `true` in `mcp:<name>:config` to open the connection again.
The status buffer is updated to `"status": "connected"` and all tools,
resources, and prompts are re-registered.
"""


# -------------------------------------------------------------------------------------------------------------------- #
# Utils                                                                                                             #
# -------------------------------------------------------------------------------------------------------------------- #


_logger = logging.getLogger("mcp_client")


@dataclass
class _ConnectionContext:
    """Holds the entered MCP Client and its session for the lifetime of a connection.
    
    The client is entered once in _open_mcp_client and exited once in _close_mcp_client.
    Callers (e.g. the tool wrapper) use ctx.session to make calls without re-entering.
    """
    client: Client
    session: ClientSession


def _config_buffer_name(name: str) -> str:
    # Derive the canonical config buffer name for the given connection name.
    return f"{_MCP_CLIENT_SCOPE}{name}:config"


def _status_buffer_name(name: str) -> str:
    # Derive the canonical status buffer name for the given connection name.
    return f"{_MCP_CLIENT_SCOPE}{name}:status"


def _tool_buffer_name(name: str, tool_name: str) -> str:
    # Derive the canonical buffer name for a single tool in a connection.
    return f"{_MCP_CLIENT_SCOPE}{name}{_MCP_CLIENT_TOOLS_SUFFIX}{tool_name}"


def _resource_buffer_name(name: str, resource_name: str) -> str:
    # Derive the canonical buffer name for a single resource in a connection.
    return f"{_MCP_CLIENT_SCOPE}{name}{_MCP_CLIENT_RESOURCES_SUFFIX}{resource_name}"


def _prompt_buffer_name(name: str, prompt_name: str) -> str:
    # Derive the canonical buffer name for a single prompt in a connection.
    return f"{_MCP_CLIENT_SCOPE}{name}{_MCP_CLIENT_PROMPTS_SUFFIX}{prompt_name}"


def _read_config(config_buffer: Any) -> dict | str:
    """Parse a config buffer as JSON and return the dict, or an error string on failure."""
    # Attempt to parse the buffer text as JSON; report the error if it fails.
    try:
        config = json.loads(config_buffer.get_text() or "{}")
    except json.JSONDecodeError as e:
        return f"config is not valid JSON: {e}"

    # Reject the parse if the result is not a JSON object.
    if not isinstance(config, dict):
        return "config must be a JSON object"

    return config


async def _open_mcp_client(
    mcp_client: MCPClient,
    name: str,
    config: dict,
) -> None:
    """Open and register an MCP connection for the given connection name and config.

    The config must contain either a `url` or a `command` with `args`.
    Stores the connection context in mcp_client._connections[name].
    The connection is established by entering the client context.
    """
    # Determine which transport mode applies based on the config fields.
    if "url" in config:
        # URL mode — HTTP/SSE or WebSocket; construct with the URL string.
        url = config["url"]
        client = Client(url)
    else:
        # Command mode — stdio (subprocess); construct with StdioServerParameters.
        command = config["command"]
        args = config.get("args", [])
        params = StdioServerParameters(command=command, args=args)
        client = Client(params)

    # Enter the client context and store the connection context on success.
    await client.__aenter__()
    mcp_client._connections[name] = _ConnectionContext(client, client.session)

    # List available tools, resources, and prompts from the connected MCP server.
    caps = client.server_capabilities
    tools = (await client.session.list_tools()).tools if caps.tools else []
    resources = (await client.session.list_resources()).resources if caps.resources else []
    prompts = (await client.session.list_prompts()).prompts if caps.prompts else []

    # Collect the full buffer names of all items for the status summary buffer.
    tool_names = [_tool_buffer_name(name, t.name) for t in tools]
    resource_names = [_resource_buffer_name(name, r.name) for r in resources]
    prompt_names = [_prompt_buffer_name(name, p.name) for p in prompts]
    status_content = {
        "status": "connected",
        "tools": tool_names,
        "resources": resource_names,
        "prompts": prompt_names,
    }

    # Write the status summary buffer at mcp:<name>:status.
    status_name = _status_buffer_name(name)
    status_result = await mcp_client.create_buffer(
        status_name,
        text=json.dumps(status_content, indent=2, ensure_ascii=False),
        overwrite=True,
    )
    if not status_result.get("ok"):
        _logger.error(f"[MCPClient] failed to create status buffer '{status_name}': {status_result.get('error')}")

    # For each tool, create a detail buffer at mcp:<name>:tools/<tool_name>.
    # Serialize the tool object directly and write it as JSON.
    for tool in tools:
        # Register the tool as a callable in the function manager.
        func_name = tool.name
        tool_buffer_name = _tool_buffer_name(name, func_name)
        wrapper = _make_mcp_tool_callable(mcp_client, name, func_name)
        create_result = await mcp_client.create_function(
            tool_buffer_name,
            wrapper,
            tool.description or wrapper.__doc__,
            tool.description or "",
        )
        if not create_result.get("ok"):
            _logger.error(f"[MCPClient] failed to register function '{func_name}': {create_result.get('error')}")

        # Write the tool detail buffer at mcp:<name>:tools/<tool_name>.
        tool_result = await mcp_client.create_buffer(
            tool_buffer_name,
            text=json.dumps(tool.model_dump(), indent=2, ensure_ascii=False),
            overwrite=True,
        )
        if not tool_result.get("ok"):
            _logger.error(f"[MCPClient] failed to create tool buffer '{tool_buffer_name}': {tool_result.get('error')}")

    # For each resource, create a detail buffer at mcp:<name>:resources/<resource_name>.
    # Serialize the resource object directly and write it as JSON.
    for resource in resources:
        resource_name = _resource_buffer_name(name, resource.name)
        resource_result = await mcp_client.create_buffer(
            resource_name,
            text=json.dumps(resource.model_dump(), indent=2, ensure_ascii=False),
            overwrite=True,
        )
        if not resource_result.get("ok"):
            _logger.error(f"[MCPClient] failed to create resource buffer '{resource_name}': {resource_result.get('error')}")

    # For each prompt, create a detail buffer at mcp:<name>:prompts/<prompt_name>.
    # Serialize the prompt object directly and write it as JSON.
    for prompt in prompts:
        prompt_name = _prompt_buffer_name(name, prompt.name)
        prompt_result = await mcp_client.create_buffer(
            prompt_name,
            text=json.dumps(prompt.model_dump(), indent=2, ensure_ascii=False),
            overwrite=True,
        )
        if not prompt_result.get("ok"):
            _logger.error(f"[MCPClient] failed to create prompt buffer '{prompt_name}': {prompt_result.get('error')}")

    return


async def _close_mcp_client(
    mcp_client: MCPClient,
    name: str,
) -> None:
    """Close the given MCP connection gracefully.

    Reads the status buffer to enumerate all registered tools, resources, and
    prompts, then removes their system buffers before exiting the connection.
    """
    # Locate the status buffer for this connection.
    status_name = _status_buffer_name(name)
    status_buffer = mcp_client.get_buffer(status_name)

    # If no status buffer exists, proceed directly to closing the connection.
    if status_buffer is None:
        # Exit the client context; ignore any exception during teardown.
        try:
            await ctx.client.__aexit__(None, None, None)
        except Exception:
            pass
        return

    # Parse the status buffer as JSON to read the registered function names.
    try:
        status = json.loads(status_buffer.get_text() or "{}")
    except json.JSONDecodeError:
        # Exit the client context; ignore any exception during teardown.
        try:
            await ctx.client.__aexit__(None, None, None)
        except Exception:
            pass
        return

    # Drop each tool buffer at mcp:<name>:tools/<tool_name>.
    # Status already contains full buffer names, so use them directly.
    tool_names = status.get("tools", [])

    # Unregister each tool from the function manager and drop its detail buffer.
    for tool_buffer_name in tool_names:
        drop_func_result = await mcp_client.drop_function(tool_buffer_name)

        # Log an error if the function could not be unregistered.
        if not drop_func_result.get("ok"):
            _logger.error(f"[MCPClient] failed to unregister function '{tool_buffer_name}': {drop_func_result.get('error')}")

        # Drop the tool detail buffer from the buffer manager.
        drop_result = await mcp_client.drop_buffer(tool_buffer_name)

        # Log an error if the buffer could not be dropped.
        if not drop_result.get("ok"):
            _logger.error(f"[MCPClient] failed to drop tool buffer '{tool_buffer_name}': {drop_result.get('error')}")

    # Drop each resource buffer at mcp:<name>:resources/<resource_name>.
    # Status already contains full buffer names, so use them directly.
    resource_names = status.get("resources", [])
    for resource_buffer_name in resource_names:
        drop_result = await mcp_client.drop_buffer(resource_buffer_name)

        # Log an error if the buffer could not be dropped.
        if not drop_result.get("ok"):
            _logger.error(f"[MCPClient] failed to drop resource buffer '{resource_buffer_name}': {drop_result.get('error')}")

    # Drop each prompt buffer at mcp:<name>:prompts/<prompt_name>.
    # Status already contains full buffer names, so use them directly.
    prompt_names = status.get("prompts", [])
    for prompt_buffer_name in prompt_names:
        drop_result = await mcp_client.drop_buffer(prompt_buffer_name)

        # Log an error if the buffer could not be dropped.
        if not drop_result.get("ok"):
            _logger.error(f"[MCPClient] failed to drop prompt buffer '{prompt_buffer_name}': {drop_result.get('error')}")

    # Update the status buffer to mark the connection as closed; keep it for reference.
    status["status"] = "disconnected"
    status_update_result = await mcp_client.create_buffer(
        status_name,
        text=json.dumps(status, indent=2, ensure_ascii=False),
        overwrite=True,
    )
    if not status_update_result.get("ok"):
        _logger.error(f"[MCPClient] failed to update status buffer to closed '{status_name}': {status_update_result.get('error')}")

    # Exit the client context; ignore any exception during teardown.
    try:
        await ctx.client.__aexit__(None, None, None)
    except Exception:
        pass

    # Remove the connection context from the registry.
    del mcp_client._connections[name]


def _make_mcp_tool_callable(
    mcp_client: MCPClient,
    name: str,
    tool_name: str,
) -> Callable:
    """Factory producing a callable that forwards to session.call_tool at call time.

    The returned closure captures the connection name and the tool name.
    At call time it looks up the live connection context from
    mcp_client._connections[name] and forwards the call to ctx.session.call_tool.
    """
    async def wrapper(**kwargs: Any) -> Any:
        # Look up the live connection context for this connection at call time.
        ctx = mcp_client._connections.get(name)
        if ctx is None:
            raise RuntimeError(f"MCP connection '{name}' is not active")

        # Forward the call to the client with the captured tool name.
        result = await ctx.client.call_tool(tool_name, **kwargs)
        return result

    # Describe the signature buffer so the agent can read the tool's parameter schema.
    tool_buffer_name = _tool_buffer_name(name, tool_name)
    wrapper.__doc__ = (
        f"Calls the MCP tool '{tool_name}' on connection '{name}'.\n"
        f"The tool's parameter signature is described in the buffer: {tool_buffer_name}"
    )

    return wrapper


# -------------------------------------------------------------------------------------------------------------------- #
# Hooks                                                                                                            #
# -------------------------------------------------------------------------------------------------------------------- #


async def _before_update_config(
    mcp_client: MCPClient,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Called before a config buffer is updated or first created.

    Validates that the new configuration buffer holds a valid JSON object
    with either a url or a command+args pair, and a boolean active flag.
    old_buffer is None on first creation.
    Returns None to allow the update, or an error string to reject it.
    """
    # Parse the buffer as a config dict; propagate any parse error.
    result = _read_config(new_buffer)
    if isinstance(result, str):
        return result
    config = result

    # Require exactly one of url or command to be present.
    has_url = "url" in config
    has_command = "command" in config
    if has_url and has_command:
        return "config must contain either 'url' or 'command', not both"
    if not has_url and not has_command:
        return "config must contain either 'url' or 'command'"

    # Validate url mode: url must be a non-empty string.
    if has_url:
        url = config.get("url", "")

        # Reject if url is absent or empty after stripping whitespace.
        if not isinstance(url, str) or not url.strip():
            return "config must contain a non-empty 'url' string"

    # Validate command mode: command must be a non-empty string, args must be a list of strings.
    if has_command:
        command = config.get("command", "")

        # Reject if command is absent or empty after stripping whitespace.
        if not isinstance(command, str) or not command.strip():
            return "config must contain a non-empty 'command' string"

        # Read the args list from the config.
        args = config.get("args", None)

        # Reject if args is absent or not a list.
        if not isinstance(args, list):
            return "config must contain an 'args' list"

        # Reject if any element in args is not a string.
        if not all(isinstance(a, str) for a in args):
            return "config 'args' must be a list of strings"

    # Require the active field as a boolean.
    if "active" not in config or not isinstance(config.get("active"), bool):
        return "config must contain a boolean 'active' field"

    # All checks passed — allow the update to proceed.
    return None


async def _after_update_config(
    mcp_client: MCPClient,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Called after a config buffer is updated or first created.

    When the active flag is true, opens the connection.
    When the URL or command changes, closes and reopens the connection.
    When the active flag becomes false, closes the connection.
    old_buffer is None on first creation.
    """
    # Parse the old config buffer if one exists; log and return if it is corrupt.
    old_config: dict = {}
    if old_buffer is not None:
        old_config = _read_config(old_buffer)
        if isinstance(old_config, str):
            _logger.error(f"[MCPClient] failed to parse old config: {old_config}")
            return None

    # Parse the new config buffer; abort if it is corrupt.
    new_config = _read_config(new_buffer)
    if isinstance(new_config, str):
        _logger.error(f"[MCPClient] failed to parse config: {new_config}")
        return None

    # Derive the connection name from the buffer name.
    config_name = new_buffer.name
    connection_name = config_name.removeprefix(_MCP_CLIENT_SCOPE).removesuffix(":config")

    # Read the active flag, URL, and command from both configs.
    old_active = old_config.get("active", False)
    new_active = new_config.get("active", False)
    old_url = old_config.get("url")
    new_url = new_config.get("url")
    old_command = old_config.get("command")
    new_command = new_config.get("command")

    # Detect a transport change: URL vs command or their values changed.
    transport_changed = (old_url != new_url) or (old_command != new_command)

    # If the connection is not active and the new state is also not active, do nothing.
    if not new_active and not old_active:
        return None

    # If the connection was active and is now inactive, close it.
    if old_active and not new_active:
        existing_ctx = mcp_client._connections.get(connection_name)
        if existing_ctx is not None:
            await _close_mcp_client(mcp_client, connection_name)
        return None

    # If the connection is now active, open a new connection.
    if new_active:
        # If a transport change occurred, close any existing connection first.
        if transport_changed:
            existing_ctx = mcp_client._connections.get(connection_name)
            if existing_ctx is not None:
                await _close_mcp_client(mcp_client, connection_name)

        # Open a new connection with the updated config.
        await _open_mcp_client(mcp_client, connection_name, new_config)

    return None


async def _after_drop_config(
    mcp_client: MCPClient,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Called after a config buffer is dropped.

    Closes the MCP connection if one exists, removes it from the connections
    registry, and drops the associated status buffer.
    """
    # Derive the connection name from the dropped config buffer name.
    # The config buffer is mcp:<name>:config.
    config_name = buffer.name
    connection_name = config_name.removeprefix(_MCP_CLIENT_SCOPE).removesuffix(":config")

    # Close the connection and remove it from the registry.
    existing_ctx = mcp_client._connections.get(connection_name)
    if existing_ctx is not None:
        await _close_mcp_client(mcp_client, connection_name)

    # Drop the associated status buffer if it still exists.
    status_name = _status_buffer_name(connection_name)
    status_exists = mcp_client.get_buffer(status_name)
    if status_exists is not None:
        drop_result = await mcp_client.drop_buffer(status_name)
        if not drop_result.get("ok"):
            _logger.error(f"[MCPClient] failed to drop status buffer '{status_name}': {drop_result.get('error')}")

    return None


async def _before_drop_status(
    mcp_client: MCPClient,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Called before a status buffer is dropped.

    Rejects the drop if the associated config buffer still exists.
    Status buffers are system-maintained and must not be dropped manually.
    """
    # Derive the connection name from the status buffer name.
    # The status buffer is mcp:<name>:status, so we strip the prefix and :status suffix.
    buffer_name = buffer.name
    if not buffer_name.endswith(":status"):
        # This should never happen — the hook pattern already enforces the suffix.
        # Log the anomaly and return an internal error so it is never silently swallowed.
        _logger.error(f"[MCPClient] _before_drop_status received non-status buffer '{buffer_name}' — hook pattern may be wrong")
        return "internal error: _before_drop_status received a non-status buffer"
    name = buffer_name.removeprefix(_MCP_CLIENT_SCOPE).removesuffix(":status")

    # Locate the associated config buffer; if it still exists, reject the drop.
    config_name = _config_buffer_name(name)
    if mcp_client.get_buffer(config_name) is not None:
        return (
            f"you cannot drop {buffer_name} — "
            f"it is a system buffer that is automatically maintained while "
            f"{config_name} exists"
        )

    return None


# -------------------------------------------------------------------------------------------------------------------- #
# MCPClient                                                                                                        #
# -------------------------------------------------------------------------------------------------------------------- #


class MCPClient(FunctionCaller, AgenticObject):
    """I connect to MCP servers and expose their tools as callable functions.
    `doc:mcp_client` holds details.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Inherit the parent constructor to initialise the FunctionCaller base.
        super().__init__(**kwargs)

        # Seed the doc buffer with the module documentation.
        self._create_buffer(_MCP_CLIENT_DOC_BUFFER, text=_MCP_CLIENT_DOC)

        # Initialise the connections registry as an empty name-to-client map.
        self._connections: dict[str, _ConnectionContext] = {}

        # Register the BEFORE hook on config buffers to validate host, port, active.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_UPDATE_BUFFER,
            "mcp_validate_config",
            rf"^{_MCP_CLIENT_SCOPE}.+:config$",
            _before_update_config,
            apply_to_existing=True,
        )

        # Also register on first creation  --  the CREATE signature takes 3 args
        # so a lambda bridges the gap by inserting None for old_buffer.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "mcp_validate_config",
            rf"^{_MCP_CLIENT_SCOPE}.+:config$",
            lambda bm, new_buffer, args: _before_update_config(bm, None, new_buffer, args),
            apply_to_existing=True,
        )

        # Register the AFTER hook on config buffers to trigger connect/disconnect.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_UPDATE_BUFFER,
            "mcp_connect_config",
            rf"^{_MCP_CLIENT_SCOPE}.+:config$",
            _after_update_config,
            apply_to_existing=True,
        )

        # Also register on first creation with the same lambda bridge.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "mcp_connect_config",
            rf"^{_MCP_CLIENT_SCOPE}.+:config$",
            lambda bm, new_buffer, args: _after_update_config(bm, None, new_buffer, args),
            apply_to_existing=True,
        )

        # Register the AFTER_DROP hook on config buffers to handle teardown.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            "mcp_drop_config",
            rf"^{_MCP_CLIENT_SCOPE}.+:config$",
            _after_drop_config,
            apply_to_existing=True,
        )

        # Register the BEFORE_DROP hook on status buffers for pre-drop validation.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_DROP_BUFFER,
            "mcp_drop_status",
            rf"^{_MCP_CLIENT_SCOPE}.+:status$",
            _before_drop_status,
            apply_to_existing=True,
        )
