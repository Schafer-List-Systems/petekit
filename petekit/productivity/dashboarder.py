from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.websockets import WebSocket

from peteos.oap.agentic_object import AgenticObject

from petekit.text.buffer_manager import Buffer, BufferManager, HookPoint
from petekit.text.option_buffer_manager import _origin_name_from_option_buffer_name


_logger = logging.getLogger("dashboarder")


# -------------------------------------------------------------------------------------------------------------------- #
# Constants                                                                                                            #
# -------------------------------------------------------------------------------------------------------------------- #

_DASHBOARDER_DOC_BUFFER = "doc:dashboarder"

_DASHBOARDER_DOC = """
## Dashboarder

You open an HTTP + WebSocket server per dashboard and fan out
create/update/remove events to all connected browsers. Cards are placed on
content buffers and subscribed to one or more dashboard servers.

### Workflow

1. Create the main HTML template buffer:
   e.g. `my_directory/my_dashboard.html` — a content buffer with HTML + JS.
   This page opens a WebSocket to the server and handles the four event types.
   -> The Dashboarder opens an HTTP server on host:port and starts serving the page.

2. Create the dashboard server option buffer on that HTML template buffer:
   e.g. `my_directory/my_dashboard.html[dashboard]` stores:
   {id: "mysrv", port: 8000, host: "0.0.0.0", title: "My Dashboard"}
   -> The Dashboarder opens an HTTP server on host:port and starts serving the main page.

3. Create the card HTML template buffers (one per card type):
   e.g. `my_directory/my_card.html` — with a `<template>` element and a JS namespace that exposes window["my_card"].updateContent(event).

4. Create the content JSON buffer:
   e.g. `any_directory/my_data.json` stores:
   {"title": "Server A", "value": "All systems nominal", "status": "online"}

5. Create the dashboard card option buffer on that content buffer:
   e.g. `any_directory/my_data.json[dashboard-card]` stores:
   {id: "my-card", template: "my_directory/my_card.html", servers: ["mysrv"]}

6. The browser receives create_card and adds a layer to the page.
   The browser calls `updateContent(event)` from the card template to render.

7. When the content buffer updates, the Dashboarder streams the new data to all cards.
   -> The raw text payload is read from the content buffer and forwarded as-is.
   -> The Dashboarder broadcasts an "update_card" event to all servers:
     {event: "update_card", id: "my_card", payload: <raw-text>}
   -> The browser re-renders by calling updateContent on that card.

8. When the dashboard card option buffer drops, the Dashboarder closes the card on all servers.
   -> The Dashboarder broadcasts "remove_card" to all servers.
   -> The browser removes the card layer.

9. When the dashboard server option buffer drops, the HTTP server stops.

### Buffer Formats

<main-template-buffer>             Main page template buffer (HTML + JS)
  Full HTML page served to the browser.
  Must include JS that opens a WebSocket to the server.
  WebSocketListener parses JSON events: create_card, update_card, remove_card, template_updated.
  The main page adds/removes/updates card layers via the DOM.

<main-template-buffer>[dashboard]  Dashboard server option buffer (lives on the main template buffer)
  id: str                          Unique server identifier
  port: int                        TCP port (1-65535)
  host: str                        Bind address (default "0.0.0.0")
  title: str                       Page title shown in the browser
  template: str                    Buffer name of the main template (see <main-template-buffer> above)

<content-buffer>                   Your data buffer. Any format — the card template parses the raw text payload as needed.
<content-buffer>[dashboard-card]   Card configuration
  id: str                          Unique card identifier (used as JS namespace)
  template: str                    Name of the card template buffer (see <card-template-buffer> below)
  servers: list[str]               Dashboard server IDs this card is published to

<card-template-buffer>             Card template buffer (HTML + JS)
  HTML fragment inserted into the card layer.
  JavaScript defines window["<card-id>"].updateContent(event):
    - Called by the browser on every update_card event for this card.
    - event = {event: "update_card", id: "<card-id>", payload: <raw-text>}
    - Mutates the card DOM in place.

### Examples

#### Dashboard main page template

The main page template buffer is served as a full HTML page. It must open a
WebSocket to /ws and handle the four event types. Use `location.host` so the
browser connects to the same host the page was served from.

See `doc:dashboarder:main.html` for a working example.

#### Card template

Each card type needs one template buffer with a `<template>` element and a
JavaScript namespace that exposes `updateContent(event)`.

See `doc:dashboarder:card.html` for a working example.

#### Minimal workflow

1. Create the main HTML template buffer `my_directory/my_dashboard.html`.

2. Create the dashboard server option buffer on that HTML template buffer:
   `my_directory/my_dashboard.html[dashboard]` stores:
   {id: "mysrv", port: 8000, host: "0.0.0.0", title: "My Dashboard"}

3. Create the card HTML template buffer `my_directory/my_card.html`.

4. Create the content JSON buffer `any_directory/my_data.json`:
   {"title": "Server A", "value": "All systems nominal", "status": "online"}

5. Create the dashboard card option buffer on that content buffer:
   `any_directory/my_data.json[dashboard-card]` stores:
   {id: "my-card", template: "my_directory/my_card.html", servers: ["mysrv"]}

The Dashboarder opens the HTTP server, the browser loads the page, opens the
WebSocket, receives the create_card event, renders the card, and re-renders on
every update to my-data.
"""

# Default HTML served to the browser when the main page buffer is empty.
_DEFAULT_DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Log Monitor Dashboard</title>
  <style>
    #card-list {
      display: grid;
      gap: 12px;
      padding: 0;
      list-style: none;
    }
  </style>
</head>
<body>
  <h1 id="page-title">Log Monitor Dashboard</h1>
  <ul id="card-list"></ul>
  <script>
    const cardList = document.querySelector("#card-list");

    function createCardLayer(id, html) {
      console.log("[Dashboarder] === createCardLayer START ===");
      console.log("[Dashboarder] Creating card:", id);
      const li = document.createElement("li");
      li.id = "card-" + id;
      
      // Parse the HTML
      const parser = new DOMParser();
      const doc = parser.parseFromString(html, 'text/html');
      
      // Extract script content and execute immediately
      const scripts = doc.querySelectorAll('script');
      console.log("[Dashboarder] Found", scripts.length, "script tags");
      scripts.forEach(script => {
        if (script.textContent.trim()) {
          console.log("[Dashboarder] Executing script:", script.textContent.substring(0, 100));
          eval(script.textContent);
          console.log("[Dashboarder] After eval, window[" + id + "] =", window[id]);
        }
      });
      
      // Extract template content
      const tempElement = doc.querySelector('template');
      if (tempElement) {
        li.innerHTML = tempElement.innerHTML;
        li.appendChild(tempElement);
      }
      
      cardList.appendChild(li);
      console.log("[Dashboarder] === createCardLayer END ===");
    }

    function removeCardLayer(id) {
      const el = document.getElementById("card-" + id);
      if (el) el.remove();
    }

    function updateCardLayer(id, event) {
      console.log("[Dashboarder] updateCardLayer called with id:", id, "event:", JSON.stringify(event));
      const el = document.getElementById("card-" + id);
      if (!el) {
        console.error("[Dashboarder] ERROR: Card element not found for id:", id);
        return;
      }
      console.log("[Dashboarder] Card element found:", el);
      const ns = window[id];
      if (!ns) {
        console.error("[Dashboarder] ERROR: Namespace window[" + id + "] not found");
        return;
      }
      console.log("[Dashboarder] Namespace found:", ns);
      if (typeof ns.updateContent !== "function") {
        console.error("[Dashboarder] ERROR: window[" + id + "].updateContent is not a function");
        console.log("[Dashboarder] Available properties in ns:", Object.keys(ns));
        return;
      }
      console.log("[Dashboarder] updateContent function found, calling it");
      ns.updateContent(event);
    }

    const socket = new WebSocket("ws://" + location.host + "/ws");
    socket.addEventListener("open", () => {
      console.log("[Dashboarder] WebSocket connected to ws://" + location.host + "/ws");
    });
    socket.addEventListener("error", (e) => {
      console.error("[Dashboarder] WebSocket error:", e);
    });
    socket.addEventListener("message", (e) => {
      try {
        console.log("[Dashboarder] Raw message received:", e.data);
        const d = JSON.parse(e.data);
        console.log("[Dashboarder] Parsed event:", d);
        switch (d.event) {
          case "create_card":
            console.log("[Dashboarder] === CREATE_CARD START ===");
            console.log("[Dashboarder] Creating card:", d.id);
            console.log("[Dashboarder] Card template length:", d.template.length);
            console.log("[Dashboarder] Before createCardLayer");
            createCardLayer(d.id, d.template);
            console.log("[Dashboarder] After createCardLayer, calling updateCardLayer");
            updateCardLayer(d.id, d);
            console.log("[Dashboarder] === CREATE_CARD END ===");
            break;
          case "update_card":
            console.log("[Dashboarder] Updating card:", d.id);
            console.log("[Dashboarder] Update payload:", d.payload);
            updateCardLayer(d.id, d);
            break;
          case "remove_card":
            console.log("[Dashboarder] Removing card:", d.id);
            removeCardLayer(d.id);
            break;
          case "template_updated":
            console.log("[Dashboarder] Template updated, reloading...");
            location.reload();
            break;
          default:
            console.warn("[Dashboarder] Unknown event type:", d.event);
        }
      } catch (err) {
        console.error("[Dashboarder] Failed to parse message:", err);
        console.error("[Dashboarder] Raw message was:", e.data);
      }
    });
  </script>
</body>
</html>
"""

# Example card template. Define one buffer per card type with a <template> and a JS namespace.
_EXAMPLE_CARD_TEMPLATE = """
<template id="test-card">
  <div class="test-card">
    <h3 class="card-title">Test Card</h3>
    <p class="card-value">No data</p>
  </div>
</template>
<script>
  window["test-card"] = {
    updateContent(event) {
      console.log("[test-card] updateContent called with:", JSON.stringify(event, null, 2));
      console.log("[test-card] Event payload:", JSON.stringify(event.payload, null, 2));
      const el = document.querySelector("#card-" + event.id);
      if (!el) {
        console.log("[test-card] ERROR: Card element not found for id:", event.id);
        console.log("[test-card] Available elements:", document.querySelectorAll("li").length, "list items in DOM");
        return;
      }
      console.log("[test-card] Found card element:", el);
      const title = el.querySelector(".card-title");
      const value = el.querySelector(".card-value");
      console.log("[test-card] Title element found:", !!title);
      console.log("[test-card] Value element found:", !!value);
      if (event.payload.title) {
        console.log("[test-card] Setting title to:", event.payload.title);
        title.textContent = event.payload.title;
      }
      if (event.payload.value !== undefined) {
        console.log("[test-card] Setting value to:", event.payload.value);
        value.textContent = event.payload.value;
      }
      if (event.payload.status !== undefined) {
        console.log("[test-card] Setting status to:", event.payload.status);
        if (el.querySelector(".card-status")) {
          el.querySelector(".card-status").textContent = event.payload.status;
        }
      }
      console.log("[test-card] updateContent complete");
    }
  };
</script>
"""

# Pattern for dashboard card option buffers on content buffers: "BUFFER_NAME[dashboard-card]".
_DASHBOARD_CARD_OPTION_BUFFER_PATTERN = r"^.+\[dashboard-card\]$"

# Pattern for dashboard server option buffers on main page buffers: "MAIN_PAGE[dashboard]".
_DASHBOARD_SERVER_OPTION_BUFFER_PATTERN = r"^.+\[dashboard\]$"


# -------------------------------------------------------------------------------------------------------------------- #
# Data Classes                                                                                                        #
# -------------------------------------------------------------------------------------------------------------------- #


@dataclass
class Dashboard:
    """Represents a single running dashboard instance."""
    id: str
    title: str
    config: dict[str, Any]
    task: asyncio.Task | None = None
    clients: set[DashboardClient] = field(default_factory=set)


@dataclass(frozen=True)
class DashboardClient:
    """Represents a single connected client for a dashboard."""
    id: str
    websocket: WebSocket


# -------------------------------------------------------------------------------------------------------------------- #
# Utils                                                                                                              #
# -------------------------------------------------------------------------------------------------------------------- #


async def _start_server(
    dashboarder: Dashboarder,
    config: dict[str, Any],
    dashboard_id: str,
) -> Dashboard | str:
    """Open the HTTP server for a dashboard and register the GET and WS handlers."""
    # Guard that the dashboard is not already running.
    if dashboard_id in dashboarder._dashboards:
        return f"dashboard '{dashboard_id}' is already running"

    # Validate required config keys.
    port = config.get("port")
    host = config.get("host")
    title = config.get("title")
    if not isinstance(port, int) or not (1 <= port <= 65535):
        return f"port must be an integer between 1 and 65535"
    if not isinstance(host, str) or not host:
        return "host must be a non-empty string"
    if not isinstance(title, str):
        title = ""

    # Create a per-dashboard FastAPI app and register the GET and WS handlers.
    app = FastAPI()

    # Serve the dashboard HTML page at the root path.
    @app.get("/")
    async def get_page() -> PlainTextResponse:
        return PlainTextResponse(
            await _http_get_handler(dashboarder, dashboard),
            media_type="text/html",
        )

    # Accept incoming WebSocket connections and relay events to and from the browser.
    @app.websocket("/ws")
    async def ws_route(websocket: WebSocket) -> None:
        # Accept the WebSocket connection and establish the session.
        await websocket.accept()

        # Register the new client connection on the dashboard.
        client = DashboardClient(id=str(uuid.uuid4()), websocket=websocket)
        dashboarder._dashboards[dashboard_id].clients.add(client)

        # Replay all existing dashboard cards to the newly connected client.
        dashboard = dashboarder._dashboards[dashboard_id]
        await _replay_cards_to_client(dashboarder, dashboard, client)

        # Pump the WebSocket for incoming JSON events until the connection drops.
        while True:
            try:
                # Receive a JSON event from the browser; exit on any error.
                data = await websocket.receive_json()
            except Exception as exc:
                # Log the error and exit the loop.
                _logger.warning(f"[Dashboarder] WS[{dashboard_id}] connection error: {exc}")
                break

            # Log the received event and forward it to the dashboarder for processing.
            _logger.debug(f"[Dashboarder] WS[{dashboard_id}] received: {data}")
            await _ws_handler(dashboarder, dashboard, data)

    # Configure and start the uvicorn server as a background task in the dashboarder event loop.
    async def _run_server() -> None:
        # Build the uvicorn config with host, port, and silent logging.
        config_uvicorn = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
            lifespan="off",
        )

        # Wrap in a Server and register the dashboard in the live servers dict.
        server = uvicorn.Server(config_uvicorn)

        # Run the server; clean up the registry when the task completes.
        try:
            _logger.info(f"[Dashboarder] server '{dashboard_id}' starting on {host}:{port}")
            await server.serve()
        finally:
            # Log the server shutdown and evict it from the registry.
            _logger.info(f"[Dashboarder] server '{dashboard_id}' stopped")
            dashboarder._dashboards.pop(dashboard_id, None)

    # Create a background task for the server and store its reference.
    task = asyncio.create_task(_run_server())

    # Create the Dashboard instance and return it immediately.
    dashboard = Dashboard(id=dashboard_id, title=title, config=config, task=task)

    # Store the dashboard in the registry so _stop_server can find it by ID.
    dashboarder._dashboards[dashboard_id] = dashboard

    # Return the dashboard to the caller.
    return dashboard


async def _stop_server(
    dashboarder: Dashboarder,
    dashboard_id: str,
) -> str | bool:
    """Close the HTTP server for a dashboard and disconnect all its WS clients."""
    # Guard that the dashboard is currently running.
    if dashboard_id not in dashboarder._dashboards:
        return f"dashboard '{dashboard_id}' is not running"

    # Cancel the background task and remove the dashboard from the registry.
    dashboard = dashboarder._dashboards[dashboard_id]
    if dashboard.task is not None:
        dashboard.task.cancel()

    # Remove the dashboard from the registry.
    del dashboarder._dashboards[dashboard_id]

    # Log the successful shutdown and return True.
    _logger.info(f"[Dashboarder] stopped dashboard '{dashboard_id}'")
    return True


async def _broadcast_to_dashboard(
    dashboarder: Dashboarder,
    event: dict[str, Any],
    dashboard_id: str,
) -> str | None:
    """Send a WS event dict to all connected clients of a specific dashboard."""
    # Guard that the dashboard is currently running.
    if dashboard_id not in dashboarder._dashboards:
        return f"dashboard '{dashboard_id}' is not running"

    # Deliver the event to every connected WebSocket client of this dashboard.
    dashboard = dashboarder._dashboards[dashboard_id]
    for client in dashboard.clients:
        # Deliver the event to each client individually; log and skip a failing client.
        try:
            await client.websocket.send_json(event)
        except Exception as exc:
            _logger.error(f"[Dashboarder] failed to deliver event to client: {exc}")

    return None


async def _replay_cards_to_client(
    dashboarder: Dashboarder,
    dashboard: Dashboard,
    client: DashboardClient,
) -> None:
    """Replay all existing dashboard cards to a newly connected client.

    Iterate all card option buffers whose servers list includes this dashboard.
    For each card, send a create_card event to the single new client so the
    browser can render the full current state of the dashboard on connect.
    """
    # Derive the target dashboard ID to filter cards by subscription.
    dashboard_id = dashboard.id

    # Scan all buffers to find card option buffers that target this dashboard.
    for buf_name in dashboarder.list_buffers():
        # Skip buffers that are not dashboard card option buffers.
        if not re.match(_DASHBOARD_CARD_OPTION_BUFFER_PATTERN, buf_name):
            continue

        # Resolve the option buffer for this card.
        option_buffer = dashboarder.get_buffer(buf_name)
        if option_buffer is None:
            continue

        # Read and parse the card config from the option buffer text.
        text = option_buffer.get_text()
        if not text:
            continue

        # Parse the config JSON and guard that it is a dict.
        try:
            config: dict[str, Any] = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(config, dict):
            continue

        # Extract the card ID, template name, and servers list from the config.
        card_id = config.get("id")
        template_name = config.get("template")
        servers: list[str] = config.get("servers", [])
        if not card_id or not template_name:
            continue

        # Skip cards not served by this dashboard.
        if dashboard_id not in servers:
            continue

        # Resolve the template buffer to get the card HTML.
        template_buffer = dashboarder.get_buffer(template_name)
        if template_buffer is None:
            continue

        # Read the template text for the card.
        template_content = template_buffer.get_text()

        # Resolve the origin content buffer to get the card payload.
        origin_name = _origin_name_from_option_buffer_name(buf_name)
        if origin_name is None:
            continue

        # Resolve the content buffer from the origin name.
        content_buffer = dashboarder.get_buffer(origin_name)
        if content_buffer is None:
            continue

        # Read the payload from the content buffer.
        payload = content_buffer.get_text()

        # Build the create_card event payload.
        event = {
            "event": "create_card",
            "id": card_id,
            "template": template_content,
            "payload": payload,
        }

        # Send the create_card event to the single new client.
        try:
            await client.websocket.send_json(event)
        except Exception as exc:
            _logger.error(f"[Dashboarder] failed to replay card '{card_id}' to new client: {exc}")


# -------------------------------------------------------------------------------------------------------------------- #
# Web Handlers                                                                                                        #
# -------------------------------------------------------------------------------------------------------------------- #
# HTTP GET and WebSocket handlers for the dashboard server.
# These are registered on the HTTP server and called on each incoming request.


async def _http_get_handler(
    dashboarder: Dashboarder,
    dashboard: Dashboard,
) -> str:
    """Handle an incoming HTTP GET request for the dashboard.

    Return the HTML page to be served to the browser. The page sets up
    the WebSocket connection and renders the initial set of active cards.
    """
    # Resolve the template buffer name from the dashboard config.
    template_name = dashboard.config.get("template")
    if not template_name:
        return ""

    # Resolve the template buffer and return its content.
    template_buffer = dashboarder.get_buffer(template_name)
    if template_buffer is None:
        return ""

    return template_buffer.get_text()


async def _ws_handler(
    dashboarder: Dashboarder,
    dashboard: Dashboard,
    data: dict[str, Any] | None = None,
) -> None:
    """Handle the WebSocket stream for the dashboard frontend.

    Receives and dispatches create_card, update_card, and remove_card events
    to and from the browser. Maintains the set of connected clients.
    """
    pass


# -------------------------------------------------------------------------------------------------------------------- #
# Hooks                                                                                                              #
# -------------------------------------------------------------------------------------------------------------------- #
# Dashboard card option buffer lives on the content buffer: CONTENT[dashboard-card].
# It holds the card config: {id: "<card-id>", template: "<template-buffer-name>"}.
#
# Dashboard server option buffer lives on the main page buffer: MAIN[dashboard].
# It holds the server config: {port: int, host: str, title: str}.
#
# Template buffers are regular buffers named by the template field in the card config.
# The dashboarder does not pattern-match templates -- they are discovered by buffer name.


# -------------------------------------------------------------------------------------------------------------------- #
# Template-Side Hooks                                                                                                  #
# -------------------------------------------------------------------------------------------------------------------- #
# Fired when the template buffer itself is created, updated, or dropped.
# These hooks are registered directly on the template buffer instance at card
# creation time (not via a pattern).


async def _after_update_template(
    dashboarder: Dashboarder,
    old_template_buffer: Buffer,
    new_template_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after the template buffer is updated.

    Re-parse the new template buffer content so that subsequent card
    renders pick up the refreshed HTML and update function.
    """
    # Skip when there are no running dashboards.
    if not dashboarder._dashboards:
        return

    # Fan out the refresh event to every dashboard that uses this template.
    template_name = new_template_buffer.name
    for dashboard in dashboarder._dashboards.values():
        # Skip dashboards that are not using this template.
        if dashboard.config.get("template") != template_name:
            continue

        # Broadcast the template_updated event -- this will cause a refresh in the browser.
        event = {"event": "template_updated", "template": template_name}
        await _broadcast_to_dashboard(dashboarder, event, dashboard.id)


def _before_drop_template(
    dashboarder: Dashboarder,
    template_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Called before the template buffer is dropped.

    Reject the drop if any dashboard is still using this template as its
    source. Return an error message to block the drop.
    """
    return "Dropping the template buffer is not allowed as long as it is the template for a dashboard"


# -------------------------------------------------------------------------------------------------------------------- #
# Dashboard Card Option Buffer Hooks                                                          #
# -------------------------------------------------------------------------------------------------------------------- #
# Fired when a dashboard card option buffer is created on or dropped from a content buffer.
# The dashboard card option buffer lives on its content buffer: CONTENT[dashboard-card].
# It holds {id: "<card-id>", template: "<template-buffer-name>"}.


def _before_drop_content_buffer(
    dashboarder: Dashboarder,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Called before the content buffer is dropped."""
    return "content buffer is in use by a dashboard card"


async def _after_update_content_buffer(
    dashboarder: Dashboarder,
    old_buffer: Buffer,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after the content buffer is updated.

    Push the new content to all connected clients of dashboards that
    subscribe to this buffer so the card re-renders without a full
    re-create.
    """
    # Derive the dashboard card option buffer name from the content buffer name.
    origin_name = new_buffer.name
    option_name = f"{origin_name}_dashboard_card"

    # Resolve the option buffer and guard that it exists.
    option_buffer = dashboarder.get_buffer(option_name)
    if option_buffer is None:
        return

    # Parse the config JSON and guard that it has a servers list.
    text = option_buffer.get_text()
    if not text:
        return

    # Parse the config JSON and guard that it is a dict.
    try:
        config: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError:
        return
    if not isinstance(config, dict):
        return

    # Read the servers list from the config.
    servers: list[str] = config.get("servers", [])
    if not servers:
        return

    # Read the new content to broadcast as the card payload.
    payload = new_buffer.get_text()

    # Build the update_card event payload.
    event = {
        "event": "update_card",
        "id": config.get("id"),
        "payload": payload,
    }

    # Broadcast the update_card event to each target dashboard server.
    for server_id in servers:
        # Skip servers that are not running.
        if server_id not in dashboarder._dashboards:
            continue

        # Broadcast the update_card event -- this will cause a refresh in the browser.
        await _broadcast_to_dashboard(dashboarder, event, server_id)


def _before_create_dashboard_card(
    dashboarder: Dashboarder,
    option_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Called before the dashboard card option buffer is created on a content buffer.

    Validate the option buffer JSON: it must be a dict with an 'id' key
    (string) and a 'template' key (string naming a valid template buffer).
    Reject the create if validation fails, which also blocks the
    AFTER_CREATE hook that would register the card on the frontend.
    """
    # Read and parse the option buffer as JSON.
    text = option_buffer.get_text()
    if not text:
        return f"option buffer '{option_buffer.name}' is empty -- 'id' and 'template' are required"

    # Parse the config JSON and guard that it is a dict.
    try:
        config: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError:
        return f"option buffer '{option_buffer.name}' is not valid JSON"

    # Guard that the parsed config is a JSON object.
    if not isinstance(config, dict):
        return f"option buffer '{option_buffer.name}' must be a JSON object"

    # Guard that 'id' is present and is a string.
    if "id" not in config:
        return f"'id' is required in '{option_buffer.name}'"
    if not isinstance(config["id"], str) or not config["id"]:
        return f"'id' must be a non-empty string in '{option_buffer.name}'"

    # Guard that 'template' is present and is a string.
    if "template" not in config:
        return f"'template' is required in '{option_buffer.name}'"
    if not isinstance(config["template"], str) or not config["template"]:
        return f"'template' must be a non-empty string in '{option_buffer.name}'"

    # Guard 'servers' only when it is present -- it is optional.
    if "servers" in config:
        if not isinstance(config["servers"], list):
            return f"'servers' must be a list in '{option_buffer.name}'"

        # Validate each server ID and reject unknown servers.
        for server_id in config["servers"]:
            if not isinstance(server_id, str):
                return f"each 'servers' entry must be a string in '{option_buffer.name}'"
            if server_id not in dashboarder._dashboards:
                return f"dashboard server with id '{server_id}' does not exist"

    return None


async def _after_create_dashboard_card(
    dashboarder: Dashboarder,
    option_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after the dashboard card option buffer is created on a content buffer.

    Extract the card 'id' and 'template' from the option buffer. Look up the
    template buffer by name to get the HTML shell and the update function.
    Then:
      - register the card with the frontend over the WebSocket (send "create_card")
      - register _after_update_content_card on the content buffer to push live updates
    """
    # Read and parse the option buffer as JSON.
    text = option_buffer.get_text()

    # Guard that the option buffer has content.
    if not text:
        return

    # Parse the config JSON and guard that it is a dict.
    try:
        config: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError:
        return
    if not isinstance(config, dict):
        return

    # Extract card metadata and servers list from the config.
    card_id = config.get("id")
    template_name = config.get("template")
    servers: list[str] = config.get("servers", [])

    # Guard that the required card ID and template name are present.
    if not card_id or not template_name:
        return

    # Resolve the template buffer to get the card HTML and update function.
    template_buffer = dashboarder.get_buffer(template_name)
    if template_buffer is None:
        return

    # Read the raw template text for the card.
    template_content = template_buffer.get_text()

    # Guard that the origin name was derived successfully.
    origin_name = _origin_name_from_option_buffer_name(option_buffer.name)
    if origin_name is None:
        return

    # Resolve the content buffer from the origin name to get the card payload.
    content_buffer = dashboarder.get_buffer(origin_name)
    if content_buffer is None:
        return

    # Register the content buffer hooks for live updates and drop guard.
    content_buffer.register_hook(
        HookPoint.AFTER_UPDATE_BUFFER,
        "_dashboard_content_refresh",
        _after_update_content_buffer,
    )
    content_buffer.register_hook(
        HookPoint.BEFORE_DROP_BUFFER,
        "_dashboard_content_guard",
        _before_drop_content_buffer,
    )

    # Read the payload from the content buffer.
    payload = content_buffer.get_text()

    # Build the create_card event payload.
    event = {
        "event": "create_card",
        "id": card_id,
        "template": template_content,
        "payload": payload,
    }

    # Broadcast the create_card event to each target dashboard server.
    for server_id in servers:
        # Skip servers that are not running.
        if server_id not in dashboarder._dashboards:
            continue

        # Broadcast the create_card event to the target dashboard.
        await _broadcast_to_dashboard(dashboarder, event, server_id)


async def _after_drop_dashboard_card(
    dashboarder: Dashboarder,
    option_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after the dashboard card option buffer is dropped from a content buffer.

    Extract the card 'id' from the option buffer before it is gone. Then:
      - send "remove_card" over the WebSocket to tear the card down on the frontend
      - unregister _after_update_content_card from the content buffer to stop updates
    """
    # Derive the content buffer name from the option buffer name.
    origin_name = _origin_name_from_option_buffer_name(option_buffer.name)
    if origin_name is None:
        return

    # Resolve the content buffer to unregister the hooks.
    content_buffer = dashboarder.get_buffer(origin_name)
    if content_buffer is None:
        _logger.error(f"[Dashboarder] content buffer '{origin_name}' not found")
        return

    # Unregister the content buffer hooks to stop updates and drop guard.
    content_buffer.unregister_hook(
        HookPoint.AFTER_UPDATE_BUFFER,
        "_dashboard_content_refresh",
    )
    content_buffer.unregister_hook(
        HookPoint.BEFORE_DROP_BUFFER,
        "_dashboard_content_guard",
    )

    # Read and parse the option buffer as JSON.
    text = option_buffer.get_text()
    if not text:
        return

    # Parse the config JSON and guard that it is a dict.
    try:
        config: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError:
        return
    if not isinstance(config, dict):
        return

    # Extract the card ID and servers list from the config.
    card_id = config.get("id")
    servers: list[str] = config.get("servers", [])
    if not servers:
        return

    # Build the remove_card event payload.
    event = {"event": "remove_card", "id": card_id}

    # Broadcast the remove_card event to each target dashboard server.
    for server_id in servers:
        if server_id not in dashboarder._dashboards:
            continue

        # Broadcast the remove_card event to the target dashboard.
        await _broadcast_to_dashboard(dashboarder, event, server_id)


async def _after_update_content_card(
    dashboarder: Dashboarder,
    old_buffer: Buffer,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after the content buffer is updated.

    This hook is registered on the content buffer when its dashboard card
    option buffer is created. It pushes a "update_card" event to the frontend
    so the card re-renders or refreshes its data without a full re-create.
    """
    # Derive the content buffer name from the option buffer name.
    origin_name = _origin_name_from_option_buffer_name(new_buffer.name)
    if origin_name is None:
        return

    # Resolve the content buffer to unregister the hooks.
    content_buffer = dashboarder.get_buffer(origin_name)
    if content_buffer is None:
        _logger.error(f"[Dashboarder] content buffer '{origin_name}' not found")
        return

    # Read and parse the option buffer as JSON.
    text = new_buffer.get_text()
    if not text:
        return

    # Parse the config JSON and guard that it is a dict.
    try:
        config: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError:
        return
    if not isinstance(config, dict):
        return

    # Extract the card ID and servers list from the config.
    card_id = config.get("id")
    servers: list[str] = config.get("servers", [])
    if not servers:
        return

    # Read the payload from the content buffer.
    payload = content_buffer.get_text()

    # Build the remove_card event payload.
    event = {
        "event": "update_card",
        "id": card_id,
        "payload": payload,
    }

    # Broadcast the remove_card event to each target dashboard server.
    for server_id in servers:
        if server_id not in dashboarder._dashboards:
            continue

        # Broadcast the remove_card event to the target dashboard.
        await _broadcast_to_dashboard(dashboarder, event, server_id)


# -------------------------------------------------------------------------------------------------------------------- #
# Dashboard Server Option Buffer Hooks                                                        #
# -------------------------------------------------------------------------------------------------------------------- #
# Fired when a dashboard server option buffer is created on or dropped from a main page buffer.
# The dashboard server option buffer lives on its main page buffer: MAIN[dashboard].
# It holds {port: int, host: str, title: str}.


def _before_create_dashboard_server(
    dashboarder: Dashboarder,
    option_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Called before the dashboard server option buffer is created on a main page buffer.

    Validate the option buffer JSON: it must be a dict with 'port' (int), 'host' (str),
    and 'title' (str). Reject the create if validation fails, which also blocks the
    AFTER_CREATE hook that would open the HTTP server.
    """
    # Read the option buffer text and parse it as JSON.
    text = option_buffer.get_text()
    if not text:
        return f"option buffer '{option_buffer.name}' is empty -- port, host, and title are required"

    # Parse the config JSON and guard that it is a dict.
    try:
        config: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError:
        return f"option buffer '{option_buffer.name}' is not valid JSON"

    # Guard that the parsed config is a JSON object.
    if not isinstance(config, dict):
        return f"option buffer '{option_buffer.name}' must be a JSON object"

    # Guard that port is present and valid.
    port = config.get("port")
    if not isinstance(port, int) or not (1 <= port <= 65535):
        return f"port must be an integer between 1 and 65535"

    # Default host to all interfaces when not provided.
    host = config.get("host", "0.0.0.0")

    # Guard that title is present.
    if "title" not in config:
        return f"'title' is required in '{option_buffer.name}'"

    # Generate a new ID only when absent from the config; otherwise guard for duplicates.
    server_id = config.get("id")
    if server_id is None:
        server_id = str(uuid.uuid4())
        config["id"] = server_id
    elif server_id in dashboarder._dashboards:
        return f"dashboard with id '{server_id}' is already running"
    config["host"] = host

    # Persist the enriched config (with id and resolved host) back to the option buffer.
    option_buffer.set_text(json.dumps(config, indent=2))

    # We ensure that the template is filled. To that, we first derive the origin buffer name from the option buffer name.
    origin_name = _origin_name_from_option_buffer_name(option_buffer.name)
    if origin_name is None:
        _logger.error(f"[Dashboarder] cannot derive origin from '{option_buffer.name}'")
        return

    # Resolve the origin buffer to read its content.
    origin_buffer = dashboarder.get_buffer(origin_name)
    if origin_buffer is None:
        _logger.error(f"[Dashboarder] origin buffer '{origin_name}' not found")
        return

    # Read the origin buffer content for the dashboard page.
    dashboard_html = origin_buffer.get_text()

    # Fall back to the default dashboard HTML when the origin buffer is empty.
    if not dashboard_html:
        origin_buffer.set_text(text=_DEFAULT_DASHBOARD_HTML)

    return None


async def _after_create_dashboard_server(
    dashboarder: Dashboarder,
    option_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after the dashboard server option buffer is created on a main page buffer.

    Extract the 'port', 'host', and 'title' from the option buffer. Open the HTTP
    server and WebSocket server on the configured port and host. Register the
    main dashboard page from the main page buffer so it is served to the frontend.
    """
    # Derive the origin buffer name from the option buffer name.
    origin_name = _origin_name_from_option_buffer_name(option_buffer.name)
    if origin_name is None:
        _logger.error(f"[Dashboarder] cannot derive origin from '{option_buffer.name}'")
        return

    # Resolve the origin buffer to read its content.
    origin_buffer = dashboarder.get_buffer(origin_name)
    if origin_buffer is None:
        _logger.error(f"[Dashboarder] origin buffer '{origin_name}' not found")
        return

    # Read the option buffer JSON for the server config and the server ID.
    try:
        server_config: dict[str, Any] = json.loads(option_buffer.get_text())
    except json.JSONDecodeError:
        _logger.error(f"[Dashboarder] option buffer '{option_buffer.name}' is not valid JSON")
        return

    # Get the server ID from the config. When omitted, the before_create hook created that ID.
    server_id = server_config.get("id")
    if server_id is None:
        _logger.error(f"[Dashboarder] 'id' not found in '{option_buffer.name}'")
        return

    # Open the HTTP server for this dashboard before registering the hooks.
    server_config["template"] = origin_name
    start_result = await _start_server(dashboarder, server_config, server_id)
    if not isinstance(start_result, Dashboard):
        _logger.error(f"[Dashboarder] failed to start server: {start_result}")
        return

    # Register the template refresh hook on the origin buffer to re-cache on update.
    origin_buffer.register_hook(
        HookPoint.AFTER_UPDATE_BUFFER,
        f"_dashboard_template_refresh_{server_id}",
        _after_update_template,
    )

    # Register the drop guard on the origin buffer to prevent removal while in use.
    origin_buffer.register_hook(
        HookPoint.BEFORE_DROP_BUFFER,
        f"_dashboard_template_guard_{server_id}",
        _before_drop_template,
    )


async def _after_drop_dashboard_server(
    dashboarder: Dashboarder,
    option_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after the dashboard server option buffer is dropped from a main page buffer.

    Close the HTTP server and WebSocket server that were opened for this dashboard.
    Stop serving the dashboard page and all active cards.
    """
    # Derive the server ID from the option buffer config.
    try:
        server_config: dict[str, Any] = json.loads(option_buffer.get_text())
    except json.JSONDecodeError:
        _logger.error(f"[Dashboarder] option buffer '{option_buffer.name}' is not valid JSON")
        return

    # Guard that the server ID is present in the config.
    server_id = server_config.get("id")
    if server_id is None:
        _logger.error(f"[Dashboarder] 'id' not found in '{option_buffer.name}'")
        return

    # Derive the origin buffer name from the option buffer name.
    origin_name = _origin_name_from_option_buffer_name(option_buffer.name)
    if origin_name is None:
        _logger.error(f"[Dashboarder] cannot derive origin from '{option_buffer.name}'")
        return

    # Stop the HTTP server for this dashboard.
    try:
        stop_result = await _stop_server(dashboarder, server_id)
        if stop_result is not True:
            _logger.warning(f"[Dashboarder] failed to stop server: {stop_result}")
    except Exception as exc:
        _logger.error(f"[Dashboarder] exception while stopping server: {exc}")

    # Resolve the origin buffer to unregister the hooks.
    origin_buffer = dashboarder.get_buffer(origin_name)
    if origin_buffer is None:
        _logger.error(f"[Dashboarder] origin buffer '{origin_name}' not found")
        return

    # Unregister the template hooks that were registered for this specific server.
    origin_buffer.unregister_hook(
        HookPoint.AFTER_UPDATE_BUFFER,
        f"_dashboard_template_refresh_{server_id}",
    )
    origin_buffer.unregister_hook(
        HookPoint.BEFORE_DROP_BUFFER,
        f"_dashboard_template_guard_{server_id}",
    )


# -------------------------------------------------------------------------------------------------------------------- #
# Dashboarder                                                                                                       #
# -------------------------------------------------------------------------------------------------------------------- #


class Dashboarder(BufferManager, AgenticObject):
    """I aggregate and surface runtime metrics and system health for AIOS.

    `doc:dashboarder` holds details.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Seed the doc buffer with the module documentation.
        self._create_buffer(_DASHBOARDER_DOC_BUFFER, text=_DASHBOARDER_DOC)
        self._create_buffer(f"{_DASHBOARDER_DOC_BUFFER}:main.html", text=_DEFAULT_DASHBOARD_HTML)
        self._create_buffer(f"{_DASHBOARDER_DOC_BUFFER}:card.html", text=_EXAMPLE_CARD_TEMPLATE)

        # In-memory registry of active dashboards, keyed by dashboard ID.
        self._dashboards: dict[str, Dashboard] = {}

        # Register pattern hooks on dashboard card option buffers to manage card lifecycle.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "dashboarder_validate_card",
            _DASHBOARD_CARD_OPTION_BUFFER_PATTERN,
            _before_create_dashboard_card,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "dashboarder_register_card",
            _DASHBOARD_CARD_OPTION_BUFFER_PATTERN,
            _after_create_dashboard_card,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_UPDATE_BUFFER,
            "dashboarder_validate_card_update",
            _DASHBOARD_CARD_OPTION_BUFFER_PATTERN,
            lambda db, _, buf, args: _before_create_dashboard_card(db, buf, args),
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            "dashboarder_unregister_card",
            _DASHBOARD_CARD_OPTION_BUFFER_PATTERN,
            _after_drop_dashboard_card,
            apply_to_existing=True,
        )

        # Register pattern hooks on dashboard server option buffers to manage server lifecycle.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "dashboarder_validate_server",
            _DASHBOARD_SERVER_OPTION_BUFFER_PATTERN,
            _before_create_dashboard_server,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "dashboarder_open_server",
            _DASHBOARD_SERVER_OPTION_BUFFER_PATTERN,
            _after_create_dashboard_server,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            "dashboarder_close_server",
            _DASHBOARD_SERVER_OPTION_BUFFER_PATTERN,
            _after_drop_dashboard_server,
            apply_to_existing=True,
        )