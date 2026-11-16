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
from petekit.connectivity.web_server import get_server


_logger = logging.getLogger("dashboarder")


# -------------------------------------------------------------------------------------------------------------------- #
# Constants                                                                                                            #
# -------------------------------------------------------------------------------------------------------------------- #


# Pattern for dashboard card option buffers on content buffers: "BUFFER_NAME[dashboard-card]".
_DASHBOARD_CARD_OPTION_BUFFER_PATTERN = r"^.+\[dashboard-card\]$"

# Pattern for dashboard server option buffers on main page buffers: "MAIN_PAGE[dashboard]".
_DASHBOARD_SERVER_OPTION_BUFFER_PATTERN = r"^.+\[dashboard\]$"


_DASHBOARDER_DOC_BUFFER = "doc:dashboarder"

_DASHBOARDER_DOC = """
## Dashboarder

You mount GET and WebSocket endpoints on a pre-configured web server
and fan out create/update/remove events to all connected browsers.
Cards are placed on content buffers and subscribed to one or more dashboard servers.

### Workflow

1. Create the main HTML template buffer:
   e.g. `my_directory/my_dashboard.html` — a content buffer with HTML + JS.
   This page opens a WebSocket to the server and handles the four event types.
   -> The Dashboarder mounts the page on a configured web server at the given path.

2. Create the dashboard server option buffer on that HTML template buffer:
   e.g. `my_directory/my_dashboard.html[dashboard]` stores:
   {id: "mysrv", server: "default", path: "/dashboards/my-dashboard", title: "My Dashboard"}
   -> The Dashboarder mounts GET and WS at /dashboards/my-dashboard on the "default" web server.

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

9. When the dashboard server option buffer drops, the endpoints are unmounted.

### Buffer Formats

<main-template-buffer>             Main page template buffer (HTML + JS)
  Full HTML page served to the browser.
  Must include JS that opens a WebSocket to the server.
  WebSocketListener parses JSON events: create_card, update_card, remove_card, template_updated.
  The main page adds/removes/updates card layers via the DOM.

<main-template-buffer>[dashboard]  Dashboard server option buffer (lives on the main template buffer)
  id: str                          Unique dashboard identifier
  server: str                      Name of the web server to mount on
  path: str                        URL path where the dashboard is served (e.g. "/dashboards/my-dashboard")
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
WebSocket by converting the page URL from HTTP to WebSocket using
`location.href.replace('http', 'ws').replace('https', 'wss')`.

See `doc:dashboarder:main.html` for a working example.

#### Card template

Each card type needs one template buffer with a `<template>` element and a
JavaScript namespace that exposes `updateContent(event)`.

See `doc:dashboarder:card.html` for a working example.

#### Minimal workflow

1. Create the main HTML template buffer `my_directory/my_dashboard.html`.

2. Create the dashboard server option buffer on that HTML template buffer:
   `my_directory/my_dashboard.html[dashboard]` stores:
   {id: "mysrv", server: "default", path: "/dashboards/my-dashboard", title: "My Dashboard"}

3. Create the card HTML template buffer `my_directory/my_card.html`.

4. Create the content JSON buffer `any_directory/my_data.json`:
   {"title": "Server A", "value": "All systems nominal", "status": "online"}

5. Create the dashboard card option buffer on that content buffer:
   `any_directory/my_data.json[dashboard-card]` stores:
   {id: "my-card", template: "my_directory/my_card.html", servers: ["mysrv"]}

The Dashboarder mounts the page on the web server, the browser loads the page,
opens the WebSocket, receives the create_card event, renders the card, and
re-renders on every update to my-data.
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

    const socket = new WebSocket(location.href.replace('http', 'ws').replace('https', 'wss'));
    socket.addEventListener("open", () => {
      console.log("[Dashboarder] WebSocket connected to " + socket.url);
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


# -------------------------------------------------------------------------------------------------------------------- #
# Data Classes                                                                                                         #
# -------------------------------------------------------------------------------------------------------------------- #


@dataclass
class Dashboard:
    """Represents a single running dashboard instance."""
    id: str
    path: str
    title: str
    config: dict[str, Any]
    clients: set[DashboardClient] = field(default_factory=set)


@dataclass(frozen=True)
class DashboardClient:
    """Represents a single connected client for a dashboard."""
    id: str
    websocket: WebSocket


# -------------------------------------------------------------------------------------------------------------------- #
# Utils                                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #


async def serve_dashboard(
    dashboarder: Dashboarder,
    config: dict[str, Any],
    dashboard_id: str,
) -> Dashboard | str:
    """Get the web server, mount GET and WS endpoints, and register the dashboard."""
    # Guard that the dashboard is not already running.
    # When already running, return an error.
    if dashboard_id in dashboarder._dashboards:
        return f"dashboard '{dashboard_id}' is already running"

    # Get the web server from the registry using the server name from the config.
    # When the server is not found, return an error.
    server = get_server(config["server"])
    if isinstance(server, str):
        return server

    # Get the path from the config.
    path = config["path"]

    # Create the Dashboard instance first, before the handlers, so the handlers can capture it.
    dashboard = Dashboard(
        id=dashboard_id,
        path=path,
        title=config.get("title", ""),
        config=config,
        clients=set(),
    )

    # Create the GET handler for this dashboard.
    def get_page() -> PlainTextResponse:
        return PlainTextResponse(_http_get_handler(dashboarder, dashboard), media_type="text/html")

    # Create the WebSocket handler for this dashboard.
    async def ws_handler(websocket: WebSocket) -> None:
        await _ws_handler(dashboarder, dashboard, websocket)

    # Mount the GET handler on the server at the path from the config.
    # When mounting fails, return the error.
    error = server.mount_get(path, get_page)
    if error is not None:
        return error

    # Mount the WebSocket handler on the server at the path from the config.
    # When mounting fails, unmount the GET handler and return the error.
    error = server.mount_ws(path, ws_handler)
    if error is not None:
        server.unmount_get(path)
        return error

    # Register the dashboard in the dashboarder's registry.
    dashboarder._dashboards[dashboard_id] = dashboard

    # Return the Dashboard.
    return dashboard


async def unserve_dashboard(
    dashboarder: Dashboarder,
    dashboard_id: str,
) -> str | None:
    """Unmount GET and WS endpoints and remove the dashboard from the registry."""
    # Get the dashboard from the registry.
    # When not found, return an error.
    if dashboard_id not in dashboarder._dashboards:
        return f"dashboard '{dashboard_id}' is not running"

    # Unmount the GET handler from the server using the path from the dashboard.
    # Unmount the WebSocket handler from the server using the path from the dashboard.
    dashboard = dashboarder._dashboards[dashboard_id]
    server = get_server(dashboard.config["server"])
    if isinstance(server, str):
        return server
    server.unmount_get(dashboard.path)
    server.unmount_ws(dashboard.path)

    # Remove the dashboard from the registry.
    del dashboarder._dashboards[dashboard_id]

    # Return None.
    return None


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
# Web Handlers                                                                                                         #
# -------------------------------------------------------------------------------------------------------------------- #
# HTTP GET and WebSocket handlers for the dashboard server.
# These are registered on the HTTP server and called on each incoming request.


def _http_get_handler(
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
    websocket: WebSocket,
) -> None:
    """Handle the WebSocket stream for the dashboard frontend.

    Accepts the WebSocket, registers the client, replays existing cards,
    and pumps incoming events until the connection drops.
    """
    # Accept the WebSocket connection and establish the session.
    await websocket.accept()

    # Register the new client connection on the dashboard.
    client = DashboardClient(id=str(uuid.uuid4()), websocket=websocket)
    dashboard.clients.add(client)

    # Replay all existing dashboard cards to the newly connected client.
    await _replay_cards_to_client(dashboarder, dashboard, client)

    # Pump the WebSocket for incoming JSON events until the connection drops.
    while True:
        # Receive a JSON event from the browser.
        # Exit on any error.
        try:
            data = await websocket.receive_json()
        except Exception as exc:
            _logger.warning(f"[Dashboarder] WS[{dashboard.id}] connection error: {exc}")
            break

        # Log the received event and forward it to the dashboarder for processing.
        _logger.debug(f"[Dashboarder] WS[{dashboard.id}] received: {data}")
        await _ws_message_handler(dashboarder, dashboard, data)


async def _ws_message_handler(
    dashboarder: Dashboarder,
    dashboard: Dashboard,
    data: dict[str, Any] | None = None,
) -> None:
    """Handle a single WebSocket message from the browser.

    Receives and dispatches create_card, update_card, and remove_card events
    to and from the browser. Maintains the set of connected clients.
    """
    pass


# -------------------------------------------------------------------------------------------------------------------- #
# Hooks                                                                                                                #
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
# Card Template Buffer Hooks                                                                                           #
# -------------------------------------------------------------------------------------------------------------------- #
# Fired when a card template buffer is updated or dropped.
# These hooks are registered on the card template buffer when a dashboard card
# option buffer that references it is created.
# The template name is the buffer name used in the 'template' field of the card config.


def _before_drop_card_template(
    dashboarder: Dashboarder,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Called before the card template buffer is dropped.

    Reject the drop if any dashboard card is still using this template.
    """
    return "cannot drop card template while a dashboard card is using it"


async def _after_update_card_template(
    dashboarder: Dashboarder,
    old_buffer: Buffer,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after the card template buffer is updated.

    Broadcast a template_updated event to all dashboard servers that have
    a card using this template so the browser picks up the refreshed HTML.
    """
    # Guard: nothing to do when no dashboards are running.
    if not dashboarder._dashboards:
        return

    # Derive the template name from the updated buffer.
    template_name = new_buffer.name

    # Fan out to all servers that have a card using this template.
    for opt_buf in dashboarder.list_buffers().values():
        # Skip buffers that are not card option buffers.
        if not re.match(_DASHBOARD_CARD_OPTION_BUFFER_PATTERN, opt_buf.name):
            continue

        # Parse the config and skip if this template is not referenced.
        try:
            config: dict[str, Any] = json.loads(opt_buf.get_text())
        except json.JSONDecodeError:
            continue
        if config.get("template") != template_name:
            continue

        # Fan out the template_updated event to each target server.
        servers: list[str] = config.get("servers", [])
        for server_id in servers:
            if server_id not in dashboarder._dashboards:
                continue
            event = {"event": "template_updated", "template": template_name}
            await _broadcast_to_dashboard(dashboarder, event, server_id)


# -------------------------------------------------------------------------------------------------------------------- #
# Content Buffer Hooks                                                                                                 #
# -------------------------------------------------------------------------------------------------------------------- #
# Fired when the content buffer (the data source for a card) changes or is dropped.
# These hooks are registered on the content buffer when the card option buffer is created.


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
    option_name = f"{origin_name}[dashboard-card]"

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


# -------------------------------------------------------------------------------------------------------------------- #
# Dashboard Card Option Buffer Hooks                                                                                   #
# -------------------------------------------------------------------------------------------------------------------- #
# Fired when the dashboard card option buffer is created on or dropped from a content buffer.
# The dashboard card option buffer lives on its content buffer: CONTENT[dashboard-card].
# It holds {id: "<card-id>", template: "<template-buffer-name>", servers: [...]}.
# The BEFORE hook on the option buffer triggers validation;
# the AFTER hook registers content-buffer hooks and sends create_card to browsers.


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

    # Register the card template buffer hooks for refresh and drop guard.
    template_buffer.register_hook(
        HookPoint.AFTER_UPDATE_BUFFER,
        f"_dashboard_card_template_refresh_{card_id}",
        _after_update_card_template,
    )
    template_buffer.register_hook(
        HookPoint.BEFORE_DROP_BUFFER,
        f"_dashboard_card_template_guard_{card_id}",
        _before_drop_card_template,
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
    # Read and parse the option buffer as JSON to get card config.
    text = option_buffer.get_text()
    if not text:
        return
    try:
        config: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError:
        return
    if not isinstance(config, dict):
        return

    # Extract the card ID, template, and servers list from the config.
    card_id = config.get("id")
    template_name = config.get("template")
    servers: list[str] = config.get("servers", [])

    # Resolve the content buffer to unregister the hooks.
    origin_name = _origin_name_from_option_buffer_name(option_buffer.name)
    if origin_name is None:
        return
    content_buffer = dashboarder.get_buffer(origin_name)
    if content_buffer is not None:
        content_buffer.unregister_hook(
            HookPoint.AFTER_UPDATE_BUFFER,
            "_dashboard_content_refresh",
        )
        content_buffer.unregister_hook(
            HookPoint.BEFORE_DROP_BUFFER,
            "_dashboard_content_guard",
        )

    # Unregister the card template buffer hooks.
    if template_name:
        template_buffer = dashboarder.get_buffer(template_name)
        if template_buffer is not None:
            template_buffer.unregister_hook(
                HookPoint.AFTER_UPDATE_BUFFER,
                f"_dashboard_card_template_refresh_{card_id}",
            )
            template_buffer.unregister_hook(
                HookPoint.BEFORE_DROP_BUFFER,
                f"_dashboard_card_template_guard_{card_id}",
            )
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
# Dashboard Server Option Buffer Hooks                                                                                 #
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

    Validate the option buffer JSON: it must be a dict with 'server' (str), 'path' (str),
    and 'title' (str). Reject the create if validation fails, which also blocks the
    AFTER_CREATE hook that would mount the dashboard on the web server.
    """
    # Read the option buffer text and parse it as JSON.
    text = option_buffer.get_text()
    if not text:
        return f"option buffer '{option_buffer.name}' is empty -- server, path, and title are required"

    # Parse the config JSON and guard that it is a dict.
    try:
        config: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError:
        return f"option buffer '{option_buffer.name}' is not valid JSON"

    # Guard that the parsed config is a JSON object.
    if not isinstance(config, dict):
        return f"option buffer '{option_buffer.name}' must be a JSON object"

    # Guard that server is present and is a string.
    # Return an error when no web server with that name is registered.
    server = config.get("server")
    if not isinstance(server, str) or not server:
        return f"'server' must be a non-empty string in '{option_buffer.name}'"

    # Check that a web server with the given name is registered in the WebRegistry.
    # When not found, return an error.
    result = get_server(server)
    if isinstance(result, str):
        return f"web server '{server}' not found: {result}"

    # Guard that path is present and is a string.
    # Return an error when path is not a non-empty string.
    path = config.get("path")
    if not isinstance(path, str) or not path:
        return f"'path' must be a non-empty string in '{option_buffer.name}'"

    # Guard that title is present.
    if "title" not in config:
        return f"'title' is required in '{option_buffer.name}'"

    # Generate a new ID only when absent from the config; otherwise guard for duplicates.
    dashboard_id = config.get("id")
    if dashboard_id is None:
        dashboard_id = str(uuid.uuid4())
        config["id"] = dashboard_id
    elif dashboard_id in dashboarder._dashboards:
        return f"dashboard with id '{dashboard_id}' is already running"

    # Persist the enriched config back to the option buffer.
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

    Extract the 'server', 'path', and 'title' from the option buffer. Mount the
    GET and WebSocket handlers on the named web server at the given path.
    Register the dashboard so it can receive broadcasts.
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
    start_result = await serve_dashboard(dashboarder, server_config, server_id)
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

    Unmount the GET and WebSocket handlers from the web server at the given path.
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

    # Unmount the GET and WS endpoints for this dashboard.
    try:
        stop_result = await unserve_dashboard(dashboarder, server_id)
        if stop_result is not None:
            _logger.warning(f"[Dashboarder] failed to unserve dashboard: {stop_result}")
    except Exception as exc:
        _logger.error(f"[Dashboarder] exception while unserving dashboard: {exc}")

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
# Dashboarder                                                                                                          #
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