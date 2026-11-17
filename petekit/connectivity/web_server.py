# Web Server Module

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import asyncio
import re
import fastapi
import uvicorn
from fastapi import FastAPI
from starlette.routing import Route


# -------------------------------------------------------------------------------------------------------------------- #
# Hook Infrastructure                                                                                                  #
# -------------------------------------------------------------------------------------------------------------------- #

# MountHook type — called when an endpoint is mounted or unmounted.
# Args: event (str) — "mount" or "unmount"
# Args: method (str) — "GET", "POST", "WebSocket", etc.
# Args: path (str) — the path being mounted or unmounted
MountHook = Callable[[str, str, str], None]


# -------------------------------------------------------------------------------------------------------------------- #
# Public API                                                                                                           #
# -------------------------------------------------------------------------------------------------------------------- #


async def create_server(*, name: str, port: int, host: str = "localhost") -> str | None:
    """Create a FastAPI + uvicorn web server and register it under the given name.

    Returns None on success, or an error string if the port is already in use.
    """
    return await _registry.create_server(name=name, port=port, host=host)


def get_server(name: str) -> WebServer | str:
    """Return the server registered under this name, or an error string if not registered."""
    return _registry.get_server(name)


async def register_server(name: str, server: WebServer) -> str | None:
    """Register a running WebServer under the given name.

    Returns None on success, or an error string if registration fails.
    """
    return await _registry.register_server(name, server)


async def unregister_server(name: str) -> str | None:
    """Remove a stopped server from the registry.

    The server must be stopped before it can be unregistered.
    Returns None on success, or an error string if unregistration fails.
    """
    return await _registry.unregister_server(name)


# -------------------------------------------------------------------------------------------------------------------- #
# Data Classes                                                                                                         #
# -------------------------------------------------------------------------------------------------------------------- #

@dataclass
class Endpoint:
    """Holds the route and handler for a mounted endpoint."""
    route: Route
    handler: callable


@dataclass
class Hook:
    """Holds a pattern and a callable for hook-based routing."""
    pattern: re.Pattern
    handler: MountHook


# -------------------------------------------------------------------------------------------------------------------- #
# WebServer                                                                                                            #
# -------------------------------------------------------------------------------------------------------------------- #

class WebServer:
    """Holds the FastAPI app, the uvicorn config, and the running task for a server."""

    def __init__(self, app: FastAPI, config: uvicorn.Config, timeout: float = 5.0) -> None:
        self.app = app
        self.config = config
        self.task: asyncio.Task[None] | None = None
        self.config.timeout = timeout
        self._get_handlers: dict[str, Endpoint] = {}
        self._ws_handlers: dict[str, Endpoint] = {}
        self._hooks: dict[str, Hook] = {}

    def register_hook(self, name: str, pattern: str, handler: MountHook) -> str | None:
        """Register a hook under the given name with a path-matching pattern.

        Compiles the pattern string to a regex and stores a Hook object
        with the pattern and handler. The hook is called on mount events
        with (event, method, path): event is "mount" or "unmount", method
        is "GET" or "WebSocket", path is the mount path. Returns None on
        success, or an error string if a hook is already registered.
        """
        # Guard that no hook is already registered under this name.
        # When a hook already exists, return an error to prevent overwriting.
        if name in self._hooks:
            return f"hook '{name}' already registered"

        # Compile the pattern string to a regex for path matching.
        # When the pattern is invalid, return an error with the position of the error.
        try:
            compiled_pattern = re.compile(pattern)
        except re.error as exc:
            return f"invalid regex pattern '{pattern}': {exc.msg} (position {exc.pos})"

        # Build the Hook object and store it in the hooks registry.
        self._hooks[name] = Hook(pattern=compiled_pattern, handler=handler)

        return None

    def _call_hooks(self, event: str, method: str, path: str) -> None:
        """Call all registered mount hooks whose pattern matches the path.

        Iterates all registered hooks and calls each hook whose pattern
        matches the given path. Passes event, method, and path as arguments.
        """
        # Iterate all registered hooks and call those whose pattern matches.
        for hook in self._hooks.values():
            # Skip hooks whose pattern does not match the path.
            if not hook.pattern.match(path):
                continue

            # Call the hook with the event, method, and path.
            hook.handler(event, method, path)

    def unregister_hook(self, name: str) -> str | None:
        """Unregister the hook under the given name.

        Returns None on success, or an error string if no hook was
        registered under this name.
        """
        # Guard that a hook is registered under this name.
        # When not found, return an error.
        if name not in self._hooks:
            return f"no hook registered under '{name}'"

        # Remove the hook from the hooks registry.
        del self._hooks[name]

        return None

    async def start(self) -> str | None:
        """Start the server and wait until it is ready to accept connections.

        Returns None when the server started successfully, or an error
        message when a server is already running on this port.
        """
        # Check whether a server task is already running.
        # Return early with an error when the server is already up.
        if self.is_started():
            return f"server already running on {self.config.bind}"

        # Create the error event to receive the failure signal from _run().
        error = asyncio.Event()

        # Create a background task that calls _run() and passes the error event.
        # Store the task reference so is_running returns True.
        self.task = asyncio.create_task(self._run(error))

        # Define the polling function that checks server.started until it is True.
        async def poll_started() -> bool:
            # Poll server.started until it becomes True.
            while not self.config.server.started:
                # Wait 100ms before polling again.
                await asyncio.sleep(0.1)
            return True

        # Create the polling task to detect when the server is ready.
        polling = asyncio.create_task(poll_started())

        # Wait for whichever completes first: the polling or the error event.
        done, pending = await asyncio.wait([
            polling,
            asyncio.create_task(error.wait())
        ], return_when=asyncio.FIRST_COMPLETED)

        # Cancel the pending task to prevent it from leaking.
        for t in pending:
            t.cancel()

        # When the error event is set first, the server failed to start.
        if polling not in done:
            return "server failed to start, call stop() for details"

        # When the startup event is set first, the server is listening.
        return None

    async def _run(self, error: asyncio.Event) -> str | None:
        """Run the uvicorn server and signal readiness to the caller.

        Blocks for the duration of the server. Does not clear the task
        reference -- only stop() does that. Returns None on success, or
        an error string on failure.
        """
        # Call the server serve function to start the server.
        try:
            await self.config.server.serve()
        except OSError as exc:
            # Set the error event and return the error string.
            error.set()
            return str(exc)

        # Return None on success, or an error string on failure.
        return None

    def is_started(self) -> bool:
        """Return True when the server was started, False when it is stopped.

        Returns True even when the server has stopped with an error. Use
        is_started to decide whether to call start() again.
        """
        # Return True when the task is not None, meaning the server was started.
        # Return False when the task is None, meaning the server was never started.
        return self.task is not None

    def is_running(self) -> bool:
        """Return True when the server is actively listening, False otherwise.

        Returns False when the server was never started or has already stopped.
        """
        # Guard that the server was started.
        # Return False when not started.
        if not self.is_started():
            return False

        # Return True when the task is still running.
        # Return False when the task has finished (even with an error).
        return not self.task.done()

    async def stop(self) -> str | None:
        """Stop the server and wait for it to shut down.

        Returns an error string if the server was not running, or the
        error from _run() if the server failed during its run.
        """
        # Guard that the server was started.
        # Return early when the server was never started.
        if not self.is_started():
            return "server was never started"

        # Signal uvicorn to shut down by setting should_exit.
        self.config.server.should_exit = True

        # Await the task so that server.serve() can return and the server stops.
        # When the timeout fires, cancel the task and return an error.
        done, _ = await asyncio.wait([self.task], timeout=self.config.timeout)
        if self.task not in done:
            self.task.cancel()
            return "server failed to stop within timeout"

        # Read the return value from _run() to check for errors.
        # Return the error string if _run() returned one.
        error = self.task.result()
        if error is not None:
            return error

        # Clear the task reference to mark the server as stopped.
        self.task = None

        return None

    def mount_get(self, path: str, handler: callable) -> str | None:
        """Mount an async HTTP GET handler on the given path.

        The handler is immediately callable. Returns None on success,
        or an error string if a handler is already mounted on this path."""
        # Check if a handler is already registered under this path.
        # When already registered, return an error.
        if path in self._get_handlers:
            return f"GET handler already mounted on {path}"

        # Mount the handler on the FastAPI app for the given path.
        self.app.add_api_route(path, handler, methods=["GET"])

        # Capture the newly created route and store it with the handler.
        # Iterate in reverse order — the new route is near the end, found quickly.
        from fastapi.routing import APIRoute
        route = next(
            route
            for route in reversed(self.app.routes)
            if isinstance(route, APIRoute)
            and route.endpoint is handler
            and route.path == path
            and "GET" in route.methods
        )
        self._get_handlers[path] = Endpoint(route, handler)

        # Call all registered mount hooks whose pattern matches the path.
        self._call_hooks("mount", "GET", path)

    def unmount_get(self, path: str) -> str | None:
        """Unmount the HTTP GET handler on the given path.

        After unmounting, requests to this path return 404.
        Returns None on success, or an error string if no handler
        was mounted on this path."""
        # Get the endpoint from the get handlers dictionary.
        # When not found, return an error.
        endpoint = self._get_handlers.get(path)
        if endpoint is None:
            return f"no GET handler mounted on {path}"

        # Remove the route from the FastAPI app.
        self.app.routes.remove(endpoint.route)

        # Remove the endpoint handler from the get handlers dictionary.
        del self._get_handlers[path]

        # Call all registered mount hooks whose pattern matches the path.
        self._call_hooks("unmount", "GET", path)

    def mount_ws(self, path: str, handler: callable) -> str | None:
        """Mount an async WebSocket handler on the given path.

        WebSocket connections on this path are routed to this handler.
        Returns None on success, or an error string if a WebSocket
        handler is already mounted on this path."""
        # Check if a handler is already registered under this path.
        # When already registered, return an error.
        if path in self._ws_handlers:
            return f"WebSocket handler already mounted on {path}"

        # Mount the handler on the FastAPI app for the given path.
        # app.websocket(path) returns a decorator — apply it to the handler.
        self.app.websocket(path)(handler)

        # Capture the newly created route and store it with the handler.
        # Iterate in reverse order — the new route is near the end, found quickly.
        from fastapi.routing import APIWebSocketRoute
        route = next(
            route
            for route in reversed(self.app.routes)
            if isinstance(route, APIWebSocketRoute)
            and route.endpoint is handler
            and route.path == path
        )
        self._ws_handlers[path] = Endpoint(route, handler)

        # Call all registered mount hooks whose pattern matches the path.
        self._call_hooks("mount", "WebSocket", path)

    def unmount_ws(self, path: str) -> str | None:
        """Unmount the WebSocket handler on the given path.

        After unmounting, WebSocket connections on this path are rejected.
        Returns None on success, or an error string if no WebSocket
        handler was mounted on this path."""
        # Get the endpoint from the web socket handlers dictionary.
        # When not found, return an error.
        endpoint = self._ws_handlers.get(path)
        if endpoint is None:
            return f"no WebSocket handler mounted on {path}"

        # Remove the route from the FastAPI app.
        self.app.routes.remove(endpoint.route)

        # Remove the endpoint handler from the web socket handlers dictionary.
        del self._ws_handlers[path]

        # Call all registered mount hooks whose pattern matches the path.
        self._call_hooks("unmount", "WebSocket", path)

    def mounted_endpoints(self) -> dict[str, list[str]]:
        """Return all currently mounted endpoints as a mapping of method to paths.

        The mapping holds one entry per transport method -- "GET" and
        "WebSocket" -- listing the paths mounted for that method. Mount
        hook subscribers use this to replay the current mount state.

        Contracts:
            - The snapshot is taken atomically per method list -- mounts
              happening during the call may or may not be included.
        """
        # Snapshot both handler registries -- one path list per method.
        return {
            "GET": list(self._get_handlers),
            "WebSocket": list(self._ws_handlers),
        }

# -------------------------------------------------------------------------------------------------------------------- #
# Web Registry                                                                                                         #
# -------------------------------------------------------------------------------------------------------------------- #


class WebRegistry:
    def __init__(self) -> None:
        self._servers: dict[str, WebServer] = {}

    async def create_server(self, *, name: str, port: int, host: str = "localhost") -> str | None:
        """Create a FastAPI + uvicorn web server on host:port and register it.

        Creates a FastAPI app, configures uvicorn, and starts the server as a
        background task. The server is registered under the given name for later access.
        Returns None on success, or an error string if the port is already in use.
        """
        # Guard that no server is already registered under this name.
        # Return early when a server already occupies this name.
        if self._servers.get(name) is not None:
            return f"name {name} already registered"

        # Create a FastAPI app instance for this server.
        app = FastAPI()

        # Configure the uvicorn config with the app, host, and port.
        config = uvicorn.Config(app, host=host, port=port)

        # Wrap in a uvicorn Server instance.
        config.server = uvicorn.Server(config)

        # Create a WebServer instance from the app and config.
        web_server = WebServer(app, config)

        # Start the server by calling start.
        # When start fails, return the error without registering.
        error = await web_server.start()
        if error is not None:
            return error

        # Register the new server in the registry by calling register_server.
        # When registration fails, return the error.
        error = await self.register_server(name, web_server)
        if error is not None:
            return error

    def get_server(self, name: str) -> WebServer | str:
        """Return the server registered under this name, or an error string if not registered."""
        # Return the server registered under this name.
        # When no server is registered under this name, return an error string.
        server = self._servers.get(name)
        if server is None:
            return f"no server registered under name {name}"
        return server

    async def register_server(self, name: str, server: WebServer) -> str | None:
        """Register a running WebServer in the port registry.

        Checks that the server is running and that the name is free.
        Returns None on success, or an error string if registration fails.
        """
        # Check if the server is running.
        # When the server is not running, reject the registration.
        running = server.is_running()
        if not running:
            return "server is not running"

        # Get the existing server from the dictionary if any.
        # Check if the name is already registered in the dictionary.
        # When the name is not free, return an error.
        existing = self._servers.get(name)
        if existing is not None:
            return f"name {name} already registered"

        # Register the server by putting it into the dictionary.
        self._servers[name] = server

    async def unregister_server(self, name: str) -> str | None:
        """Remove a stopped server from the port registry.

        The server must be stopped before it can be unregistered.
        Returns None on success, or an error string if unregistration fails.
        """
        # Get the server from the registry.
        # When no server is registered, return an error message.
        server = self._servers.get(name)
        if server is None:
            return f"no server registered under name {name}"

        # Check if the server is stopped.
        # When the server is still running, return an error message.
        if server.is_started():
            return "server is still running"

        # Remove the server from the registry.
        del self._servers[name]


# -------------------------------------------------------------------------------------------------------------------- #
# Singleton                                                                                                            #
# -------------------------------------------------------------------------------------------------------------------- #

_registry = WebRegistry()
