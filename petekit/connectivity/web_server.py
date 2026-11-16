# Web Server Module

from __future__ import annotations

from dataclasses import dataclass

import asyncio
import fastapi
import uvicorn
from fastapi import FastAPI
from starlette.routing import Route


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
        route = self.app.add_api_route(path, handler, methods=["GET"])

        # Store the endpoint handler in the get handlers dictionary.
        self._get_handlers[path] = Endpoint(route, handler)

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

        # Remove the handler from the FastAPI app using the stored route.
        self.app.routes.remove(endpoint.route)

        # Remove the endpoint handler from the get handlers dictionary.
        del self._get_handlers[path]

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
        route = self.app.add_websocket_route(path, handler)

        # Store the endpoint handler in the web socket handlers dictionary.
        self._ws_handlers[path] = Endpoint(route, handler)

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

        # Remove the handler from the FastAPI app using the stored route.
        self.app.routes.remove(endpoint.route)

        # Remove the endpoint handler from the web socket handlers dictionary.
        del self._ws_handlers[path]

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
