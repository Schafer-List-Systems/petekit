from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any
from typing import Callable

from peteos.oap.agentic_object import AgenticObject


_logger = logging.getLogger("grapher")

from petekit.text.buffer_manager import Buffer
from petekit.text.buffer_manager import BufferManager
from petekit.text.buffer_manager import HookPoint
from petekit.text.option_buffer_manager import OptionBufferManager


# -------------------------------------------------------------------------------------------------------------------- #
# Constants                                                                                                          #
# -------------------------------------------------------------------------------------------------------------------- #


_GRAPHER_NODE_SUFFIX = "[node]"
_GRAPHER_NODE_BUFFER_PATTERN = rf"^.+\[{_GRAPHER_NODE_SUFFIX[1:-1]}\]$"

_GRAPHER_DOC_BUFFER = "doc:graph"

_GRAPHER_DOC = f"""\
## Grapher

The Grapher manages a directed graph of named nodes, stored as option buffers.

### Buffer Naming

- `<buffer-name>{_GRAPHER_NODE_SUFFIX}`  --  a single node in the graph; stores its outgoing edges as JSON

### Node Config

Each node buffer holds a flat JSON object with an optional `successors` dict.
Each dict key is the bare name of a connected node, and each value is the label for that edge.
The key must be the node name only — never the buffer name with suffix.

```json
{{
  "successors": {{
    "node-b": "label for the edge to node-b"
  }}
}}
```

The `successors` dict is optional. Omit it to declare a node with no outgoing edges.

### Key Format

The successor key must always be the bare node name (e.g., `"node-b"`), not the buffer
name with the `{_GRAPHER_NODE_SUFFIX}` suffix (e.g., not `"node-b{_GRAPHER_NODE_SUFFIX}"`).
The Grapher stores the key as-is and uses it for lookups. If you store the full buffer
name, lookups will fail because the graph only knows bare node names.

### Creating a Node

Create the option buffer `<node-name>{_GRAPHER_NODE_SUFFIX}` with the config above.
The node is registered in the in-memory graph on create and update. The node referenced
in `successors` does not need to exist yet — the key is stored as-is and resolved on
lookup. You can create the target node afterward. The graph is stored in memory only
and is not persisted across restarts.

### Updating Edges

Edit the `successors` dict to add or remove edges. The Grapher diffs the old and new
sets on every update and reconciles the in-memory graph accordingly.

### Deleting a Node

Drop the `<node-name>{_GRAPHER_NODE_SUFFIX}` buffer to remove the node and unlink all its edges from
both sides (outgoing to successors and incoming from predecessors).
"""


# -------------------------------------------------------------------------------------------------------------------- #
# Data Classes                                                                                                       #
# -------------------------------------------------------------------------------------------------------------------- #


@dataclass
class _Node:
    """A directed graph node with predecessor and successor sets."""

    name: str
    predecessors: set[str] = field(default_factory=set)
    successors: set[str] = field(default_factory=set)
    pending: bool = False


# -------------------------------------------------------------------------------------------------------------------- #
# Utils                                                                                                             #
# -------------------------------------------------------------------------------------------------------------------- #


def _get_node_config(node_buffer: Buffer) -> dict[str, Any] | str:
    """Read and parse the JSON config from a node buffer.

    Returns the parsed config dict on success, or an error string if the buffer
    is absent, empty, or contains invalid JSON.
    """
    # Extract the text from the node buffer.
    text = node_buffer.get_text()
    if not text or not text.strip():
        return f"[Grapher] node buffer '{node_buffer.name}' is empty"

    # Parse the JSON config from the buffer text.
    try:
        config = json.loads(text)
    except json.JSONDecodeError as e:
        return f"[Grapher] node buffer '{node_buffer.name}' has invalid JSON: {e}"

    return config


def get_node_config(grapher: Grapher, node_name: str) -> dict[str, Any] | str:
    """Read and parse the config from a named node's option buffer.

    Returns the parsed config dict on success, or an error string if the node
    does not exist or its buffer is absent, empty, or contains invalid JSON.
    """
    # Guard: reject if the node is not registered in the graph.
    node = get_node(grapher, node_name)
    if node is None:
        return f"[Grapher] node '{node_name}' does not exist"

    # Look up the node's option buffer.
    node_buffer = grapher.get_buffer(node_name + _GRAPHER_NODE_SUFFIX)
    if node_buffer is None:
        return f"[Grapher] node '{node_name}' has no associated option buffer"

    # Parse and return the config from the buffer text.
    return _get_node_config(node_buffer)


def get_node(grapher: Grapher, node_name: str) -> _Node | None:
    """Look up a node by name in the graph.

    Returns the _Node if it exists, or None if it has not been registered.
    """
    return grapher._grapher_nodes.get(node_name)


def get_predecessors(grapher: Grapher, node_name: str) -> set[str] | None:
    """Return the set of predecessor node names for the given node.

    Returns None if the node does not exist.
    """
    node = get_node(grapher, node_name)
    if node is None:
        return None
    return node.predecessors


def get_successors(grapher: Grapher, node_name: str) -> set[str] | None:
    """Return the set of successor node names for the given node.

    Returns None if the node does not exist.
    """
    node = get_node(grapher, node_name)
    if node is None:
        return None
    return node.successors


def _add_successor(
    grapher: Grapher,
    node: _Node,
    successor_name: str,
) -> str | None:
    """Add a directed edge from node to successor on both sides.

    Adds successor_name to node.successors and adds node.name to
    the successor node's predecessors. Returns an error string if the
    successor node does not exist in the graph, or None on success.
    """
    # Create a placeholder node when the successor is not yet registered.
    successor = grapher._grapher_nodes.get(successor_name)
    if successor is None:
        successor = _Node(name=successor_name, pending=True)
        grapher._grapher_nodes[successor_name] = successor

    # Add the edge in both directions.
    node.successors.add(successor_name)
    successor.predecessors.add(node.name)

    return None


def _remove_successor(
    grapher: Grapher,
    node: _Node,
    successor_name: str,
) -> None:
    """Remove a directed edge from node to successor on both sides.

    Removes successor_name from node.successors and removes node.name
    from the successor node's predecessors. The successor node itself
    is not removed -- only the edge between the two nodes.
    """
    # Remove the edge from node to successor.
    node.successors.discard(successor_name)

    # Remove the back-reference from the successor node.
    successor = grapher._grapher_nodes.get(successor_name)
    if successor is None:
        _logger.warning(f"[_remove_successor] successor '{successor_name}' not found in graph -- skipping edge removal")
        return
    successor.predecessors.discard(node.name)


# -------------------------------------------------------------------------------------------------------------------- #
# Hooks                                                                                                            #
# -------------------------------------------------------------------------------------------------------------------- #


def _before_update_node(
    grapher: Grapher,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Called before a node buffer is updated.

    Returns None to allow the update, or an error string to block it.
    """
    # Guard against empty or invalid JSON in the node buffer.
    config_or_error = _get_node_config(new_buffer)
    if isinstance(config_or_error, str):
        return config_or_error

    # Extract the successors dict from the node config.
    successors = config_or_error.get("successors", {})
    if not isinstance(successors, dict):
        return f"[Grapher] 'successors' must be a dict of node-name to edge-label, got: {type(successors).__name__}"

    # Reject when a referenced successor has no origin buffer.
    for successor_name in successors:
        if grapher.get_buffer(successor_name) is None:
            return f"[Grapher] node '{new_buffer.name}' references unknown origin buffer '{successor_name}' (hint: create the origin buffer '{successor_name}' first)"

    return None


async def _after_update_node(
    grapher: Grapher,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after a node buffer is created or updated.

    Reads the new buffer config and reconciles the in-memory node graph.
    """
    # Get the new buffer config -- reject invalid or empty buffers.
    new_config_or_error = _get_node_config(new_buffer)
    if isinstance(new_config_or_error, str):
        _logger.error(f"[_after_update_node] {new_config_or_error}")
        return
    new_config = new_config_or_error

    # Get the old buffer config -- empty dict when this is a create (no prior state).
    old_config: dict[str, Any] = {}
    if old_buffer is not None:
        old_config_or_error = _get_node_config(old_buffer)
        if isinstance(old_config_or_error, str):
            _logger.error(f"[_after_update_node] {old_config_or_error}")
        else:
            old_config = old_config_or_error

    # Extract the old and new successor sets (keys of the successors dict).
    old_successors = set(old_config.get("successors", {}).keys())
    new_successors = set(new_config.get("successors", {}).keys())

    # Compute the diff: which successors were removed and which were added.
    removed_successors = old_successors - new_successors
    added_successors = new_successors - old_successors

    # Derive the node name from the buffer name and get or create the node instance.
    node_name = new_buffer.name.removesuffix(_GRAPHER_NODE_SUFFIX)
    node = grapher._grapher_nodes.get(node_name)
    if node is None:
        node = _Node(name=node_name)
        grapher._grapher_nodes[node_name] = node
    else:
        node.pending = False

    # Remove edges for successors that are no longer in the config.
    for successor_name in removed_successors:
        _remove_successor(grapher, node, successor_name)

    # Add edges for newly declared successors.
    for successor_name in added_successors:
        error = _add_successor(grapher, node, successor_name)
        if error:
            _logger.error(f"[_after_update_node] {error}")


async def _after_drop_node(
    grapher: Grapher,
    old_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after a node buffer is dropped.

    Extracts the node name from the dropped buffer and removes it from the
    in-memory node graph. Unlinks all successor and predecessor edges first.
    """
    # Derive the node name from the dropped buffer name.
    node_name = old_buffer.name.removesuffix(_GRAPHER_NODE_SUFFIX)

    # Look up the node in the graph -- error if it was never registered.
    node = grapher._grapher_nodes.get(node_name)
    if node is None:
        _logger.error(f"[_after_drop_node] node '{node_name}' not found in graph -- inconsistent state")
        return

    # Unlink all successor edges (this node -> others).
    for successor_name in list(node.successors):
        _remove_successor(grapher, node, successor_name)

    # Unlink all predecessor edges (others -> this node).
    for predecessor_name in list(node.predecessors):
        # Look up the predecessor node in the graph.
        predecessor = grapher._grapher_nodes.get(predecessor_name)

        # Guard: reject if the predecessor node is missing -- an inconsistent graph.
        if predecessor is None:
            _logger.error(f"[_after_drop_node] predecessor '{predecessor_name}' not found in graph -- inconsistent state")
            node.predecessors.discard(predecessor_name)
            continue

        # Remove the back-reference from the predecessor and clear it locally.
        predecessor.successors.discard(node_name)
        node.predecessors.discard(predecessor_name)

    # Remove the node from the graph.
    grapher._grapher_nodes.pop(node_name, None)


# -------------------------------------------------------------------------------------------------------------------- #
# Grapher                                                                                                            #
# -------------------------------------------------------------------------------------------------------------------- #


class Grapher(OptionBufferManager, AgenticObject):
    """I manage graph data structures and provide traversal and pathfinding operations."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Node storage keyed by node name.
        self._grapher_nodes: dict[str, _Node] = {}

        # Seed the doc buffer with the module documentation.
        self._create_buffer(_GRAPHER_DOC_BUFFER, text=_GRAPHER_DOC)

        # Register the before and after update hooks for node buffers.
        # BEFORE: guard the update of node buffers.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_UPDATE_BUFFER,
            "before_update_grapher_node",
            _GRAPHER_NODE_BUFFER_PATTERN,
            _before_update_node,
            apply_to_existing=True,
        )
        # BEFORE CREATE: same guard with a None bridge for old_buffer.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "before_update_grapher_node",
            _GRAPHER_NODE_BUFFER_PATTERN,
            lambda bm, new_buffer, args: _before_update_node(bm, None, new_buffer, args),
            apply_to_existing=True,
        )

        # AFTER: handle the creation or update of node buffers.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_UPDATE_BUFFER,
            "after_update_grapher_node",
            _GRAPHER_NODE_BUFFER_PATTERN,
            _after_update_node,
            apply_to_existing=True,
        )
        # AFTER CREATE: same handler with a None bridge for old_buffer.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "after_update_grapher_node",
            _GRAPHER_NODE_BUFFER_PATTERN,
            lambda bm, new_buffer, args: _after_update_node(bm, None, new_buffer, args),
            apply_to_existing=True,
        )

        # AFTER DROP: remove the node from the in-memory graph when its option buffer is dropped.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            "after_drop_grapher_node",
            _GRAPHER_NODE_BUFFER_PATTERN,
            _after_drop_node,
            apply_to_existing=True,
        )
