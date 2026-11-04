from __future__ import annotations

import asyncio
import json
import logging
import time

from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable

from peteos.oap.agentic_object import AgenticObject


_logger = logging.getLogger("petri")

from petekit.graph.grapher import (
    _GRAPHER_DOC_BUFFER,
    _GRAPHER_NODE_BUFFER_PATTERN,
    _GRAPHER_NODE_SUFFIX,
    _Node,
    _get_node_config,
    get_node,
    get_node_config,
    get_predecessors,
    get_successors,
    Grapher,
)
from petekit.text.buffer_manager import HookPoint
from petekit.text.buffer_manager import Buffer
from petekit.text.option_buffer_manager import validate_and_prepare_action_config


# -------------------------------------------------------------------------------------------------------------------- #
# Constants                                                                                                          #
# -------------------------------------------------------------------------------------------------------------------- #


_PETRI_DOC_BUFFER = "doc:petri"

_PETRI_DOC = f"""\
## PetriNetor

The PetriNetor manages a Petri net -- a directed bipartite graph of places and
transitions. Places hold tokens; transitions move tokens along edges.

### Buffer Naming

- `<buffer-name>{_GRAPHER_NODE_SUFFIX}`  --  a single node that is either a place or a transition

### Node Config

Each node buffer extends the base graph config (see `{_GRAPHER_DOC_BUFFER}`) with a `petri` key.
The value is a dict with a `type` field set to either `place` or `transition`.

**Base fields (all nodes):**

- `successors`  --  dict of successor node names to edge labels (see `{_GRAPHER_DOC_BUFFER}`)

**Place node:**

```json
{{
  "petri": {{
    "type": "place",
    "tokens": 0,
    "last_updated_at": 0.0
  }},
  "successors": {{
    "fire-transition": "description of the edge to fire-transition"
  }}
}}
```

- `tokens`  --  integer count of tokens in this place (default: 0)
- `last_updated_at`  --  Unix timestamp of the last token count change (managed automatically)

**Transition node:**

```json
{{
  "petri": {{
    "type": "transition",
    "action": {{
      "name": "!notify",
      "action_config": {{
        "prompt": "hello world"
      }}
    }},
    "consume": {{"p1": 1, "p2": 2}},
    "produce": {{"p3": 1}}
  }},
  "successors": {{
    "p3": "description of the edge to place p3"
  }}
}}
```

- `action`  --  optional; when a string it names the function to call (including built-in actions `!read`, `!touch`, `!notify`); when a dict it must have a `name` field (the function or action to call) and an optional `kwargs` dict of keyword arguments; when `name` is `!notify` an optional `action_config` sub-dict may be present with `prompt` (notification text, defaults to a timestamp header and buffer content) and `persistent_thread_id` (conversation thread, auto-injected when absent); omit the entire field when no action is needed
- `earliest_fire`  --  optional minimum delay in seconds before the transition can fire; omit when not needed
- `latest_fire`  --  optional maximum delay in seconds before the transition must fire; omit when not needed
- `consume`  --  optional dict of `{{"place_name": token_count}}` deducted from each pre-place when fired; omit when not needed
- `produce`  --  optional dict of `{{"place_name": token_count}}` added to each post-place when fired; omit when not needed

### Firing Rules

A transition fires when:
- All pre-places have at least the `consume` token count
- `earliest_fire` has elapsed since the last fire (if set)
- `latest_fire` has not passed since the last fire (if set)

On fire, tokens are deducted from each pre-place and added to each post-place per the `consume`/`produce` dicts.
The `action` function is called with the `kwargs` if provided.

### Creating a Node

Create the option buffer `<node-name>{_GRAPHER_NODE_SUFFIX}` with the config above.
The `petri.type` field determines whether it is a place or a transition.
"""


# -------------------------------------------------------------------------------------------------------------------- #
# Data Classes                                                                                                       #
# -------------------------------------------------------------------------------------------------------------------- #


@dataclass
class _Transition:
    """A Petri net transition with a firing timer.
    
    The task field is the timer that starts when the transition is activated.
    When the timer reaches the `earliest_fire` delay, the transition becomes
    eligible to fire (subject to token availability and other constraints).
    """

    name: str
    # The asyncio task acting as the firing timer. Started when the transition
    # is activated; cancelled or collected when the transition fires or is
    # dropped. None when no earliest_fire delay is configured.
    task: asyncio.Task[None] | None = None


# -------------------------------------------------------------------------------------------------------------------- #
# Utils                                                                                                             #
# -------------------------------------------------------------------------------------------------------------------- #


def _make_transition_action(petri: PetriNetor, node_name: str) -> Callable[[], None]:
    """Factory that returns a closure for invoking the transition's action.

    Captures the grapher and transition name so the closure can read the
    current action config at call time and execute it.
    """
    async def transition_action() -> None:
        # Read the current transition config at call time.
        config_or_error = get_node_config(petri, node_name)
        if isinstance(config_or_error, str):
            _logger.error(f"[_transition_action] '{node_name}' config unreadable: {config_or_error}")
            return

        # Guard: reject if this node is not a transition.
        petri_config = config_or_error.get("petri")
        if not isinstance(petri_config, dict) or petri_config.get("type") != "transition":
            return

        # Wait for the earliest-fire delay if one is configured.
        earliest_fire = petri_config.get("earliest_fire")
        if isinstance(earliest_fire, (int, float)):
            await asyncio.sleep(earliest_fire)

        # Serialize token mutations across the net to prevent race conditions.
        async with petri._petri_lock:
            # Guard: re-check fire eligibility after the delay, another transition may have fired.
            can_fire = transition_can_fire(petri, node_name)
            if isinstance(can_fire, str):
                _logger.error(f"[_transition_action] '{node_name}' fire check failed: {can_fire}")
                return

            if not can_fire:
                # The transition cannot fire -- leave all pre-place tokens untouched.
                return

            # Reduce the token count of each pre-place.
            reduced = await _reduce_preplaces(petri, node_name, config_or_error)
            if not reduced:
                # The transition was preempted -- leave all pre-place tokens untouched.
                return

            # Call the configured action if one is present.
            try:
                await _fire_action(petri, node_name)
            except Exception as e:
                _logger.error(f"[_transition_action] '{node_name}' action failed: {e}")

            # Increase the token count of each post-place.
            await _increase_postplaces(petri, node_name, config_or_error)

    return transition_action


async def _fire_action(petri: PetriNetor, node_name: str) -> None:
    """Fire the configured action for a transition node.

    Reads the node config, extracts the action sub-dict, and dispatches it
    via the option_buffer_manager's _fire_action helper.
    """
    # Read the transition config to get the action field.
    config_or_error = get_node_config(petri, node_name)
    if isinstance(config_or_error, str):
        _logger.error(f"[_fire_action] '{node_name}' config unreadable: {config_or_error}")
        return

    # Guard: skip when this node has no action configured.
    petri_config = config_or_error.get("petri")
    if not isinstance(petri_config, dict):
        return

    action = petri_config.get("action")
    if not action:
        return

    # Dispatch to the option buffer manager's fire_action helper.
    await _fire_action(petri, action, node_name)


async def _reduce_preplaces(petri: PetriNetor, node_name: str, config: dict[str, Any]) -> bool:
    """Reduce the token count of each pre-place according to consume weights.

    Returns True on success, or False when the transition was preempted by
    another firing transition (insufficient tokens). All pre-place tokens
    are only modified when every pre-place has sufficient tokens.
    """
    # Use the helper to verify every pre-place has enough tokens.
    can_fire = transition_can_fire(petri, node_name)
    if isinstance(can_fire, str):
        _logger.error(f"[_reduce_preplaces] '{node_name}' pre-place check failed: {can_fire}")
        return False

    if not can_fire:
        return False

    # Read the consume weights from the transition config.
    petri_config = config.get("petri")
    consume = petri_config.get("consume") if isinstance(petri_config, dict) else {}
    if not isinstance(consume, dict):
        consume = {}

    # Get the predecessor places.
    predecessors = get_predecessors(petri, node_name)
    if predecessors is None:
        return False

    # Second pass: actually decrement the token count.
    for predecessor_name in predecessors:
        # Determine the consume requirement for this pre-place (default: 1).
        required = consume.get(predecessor_name, 1)

        # Read the current config of the pre-place.
        pred_config_or_error = get_node_config(petri, predecessor_name)
        if isinstance(pred_config_or_error, str):
            _logger.error(f"[_reduce_preplaces] '{node_name}' predecessor '{predecessor_name}' config unreadable during reduction: {pred_config_or_error}")
            return False

        # Decrement the token count and update the last_updated_at timestamp.
        pred_config_or_error["petri"]["tokens"] = pred_config_or_error["petri"].get("tokens", 0) - required
        pred_config_or_error["petri"]["last_updated_at"] = time.time()

        # Write the updated config back to the place buffer.
        buffer_name = predecessor_name + _GRAPHER_NODE_SUFFIX

        # Use create_buffer with overwrite=True to replace the buffer content.
        create_result = await petri.create_buffer(
            buffer_name,
            text=json.dumps(pred_config_or_error, indent=2),
            overwrite=True,
        )
        if not create_result.get("ok"):
            _logger.error(f"[_reduce_preplaces] '{node_name}' predecessor '{predecessor_name}' write failed: {create_result.get('error')}")
            return False

    return True


async def _increase_postplaces(petri: PetriNetor, node_name: str, config: dict[str, Any]) -> None:
    """Increase the token count of each post-place according to produce weights.

    Produces tokens into each successor place from the transition's produce
    config (default: 1 per place).
    """
    # Read the produce weights from the transition config.
    petri_config = config.get("petri")
    if not isinstance(petri_config, dict):
        _logger.error(f"[_increase_postplaces] '{node_name}' has no petri config")
        return

    # Read the produce weights from the transition config.
    produce = petri_config.get("produce")
    if not isinstance(produce, dict):
        produce = {}

    # Get the successor places.
    successors = get_successors(petri, node_name)
    if successors is None:
        return

    # Iterate over each successor post-place and produce tokens into it.
    for successor_name in successors:
        # Determine the produce amount for this post-place (default: 1).
        amount = produce.get(successor_name, 1)

        # Read and validate the post-place config.
        succ_config_or_error = get_node_config(petri, successor_name)
        if isinstance(succ_config_or_error, str):
            _logger.error(f"[_increase_postplaces] '{node_name}' successor '{successor_name}' config unreadable: {succ_config_or_error}")
            continue

        # Read and validate the petri config.
        succ_petri = succ_config_or_error.get("petri")
        if not isinstance(succ_petri, dict):
            _logger.error(f"[_increase_postplaces] '{node_name}' successor '{successor_name}' has no petri config")
            continue

        # Increment the token count and update the last_updated_at timestamp.
        succ_config_or_error["petri"]["tokens"] = succ_config_or_error["petri"].get("tokens", 0) + amount
        succ_config_or_error["petri"]["last_updated_at"] = time.time()

        # Write the updated config back to the place buffer.
        buffer_name = successor_name + _GRAPHER_NODE_SUFFIX

        # Use create_buffer with overwrite=True to replace the buffer content.
        create_result = await petri.create_buffer(
            buffer_name,
            text=json.dumps(succ_config_or_error, indent=2),
            overwrite=True,
        )
        if not create_result.get("ok"):
            _logger.error(f"[_increase_postplaces] '{node_name}' successor '{successor_name}' write failed: {create_result.get('error')}")


def transition_can_fire(petri: PetriNetor, node_name: str) -> bool | str:
    """Return True when a transition has sufficient tokens at all pre-places to fire.

    Returns an error string if the config cannot be read, the node is absent,
    or the node is not a transition. Returns False when the transition exists
    but cannot fire due to insufficient tokens. Returns True only when all
    pre-places have at least the required token count (from `consume`, default 1).
    """
    # Guard: reject if the transition config cannot be read.
    config_or_error = get_node_config(petri, node_name)
    if isinstance(config_or_error, str):
        return config_or_error

    # Guard: reject if this is not a transition.
    petri_config = config_or_error.get("petri")
    if not isinstance(petri_config, dict) or petri_config.get("type") != "transition":
        return (
            f"[PetriNetor] node '{node_name}' is not a transition "
            f"(type={petri_config.get('type') if isinstance(petri_config, dict) else petri_config})"
        )

    # Read the consume requirements from the transition config.
    consume = petri_config.get("consume")
    if not isinstance(consume, dict):
        consume = {}

    # Guard: reject if there are no predecessors registered.
    predecessors = get_predecessors(petri, node_name)
    if predecessors is None:
        return f"[PetriNetor] transition '{node_name}' has no registered predecessors in the graph"

    # Check every predecessor has sufficient tokens.
    for predecessor_name in predecessors:
        # Determine the consume requirement for this pre-place (default: 1).
        required = consume.get(predecessor_name, 1)

        # Guard: reject if the pre-place config cannot be read.
        pred_config_or_error = get_node_config(petri, predecessor_name)
        if isinstance(pred_config_or_error, str):
            return pred_config_or_error

        # Guard: reject if the pre-place config is malformed.
        pred_petri = pred_config_or_error.get("petri")
        if not isinstance(pred_petri, dict):
            return f"[PetriNetor] predecessor '{predecessor_name}' of transition '{node_name}' has no petri config"

        # Reject when the pre-place has insufficient tokens.
        tokens = pred_petri.get("tokens", 0)
        if tokens < required:
            return False

    return True


def transition_is_active(petri: PetriNetor, node_name: str) -> bool:
    """Return True when a transition has an active firing timer task.

    Checks whether the transition entry in `petri._petri_transitions` has a
    non-None task. An active task means the earliest-fire timer has started
    but the transition has not yet fired.
    """
    transition = petri._petri_transitions.get(node_name)
    return transition is not None and transition.task is not None


async def _activate_transition(petri: PetriNetor, node_name: str) -> None:
    """Activate a transition that was not active before.

    Creates the transition action and wraps it in an asyncio task to
    start the firing timer. The task is stored in the transition registry.
    """
    # Guard: the transition entry must exist in the registry (registered on create or update).
    transition = petri._petri_transitions.get(node_name)
    if transition is None:
        _logger.error(f"[_activate_transition] transition '{node_name}' has no entry in the registry")
        return

    # Guard: reject when the transition is already active.
    if transition.task is not None:
        return

    # Create the transition action closure and wrap it in an asyncio task.
    action = _make_transition_action(petri, node_name)
    transition.task = asyncio.create_task(action())


async def _deactivate_transition(petri: PetriNetor, node_name: str) -> None:
    """Deactivate an active transition.

    Cancels the firing timer task and removes the transition from the
    active registry. Used when a pre-place token count decreased and
    the transition is no longer eligible to fire.
    """
    # Look up the transition entry.
    transition = petri._petri_transitions.get(node_name)
    if transition is None or transition.task is None:
        return

    # Cancel the firing timer task.
    transition.task.cancel()

    # Clear the task to mark the transition as inactive (keep the entry for reactivation).
    transition.task = None


def _validate_transition_config(
    petri: PetriNetor,
    config_buffer: Buffer,
    buffer_config: dict[str, Any],
) -> str | None:
    """Validate that all predecessors and successors of a transition are places.

    In a Petri net, transitions may only be connected to places. Also validates
    and prepares the action config if one is present. Returns None if all
    checks pass, or an error string describing the first violation found.
    """
    # Derive the node name from the buffer name.
    node_name = config_buffer.name.removesuffix(_GRAPHER_NODE_SUFFIX)

    # Guard: reject if the node is not a transition.
    petri_config = buffer_config.get("petri")
    if not isinstance(petri_config, dict):
        return f"[PetriNetor] node '{node_name}' is missing a 'petri' config dict"

    # Guard: skip validation when the node is not a transition.
    if petri_config.get("type") != "transition":
        return None

    # Validate and prepare the action config if one is present.
    action = petri_config.get("action")
    if isinstance(action, dict):
        error = validate_and_prepare_action_config(
            petri,
            config_buffer,
            buffer_config,
            action,
        )
        if error:
            _logger.error(f"[_validate_transition_config] transition '{node_name}' action config error: {error}")

    # Validate that every predecessor is a place.
    predecessors = get_predecessors(petri, node_name)
    if predecessors is not None:
        # Each predecessor must be a place -- reject the first violation.
        for predecessor_name in predecessors:
            # Reject if the predecessor config cannot be read.
            pred_config_or_error = get_node_config(petri, predecessor_name)
            if isinstance(pred_config_or_error, str):
                return pred_config_or_error

            # Reject if the predecessor is not a place.
            pred_petri = pred_config_or_error.get("petri")
            if not isinstance(pred_petri, dict) or pred_petri.get("type") != "place":
                return (
                    f"[PetriNetor] predecessor '{predecessor_name}' of transition "
                    f"'{node_name}' must be a place, got: {pred_petri}"
                )

    # Validate that every successor is a place.
    successors = get_successors(petri, node_name)
    if successors is not None:
        # Each successor must be a place -- reject the first violation.
        for successor_name in successors:
            # Reject if the successor config cannot be read.
            succ_config_or_error = get_node_config(petri, successor_name)
            if isinstance(succ_config_or_error, str):
                return succ_config_or_error

            # Reject if the successor is not a place.
            succ_petri = succ_config_or_error.get("petri")
            if not isinstance(succ_petri, dict) or succ_petri.get("type") != "place":
                return (
                    f"[PetriNetor] successor '{successor_name}' of transition "
                    f"'{node_name}' must be a place, got: {succ_petri}"
                )

    # Validate that every key in consume is a registered predecessor.
    consume = petri_config.get("consume")
    if isinstance(consume, dict):
        for place_name in consume:
            if predecessors is not None and place_name not in predecessors:
                return (
                    f"[PetriNetor] transition '{node_name}' consume references "
                    f"'{place_name}' which is not a registered predecessor"
                )

    # Validate that every key in produce is a registered successor.
    produce = petri_config.get("produce")
    if isinstance(produce, dict):
        for place_name in produce:
            if successors is not None and place_name not in successors:
                return (
                    f"[PetriNetor] transition '{node_name}' produce references "
                    f"'{place_name}' which is not a registered successor"
                )

    return None


# -------------------------------------------------------------------------------------------------------------------- #
# Hooks                                                                                                            #
# -------------------------------------------------------------------------------------------------------------------- #


def _before_update_petri_node(
    petri: PetriNetor,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Called before a petri-node buffer is updated.

    Returns None to allow the update, or an error string to block it.
    """
    # Guard against empty or invalid JSON in the petri-node buffer.
    config_or_error = _get_node_config(new_buffer)
    if isinstance(config_or_error, str):
        return config_or_error

    # Guard: validate that all predecessors and successors are places when
    # this node is a transition, and validate the action config if present.
    validation_error = _validate_transition_config(
        petri,
        new_buffer,
        config_or_error,
    )
    if validation_error is not None:
        return validation_error

    return None


async def _after_update_petri_node(
    petri: PetriNetor,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after a petri-node buffer is created or updated.

    When the token count of a place changed, this function activates
    successor transitions that became newly eligible, or deactivates
    those that are no longer eligible.
    """
    # Derive the node name from the buffer name.
    node_name = new_buffer.name.removesuffix(_GRAPHER_NODE_SUFFIX)

    # Guard: reject if the node config cannot be read.
    config_or_error = get_node_config(petri, node_name)
    if isinstance(config_or_error, str):
        return

    # Guard: skip post-processing for transitions -- no token state to manage here.
    petri_config = config_or_error.get("petri")
    if isinstance(petri_config, dict) and petri_config.get("type") == "transition":
        # Ensure the transition entry exists in the registry.
        if node_name not in petri._petri_transitions:
            petri._petri_transitions[node_name] = _Transition(name=node_name, task=None)

        return

    # Determine the new token count.
    new_tokens = petri_config.get("tokens", 0) if isinstance(petri_config, dict) else 0

    # Determine the old token count directly from the old buffer.
    old_tokens = 0
    if old_buffer is not None:
        # Guard: skip when the old buffer config cannot be parsed.
        old_config_or_error = _get_node_config(old_buffer)
        if not isinstance(old_config_or_error, str):
            # Guard: skip when the old petri config is absent.
            old_petri = old_config_or_error.get("petri")
            if isinstance(old_petri, dict):
                # Read the old token count from the petri config.
                old_tokens = old_petri.get("tokens", 0)

    # Reject early when the token count did not change.
    if new_tokens == old_tokens:
        return

    # Get the successor nodes to propagate the change.
    successors = get_successors(petri, node_name)
    if successors is None or len(successors) == 0:
        return

    # Handle token increase: activate inactive transitions that became newly eligible.
    if new_tokens > old_tokens:
        for successor_name in successors:
            # Guard: skip non-transition successors.
            succ_config_or_error = get_node_config(petri, successor_name)
            if isinstance(succ_config_or_error, str):
                _logger.error(f"[_after_update_petri_node] successor '{successor_name}' config unreadable: {succ_config_or_error}")
                continue

            # Guard: a place may only have transition successors -- log error and skip.
            succ_petri = succ_config_or_error.get("petri")
            if not isinstance(succ_petri, dict) or succ_petri.get("type") != "transition":
                _logger.error(
                    f"[_after_update_petri_node] place '{node_name}' has non-transition "
                    f"successor '{successor_name}' (type={succ_petri.get('type') if isinstance(succ_petri, dict) else succ_petri}) -- "
                    f"Places may only be connected to transitions in a Petri net"
                )
                continue

            # Only activate when the transition is inactive and now has sufficient tokens.
            if transition_is_active(petri, successor_name):
                continue

            # Check if the transition can fire after the token increase.
            can_fire = transition_can_fire(petri, successor_name)
            if isinstance(can_fire, str):
                _logger.error(f"[_after_update_petri_node] could not determine fire eligibility for '{successor_name}': {can_fire}")
                continue

            # Skip activation when the transition still cannot fire.
            if not can_fire:
                continue

            # Activate the successor transition when it became eligible.
            await _activate_transition(petri, successor_name)

    # Handle token decrease: deactivate active transitions that are no longer eligible.
    else:
        for successor_name in successors:
            # Guard: skip non-transition successors.
            succ_config_or_error = get_node_config(petri, successor_name)
            if isinstance(succ_config_or_error, str):
                _logger.error(f"[_after_update_petri_node] successor '{successor_name}' config unreadable: {succ_config_or_error}")
                continue

            # Guard: a place may only have transition successors -- log error and skip.
            succ_petri = succ_config_or_error.get("petri")
            if not isinstance(succ_petri, dict) or succ_petri.get("type") != "transition":
                _logger.error(
                    f"[_after_update_petri_node] place '{node_name}' has non-transition "
                    f"successor '{successor_name}' (type={succ_petri.get('type') if isinstance(succ_petri, dict) else succ_petri}) -- "
                    f"Places may only be connected to transitions in a Petri net"
                )
                continue

            # Only deactivate when the transition was active and is now ineligible to fire.
            if not transition_is_active(petri, successor_name):
                continue

            # Check if the transition can still fire after the token decrease.
            can_fire = transition_can_fire(petri, successor_name)
            if isinstance(can_fire, str):
                _logger.error(f"[_after_update_petri_node] could not determine fire eligibility for '{successor_name}': {can_fire}")
                continue

            # Skip deactivation when the transition is still eligible to fire.
            if can_fire:
                continue

            # Deactivate the successor transition when it became ineligible.
            await _deactivate_transition(petri, successor_name)


async def _before_drop_petri_node(
    petri: PetriNetor,
    old_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Called before a petri-node buffer is dropped.

    Returns None to allow the drop, or an error string to block it.
    """
    # Guard: reject if the node does not exist in the graph.
    node_name = old_buffer.name.removesuffix(_GRAPHER_NODE_SUFFIX)
    node = get_node(petri, node_name)
    if node is None:
        return f"[PetriNetor] node '{node_name}' not found in graph"

    # Guard: reject if the node config cannot be read.
    config_or_error = get_node_config(petri, node_name)
    if isinstance(config_or_error, str):
        return config_or_error

    # Guard: reject if the node is a place that still has connected transitions.
    petri_config = config_or_error.get("petri")
    if isinstance(petri_config, dict) and petri_config.get("type") == "place":
        successors = get_successors(petri, node_name)
        if successors is not None and len(successors) > 0:
            # Reject: a place with outgoing edges cannot be dropped.
            connected = ", ".join(sorted(successors))
            return (
                f"[PetriNetor] place '{node_name}' is still connected to "
                f"transition(s): {connected} -- drop the transition(s) first"
            )

    return None


async def _after_drop_petri_node(
    petri: PetriNetor,
    old_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after a petri-node buffer is dropped.

    Cleans up the petri-specific state for the dropped node.
    Note: edge unlinking from the graph is handled by Grapher's after_drop_node
    hook independently.
    """
    # Derive the node name from the dropped buffer name.
    node_name = old_buffer.name.removesuffix(_GRAPHER_NODE_SUFFIX)

    # Clean up the transition-specific state if this was a transition.
    transition = petri._petri_transitions.pop(node_name, None)
    if transition is not None and transition.task is not None:
        transition.task.cancel()


# -------------------------------------------------------------------------------------------------------------------- #
# PetriNetor                                                                                                        #
# -------------------------------------------------------------------------------------------------------------------- #


class PetriNetor(Grapher, AgenticObject):
    """I manage a Petri net."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Mutex that serializes all token mutations across the net.
        self._petri_lock = asyncio.Lock()

        # In-memory dict of active transition tasks keyed by transition name.
        self._petri_transitions: dict[str, _Transition] = {}

        # Seed the doc buffer with the module documentation.
        self._create_buffer(_PETRI_DOC_BUFFER, text=_PETRI_DOC)

        # Register the before and after update hooks for petri-node buffers.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_UPDATE_BUFFER,
            "before_update_petri_node",
            _GRAPHER_NODE_BUFFER_PATTERN,
            _before_update_petri_node,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "before_update_petri_node",
            _GRAPHER_NODE_BUFFER_PATTERN,
            lambda bm, new_buffer, args: _before_update_petri_node(bm, None, new_buffer, args),
            apply_to_existing=True,
        )

        # AFTER: handle the creation or update of petri-node buffers.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_UPDATE_BUFFER,
            "after_update_petri_node",
            _GRAPHER_NODE_BUFFER_PATTERN,
            _after_update_petri_node,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "after_update_petri_node",
            _GRAPHER_NODE_BUFFER_PATTERN,
            lambda bm, new_buffer, args: _after_update_petri_node(bm, None, new_buffer, args),
            apply_to_existing=True,
        )

        # Register the before and after drop hooks for petri-node buffers.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_DROP_BUFFER,
            "before_drop_petri_node",
            _GRAPHER_NODE_BUFFER_PATTERN,
            _before_drop_petri_node,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            "after_drop_petri_node",
            _GRAPHER_NODE_BUFFER_PATTERN,
            _after_drop_petri_node,
            apply_to_existing=True,
        )
