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
from petekit.text.option_buffer_manager import _fire_action
from petekit.text.option_buffer_manager import validate_and_prepare_action_config


# -------------------------------------------------------------------------------------------------------------------- #
# Constants                                                                                                          #
# -------------------------------------------------------------------------------------------------------------------- #


_PETRI_NODE_SUFFIX = "[petri]"
_PETRI_NODE_BUFFER_PATTERN = rf"^(.+)\[{_PETRI_NODE_SUFFIX[1:-1]}\]$"
_PETRI_DOC_BUFFER = "doc:petri"

_PETRI_DOC = f"""\
## PetriNetor

The Petri net layer sits on top of the Grapher graph as an option buffer.
It does NOT manage the graph structure -- that is the Grapher's job.
Instead, the Petri net layer adds token semantics to nodes that already exist in the graph.

### Buffer Naming

- `<node-name>{_GRAPHER_NODE_SUFFIX}{_PETRI_NODE_SUFFIX}`  --  a Petri net config buffer (option buffer of a graph node)

The Petri net buffer is an option buffer of the graph node buffer.
For example, a graph node `p1[node]` has an optional `p1[node][petri]` Petri net config.

### Creating a Petri Buffer

Create the Petri net buffer `<node-name>{_GRAPHER_NODE_SUFFIX}{_PETRI_NODE_SUFFIX}`.
The graph node `<node-name>{_GRAPHER_NODE_SUFFIX}` must already exist.

### Petri Config

A Petri net buffer is a JSON object with a `type` field set to either `place` or `transition`.

**Place node:**

```json
{{
  "type": "place",
  "tokens": 0,
  "last_updated_at": 0.0
}}
```

- `tokens`  --  integer count of tokens in this place (default: 0); can be set manually to add or remove tokens
  the PetriNetor activates or deactivates transitions based on the current value
- `last_updated_at`  --  Unix timestamp of the last token count change (managed automatically)

**Transition node:**

```json
{{
  "type": "transition",
  "action": {{
    "name": "!notify",
    "action_config": {{
      "prompt": "hello world"
    }}
  }},
  "consume": {{"p1": 1, "p2": 2}},
  "produce": {{"p3": 1}}
}}
```

- `action`  --  optional; when a string it names the function to call
  (including built-in actions `!read`, `!touch`, `!notify`); when a dict it must have a `name` field
  (the function or action to call) and an optional `kwargs` dict of keyword arguments
  when `name` is `!notify` an optional `action_config` sub-dict may be present with `prompt`
  (notification text, defaults to a timestamp header and buffer content)
  and `persistent_thread_id` (conversation thread, auto-injected when absent)
  omit the entire field when no action is needed
- `earliest_fire`  --  optional minimum delay in seconds before the transition can fire; omit when not needed
- `latest_fire`  --  optional maximum delay in seconds before the transition must fire; omit when not needed
- `consume`  --  optional dict of `{{"place_name": token_count}}` deducted from each pre-place when fired
  omit when not needed
- `produce`  --  optional dict of `{{"place_name": token_count}}` added to each post-place when fired
  omit when not needed

### Firing Rules

A transition is **inactive** by default and becomes **active** when all pre-places have
at least the `consume` token count.
While active, a timer counts from the moment of activation.
If a pre-place loses tokens the transition is deactivated and the timer is cancelled.

A transition fires when:
- It is **active**
- `earliest_fire` has elapsed since activation (if set)
- `latest_fire` has not passed since activation (if set)

On fire, tokens are deducted from each pre-place and added to each post-place per the `consume`/`produce` dicts.
The `action` function is called with the `kwargs` if provided.

### Workflow

Create the Petri net in the following order:

1. **Graph layer first** -- create all graph nodes `<node-name>[node]` with their `successors` dict.
   This defines the structure of the net.

2. **Places** -- create a Petri net buffer `<node-name>[node][petri]` with `type=place` for each place.

3. **Transitions** -- create a Petri net buffer `<node-name>[node][petri]` with `type=transition`.
   The `consume` dict references pre-place names, the `produce` dict references post-place names.
   All places referenced in `consume` and `produce` must already exist when the transition is created.

A transition can be created as soon as the places it links to exist.
New places can be added later -- just ensure that no existing transition references them in `consume` or `produce`.
"""


# -------------------------------------------------------------------------------------------------------------------- #
# Data Classes                                                                                                         #
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
# Config Utils                                                                                                         #
# -------------------------------------------------------------------------------------------------------------------- #


def _node_name_from_petri_buffer_name(petri_buffer_name: str) -> str:
    """Derive the node name from a Petri option buffer name.

    Strips both suffixes from '<node>[node][petri]' to return the base node name.
    """
    return petri_buffer_name.removesuffix(_PETRI_NODE_SUFFIX).removesuffix(_GRAPHER_NODE_SUFFIX)


def _petri_buffer_name_from_node_name(node_name: str) -> str:
    """Derive the Petri option buffer name from a node name.

    Appends both suffixes to '<node>' to return '<node>[node][petri]'.
    """
    return node_name + _GRAPHER_NODE_SUFFIX + _PETRI_NODE_SUFFIX


def _get_petri_config(petri: PetriNetor, buffer: Buffer) -> dict[str, Any] | str:
    """Read and parse the Petri net config from a buffer reference.

    Returns the parsed config dict on success, or an error string if the buffer
    is absent, empty, or contains invalid JSON.
    """
    # Reject when the buffer is None.
    if buffer is None:
        return "[PetriNetor] buffer is None"

    # Extract the text from the petri buffer.
    petri_text = buffer.get_text()
    if not petri_text:
        return f"[PetriNetor] petri buffer '{buffer.name}' is empty"

    # Parse the JSON config from the petri buffer text.
    try:
        config = json.loads(petri_text)
    except json.JSONDecodeError:
        return f"[PetriNetor] petri buffer '{buffer.name}' has invalid JSON"

    return config


def _get_petri_config_by_node(petri: PetriNetor, node_name: str) -> dict[str, Any] | str:
    """Read and parse the Petri net config for a node by node name.

    Returns the parsed config dict on success, or an error string if the buffer
    is absent, empty, or contains invalid JSON.
    """
    # Derive the petri buffer name and look up the buffer reference.
    petri_buffer_name = _petri_buffer_name_from_node_name(node_name)
    petri_buffer = petri.get_buffer(petri_buffer_name)
    if petri_buffer is None:
        return f"[PetriNetor] petri buffer '{petri_buffer_name}' not found"

    # Delegate to the buffer-level reader.
    return _get_petri_config(petri, petri_buffer)


def _validate_petri_config(
    petri: PetriNetor,
    config_buffer: Buffer,
    petri_config: dict[str, Any],
) -> str | None:
    """Validate that all predecessors and successors of a transition are places.

    In a Petri net, transitions may only be connected to places. Also validates
    and prepares the action config if one is present. Returns None if all
    checks pass, or an error string describing the first violation found.
    """
    # Derive the node name from the buffer name.
    node_name = _node_name_from_petri_buffer_name(config_buffer.name)

    # Dispatch the configuration according to the node type
    if petri_config.get("type") == "transition":
        return _validate_transition_config(petri, config_buffer, petri_config)
    elif petri_config.get("type") == "place":
        return _validate_place_config(petri, config_buffer, petri_config)
    else:
        _logger.error(f"[_validate_petri_config] '{node_name}' config error: The type must be either `place` or `transition`")


def _validate_place_config(
    petri: PetriNetor,
    config_buffer: Buffer,
    petri_config: dict[str, Any],
) -> str | None:
    """Validate the petri config for a place node.

    A place must have a `tokens` field with an integer value.
    """
    # Reject when tokens is missing or not an integer.
    tokens = petri_config.get("tokens")
    if not isinstance(tokens, int):
        return f"[PetriNetor] place config must have 'tokens' as int, got: {type(tokens).__name__}"

    return None


def _validate_transition_config(
    petri: PetriNetor,
    config_buffer: Buffer,
    petri_config: dict[str, Any],
) -> str | None:
    # Derive the node name from the buffer name.
    node_name = _node_name_from_petri_buffer_name(config_buffer.name)

    # Validate and prepare the action config if one is present.
    action = petri_config.get("action")
    if isinstance(action, dict):
        error = validate_and_prepare_action_config(
            petri,
            config_buffer,
            petri_config,
            action,
        )
        if error:
            _logger.error(f"[_validate_transition_config] transition '{node_name}' action config error: {error}")

    # Validate that every predecessor is a place.
    predecessors = get_predecessors(petri, node_name)
    if predecessors is None:
        _logger.error(f"[_validate_transition_config] node '{node_name}' is not registered in the graph")
        return f"[PetriNetor] node '{node_name}' has no registered predecessors in the graph (hint: the graph node buffer must be created first)"

    # Each predecessor must be a place -- reject the first violation.
    for predecessor_name in predecessors:
        # Reject if the predecessor petri config cannot be read.
        pred_config_or_error = _get_petri_config_by_node(petri, predecessor_name)
        if isinstance(pred_config_or_error, str):
            return pred_config_or_error

        # Reject if the predecessor is not a place.
        if pred_config_or_error.get("type") != "place":
            return (
                f"[PetriNetor] predecessor '{predecessor_name}' of transition "
                f"'{node_name}' must be a place, got: {pred_config_or_error}"
            )

    # Validate that every successor is a place.
    successors = get_successors(petri, node_name)
    if successors is None:
        _logger.error(f"[_validate_transition_config] node '{node_name}' is not registered in the graph")
        return f"[PetriNetor] node '{node_name}' has no registered successors in the graph (hint: the graph node buffer must be created first)"

    # Each successor must be a place -- reject the first violation.
    for successor_name in successors:
        # Reject if the successor petri config cannot be read.
        succ_config_or_error = _get_petri_config_by_node(petri, successor_name)
        if isinstance(succ_config_or_error, str):
            return succ_config_or_error

        # Reject if the successor is not a place.
        if succ_config_or_error.get("type") != "place":
            return (
                f"[PetriNetor] successor '{successor_name}' of transition "
                f"'{node_name}' must be a place, got: {succ_config_or_error}"
            )

    # Validate every key in consume when the node buffer already exists.
    consume = petri_config.get("consume", {})
    if not isinstance(consume, dict):
        return f"[PetriNetor] transition '{node_name}' consume must be a dict, got: {type(consume).__name__}"

    # Verify every place in consume is a registered predecessor of this transition.
    for place_name in consume:
        if place_name not in predecessors:
            return (
                f"[PetriNetor] transition '{node_name}' consume references "
                f"'{place_name}' which is not a registered predecessor"
            )

    # Validate every key in produce when the node buffer already exists.
    produce = petri_config.get("produce", {})
    if not isinstance(produce, dict):
        return f"[PetriNetor] transition '{node_name}' produce must be a dict, got: {type(produce).__name__}"

    # Verify every place in produce is a registered successor of this transition.
    for place_name in produce:
        if place_name not in successors:
            return (
                f"[PetriNetor] transition '{node_name}' produce references "
                f"'{place_name}' which is not a registered successor"
            )

    return None


# -------------------------------------------------------------------------------------------------------------------- #
# Runtime Utils                                                                                                        #
# -------------------------------------------------------------------------------------------------------------------- #


def _make_transition_action(petri: PetriNetor, node_name: str) -> Callable[[], None]:
    """Factory that returns a closure for invoking the transition's action.

    Captures the grapher and transition name so the closure can read the
    current action config at call time and execute it.
    """
    async def transition_action() -> None:
        # Read the current transition config at call time.
        config_or_error = _get_petri_config_by_node(petri, node_name)
        if isinstance(config_or_error, str):
            _logger.error(f"[_transition_action] '{node_name}' config unreadable: {config_or_error}")
            return

        # Wait for the earliest-fire delay if one is configured.
        earliest_fire = config_or_error.get("earliest_fire")
        if isinstance(earliest_fire, (int, float)):
            await asyncio.sleep(earliest_fire)

        # Serialize token mutations across the net to prevent race conditions.
        async with petri._petri_lock:
            # Re-check fire eligibility after the delay, another transition may have fired.
            can_fire = transition_can_fire(petri, node_name)
            if isinstance(can_fire, str):
                _logger.error(f"[_transition_action] '{node_name}' fire check failed: {can_fire}")
                return

            # If the transition cannot fire, leave all pre-place tokens untouched.
            if not can_fire:
                return

            # Try to reduce the token count of each pre-place.
            reduced = await _reduce_preplaces(petri, node_name, config_or_error)
            if not reduced:
                # The transition was preempted -- leave all pre-place tokens untouched.
                return

            # Call the configured action if one is present.
            try:
                await _fire_petri_action(petri, node_name)
            except Exception as e:
                _logger.error(f"[_transition_action] '{node_name}' action failed: {e}")

            # Increase the token count of each post-place.
            await _increase_postplaces(petri, node_name, config_or_error)

        # Deactivate after firing.
        await _deactivate_transition(petri, node_name)

        # Re-activate when still eligible after firing (pre-places may have gained tokens again).
        can_fire = transition_can_fire(petri, node_name)
        if can_fire is True and not transition_is_active(petri, node_name):
            await _activate_transition(petri, node_name)

    return transition_action


async def _fire_petri_action(petri: PetriNetor, node_name: str) -> None:
    """Fire the configured action for a transition node.

    Reads the node config, extracts the action sub-dict, and dispatches it
    via the option_buffer_manager's _fire_action helper.
    """
    # Read the transition config to get the action field.
    config_or_error = _get_petri_config_by_node(petri, node_name)
    if isinstance(config_or_error, str):
        _logger.error(f"[_fire_action] '{node_name}' config unreadable: {config_or_error}")
        return

    action = config_or_error.get("action")
    if not action:
        return

    # Dispatch to the option buffer manager's fire_action helper.
    await _fire_action(petri, action, node_name)


async def _reduce_preplaces(petri: PetriNetor, node_name: str, petri_config: dict[str, Any]) -> bool:
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
    consume = petri_config.get("consume", {})

    # Get the predecessor places.
    predecessors = get_predecessors(petri, node_name)
    if predecessors is None:
        return False

    # Second pass: actually decrement the token count.
    for predecessor_name in predecessors:
        # Determine the consume requirement for this pre-place (default: 1).
        required = consume.get(predecessor_name, 1)

        # Read the current config of the pre-place.
        pred_config_or_error = _get_petri_config_by_node(petri, predecessor_name)
        if isinstance(pred_config_or_error, str):
            _logger.error(f"[_reduce_preplaces] '{node_name}' predecessor '{predecessor_name}' config unreadable during reduction: {pred_config_or_error}")
            return False

        # Decrement the token count and update the last_updated_at timestamp.
        pred_config_or_error["tokens"] = pred_config_or_error.get("tokens", 0) - required
        pred_config_or_error["last_updated_at"] = time.time()

        # Write the updated config back to the place buffer.
        buffer_name = _petri_buffer_name_from_node_name(predecessor_name)

        # Use create_buffer with overwrite=True to replace the buffer content.
        create_result = await petri.create_buffer(
            buffer_name,
            text=json.dumps(pred_config_or_error, indent=2),
            overwrite=True,
        )
        if not create_result.get("ok"):
            _logger.error(f"[_reduce_preplaces] '{node_name}' predecessor '{predecessor_name}' write failed: {create_result.get('error')}")

    return True


async def _increase_postplaces(petri: PetriNetor, node_name: str, petri_config: dict[str, Any]) -> None:
    """Increase the token count of each post-place according to produce weights.

    Produces tokens into each successor place from the transition's produce
    config (default: 1 per place).
    """
    # Read the produce weights from the transition config.
    produce = petri_config.get("produce", {})

    # Get the successor places.
    successors = get_successors(petri, node_name)
    if successors is None:
        return

    # Iterate over each successor post-place and produce tokens into it.
    for successor_name in successors:
        # Determine the produce amount for this post-place (default: 1).
        amount = produce.get(successor_name, 1)

        # Read the post-place config.
        succ_config_or_error = _get_petri_config_by_node(petri, successor_name)
        if isinstance(succ_config_or_error, str):
            _logger.error(f"[_increase_postplaces] '{node_name}' successor '{successor_name}' config unreadable: {succ_config_or_error}")
            continue

        # Increment the token count and update the last_updated_at timestamp.
        succ_config_or_error["tokens"] = succ_config_or_error.get("tokens", 0) + amount
        succ_config_or_error["last_updated_at"] = time.time()

        # Write the updated config back to the place buffer.
        buffer_name = _petri_buffer_name_from_node_name(successor_name)

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
    # Reject if this node is not a registered transition.
    if petri._petri_transitions.get(node_name) is None:
        return f"[PetriNetor] node '{node_name}' is not a registered transition"

    # Read the transition config.
    petri_config = _get_petri_config_by_node(petri, node_name)
    if isinstance(petri_config, str):
        return petri_config

    # Read the consume requirements from the transition config.
    consume = petri_config.get("consume")
    if not isinstance(consume, dict):
        consume = {}

    # Reject if there are no predecessors registered.
    predecessors = get_predecessors(petri, node_name)
    if predecessors is None:
        return f"[PetriNetor] transition '{node_name}' has no registered predecessors in the graph"

    # Check every predecessor has sufficient tokens.
    for predecessor_name in predecessors:
        # Determine the consume requirement for this pre-place (default: 1).
        required = consume.get(predecessor_name, 1)

        # Read the pre-place config.
        pred_config_or_error = _get_petri_config_by_node(petri, predecessor_name)
        if isinstance(pred_config_or_error, str):
            return pred_config_or_error

        # Reject when the pre-place has insufficient tokens.
        tokens = pred_config_or_error.get("tokens", 0)
        if tokens < required:
            return False

    return True


def transition_is_active(petri: PetriNetor, node_name: str) -> bool:
    """Return True when a transition has an active firing timer task.

    Checks whether the transition entry in `petri._petri_transitions` has a
    non-None task. An active task means the earliest-fire timer has started
    but the transition has not yet fired.
    """
    # Return True when the transition has a non-None firing timer task.
    transition = petri._petri_transitions.get(node_name)
    return transition is not None and transition.task is not None


async def _activate_transition(petri: PetriNetor, node_name: str) -> None:
    """Activate a transition that was not active before.

    Creates the transition action and wraps it in an asyncio task to
    start the firing timer. The task is stored in the transition registry.
    """
    # Verify the transition entry exists in the registry (registered on create or update).
    transition = petri._petri_transitions.get(node_name)
    if transition is None:
        _logger.error(f"[_activate_transition] transition '{node_name}' has no entry in the registry")
        return

    # Reject when the transition is already active.
    if transition.task is not None:
        return

    # Create the transition action closure and wrap it in an asyncio task.
    action = _make_transition_action(petri, node_name)
    transition.task = asyncio.create_task(action())

    # Locate the transition buffer by node name.
    petri_buffer = petri.get_buffer(_petri_buffer_name_from_node_name(node_name))
    if petri_buffer is None:
        _logger.error(f"[_activate_transition] buffer for transition '{node_name}' not found")
        return

    # Read the petri config from the buffer.
    config_or_error = _get_petri_config(petri, petri_buffer)
    if isinstance(config_or_error, str):
        _logger.error(f"[_activate_transition] transition '{node_name}' config unreadable: {config_or_error}")
        return

    # Stamp the transition buffer with the activation time and persist the change.
    config_or_error["activated_at"] = time.time()
    petri_buffer.set_text(json.dumps(config_or_error, indent=2))


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

    # Locate the transition buffer by node name.
    petri_buffer = petri.get_buffer(_petri_buffer_name_from_node_name(node_name))
    if petri_buffer is None:
        _logger.error(f"[_deactivate_transition] buffer for transition '{node_name}' not found")
        return

    # Read the petri config from the buffer.
    config_or_error = _get_petri_config(petri, petri_buffer)
    if isinstance(config_or_error, str):
        _logger.error(f"[_deactivate_transition] transition '{node_name}' config unreadable: {config_or_error}")
        return

    # Remove the activated_at field and persist when it was present.
    if "activated_at" in config_or_error:
        del config_or_error["activated_at"]
        petri_buffer.set_text(json.dumps(config_or_error, indent=2))


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
    config_or_error = _get_petri_config(petri, new_buffer)
    if isinstance(config_or_error, str):
        return config_or_error

    # Validate that all predecessors and successors are places when
    # this node is a transition, and validate the action config if present.
    validation_error = _validate_petri_config(
        petri,
        new_buffer,
        config_or_error,
    )
    if validation_error is not None:
        return validation_error

    # When the place token count changes, set the last_updated_at timestamp on
    # the new buffer and persist it before the update commits.
    if config_or_error.get("type") == "place":
        # Read the old buffer tokens or default to 0.
        old_tokens = 0
        if old_buffer is not None:
            old_config_or_error = _get_petri_config(petri, old_buffer)
            if not isinstance(old_config_or_error, str):
                old_tokens = old_config_or_error.get("tokens", 0)

        # Read the new token count.
        new_tokens = config_or_error.get("tokens", 0)

        # Persist the last_updated_at timestamp when the token count changed.
        if new_tokens != old_tokens:
            config_or_error["last_updated_at"] = time.time()
            new_buffer.set_text(json.dumps(config_or_error, indent=2))

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
    node_name = _node_name_from_petri_buffer_name(new_buffer.name)

    # Skip post-processing for transitions -- no token state to manage here.
    if petri._petri_transitions.get(node_name) is not None:
        return

    # Reject if the petri config cannot be read.
    petri_config = _get_petri_config(petri, new_buffer)
    if isinstance(petri_config, str):
        _logger.error(f"[_after_update_petri_node] node '{node_name}' petri config unreadable: {petri_config}")
        return

    # Register the transition and skip post-processing -- no token state to manage here.
    if petri_config.get("type") == "transition":
        # Note: we confirmed the transition was absent before entering this function.
        petri._petri_transitions[node_name] = _Transition(name=node_name)
        return

    # Determine the new token count.
    new_tokens = petri_config.get("tokens", 0)

    # Determine the old token count directly from the old buffer.
    old_tokens = 0
    if old_buffer is not None:
        old_config_or_error = _get_petri_config(petri, old_buffer)
        if not isinstance(old_config_or_error, str):
            old_tokens = old_config_or_error.get("tokens", 0)
        else:
            _logger.warning(f"[_after_update_petri_node] node '{node_name}' old petri config unreadable: {old_config_or_error}")

    # Reject early when the token count did not change.
    if new_tokens == old_tokens:
        return

    # Get the successor transitions to propagate the change.
    successors = get_successors(petri, node_name)
    if successors is None or len(successors) == 0:
        return

    # Handle token increase: activate inactive transitions that became newly eligible.
    if new_tokens > old_tokens:
        for successor_name in successors:
            # Skip non-transition successors -- places may only be connected to transitions.
            if petri._petri_transitions.get(successor_name) is None:
                # Only log an error when the petri option buffer exists but is not a transition.
                if petri.get_buffer(_petri_buffer_name_from_node_name(successor_name)) is not None:
                    _logger.error(
                        f"[_after_update_petri_node] place '{node_name}' has non-transition "
                        f"successor '{successor_name}' -- Places may only be connected to transitions in a Petri net"
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
            # Skip non-transition successors -- places may only be connected to transitions.
            if petri._petri_transitions.get(successor_name) is None:
                # Only log an error when the petri option buffer exists but is not a transition.
                if petri.get_buffer(_petri_buffer_name_from_node_name(successor_name)) is not None:
                    _logger.error(
                        f"[_after_update_petri_node] place '{node_name}' has non-transition "
                        f"successor '{successor_name}' -- Places may only be connected to transitions in a Petri net"
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
    # Derive the node name from the dropped buffer name.
    node_name = _node_name_from_petri_buffer_name(old_buffer.name)

    # Allow the drop immediately when the node is a transition.
    if petri._petri_transitions.get(node_name) is not None:
        return None

    # Reject if the node does not exist in the graph.
    node = get_node(petri, node_name)
    if node is None:
        _logger.warning(f"[_before_drop_petri_node] node '{node_name}' not found in graph -- allowing drop to proceed")
        return None

    # Allow the drop with a warning when the config cannot be read.
    config_or_error = _get_petri_config(petri, old_buffer)
    if isinstance(config_or_error, str):
        _logger.warning(f"[_before_drop_petri_node] node '{node_name}' config unreadable: {config_or_error} -- allowing drop to proceed")
        return None

    # It's a place. Reject when the place still has predecessors or successors.
    predecessors = get_predecessors(petri, node_name)
    successors = get_successors(petri, node_name)
    if (predecessors and len(predecessors) > 0) or (successors and len(successors) > 0):
        connected = ", ".join(sorted(set(predecessors or []) | set(successors or [])))
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
    node_name = _node_name_from_petri_buffer_name(old_buffer.name)

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
            _PETRI_NODE_BUFFER_PATTERN,
            _before_update_petri_node,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "before_update_petri_node",
            _PETRI_NODE_BUFFER_PATTERN,
            lambda bm, new_buffer, args: _before_update_petri_node(bm, None, new_buffer, args),
            apply_to_existing=True,
        )

        # AFTER: handle the creation or update of petri-node buffers.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_UPDATE_BUFFER,
            "after_update_petri_node",
            _PETRI_NODE_BUFFER_PATTERN,
            _after_update_petri_node,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "after_update_petri_node",
            _PETRI_NODE_BUFFER_PATTERN,
            lambda bm, new_buffer, args: _after_update_petri_node(bm, None, new_buffer, args),
            apply_to_existing=True,
        )

        # Register the before and after drop hooks for petri-node buffers.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_DROP_BUFFER,
            "before_drop_petri_node",
            _PETRI_NODE_BUFFER_PATTERN,
            _before_drop_petri_node,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            "after_drop_petri_node",
            _PETRI_NODE_BUFFER_PATTERN,
            _after_drop_petri_node,
            apply_to_existing=True,
        )
