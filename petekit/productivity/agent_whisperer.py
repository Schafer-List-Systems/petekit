from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Coroutine
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable

from peteos.chatbot import ContentPart
from peteos.oap.agentic_object import AgenticObject

from petekit.function.function_manager import FunctionManager
from petekit.text.buffer_manager import HookPoint

if TYPE_CHECKING:
    from peteos.chatbot import Message
    from peteos.conversation.context import Context
    from peteos.conversation.session import Session
    from peteos.engine.runner import Runner
    from petekit.text.buffer_manager import Buffer


_logger = logging.getLogger("whisperer")


# -------------------------------------------------------------------------------------------------------------------- #
# Constants                                                                                                          #
# -------------------------------------------------------------------------------------------------------------------- #


_WHISPERER_MAX_INJECT_CHARS = 1000


_WHISPERER_SCOPE = "whisperer"
_WHISPERER_DOC_BUFFER = "doc:whisperer"

_WHISPERER_DOC = f"""\
## Whisperer

You can configure automatic message augmentation. This allows you
to inject information into a message programmatically. Each message
carries tags set by its sender. When a message arrives, each matching
tag in your configuration activates a whisperer. A whisperer enriches
the message before it enters the chat history.

### Enable Whisperers via Tags

Create a config buffer for each tag you want to activate:

```
{_WHISPERER_SCOPE}:<tag>:config
```

Example tags: `memory`, `docs`, `context`. A config buffer activates that
tag; dropping the buffer deactivates it. Use as many tags as you need.

**Wildcard** -- use `*` to fire on every message unconditionally, regardless of tags:

```
whisperer:*:config
```

This is useful for system-wide context that should always be present.

### Configuration Shape

```json
{{
  "filter": {{
    "roles": ["user"],
    "content_part_types": ["text", "thinking"]
  }},
  "fire_on_empty_input": false,
  "max_chars": 1000,
  "action": "<rag:memory:result:recent>",
  "action_config": {{}}
}}
```

- `filter.roles` -- only fire when the message role is in this
  list. If omitted, fire on every role.
- `filter.content_part_types` -- collect only content parts of these
  types for the action. If omitted, use the full message.
- `fire_on_empty_input` -- if true, fire even when no content parts
  matched. Default is false (skip silently).
- `max_chars` -- maximum characters to inject. If omitted, defaults to 1000.
- `action` -- what to inject. See below.
- `action_config` -- arbitrary dict passed to the action or custom function.
  Use it to provide action-specific parameters (e.g. query constraints,
  target identifiers, or any domain-specific context). Defaults to `{{}}`.

### How to Write the Action

Three ways to define what Whisperer does when it fires:

**Buffer pipe** -- read a buffer and inject its text as a new content part:

```
<buffer_name
```

Example: `<rag:memory:result:recent` injects the content of that
buffer. The buffer is read fresh at every message.

**Built-in action** -- use a pre-defined action that ships with Whisperer:

```
!<action_name>
```

Example: `!memory_recent` injects the three most recent memory entries.
Built-in actions receive `(message_dict, action_config)` -- the same
interface as custom functions. Ask the system what built-in actions are
available.

**Custom function** -- the bare function name as the action string.
The function is looked up in the function manager. It receives two
arguments: a copy of the message as a dict, and the `action_config`
dict from the config buffer. It must return a string, which is injected
directly into the message.

### Enable or Disable a Whisperer at Runtime

Create `{_WHISPERER_SCOPE}:<tag>:config` to enable. Drop the buffer to disable.
There is no run-time imperative API -- use the buffer tools to manage it.

### What Whisperer Changes

When an action fires, Whisperer injects a new content part into the
message. The message you see in the chat history has the augmentation
already applied.
"""


# -------------------------------------------------------------------------------------------------------------------- #
# Data Classes                                                                                                        #
# -------------------------------------------------------------------------------------------------------------------- #

# A hook that receives a copy of the message dict and returns a string, or None to skip.
WhispererHintHook = Callable[[dict], str | None]

# A hook that is dispatched by the runner -- it mutates the message in place.
WhispererHook = Callable[["AgentWhisperer", "Session", "Context"], Coroutine[Any, Any, Any | None]]


# -------------------------------------------------------------------------------------------------------------------- #
# Utils                                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #


def _get_config_from_buffer(
    config_buffer: Buffer,
) -> dict[str, Any] | str:
    """Read and parse the config from a buffer.

    Returns the parsed JSON dict on success.
    Returns an error string on any failure.
    """
    # Read the text content from the config buffer.
    config_text = config_buffer.get_text()
    if not config_text or config_text == "":
        return f"config buffer '{config_buffer.name}' is empty"

    # Parse the JSON content of the config buffer.
    try:
        return json.loads(config_text)
    except json.JSONDecodeError as e:
        return f"config buffer '{config_buffer.name}' is not valid JSON: {e}"


def _get_config_for_tag(
    agent_whisperer: AgentWhisperer,
    tag_name: str,
) -> dict[str, Any] | str:
    """Read and parse the config buffer for the given tag.

    Returns the parsed JSON dict on success.
    Returns an error string on any failure.
    """
    # Locate the config buffer for this tag.
    config_buffer = agent_whisperer.get_buffer(f"{_WHISPERER_SCOPE}:{tag_name}:config")
    if config_buffer is None:
        return f"config buffer for tag '{tag_name}' not found"

    # Delegate the read and parse to the buffer-level helper.
    return _get_config_from_buffer(config_buffer)


def _get_tag_from_config_buffer_name(config_buffer_name: str) -> str | None:
    """Extract the tag name from a config buffer name.

    Example: whisperer:memory:config -> memory
    Returns None if the buffer name is malformed.
    """
    # The prefix identifies the scope; the suffix marks the config slot.
    prefix = f"{_WHISPERER_SCOPE}:"
    suffix = ":config"

    # Reject names that do not start with the scope prefix or end with the config suffix.
    if not config_buffer_name.startswith(prefix) or not config_buffer_name.endswith(suffix):
        return None

    # Remove the prefix and suffix to isolate the tag.
    return config_buffer_name.removeprefix(prefix).removesuffix(suffix)


def _augment_message(
    message: Message,
    txt: str,
    badge: str,
    config: dict[str, Any],
) -> None:
    """Inject txt into the message, or emit hint if it exceeds max_chars."""
    # Guard against empty message content -- the message must have at least one part.
    if not message.content:
        _logger.error(f"[_augment_message] cannot augment message '{message._json_dict.get('id', '?')}': message has no content parts")
        return

    # Prepend a source prefix to identify where this content comes from.
    txt_with_badge = f"[{badge}]\n{txt}"

    # Capture the current length before modifying the message.
    old_len = len(message.content)

    # Read the per-tag max_chars from the config, falling back to the module default.
    max_chars = config.get("max_chars", _WHISPERER_MAX_INJECT_CHARS)

    # Find the last relevant content part to anchor the augmentation.
    content_part_types = config.get("filter", {}).get("content_part_types", [])
    if content_part_types:
        last_relevant = next(
            (cp for cp in reversed(message.content) if getattr(cp, "type", None) in content_part_types),
            None,
        )
    else:
        last_relevant = message.content[-1] if message.content else None

    # TODO: mutating last_relevant._json_dict["text"] bypasses ContentPart -- we should add a setter to ContentPart.text and use that here once it exists.
    # Decide whether to inject the text or emit the hint based on the limit.
    if old_len + len(txt_with_badge) <= max_chars:
        last_relevant._json_dict["text"] += f"\n\n{txt_with_badge}"
    else:
        overflow_hint = f"[{badge}] --\nResult too long for message injection"
        last_relevant._json_dict["text"] += f"\n\n{overflow_hint}"


def _get_buffer_whisperer_hook(
    agent_whisperer: AgentWhisperer,
    config: dict[str, Any],
) -> WhispererHook | str:
    """Build a WhispererHook that pipes a buffer into the message.

    The action string starts with '<' -- strip it to get the buffer name.
    Returns an error string if the buffer cannot be read.
    """
    # Strip the leading '<' to get the buffer name.
    action_str: str = config["action"]
    buffer_name = action_str[1:]

    # The hook looks up the buffer fresh on every fire so it reacts to buffer changes.
    async def _buffer_hook(
        agt: AgentWhisperer,
        session: Session,
        message: Message,
    ) -> None:
        # Look up the buffer fresh on every fire -- the current reference is used.
        buf = agt.get_buffer(buffer_name)
        if buf is None:
            _logger.error(f"[_get_buffer_whisperer_hook] buffer '{buffer_name}' not found")
            return

        # Read the buffer text and guard against empty buffers.
        txt = buf.get_text()
        if not txt:
            return

        _augment_message(
            message,
            txt,
            f"whisper buffer {buffer_name}",
            config
        )

    return _buffer_hook


def _get_function_whisperer_hook(
    agent_whisperer: AgentWhisperer,
    config: dict[str, Any],
) -> WhispererHook | str:
    """Build a WhispererHook from a custom function in the config.

    The action string is the bare function name.
    The function receives a copy of the message dict and returns a string.
    """
    # The action string is the bare function name.
    action_str: str = config["action"]

    # Look up the registered function.
    result = agent_whisperer.get_function(action_str)
    if not result["ok"]:
        return f"function '{action_str}' is not registered"

    func = result["callable"]

    async def _function_hook(
        agt: AgentWhisperer,
        session: Session,
        message: Message,
    ) -> None:
        # Build a copy of the message dict for the function to safely inspect.
        msg_dict = dict(message.raw_dict)

        # Read the action_config from the config; default to empty dict.
        action_config = config.get("action_config", {})

        # Call the function with the message copy and action_config -- it returns a string.
        fn_result = func(msg_dict, action_config)
        if asyncio.iscoroutine(fn_result):
            fn_result = await fn_result

        # The function must return a string; None means skip silently.
        if not isinstance(fn_result, str) or not fn_result:
            return
        txt = fn_result

        _augment_message(
            message,
            txt,
            f"whisper function {action_str}",
            config
        )

    return _function_hook


def _get_built_in_whisperer_hook(
    agent_whisperer: AgentWhisperer,
    config: dict[str, Any],
) -> WhispererHook | str:
    """Build a WhispererHook from a built-in action name in the config.

    The action string starts with '!' -- strip it to get the built-in name.
    Returns an error string if the built-in is not registered.
    """
    # Strip the leading '!' to get the built-in action name.
    action_str: str = config["action"]
    built_in_name = action_str[1:]

    # Look up the registered built-in action.
    action_fn = agent_whisperer._built_in_actions.get(built_in_name)
    if action_fn is None:
        return f"built-in action '{built_in_name}' is not registered"

    # Wrap the built-in with the same injection logic as the function hook.
    async def _built_in_hook(
        agt: AgentWhisperer,
        session: Session,
        message: Message,
    ) -> None:
        # Build a copy of the message dict for the built-in to safely inspect.
        msg_dict = dict(message.raw_dict)

        # Read the action_config from the config; default to empty dict.
        action_config = config.get("action_config", {})

        # Call the built-in action with the message dict and action_config -- it returns a string.
        bn_result = action_fn(msg_dict, action_config)
        if asyncio.iscoroutine(bn_result):
            bn_result = await bn_result

        # The built-in must return a string; None means skip silently.
        if not isinstance(bn_result, str) or not bn_result:
            return
        txt = bn_result

        _augment_message(
            message,
            txt,
            f"whisper built-in {built_in_name}",
            config
        )

    return _built_in_hook


def _get_whisperer_hook(
    agent_whisperer: AgentWhisperer,
    config: dict[str, Any],
) -> WhispererHook | str:
    """Build a WhispererHook from the config dict.

    The action string is dispatched based on its first character:
    - '<' -> buffer pipe -> _get_buffer_whisperer_hook
    - '!' -> built-in action -> _get_built_in_whisperer_hook
    - otherwise -> custom function name -> _get_function_whisperer_hook
    """
    # The action is the required key in the config.
    action_str: str = config["action"]

    # Dispatch to the appropriate builder based on the first character.
    first_char = action_str[0]
    if first_char == "<":
        return _get_buffer_whisperer_hook(agent_whisperer, config)

    if first_char == "!":
        return _get_built_in_whisperer_hook(agent_whisperer, config)

    else:
        return _get_function_whisperer_hook(agent_whisperer, config)


async def _handle_matching_whisperer(
    agent_whisperer: AgentWhisperer,
    session: Session,
    message: Message,
    config: dict[str, Any]
) -> None:
    """Apply one whisperer's action to a message after filter and content part checks."""

    # Check the filter.roles condition -- silently pass if the role is not in the allowed list.
    allowed_roles = config.get("filter", {}).get("roles", [])
    if allowed_roles and message.role not in allowed_roles:
        return

    # Resolve the action name to a callable that injects a hint into the message.
    action = _get_whisperer_hook(agent_whisperer, config)

    # Resolve the WhispererHook from the config -- returns a hook or an error string.
    content_part_types = config.get("filter", {}).get("content_part_types", [])
    if content_part_types:
        matching_parts = [
            cp for cp in message.content
            if getattr(cp, "type", None) in content_part_types
        ]
    else:
        matching_parts = list(message.content)

    # If no matching content parts and fire_on_empty_input is false, return silently.
    if not matching_parts and not config.get("fire_on_empty_input", False):
        return

    # Fire the whisperer's action -- handle both sync and async results.
    # Guard against any exception the action itself might raise.
    try:
        await action(agent_whisperer, session, message)
    except Exception as e:
        _logger.error(f"[_handle_matching_whisperer] action '{action}' raised: {e}")


# -------------------------------------------------------------------------------------------------------------------- #
# Hooks                                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #


async def _before_message_append(
    agent_whisperer: AgentWhisperer,
    runner: Runner,
    session: Session,
    message: Message,
) -> None:
    """Fires before a message is appended to the active context.

    Every tag on the message is matched against the whisperers
    registered in the agent. For each matching tag, the configured
    action fires and the message is potentially enriched before it
    enters the chat history.
    """
    # Extract the tags the sending entity set on this message.
    message_tags = message.tags

    # Filter to only those tags that have a configured whisperer.
    matching_tags = [t for t in message_tags if t in agent_whisperer._whisperers]
    # Prepend the wildcard tag if one is registered so it fires unconditionally.
    if "*" in agent_whisperer._whisperers:
        matching_tags.append("*")

    # Handle each matching whisperer in turn.
    for tag in matching_tags:
        # Load the Whisperer config from the config buffer for this tag.
        config = _get_config_for_tag(agent_whisperer, tag)
        if isinstance(config, str):
            _logger.error(f"[_before_message_append] {config}")
            continue

        # The handler decides whether to fire and fires the action,
        # potentially augmenting the message before it is committed.
        await _handle_matching_whisperer(agent_whisperer, session, message, config)


async def _before_update_config(
    agent_whisperer: AgentWhisperer,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Fires before a config buffer is updated. Return an error string to reject the update."""
    # Read and parse the new buffer text as JSON.
    new_text = new_buffer.get_text() or ""
    try:
        config = json.loads(new_text)
    except json.JSONDecodeError as e:
        return f"config buffer is not valid JSON: {e}"

    # Reject if the config is not a dictionary.
    if not isinstance(config, dict):
        return f"config must be a JSON object, got {type(config).__name__}"

    # Reject if the required action field is absent.
    if "action" not in config:
        return f"config is missing required field 'action'"

    # Validate filter.roles if present -- must be a list.
    roles = config.get("filter", {}).get("roles")
    if roles is not None and not isinstance(roles, list):
        return f"filter.roles must be a list, got {type(roles).__name__}"

    # Validate filter.content_part_types if present -- must be a list.
    cpt = config.get("filter", {}).get("content_part_types")
    if cpt is not None and not isinstance(cpt, list):
        return f"filter.content_part_types must be a list, got {type(cpt).__name__}"

    # Validate fire_on_empty_input if present -- must be a bool.
    fire_empty = config.get("fire_on_empty_input")
    if fire_empty is not None and not isinstance(fire_empty, bool):
        return f"fire_on_empty_input must be a bool, got {type(fire_empty).__name__}"

    # Validate max_chars if present -- must be an int.
    max_chars = config.get("max_chars")
    if max_chars is not None and not isinstance(max_chars, int):
        return f"max_chars must be an int, got {type(max_chars).__name__}"

    # Try to resolve the WhispererHook to catch action-level errors early.
    hook_check = _get_whisperer_hook(agent_whisperer, config)
    if isinstance(hook_check, str):
        return hook_check


async def _after_update_config(
    agent_whisperer: AgentWhisperer,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Fires after a config buffer is updated."""
    # Load the config from the new buffer.
    new_config = _get_config_from_buffer(new_buffer)
    if isinstance(new_config, str):
        _logger.error(f"[_after_update_config] {new_config}")
        return

    # Extract the tag from the new buffer name to key the whisperer.
    tag = _get_tag_from_config_buffer_name(new_buffer.name)
    if tag is None:
        return

    # Register the tag so it fires on matching messages.
    agent_whisperer._whisperers.add(tag)


async def _after_drop_config(
    agent_whisperer: AgentWhisperer,
    buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Fires after a config buffer is dropped."""
    # Extract the tag from the dropped config buffer name.
    tag = _get_tag_from_config_buffer_name(buffer.name)
    if tag is None:
        return

    # Remove the whisperer for this tag from the agent, if it exists.
    agent_whisperer._whisperers.discard(tag)


# -------------------------------------------------------------------------------------------------------------------- #
# Main Class                                                                                                           #
# -------------------------------------------------------------------------------------------------------------------- #


class AgentWhisperer(FunctionManager, AgenticObject):
    """I can configure automatic message augmentation for incoming messages to enrich them with relevant context.
    `doc:whisperer` holds details.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Holds every Whisperer by tag name, keyed from the config buffer names.
        self._whisperers: set[str] = set()

        # Holds built-in actions registered by subclasses, keyed by name.
        self._built_in_actions: dict[str, WhispererHintHook] = {}

        # Seed the doc buffer with the module documentation.
        self._create_buffer(_WHISPERER_DOC_BUFFER, text=_WHISPERER_DOC)

        # The runner fires before_message_append on every incoming message on
        # every running session -- this is where each matching whisperer decides
        # to fire and potentially augment the message.
        self.register_runner_hook("agent_whisperer", "before_message_append", _before_message_append)

        # Validate new config buffers before they are created; the guard checks
        # the JSON structure so a malformed config cannot be stored.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "agent_whisperer",
            f"{_WHISPERER_SCOPE}:.+:config",
            lambda bm, buf, args: _before_update_config(bm, None, buf, args),
            apply_to_existing=True,
        )

        # Validate config buffer updates before the buffer is changed; same
        # structure check as on create to prevent storing corrupted config.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_UPDATE_BUFFER,
            "agent_whisperer",
            f"{_WHISPERER_SCOPE}:.+:config",
            _before_update_config,
            apply_to_existing=True,
        )

        # Seed the whisperer registry when a new config buffer lands -- this
        # activates the tag so matching messages can fire it.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "agent_whisperer",
            f"{_WHISPERER_SCOPE}:.+:config",
            lambda bm, buf, args: _after_update_config(bm, None, buf, args),
            apply_to_existing=True,
        )

        # Refresh the whisperer when its config buffer changes -- the old
        # Whisperer object is replaced with a fresh one reflecting the new config.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_UPDATE_BUFFER,
            "agent_whisperer",
            f"{_WHISPERER_SCOPE}:.+:config",
            _after_update_config,
            apply_to_existing=True,
        )

        # Remove the whisperer when its config buffer is dropped -- the tag
        # is no longer active and will not fire on any further messages.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            "agent_whisperer",
            f"{_WHISPERER_SCOPE}:.+:config",
            _after_drop_config,
            apply_to_existing=True,
        )

    def register_built_in_action(self, name: str, action: WhispererHintHook) -> None:
        """Register a built-in action by name that subclasses can reference in config."""
        # Store the hook under the given name so _get_built_in_whisperer_hook can resolve it.
        self._built_in_actions[name] = action

    def unregister_built_in_action(self, name: str) -> None:
        """Remove a built-in action by name.

        If any configured whisperer still references this built-in action,
        an error is logged before removal.
        """
        # Guard against removing a built-in that is not registered.
        if name not in self._built_in_actions:
            _logger.error(f"[unregister_built_in_action] built-in action '{name}' is not registered")
            return

        # Check every configured whisperer to see if it references this built-in.
        for tag in self._whisperers:
            # Load the config for this tag; skip if the buffer is unreadable.
            config = _get_config_for_tag(self, tag)
            if not isinstance(config, dict):
                continue

            # Extract the action string to see if it references this built-in by name.
            action_str = config.get("action", "")
            if action_str == f"!{name}":
                # The built-in is still referenced -- log a warning but proceed with removal.
                _logger.error(
                    f"[unregister_built_in_action] built-in action '{name}' "
                    f"is still in use by whisperer '{tag}' -- removing anyway"
                )

        # Remove the built-in action.
        self._built_in_actions.pop(name, None)
