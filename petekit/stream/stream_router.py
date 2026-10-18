"""StreamRouter — routes stream data based on routing table option buffers."""

from __future__ import annotations

import inspect
import json
import logging
import re
from typing import Any, Callable, Dict

from .stream_buffer_manager import StreamBufferManager
from ..function.function_manager import FunctionManager
from ..text.buffer_manager import (
    Buffer,
    BufferManager,
    HookPoint,
)


_logger = logging.getLogger("stream_router")
CONDITION_FN = Callable[[FunctionManager, str, str, dict], bool]

_LOG_BUFFER = "system:log:stream_router"


def _is_condition_fn(callable_fn: Any) -> bool:
    """Return True if callable_fn has the condition function signature.

    Condition functions take (self/factory, str, str, dict) and return bool.
    The first argument (self) is ignored — it is always the buffer manager.
    The second and third arguments must be str. The fourth must be dict.
    The return annotation must be bool.
    """
    sig = inspect.signature(callable_fn)
    params = list(sig.parameters.values())
    if len(params) != 4:
        return False
    if params[1].annotation not in (str, "str"):
        return False
    if params[2].annotation not in (str, "str"):
        return False
    if params[3].annotation not in (dict, "dict", Dict, "Dict"):
        return False
    if sig.return_annotation not in (bool, "bool"):
        return False
    return True


# Group 1 = origin buffer name, group 2 = routing table name.
_ROUTING_TABLE_PATTERN = r"^(.+)\[routing_table:([^[]+)\]$"


# -------------------------------------------------------------------------------------------------------------------- #
# Utility                                                                       #
# -------------------------------------------------------------------------------------------------------------------- #

def _is_routing_table_option(name: str) -> bool:
    """Return True if name is a valid routing_table option buffer (has the [routing_table:NAME] suffix)."""
    return re.compile(_ROUTING_TABLE_PATTERN).fullmatch(name) is not None


def _routing_table_option_name(origin_name: str, table_name: str) -> str:
    """Return the routing_table option buffer name for a given origin and table."""
    return f"{origin_name}[routing_table:{table_name}]"


def _routing_table_hook_name(table_name: str) -> str:
    """Return the hook name for a routing table."""
    return f"routing_table:{table_name}"


def _fallthrough_name(origin_name: str, table_name: str) -> str:
    """Return the default fallthrough stream buffer name for a routing table."""
    return f"{origin_name}[routing_table:{table_name}]:fallthrough"


async def _safe_log(processor: Any, message: str) -> bool:
    """Write to the log buffer; escalate to _logger.error if the in-app log write fails."""
    result = await processor.write_buffer(_LOG_BUFFER, message)
    if not result.get("ok"):
        _logger.error(f"{message}  [log write failed: {result.get('error')}]")
        return False
    return True


_STREAM_ROUTER_DOC_BUFFER = "doc:stream_router"
_STREAM_ROUTER_DOC = f"""\
## Stream Router
You can route appended text conditionally to different output streams by creating
routing table option buffers. Conditions are registered as functions with signature
`(function_manager, stream, text, metadata) -> bool`.

### Routing Table Format

The routing_table is a list of routes, evaluated top-to-bottom. Each route is a
two-element list: a rule (a list of condition names) and an output stream name.
All conditions in a route's rule must return True for the route to fire and have
its text written to the output stream. When no route matches, the fallthrough
stream receives the text.

Create a "STREAM[routing_table:NAME]" option buffer with this JSON content:

```json
{{
    "routing_table": [
        [["condition1", "condition2"], "output_stream"],
        [["true"], "fallback_stream"]
    ],
    "fallthrough": "custom_fallthrough"
}}
```

- **routing_table**: a list of [rule, output_stream] routes
- **rule**: a list of condition names; all must return True for the route to fire
- **output_stream**: the destination stream when the rule matches
- **fallthrough**: optional; overrides the default fallthrough stream

### Routing Rules

- You can create multiple routing tables on the same input stream
- Each routing table routes appended text to exactly one output stream — the first
  matching route wins and stops further evaluation within that table
- Routes are evaluated top-to-bottom; the first fully-matching route wins and stops
- Prefix a condition name with "!" to negate its result
- Use "true" as a catch-all condition that always matches
- When no route matches, text is written to the configured fallthrough stream,
  or by default to "STREAM[routing_table:NAME]:fallthrough"
- Every routing table writes appended text to exactly one output stream — whether
  that stream is the matched route's output or the table's fallthrough stream.
  With N routing tables applied, the same text appears N times in total.

### Notifications

You can enable notifications on any output stream independently via the notifications
option buffer: "STREAM[routing_table:NAME]:fallthrough[notifications] = {{...}}"

### Feedback Stream

The feedback stream `{_LOG_BUFFER}` records all routing events and errors.
"""


def _parse_routing_table_option(name: str) -> tuple[str, str] | None:
    """Parse a routing_table option buffer name into (origin, table_name).

    E.g. "ORIGIN[routing_table:foo]" -> ("ORIGIN", "foo").
    Returns None if the name is not a routing_table option buffer.
    """
    # Attempt a full match against the routing_table pattern.
    match = re.compile(_ROUTING_TABLE_PATTERN).fullmatch(name)
    if not match:
        return None

    # Return the origin and table_name from the matched groups.
    return (match.group(1), match.group(2))


# -------------------------------------------------------------------------------------------------------------------- #
# Exception                                                                       #
# -------------------------------------------------------------------------------------------------------------------- #

class RoutingConditionError(Exception):
    def __init__(
        self,
        condition: str,
        routing_table: str | None = None,
        reason: str | None = None,
    ) -> None:
        # Store fields for inspection.
        self.condition = condition
        self.routing_table = routing_table
        self.reason = reason

        # Build the message parts list.
        parts = [f"condition '{condition}'"]
        if routing_table:
            parts.append(f"routing_table='{routing_table}'")
        if reason:
            parts.append(f"reason={reason}")

        # Format the exception message from the joined parts.
        super().__init__("; ".join(parts))


# -------------------------------------------------------------------------------------------------------------------- #
# Pattern hooks                                                               #
# -------------------------------------------------------------------------------------------------------------------- #

def _make_dispatch(
    table_name: str,
) -> Any:
    """Factory producing a routing dispatch hook closure for a specific table_name."""
    # Create the dispatch closure with table_name captured in the inner scope.
    async def dispatch(
        buffer_manager: BufferManager,
        old_buffer: Buffer,
        new_buffer: Buffer,
        args: dict[str, Any],
    ) -> str | None:
        # Call route_dispatch and catch errors to the log rather than propagating.
        try:
            return await route_dispatch(buffer_manager, old_buffer, new_buffer, table_name)
        except Exception as e:
            buffer_manager.write_buffer(_LOG_BUFFER, f"ROUTING ERROR: {e}")
            return None

    return dispatch


async def after_create_routing_table(
    processor: BufferManager,
    option_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Register the routing dispatch hook and create the fallthrough buffer when a routing_table option is created."""
    # Guard: reject if the buffer name is not a routing_table option buffer.
    if not _is_routing_table_option(option_buffer.name):
        _logger.error(f"[after_create_routing_table] '{option_buffer.name}' is not a routing_table option buffer")
        return

    # Parse the origin and table_name from the option buffer name.
    parsed = _parse_routing_table_option(option_buffer.name)
    if parsed is None:
        _logger.error(f"[after_create_routing_table] cannot parse '{option_buffer.name}'")
        return
    origin_name, table_name = parsed

    # Guard: reject if the origin buffer does not exist.
    origin = processor.get_buffer(origin_name)
    if origin is None:
        _logger.error(f"[after_create_routing_table] origin '{origin_name}' not found for '{option_buffer.name}'")
        return

    # Register the routing dispatch hook on the origin's after_update.
    hook_name = _routing_table_hook_name(table_name)
    origin.register_hook(
        HookPoint.AFTER_UPDATE_BUFFER,
        hook_name,
        _make_dispatch(table_name),
    )

    # Create the default fallthrough stream buffer.
    # The agent can override this via the "fallthrough" key in the routing table JSON.
    default_fallthrough_name = _fallthrough_name(origin_name, table_name)
    create_result = await processor.create_stream_buffer(default_fallthrough_name, text="")
    if not create_result.get("ok"):
        _logger.error(f"[after_create_routing_table] failed to create fallthrough '{default_fallthrough_name}': {create_result.get('error')}")


def before_update_routing_table(
    processor: BufferManager,
    old_option_buffer: Buffer | None,
    new_option_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Validate the routing table JSON schema before allowing the update.

    Runs as a BEFORE_UPDATE_BUFFER hook on subsequent updates, and via the
    BEFORE_CREATE_BUFFER bridge on initial creation.
    Return None to allow, or an error string to block.
    """
    # Guard: skip if the buffer name is not a routing_table option buffer.
    if not _is_routing_table_option(new_option_buffer.name):
        _logger.error(f"[before_update_routing_table] '{new_option_buffer.name}' is not a routing_table option buffer — hook fired on wrong buffer")
        return None

    # Parse the origin and table_name from the option buffer name.
    parsed = _parse_routing_table_option(new_option_buffer.name)
    if parsed is None:
        _logger.error(f"[before_update_routing_table] cannot parse '{new_option_buffer.name}' after pattern match — internal bug")
        return None
    origin_name, table_name = parsed

    # Validate the JSON is well-formed.
    try:
        new_text = new_option_buffer.get_text()
        config = json.loads(new_text)
    except json.JSONDecodeError:
        _logger.error(f"[before_update_routing_table] '{new_option_buffer.name}' is not valid JSON — update blocked")
        return f"'{new_option_buffer.name}' is not valid JSON"

    # Guard: routing_table must be a list.
    if not isinstance(config.get("routing_table"), list):
        return f"'{new_option_buffer.name}' routing_table must be a list"

    # Validate each entry: [conditions, output_stream].
    for i, entry in enumerate(config.get("routing_table", [])):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            return f"'{new_option_buffer.name}' entry[{i}] must be [conditions, output_stream]"
        conditions, output_stream = entry
        # Guard: conditions must be a list of non-empty strings.
        if not isinstance(conditions, list):
            return f"'{new_option_buffer.name}' entry[{i}] conditions must be a list"
        for j, cond in enumerate(conditions):
            if not isinstance(cond, str) or not cond:
                return f"'{new_option_buffer.name}' entry[{i}] conditions[{j}] must be a non-empty string"
        if not isinstance(output_stream, str) or not output_stream:
            return f"'{new_option_buffer.name}' entry[{i}] output_stream must be a non-empty string"

    # Guard: fallthrough, if present, must be a non-empty string.
    ft = config.get("fallthrough")
    if ft is not None and (not isinstance(ft, str) or not ft):
        return f"'{new_option_buffer.name}' fallthrough must be a non-empty string"

    return None


def after_drop_routing_table(
    processor: BufferManager,
    option_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Unregister the routing dispatch hook and drop the fallthrough buffer when a routing_table option is dropped."""
    # Guard: skip if the buffer name is not a routing_table option buffer.
    if not _is_routing_table_option(option_buffer.name):
        _logger.error(f"[after_drop_routing_table] '{option_buffer.name}' is not a routing_table option buffer — hook fired on wrong buffer")
        return

    # Parse the origin and table_name from the option buffer name.
    parsed = _parse_routing_table_option(option_buffer.name)
    if parsed is None:
        _logger.error(f"[after_drop_routing_table] cannot parse '{option_buffer.name}' after pattern match — internal bug")
        return
    origin_name, table_name = parsed

    # Unregister the dispatch hook from the origin buffer if it still exists.
    origin = processor.get_buffer(origin_name)
    if origin is not None:
        hook_name = _routing_table_hook_name(table_name)
        origin.unregister_hook(HookPoint.AFTER_UPDATE_BUFFER, hook_name)

    # Drop the default fallthrough buffer.
    default_fallthrough_name = _fallthrough_name(origin_name, table_name)
    processor.drop_buffer(default_fallthrough_name)


# -------------------------------------------------------------------------------------------------------------------- #
# Routing dispatch                                                             #
# -------------------------------------------------------------------------------------------------------------------- #

async def route_dispatch(
    processor: BufferManager,
    old_buffer: Buffer,
    new_buffer: Buffer,
    table_name: str,
) -> str | None:
    """Evaluate routing conditions against newly appended text and dispatch to the matched output.

    Registered as an after_update hook on the origin stream.
    If no condition matches, write to the configured or default fallthrough stream.
    """
    # Look up the routing_table option buffer for this origin and table.
    option_buffer_name = _routing_table_option_name(new_buffer.name, table_name)
    option_buffer = processor.get_buffer(option_buffer_name)
    if option_buffer is None:
        return None

    # Parse the routing table config from the option buffer JSON.
    try:
        config = json.loads(option_buffer.get_text())
    except json.JSONDecodeError:
        await _safe_log(
            processor,
            f"ROUTING ERROR: {option_buffer_name} is not valid JSON"
        )
        
        return None

    # Extract only the newly appended text.
    appended_text = new_buffer.get_text(start=len(old_buffer)-1)
    if not appended_text.strip():
        return None

    # Evaluate routing conditions top-to-bottom; first fully-matching entry wins.
    for condition_list, output_stream in config.get("routing_table", []):
        # Try each condition in this entry — all must match for the entry to fire.
        # A RoutingConditionError or a single condition failing means this entry is skipped.
        all_match = True
        for condition_name in condition_list:
            # Evaluate the condition — RoutingConditionError means fall-through for this entry.
            try:
                matched = evaluate_condition(
                    processor,
                    condition_name,
                    appended_text,
                    new_buffer.name,
                    table_name,
                )
            except RoutingConditionError as e:
                # A condition signaled fall-through for this entry — skip to the next entry.
                await _safe_log(
                    processor,
                    f"Routing Condition Error source={new_buffer.name} table={table_name} reason={e}",
                )
                all_match = False
                break
            except Exception as e:
                # An unexpected error during condition evaluation — abort routing entirely.
                await _safe_log(
                    processor,
                    f"CONDITION ERROR condition='{condition_name}' table='{table_name}' "
                    f"text='{appended_text[:40]}' error='{e}'",
                )
                return None

            # The condition returned False — this entry does not match, move to the next.
            if not matched:
                all_match = False
                break

        # All sub-conditions in this entry matched — write to the output stream and stop.
        if all_match:
            result = await processor.write_buffer(output_stream, appended_text)
            if not result.get("ok"):
                await _safe_log(
                    processor,
                    f"SINK ERROR: failed to write to '{output_stream}': {result.get('error')}",
                )
            return None

    # No entry matched — route to the fallthrough stream.
    # Use the configured fallthrough stream, or default to STREAM[routing_table:NAME]:fallthrough.
    fallthrough_stream = config.get("fallthrough")
    if not fallthrough_stream:
        fallthrough_stream = _fallthrough_name(new_buffer.name, table_name)

    # Log the fallthrough event.
    await _safe_log(
        processor,
        f"FALLTHROUGH source={new_buffer.name} table={table_name} -> {fallthrough_stream}",
    )

    # Write to the fallthrough stream.
    result = await processor.write_buffer(fallthrough_stream, appended_text)
    if not result.get("ok"):
        _safe_log(
            processor,
            f"FALLTHROUGH ERROR: failed to write to '{fallthrough_stream}': {result.get('error')}",
        )

    return None


def evaluate_condition(
    function_manager: FunctionManager,
    condition_name: str,
    text: str,
    stream: str,
    table_name: str,
) -> bool:
    """Evaluate a single named condition against the given text (synchronous).

    Conditions are registered in the FunctionManager with signature
    (function_manager, stream, text, metadata) -> bool.
    Prefix the name with "!" to negate the result.
    The built-in names "true" and "false" return their respective boolean values.
    """
    # Handle negation prefix by recursing with the inner name.
    if condition_name.startswith("!"):
        inner = condition_name[1:]
        return not evaluate_condition(function_manager, inner, text, stream, table_name)

    # Handle the built-in "true" and "false" conditions.
    if condition_name.lower() == "true":
        return True
    if condition_name.lower() == "false":
        return False

    # Look up the condition function from the FunctionManager.
    fn_result = function_manager.get_function(condition_name)
    if not fn_result.get("ok"):
        raise RoutingConditionError(
            condition=condition_name,
            routing_table=table_name,
            reason=f"not found in function manager: {fn_result.get('error')}",
        )

    method = fn_result["callable"]

    # Dispatch only if the method has the correct condition function signature.
    if not _is_condition_fn(method):
        sig = inspect.signature(method)
        params = list(sig.parameters.keys())
        raise RoutingConditionError(
            condition=condition_name,
            routing_table=table_name,
            reason=f"incompatible signature: {params}",
        )
    result = method(function_manager, stream, text, metadata={"routing_table": table_name})
    return result


# -------------------------------------------------------------------------------------------------------------------- #
# StreamRouter class                                                           #
# -------------------------------------------------------------------------------------------------------------------- #

class StreamRouter(FunctionManager, StreamBufferManager):
    """I can route appended text on streams conditionally to different output streams by creating routing table option buffers.
    "doc:stream_router" holds details.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Initialize the base classes.
        super().__init__(**kwargs)

        # Create the feedback buffer for routing results and errors.
        self._create_buffer(_LOG_BUFFER, text="")

        # Seed the documentation buffer so agents can discover routing tables without consulting the source.
        self._create_buffer(_STREAM_ROUTER_DOC_BUFFER, text=_STREAM_ROUTER_DOC)

        # Register the routing_table pattern hooks.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "routing_table_validated",
            _ROUTING_TABLE_PATTERN,
            lambda bm, buf, args: before_update_routing_table(bm, None, buf, args),
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "routing_table_created",
            _ROUTING_TABLE_PATTERN,
            after_create_routing_table,
        )
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_UPDATE_BUFFER,
            "routing_table_updated",
            _ROUTING_TABLE_PATTERN,
            before_update_routing_table,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            "routing_table_dropped",
            _ROUTING_TABLE_PATTERN,
            after_drop_routing_table,
        )
