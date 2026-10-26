from __future__ import annotations

import logging
from typing import Any

from peteos.oap.agentic_object import AgenticObject

from petekit.productivity.agent_whisperer import AgentWhisperer, WhispererHintHook
from petekit.productivity.rager import RAGer, _RAGER_DOC_BUFFER


_logger = logging.getLogger("whisperer_rager")


# -------------------------------------------------------------------------------------------------------------------- #
# Constants                                                                                                          #
# -------------------------------------------------------------------------------------------------------------------- #


_WHISPERER_RAGER_SCOPE = "whisperer_rager"
_WHISPERER_RAGER_DOC_BUFFER = "doc:whisperer_rager"
_QUERY_RAG_NAME = "query_rag"

_WHISPERER_RAGER_DOC = f"""\
## Built-in Action: !{_QUERY_RAG_NAME}

When a whisperer with this action fires, the action reads the message
text, queries the configured RAG index, and injects the most relevant
chunks directly into the message.

### Configuration

Set the action to `!{_QUERY_RAG_NAME}` in any whisperer config:

```json
{{
  "action": "!{_QUERY_RAG_NAME}",
  "action_config": {{
    "index": "my_index",
    "query_args": {{...}}
  }}
}}
```

**What it does** -- reads the message text, queries the RAG index,
injects the top matches into the message.

**Configure it** -- `action_config` holds:

- `index` -- required. The RAG index to search.
- `query_args` -- optional. Overrides index config defaults.
  See the `{_RAGER_DOC_BUFFER}` buffer for the full schema.

If `index` is absent, the action returns None and injects nothing.
"""


# -------------------------------------------------------------------------------------------------------------------- #
# Utils                                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #




# -------------------------------------------------------------------------------------------------------------------- #
# Hooks                                                                                                                #
# -------------------------------------------------------------------------------------------------------------------- #


def _make_rag_query_action(
    whisper_rager: WhisperRAGer,
) -> WhispererHintHook:
    """Factory: build a WhispererHintHook that queries a RAG index.

    The returned hook captures whisper_rager so it can call
    create_buffer and read_buffer without a self reference.
    """

    async def _query_rag(
        msg_dict: dict,
        action_config: dict,
    ) -> str | None:
        """Built-in action: query a RAG index and return the results.

        Reads the index name and query_args from action_config, creates a
        query buffer, and returns the result buffer content as a string.
        """
        # Guard against a missing index name in the config.
        index_name = action_config.get("index")
        if not index_name:
            _logger.error(f"[WhisperRAGer] !{_QUERY_RAG_NAME} called without 'index' in action_config")
            return None

        # Build the query args from config with no query-name suffix.
        query_name = "current_whisperer"
        query_text = str(msg_dict)
        query_args = action_config.get("query_args", {})
        query_data = {"search": query_text, "query_args": query_args}

        # Create the query buffer so the RAGer system picks it up and processes it.
        import json
        create_result = await whisper_rager.create_buffer(
            name=f"rag:{index_name}:query:{query_name}",
            text=json.dumps(query_data),
            overwrite=True,
        )
        if not create_result.get("ok", False):
            _logger.error(f"[WhisperRAGer] !{_QUERY_RAG_NAME} failed to create query buffer: {create_result.get('error', 'unknown')}")
            return None

        # Read and return the result buffer content.
        result = await whisper_rager.read_buffer(
            name=f"rag:{index_name}:result:{query_name}",
            raw=True,
        )
        if isinstance(result, dict) and not result.get("ok", False):
            _logger.error(f"[WhisperRAGer] !{_QUERY_RAG_NAME} failed to read result buffer: {result.get('error', 'unknown')}")
            return None

        return result if isinstance(result, str) else None

    return _query_rag




# -------------------------------------------------------------------------------------------------------------------- #
# Main Class                                                                                                         #
# -------------------------------------------------------------------------------------------------------------------- #


class WhisperRAGer(RAGer, AgentWhisperer, AgenticObject):
    """I combine RAG indexing and search with automatic message augmentation.
    `doc:whisperer_rager` holds details.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Seed the doc buffer with the module documentation.
        self._create_buffer(_WHISPERER_RAGER_DOC_BUFFER, text=_WHISPERER_RAGER_DOC)

        # Register the built-in RAG query action so `!query_rag` works in whisperer configs.
        self.register_built_in_action(_QUERY_RAG_NAME, _make_rag_query_action(self))
