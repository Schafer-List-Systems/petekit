from __future__ import annotations

import faiss
import json
import logging
import re
import numpy as np
import requests
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from peteos.oap.agentic_object import AgenticObject

from petekit.text.buffer_manager import AfterBufferDropped, AfterUpdateBuffer, Buffer, BufferManager, HookPoint, _STATUS_BUFFER


_logger = logging.getLogger("rager")


# -------------------------------------------------------------------------------------------------------------------- #
# TODO                                                                                                                 #
# -------------------------------------------------------------------------------------------------------------------- #

# Index Status
# The status buffer holds runtime information about the index:
# 
# rag:<index_name>:status
# {
#   "buffers": 0,
#   "chunks": 0,
#   "characters": 0,
#   "tokens": 0,
#   "view_queries": 0,
#   "last_update": 1234567890.0,
#   "avg_query_time": 0.05,
#   "queries_per_second": 0.0,
#   "is_populated": false
# }
# buffers -- number of source buffers currently indexed
# chunks -- total number of chunks across all buffers
# characters -- total character count of indexed content
# tokens -- approximate token count of indexed content
# view_queries -- number of view query buffers registered on this index
# last_update -- Unix timestamp of the last index update
# avg_query_time -- average time in seconds to compute query results
# queries_per_second -- average query frequency on this index
# is_populated -- boolean: true if the index has at least one buffer indexed
# Using Index Status for Debugging:
# 
# If buffers = 0, your buffer_pattern likely doesn't match any buffers
# If last_update is very old, the index hasn't refreshed
# If avg_query_time is high with eager view, you may want to consider snapshot mode
# Query Status
# The status buffer holds runtime information about the query:
# 
# rag:<index_name>:status
# {
#   "last_query_time": 1234567890.0,
#   "queries_executed": 12,
#   "avg_query_time": 0.1,
#   "query_mode": "snapshot",
#   "query_timing": "lazy",
#   "is_active": true
# }
# last_query_time -- Unix timestamp of the last query execution (null if never queried)
# queries_executed -- total number of queries executed
# avg_query_time -- average time in seconds to compute results
# query_mode -- the mode used (snapshot or view)
# query_timing -- the timing used (eager or lazy)
# is_active -- true if the query is currently being evaluated (view mode), false if frozen (snapshot mode)
# Using Query Status for Debugging:
# 
# If last_query_time is null, the query hasn't been executed (lazy mode waiting for first read)
# If queries_per_second is unexpectedly high with eager view, you may be paying unnecessary overhead
# If is_active = false but you expected updates, check your query mode is view, not snapshot
# Practical Examples
# Example 1: Query returns zero results
# 
# Check index status: buffers = 0 → your pattern is wrong
# Check index status: buffers > 0 → your search term may be too specific
# Example 2: Query taking too long
# 
# Check query status: avg_query_time = 5.0 seconds
# Consider switching to lazy snapshot if you don't need immediate results
# Check index status: if buffers = 10000+, consider narrowing your buffer_pattern
# Example 3: Eager view with high overhead
# 
# Check index status: queries_per_second = 100
# With eager view, every source change triggers recomputation
# Consider snapshot mode if updates aren't needed in real-time

# -------------------------------------------------------------------------------------------------------------------- #
# Constants                                                                                                          #
# -------------------------------------------------------------------------------------------------------------------- #

_RAGER_SCOPE = "rag"
_RAGER_DOC_BUFFER = "doc:rager"

# Side-index files: map FAISS integer IDs to domain string labels and back.
_RAGER_IDS_TO_LABELS_FILE = "ids_to_labels.json"
_RAGER_IDS_TO_VECTORS_FILE = "ids_to_vectors.json"
_RAGER_LABELS_TO_IDS_FILE = "labels_to_ids.json"
_RAGER_INDEX_FILE = "index.faiss"
_RAGER_INDEX_CONFIG_FILE = "index.json"
_RAGER_DEFAULT_QUERY_MODE = "snapshot"
_RAGER_DEFAULT_K = 10
_RAGER_DEFAULT_TIMING = "eager"
_RAGER_DEFAULT_THRESHOLD = float("inf")
_RAGER_EMBED_TIMEOUT = 15

_RAGER_DOC = f"""\
## RAGer

You can index buffers into one or more named RAG indexes and search them
to retrieve relevant chunks. Every index is completely autonomous -- its
configuration, its source buffer selection, and its queries are all stored
in buffers, so you can manage and inspect every aspect of the system
through the buffer tools.

### Index Configuration

Each index lives under the `{_RAGER_SCOPE}:<index_name>:` namespace. Its
configuration buffer is:

```
{_RAGER_SCOPE}:<index_name>:config
```

```json
{{
  "buffer_pattern": "<regex>",
  "model": "text-embedding-nomic-embed-text-v1.5",
  "embedding": {{
    "via": "api",
    "url": "https://my-embed-endpoint/v1/embed",
    "model": "text-embedding-nomic-embed-text-v1.5"
  }},
  "query_args": {{}}
}}
```

- `buffer_pattern`  --  required; regex that matches all buffer names to index
- `model`           --  required; the model to use for embedding texts
- `embedding`       --  required; how to embed texts
- `embedding.via`   --  required; `"api"` for HTTP or `"mcp"` for MCP server
- `embedding.url`   --  required when `via` is `"api"`; the HTTP endpoint URL
- `embedding.model` --  required; the model name for the embedder
- `query_args`      --  optional; default parameters for every query (see the Queries section)

### Index Registration

Create the config buffer to register a new named index. Dropping the config
buffer removes the index entirely and all its chunks.

### Index Status

The status buffer holds runtime information about the index:

```
{_RAGER_SCOPE}:<index_name>:status
```

```json
{{
  "buffers": 0,
  "chunks": 0,
  "characters": 0,
  "tokens": 0,
  "view_queries": 0,
  "last_update": 1234567890.0
}}
```

- `buffers`      --  number of source buffers currently indexed
- `chunks`       --  total number of chunks across all buffers
- `characters`  --  total character count of all indexed content
- `tokens`      --  approximate token count of indexed content
- `view_queries` --  number of `view` query buffers registered on this index
- `last_update` --  Unix timestamp of the last index update

### Queries

Create a query buffer to define what to search for:

```
{_RAGER_SCOPE}:<index_name>:query:<query_name>
```

```json
{{
  "search": "<search string>",
  "query_args": {{
    "limit": 10,
    "threshold": null,
    "timing": "lazy",
    "mode": "snapshot"
  }}
}}
```

- `search`  --  the search string
- `query_args`  --  optional; override index config defaults for this query

### Query Arguments

The `query_args` block holds overridable defaults. If omitted in a specific query buffer,
the values from the index config's `query_args` are used instead.

- `limit`     --  optional; cap the number of results (default from config)
- `threshold` --  optional; maximum distance to include a result (default Infinity)
- `timing`   --  optional; `eager` or `lazy` (default `lazy`)
- `mode`     --  optional; `snapshot` or `view` (default `snapshot`)

### Query Modes

The `timing` and `mode` options are orthogonal and combine into four modes:

| | Snapshot | View |
|---|---|---|
| **Eager** | Computed when the query buffer is created or updated. Cached in the result buffer. Source buffer changes within the index do not trigger recompute. | Computed immediately on query creation. Cached in the result buffer. Recomputed on **every** buffer change within the index's `buffer_pattern` -- the result buffer stays current at all times. |
| **Lazy** | Computed on the first read of the result buffer. Cached. Subsequent reads return the cached result. | Computed fresh on **every** read of the result buffer. No result accumulation -- the buffer is left empty after each read. |

- **eager snapshot**  --  low read-time cost; recomputes when the query buffer is created or updated; source buffer changes within the index do not trigger recompute; good for expensive embeddings where freshness at read time matters
- **eager view**      --  result buffer is always current without reading it; a change log can react to each refresh; highest compute overhead if many buffers change frequently
- **lazy snapshot**   --  minimal overhead when the result is not being read; recomputes only when the result buffer is actually read
- **lazy view**       --  always returns fresh results on every read; no stale cache; highest read-time cost since it always re-embeds

### When to Use Which Mode

- **Snapshot** — Use this when you want stable, reproducible results. The result
  is fixed once and not recomputed on source buffer changes.

  - **Eager snapshot** — The result is computed when the query buffer is created
    or updated. Choose this when the result should reflect the index state at the
    time you wrote the query.

  - **Lazy snapshot** — The result is computed on the first read of the result
    buffer. Choose this when you want to delay the search until the result is
    actually needed — for example when the query is expensive and you are not
    sure it will be read at all.

- **View** — Use this when you want the result buffer to always reflect the
  current state of the index. The results update automatically when source buffers
  change. This is useful for dashboards, watchers, or any agent that reacts to
  result buffer changes.

  - **Eager view** — The result is recomputed on every source buffer change but
    never on read. Choose this when source buffers change rarely but you read
    often. It also enables automatic reactions: because the result updates on
    every index change, a separate process can watch the result buffer for
    changes and act on them — something that is not possible with lazy view.

  - **Lazy view** — The result is recomputed on every read. Choose this when
    source buffers change often but you read rarely. Changes to source buffers
    do not automatically update the result — the result is only refreshed when
    you read it. Automatic reactions to index changes are not possible with
    lazy view.

When you create or update a query buffer, a companion result buffer is created or updated automatically:

```
{_RAGER_SCOPE}:<index_name>:result:<query_name>
```

### Results

The result buffer holds a list of matching source buffers, sorted by
relevance (best matches first):

```json
[
  {{
    "buffer": "source:buffer/name",
    "distance": 12.47
  }}
]
```

Each entry is the name of the matching buffer. A lower distance value means higher relevance.

### How Indexing Works

Whenever a source buffer that matches the index's `buffer_pattern` is
created or updated, the index is refreshed automatically -- its old
chunks for that buffer are replaced with fresh ones. This keeps the
index current without any background polling.
"""


# -------------------------------------------------------------------------------------------------------------------- #
# Data Classes                                                                                                        #
# -------------------------------------------------------------------------------------------------------------------- #


@dataclass
class RAGIndex:
    """Holds a live FAISS index and its side-indexes for one named index."""
    name: str
    config: dict[str, Any]
    index: faiss.Index
    ids_to_labels: dict[int, str]
    ids_to_vectors: dict[int, list[float]]
    labels_to_ids: dict[str, int]
    eager_views: set[str]


# -------------------------------------------------------------------------------------------------------------------- #
# Common Utils                                                                                                        #
# -------------------------------------------------------------------------------------------------------------------- #


def _get_persistence_dir(rager: RAGer) -> Path:
    """Return the RAGer persistence directory inside the RAGer's instance dir."""
    return rager.instance_dir / "rager"


def _get_index_dir(rager: RAGer, index_name: str) -> Path:
    """Return the directory path for the index with the given name."""
    return _get_persistence_dir(rager) / index_name


def _get_index_config(
    rager: RAGer,
    index_name: str,
) -> dict[str, Any] | None:
    """Read and parse the config buffer for the given index.

    Returns the parsed JSON dict on success, None if the buffer is absent.
    """
    # Locate the config buffer for this index.
    config_buffer = rager.get_buffer(f"{_RAGER_SCOPE}:{index_name}:config")
    if config_buffer is None:
        return None

    # Parse the JSON content of the config buffer.
    config_text = config_buffer.get_text()
    if not config_text:
        _logger.error(f"[RAGer] config buffer for '{index_name}' is empty")
        return None

    try:
        return json.loads(config_text)
    except json.JSONDecodeError:
        _logger.error(f"[RAGer] config buffer for '{index_name}' is not valid JSON")
        return None


def _query_buffer_name_from_result(result_buffer_name: str) -> str:
    """Derive the query buffer name from a result buffer name.

    Example: rag:docs:result:topic -> rag:docs:query:topic
    """
    return result_buffer_name.replace(":result:", ":query:")


def _index_name_from_query(query_buffer_name: str) -> str | None:
    """Extract the index name from a query buffer name.

    Example: rag:docs:query:topic -> docs
    Returns None if the buffer name is malformed.
    """
    parts = query_buffer_name.rsplit(":", 3)
    if len(parts) < 4:
        _logger.error(f"[RAGer] query buffer name '{query_buffer_name}' is malformed")
        return None
    return parts[1]


def _index_name_from_config(config_buffer_name: str) -> str | None:
    """Extract the index name from a config buffer name.

    Example: rag:docs:config -> docs
    Returns None if the buffer name is malformed.
    """
    parts = config_buffer_name.rsplit(":", 2)
    if len(parts) < 3:
        _logger.error(f"[RAGer] config buffer name '{config_buffer_name}' is malformed")
        return None
    return parts[1]


def _result_buffer_name_from_query(query_buffer_name: str) -> str:
    """Derive the result buffer name from a query buffer name.

    Example: rag:docs:query:topic -> rag:docs:result:topic
    """
    return query_buffer_name.replace(":query:", ":result:")


def _get_query_dict(
    rager: RAGer,
    query_buffer_name: str,
) -> dict[str, Any] | None:
    """Read the query buffer and return its query dictionary.

    Parses the buffer as JSON if valid. Otherwise builds a dict with the
    buffer text as `search` and default arguments from the index config.

    Returns the query dict on success, None on failure.
    """
    # Locate and read the query buffer.
    query_buffer = rager.get_buffer(query_buffer_name)
    if query_buffer is None:
        _logger.error(f"[RAGer] query buffer '{query_buffer_name}' not found")
        return None

    # Read the text content from the query buffer.
    query_text = query_buffer.get_text() or ""

    # Attempt to parse the query buffer as JSON; fall back to raw text.
    try:
        query_data: dict[str, Any] = json.loads(query_text)
    except json.JSONDecodeError:
        query_data = {"search": query_text}

    # Extract the index name from the query buffer name to load the config.
    # The helper logs the error if the buffer name is malformed.
    index_name = _index_name_from_query(query_buffer_name)
    if index_name is None:
        return None

    # Load the index config to get the default query arguments.
    config = _get_index_config(rager, index_name)
    if config is None:
        _logger.error(f"[RAGer] config for index '{index_name}' not found")
        return None

    # Get the config's query_args as the base defaults.
    config_args = config.get("query_args", {}).copy()

    # Override with the query buffer's own query_args.
    config_args.update(query_data.get("query_args", {}))

    # Splice the merged query_args back into the query buffer's full dict.
    query_data["query_args"] = config_args

    # Consolidate defaults centrally -- ensure all four fields are always present.
    config_args.setdefault("limit", _RAGER_DEFAULT_K)
    config_args.setdefault("threshold", _RAGER_DEFAULT_THRESHOLD)
    config_args.setdefault("timing", _RAGER_DEFAULT_TIMING)
    config_args.setdefault("mode", _RAGER_DEFAULT_QUERY_MODE)

    return query_data


def _get_results_for_query(
    rager: RAGer,
    rag_index: RAGIndex,
    query_buffer_name: str,
) -> dict[str, Any]:
    """Read a query buffer and return the search results without storing them.

    Uses _get_query_dict to parse the buffer and fill defaults.
    Returns {"ok": True, "results": [...]} on success.
    Returns {"ok": False, "error": ...} on failure.
    """
    # Delegate parsing and default filling to the dedicated helper.
    query_dict = _get_query_dict(rager, query_buffer_name)
    if query_dict is None:
        return {"ok": False, "error": f"query buffer '{query_buffer_name}' not found or index config absent"}

    # Extract the search string; it must be present.
    search = query_dict.get("search", "")
    if not search:
        return {"ok": False, "error": "query buffer has no 'search' field"}

    # Read limit and threshold from the pre-filled query_args.
    limit = query_dict["query_args"]["limit"]
    threshold = query_dict["query_args"]["threshold"]

    # Embed the search string into a query vector.
    embed_result = _embed(rager, rag_index, search)
    if not embed_result["ok"]:
        return {"ok": False, "error": embed_result["error"]}
    query_vector = embed_result["vector"]

    # Query the index with the query vector and keep the result.
    query_result = _query_index(rager, rag_index, query_vector, limit)
    if not query_result["ok"]:
        return query_result

    # Filter out results whose distance exceeds the threshold.
    if threshold < float("inf"):
        results = [r for r in query_result["results"] if r["distance"] <= threshold]
        query_result["results"] = results

    return query_result


def _refresh_result(
    rager: RAGer,
    rag_index: RAGIndex,
    query_buffer: Buffer,
    result_buffer: Buffer,
) -> str | None:
    """Run the query and write its results into the result buffer.

    Uses the query buffer name to drive the search and the result buffer
    object to hold the rendered output.
    """
    # Derive the query buffer name from the buffer object itself.
    query_buffer_name = query_buffer.name

    # Execute the search and hold the results.
    query_result = _get_results_for_query(rager, rag_index, query_buffer_name)
    if not query_result["ok"]:
        _logger.error(
            f"[RAGer] query execution failed for '{query_buffer_name}': "
            f"{query_result.get('error', 'unknown')}"
        )
        return query_result.get("error")

    # Render the results as JSON and write them into the result buffer.
    results = query_result.get("results", [])
    result_json = json.dumps(results, indent=2, ensure_ascii=False)
    result_buffer.set_text(result_json)


def _read_results_for_query(
    rager: RAGer,
    query_buffer_name: str,
) -> list[dict[str, Any]] | None:
    """Read the cached result buffer for the given query and return its results list.

    Parses the result buffer as JSON. Returns the list on success, None on failure.
    """
    # Derive the result buffer name from the query buffer name.
    result_buffer_name = _result_buffer_name_from_query(query_buffer_name)

    # Locate and read the result buffer.
    result_buffer = rager.get_buffer(result_buffer_name)
    if result_buffer is None:
        _logger.error(f"[RAGer] result buffer '{result_buffer_name}' not found")
        return None

    # Parse the result buffer as JSON.
    result_text = result_buffer.get_text()
    if not result_text:
        return None

    # Deserialise the JSON results list; log the error and return None if malformed.
    try:
        return json.loads(result_text)
    except json.JSONDecodeError:
        _logger.error(f"[RAGer] result buffer '{result_buffer_name}' is not valid JSON")
        return None


# -------------------------------------------------------------------------------------------------------------------- #
# Index Utils                                                                                                        #
# -------------------------------------------------------------------------------------------------------------------- #


def _load_index(
    index_file: Path,
    index_path: Path,
    config: dict[str, Any] | None = None,
) -> RAGIndex | str:
    """Load a persisted index and its on-disk configuration.

    If config is provided, validates create_on_gpu and dim against the
    on-disk config and warns or errors accordingly.

    Returns the RAGIndex on success.
    Returns an error string if the dimension in config does not match
    the dimension of the index on disk.
    Raises on load failure.
    """
    # Load the persisted index and return early on load failure.
    try:
        index = faiss.read_index(str(index_file))
    except Exception:
        _logger.error(f"[RAGer] failed to load index from disk")
        raise

    # Load the on-disk configuration for the next validation steps.
    meta_file = index_path / _RAGER_INDEX_CONFIG_FILE
    on_disk_config: dict[str, Any] = {}
    if meta_file.exists():
        with open(meta_file) as f:
            on_disk_config = json.load(f)

    # Load the integer-to-label and label-to-integer side-index files.
    ids_to_labels: dict[int, str] = {}
    ids_to_labels_file = index_path / _RAGER_IDS_TO_LABELS_FILE
    if ids_to_labels_file.exists():
        # Parse the integer-keyed JSON back to a typed dict.
        with open(ids_to_labels_file) as f:
            raw_ids = json.load(f)
            ids_to_labels = {int(k): v for k, v in raw_ids.items()}

    labels_to_ids: dict[str, int] = {}
    labels_to_ids_file = index_path / _RAGER_LABELS_TO_IDS_FILE
    if labels_to_ids_file.exists():
        # Parse the label-keyed JSON back to a typed dict.
        with open(labels_to_ids_file) as f:
            labels_to_ids = json.load(f)

    # Load the integer-ID to embedding-vector side-index.
    ids_to_vectors: dict[int, list[float]] = {}
    ids_to_vectors_file = index_path / _RAGER_IDS_TO_VECTORS_FILE
    if ids_to_vectors_file.exists():
        # Parse the integer-keyed JSON back to a typed dict.
        with open(ids_to_vectors_file) as f:
            raw_vectors = json.load(f)
            ids_to_vectors = {int(k): v for k, v in raw_vectors.items()}

    # Validate against the provided config when one is passed.
    if config is not None:
        # Extract the create_on_gpu flag and the expected dimension.
        create_on_gpu = config.get("create_on_gpu", False)
        dim = config["dim"]

        # Warn if the GPU flag on disk differs from the one just requested.
        on_disk_gpu = on_disk_config.get("create_on_gpu", False)
        if on_disk_gpu != create_on_gpu:
            _logger.warning(
                f"[RAGer] index was created with create_on_gpu={on_disk_gpu}"
                f" but load request specified create_on_gpu={create_on_gpu}"
                f" -- loaded as {on_disk_gpu}"
            )

        # Error if the vector dimension on disk differs from the one in config.
        on_disk_dim = on_disk_config.get("dim", 0)
        if on_disk_dim != dim:
            _logger.error(
                f"[RAGer] index was created with dim={on_disk_dim}"
                f" but load request specified dim={dim} -- refusing to load"
            )
            return "dimension mismatch for index"

        # Warn if the embedding model on disk differs from the one just requested.
        on_disk_model = on_disk_config.get("model")
        requested_model = config.get("model")
        if on_disk_model != requested_model:
            _logger.warning(
                f"[RAGer] index was created with model={on_disk_model}"
                f" but load request specified model={requested_model}"
                f" -- loaded as {on_disk_model}"
            )

    return RAGIndex(
        name=index_path.name,
        config=on_disk_config,
        index=index,
        ids_to_labels=ids_to_labels,
        ids_to_vectors=ids_to_vectors,
        labels_to_ids=labels_to_ids,
        eager_views=set(),
    )


def _create_index(
    rager: RAGer,
    index_name: str,
    config: dict[str, Any],
) -> RAGIndex:
    """Create a fresh index, persist all its files, and return the RAGIndex."""
    # Derive the index directory and file path from the index name.
    index_path = _get_index_dir(rager, index_name)
    index_file = index_path / _RAGER_INDEX_FILE

    # Extract the vector dimension and GPU flag from the config.
    dim = config["dim"]
    create_on_gpu = config.get("create_on_gpu", False)

    # Attempt the GPU variant first if requested; log the error and fall through.
    if create_on_gpu:
        try:
            index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
        except Exception:
            _logger.error(f"[RAGer] GPU index creation failed for '{index_name}'")

    # Fall back to the CPU variant; re-raise if even that fails.
    try:
        index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
    except Exception:
        _logger.critical(f"[RAGer] CPU index creation failed for '{index_name}'")
        raise

    # Persist the freshly created index to disk.
    # Re-raise if the write fails -- the index is in memory and the
    # caller must decide how to handle an incomplete persist.
    faiss.write_index(index, str(index_file))

    # Persist the index configuration to a companion JSON file so the
    # config is available at reload time without consulting the config buffer.
    meta_file = index_path / _RAGER_INDEX_CONFIG_FILE
    try:
        with open(meta_file, "w") as f:
            json.dump(config, f)
    except Exception:
        _logger.error(f"[RAGer] failed to persist index config for '{index_name}'")
        raise

    # Initialise and persist the empty integer-to-label side-index.
    ids_to_labels: dict[int, str] = {}
    ids_to_labels_file = index_path / _RAGER_IDS_TO_LABELS_FILE
    try:
        with open(ids_to_labels_file, "w") as f:
            json.dump(ids_to_labels, f)
    except Exception:
        _logger.error(f"[RAGer] failed to persist ids_to_labels for '{index_name}'")
        raise

    # Initialise and persist the empty integer-ID to embedding-vector side-index.
    ids_to_vectors: dict[int, list[float]] = {}
    ids_to_vectors_file = index_path / _RAGER_IDS_TO_VECTORS_FILE
    try:
        with open(ids_to_vectors_file, "w") as f:
            json.dump(ids_to_vectors, f)
    except Exception:
        _logger.error(f"[RAGer] failed to persist ids_to_vectors for '{index_name}'")
        raise

    # Initialise and persist the empty label-to-integer side-index.
    labels_to_ids: dict[str, int] = {}
    labels_to_ids_file = index_path / _RAGER_LABELS_TO_IDS_FILE
    try:
        with open(labels_to_ids_file, "w") as f:
            json.dump(labels_to_ids, f)
    except Exception:
        _logger.error(f"[RAGer] failed to persist labels_to_ids for '{index_name}'")
        raise

    return RAGIndex(
        name=index_name,
        config=config,
        index=index,
        ids_to_labels=ids_to_labels,
        ids_to_vectors=ids_to_vectors,
        labels_to_ids=labels_to_ids,
        eager_views=set(),
    )


def _create_or_load_index(
    rager: RAGer,
    index_name: str,
    config: dict[str, Any],
) -> RAGIndex | str:
    """Create or load an index with the given name inside the persistence directory.

    Returns the RAGIndex on success.
    Returns an error string on failure.
    """
    # Derive the directory path for this index and ensure it exists on disk.
    index_path = _get_index_dir(rager, index_name)
    index_path.mkdir(parents=True, exist_ok=True)

    # Derive the index file path inside that directory.
    index_file = index_path / _RAGER_INDEX_FILE

    # If the index file already exists, load it from disk.
    if index_file.exists():
        # Delegate the load and config validation to the helper.
        rag_index = _load_index(index_file, index_path, config)

        # Propagate the error string if the helper failed.
        if isinstance(rag_index, str):
            return rag_index

        return rag_index

    # File does not exist yet -- delegate creation to the helper.
    rag_index = _create_index(rager, index_name, config)

    return rag_index


def _destroy_index(rager: RAGer, rag_index: RAGIndex) -> dict[str, Any]:
    """Destroy the index inside the persistence directory."""
    # Derive the index directory path.
    index_path = _get_index_dir(rager, rag_index.name)

    # Delete the persisted index file and its configuration.
    faiss_file = index_path / _RAGER_INDEX_FILE
    if faiss_file.exists():
        faiss_file.unlink()

    # Delete the on-disk configuration if it exists.
    meta_file = index_path / _RAGER_INDEX_CONFIG_FILE
    if meta_file.exists():
        meta_file.unlink()

    # Delete the integer-to-label side-index if it exists.
    ids_to_labels_file = index_path / _RAGER_IDS_TO_LABELS_FILE
    if ids_to_labels_file.exists():
        ids_to_labels_file.unlink()

    # Delete the label-to-integer side-index if it exists.
    labels_to_ids_file = index_path / _RAGER_LABELS_TO_IDS_FILE
    if labels_to_ids_file.exists():
        labels_to_ids_file.unlink()

    # Delete the integer-ID to embedding-vector side-index if it exists.
    ids_to_vectors_file = index_path / _RAGER_IDS_TO_VECTORS_FILE
    if ids_to_vectors_file.exists():
        ids_to_vectors_file.unlink()

    # Remove the now-empty index directory.
    if index_path.is_dir() and not any(index_path.iterdir()):
        index_path.rmdir()

    return {"ok": True}


def _get_free_id(rag_index: RAGIndex) -> int:
    """Return a pseudo-random integer ID that is not currently in use by the index.

    Generates random candidates until one is unused in ids_to_labels.
    """
    # Build a set of every ID that is already in use so lookup is O(1).
    import random
    used_ids = set(rag_index.ids_to_labels.keys())

    # Keep generating candidates until one is not in use.
    while True:
        # Draw a candidate from the full 64-bit unsigned space.
        candidate = random.randint(0, 2**63 - 1)

        # Return the candidate if it is unused.
        if candidate not in used_ids:
            return candidate


def _upsert_vector(
    rager: RAGer,
    rag_index: RAGIndex,
    vector: list[float],
    label: str,
) -> dict[str, Any] | str:
    """Insert a vector with a domain label into the index.

    If the label already exists in the index, the old vector is removed
    and the new one takes its place.

    Returns {{"ok": True, "vector": [...], "faiss_id": ...}} on success.
    Returns an error string on insert failure.
    """
    # Check whether this label is already in the index.
    if label in rag_index.labels_to_ids:
        # Remove the stale vector first so the index stays consistent.
        remove_err = _remove_vector(rager, rag_index, label)
        if isinstance(remove_err, str):
            return remove_err

    # Allocate a free integer ID for the new vector.
    faiss_id = _get_free_id(rag_index)

    # Add the new vector to the FAISS index; capture the assigned integer ID.
    # Some FAISS bindings return a new index rather than modifying in place.
    # Reassign to capture the returned index; discard the ID from ntotal since add()
    # may also return an ID that was being double-assigned.
    try:
        vec = np.array([vector], dtype=np.float32)
        id = np.array([faiss_id], dtype=np.int64)
        rag_index.index.add_with_ids(vec, id)
    except Exception as e:
        _logger.error(f"[RAGer] failed to add vector for label '{label}': {e}")
        raise

    # Update the integer-to-label side-index with the new mapping.
    rag_index.ids_to_labels[faiss_id] = label

    # Update the label-to-integer side-index with the new mapping.
    rag_index.labels_to_ids[label] = faiss_id

    # Populate the integer-ID to embedding-vector side-index with the raw vector.
    rag_index.ids_to_vectors[faiss_id] = vector

    return {"ok": True, "faiss_id": faiss_id, "vector": vector}


def _remove_vector(
    rager: RAGer,
    rag_index: RAGIndex,
    label: str,
) -> dict[str, Any] | str:
    """Remove the vector identified by label from the index.

    Returns {{"ok": True, "faiss_id": ..., "vector": [...]}} on success.
    Returns an error string if the label was not found.
    """
    # Look up the FAISS integer ID for this label.
    faiss_id = rag_index.labels_to_ids.get(label)
    if faiss_id is None:
        return f"label '{label}' not found in index"

    # Capture the vector before removing it so it can be returned.
    removed_vector = rag_index.ids_to_vectors.get(faiss_id)

    # Remove the FAISS ID from the index and the integer-to-label side-index.
    rag_index.index.remove_ids(np.array([faiss_id], dtype=np.int64))
    rag_index.ids_to_labels.pop(faiss_id, None)

    # Drop the vector from the integer-ID to embedding-vector side-index.
    rag_index.ids_to_vectors.pop(faiss_id, None)

    # Drop the label entry from the label-to-integer side-index.
    rag_index.labels_to_ids.pop(label, None)

    return {"ok": True, "faiss_id": faiss_id, "vector": removed_vector}


def _format_distance(distance: float) -> float:
    """Round a distance value to 6 decimal places for stable result comparisons."""
    return round(float(distance), 6)


def _query_index(
    rager: RAGer,
    rag_index: RAGIndex,
    query_vector: list[float],
    limit: int,
) -> dict[str, Any]:
    """Search the index for the limit nearest neighbours of the query vector.

    Returns {"ok": True, "results": [...]} sorted by distance.
    Returns {"ok": False, "error": ...} on query failure.
    """
    # Search the FAISS index for the k nearest neighbours.
    # Re-raise on search failure -- caller decides how to handle it.
    try:
        vec = np.array([query_vector], dtype=np.float32)
        distances, faiss_ids = rag_index.index.search(vec, limit)
    except Exception as e:
        _logger.error(f"[RAGer] failed to search index '{rag_index.name}': {e}")
        raise

    # FAISS returns 2D result arrays when the query is 2D; extract the first row.
    distances = distances[0]
    faiss_ids = faiss_ids[0]

    # Translate each FAISS result into a label and distance for the caller.
    results = []
    for faiss_id, distance in zip(faiss_ids, distances):
        # FAISS pads with -1 when fewer than k results exist; stop here.
        if faiss_id == -1:
            break

        # Look up the domain label for this FAISS integer ID.
        label = rag_index.ids_to_labels.get(faiss_id)
        if label is None:
            _logger.error(
                f"[RAGer] FAISS id {faiss_id} has no label in ids_to_labels"
            )
            continue

        # Append the labelled result to the output list.
        results.append({"buffer": label, "distance": _format_distance(distance)})

    return {"ok": True, "results": results}


def _flush_index(rager: RAGer, rag_index: RAGIndex) -> dict[str, Any]:
    """Flush the live RAGIndex and its side-indexes back to disk.

    Returns {"ok": True} on success.
    Returns {"ok": False, "error": ...} on any persist failure.
    """
    # Derive the index directory from the RAGer and the index name in the RAGIndex.
    index_dir = _get_index_dir(rager, rag_index.name)

    # Persist the FAISS index to its binary file.
    # Let any I/O error propagate -- caller decides how to handle it.
    index_file = index_dir / _RAGER_INDEX_FILE
    faiss.write_index(rag_index.index, str(index_file))

    # Persist the integer-to-label side-index as JSON.
    ids_to_labels_file = index_dir / _RAGER_IDS_TO_LABELS_FILE
    try:
        with open(ids_to_labels_file, "w") as f:
            json.dump(rag_index.ids_to_labels, f)
    except Exception as e:
        _logger.error(f"[RAGer] failed to persist ids_to_labels for '{rag_index.name}': {e}")
        return {"ok": False, "error": str(e)}

    # Persist the label-to-integer side-index as JSON.
    labels_to_ids_file = index_dir / _RAGER_LABELS_TO_IDS_FILE
    try:
        with open(labels_to_ids_file, "w") as f:
            json.dump(rag_index.labels_to_ids, f)
    except Exception as e:
        _logger.error(f"[RAGer] failed to persist labels_to_ids for '{rag_index.name}': {e}")
        return {"ok": False, "error": str(e)}

    # Persist the integer-ID to embedding-vector side-index as JSON.
    ids_to_vectors_file = index_dir / _RAGER_IDS_TO_VECTORS_FILE
    try:
        with open(ids_to_vectors_file, "w") as f:
            json.dump(rag_index.ids_to_vectors, f)
    except Exception as e:
        _logger.error(f"[RAGer] failed to persist ids_to_vectors for '{rag_index.name}': {e}")
        return {"ok": False, "error": str(e)}

    return {"ok": True}


# -------------------------------------------------------------------------------------------------------------------- #
# Embedding Utils                                                                                                   #
# -------------------------------------------------------------------------------------------------------------------- #


def _embed_via_mcp(
    text: str,
) -> dict[str, Any]:
    """Convert a string to a vector via an MCP server.

    Returns {"ok": True, "vector": [...]} on success.
    Returns {"ok": False, "error": ...} on failure.
    """
    raise NotImplementedError


def _embed_via_api(
    url: str,
    model: str,
    text: str,
) -> dict[str, Any]:
    """Convert a string to a vector via a remote API call.

    Returns {"ok": True, "vector": [...]} on success.
    Returns {"ok": False, "error": ...} on failure.
    """
    # POST to the embedding endpoint and parse the JSON response.
    try:
        response = requests.post(
            url,
            json={"model": model, "input": text},
            headers={"Content-Type": "application/json"},
            timeout=_RAGER_EMBED_TIMEOUT,
        )
        result = response.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

    # Extract the embedding from the nested response path data[0].embedding.
    data = result.get("data")
    if not isinstance(data, list) or not data:
        return {"ok": False, "error": "response has no 'data' array"}
    embedding = data[0].get("embedding")
    if embedding is None:
        return {"ok": False, "error": "response data[0] has no 'embedding' field"}

    return {"ok": True, "vector": embedding, "usage": result.get("usage", {})}


def _embed(
    rager: RAGer,
    rag_index: RAGIndex,
    text: str,
) -> dict[str, Any]:
    """Embed a string into a vector using the index's configured embedding method.

    Reads the index configuration to determine whether to embed via API or MCP,
    then dispatches to the appropriate backend.

    Returns {"ok": True, "vector": [...]} on success.
    Returns {"ok": False, "error": ...} on failure.
    """
    # Use the already-loaded config from the RAGIndex object.
    config = rag_index.config

    # Extract the embedding method flag from the config.
    embed_via_api = config.get("embedding", {}).get("via") == "api"
    if not embed_via_api:
        # Embed via the configured MCP server.
        return _embed_via_mcp(text)

    # Extract the URL and model from the config for the remote API.
    embed_url = config.get("embedding", {}).get("url", "")
    if not embed_url:
        return {"ok": False, "error": f"config for '{rag_index.name}' has no embedding URL"}

    embed_model = config.get("embedding", {}).get("model", "")
    if not embed_model:
        return {"ok": False, "error": f"config for '{rag_index.name}' has no embedding model"}

    # Embed via the remote API.
    return _embed_via_api(embed_url, embed_model, text)


def _discover_embedding_dimension(
    rager: RAGer,
    config: dict[str, Any],
) -> int | None:
    """Probe the configured embedder with a short test string to discover the vector dimension.

    Builds a minimal stand-in RAGIndex that carries the config so _embed can read the
    embedding settings without needing a live FAISS index.

    Returns the embedding dimension (vector length) on success.
    Returns None if the embed call fails or the result has no vector.
    """
    # Build a minimal stand-in RAGIndex to satisfy _embed's interface.
    # The FAISS index field is left as None -- _embed does not access it during probing.
    probe_index = RAGIndex(
        name="probe",
        config=config,
        index=None,  # type: ignore[arg-type]
        ids_to_labels={},
        ids_to_vectors={},
        labels_to_ids={},
        eager_views=set(),
    )

    # Probe the embedder with a short, innocuous test string.
    embed_result = _embed(rager, probe_index, "test")
    if not embed_result["ok"]:
        _logger.error(
            f"[RAGer] embedding dimension probe failed: {embed_result['error']}"
        )
        return None

    vector = embed_result.get("vector")
    if not vector:
        _logger.error("[RAGer] embedding dimension probe returned no vector")
        return None

    return len(vector)


def _l2_squared(a: list[float], b: list[float]) -> float:
    """Return the squared L2 distance between two vectors.

    Matches what IndexFlatL2.search returns so the same values can be compared
    directly without a sqrt.
    """
    # Compute the squared L2 distance; no sqrt applied.
    diff = np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)
    return float(np.dot(diff, diff))


# -------------------------------------------------------------------------------------------------------------------- #
# RAGer Utils                                                                                                   #
# -------------------------------------------------------------------------------------------------------------------- #


def _upsert_buffer(
    rager: RAGer,
    rag_index: RAGIndex,
    buffer_name: str,
) -> dict[str, Any] | str:
    """Read the buffer, embed its content, and upsert the vector into the index.

    If the buffer was already indexed, the old vector is replaced.

    Returns {{"ok": True, "vector": [...]}} on success.
    Returns an error string on failure.
    """
    # Locate the buffer by name; abort if it does not exist.
    buffer = rager.get_buffer(buffer_name)
    if buffer is None:
        return f"buffer '{buffer_name}' not found"

    # Read the text content from the buffer.
    buffer_text = buffer.get_text() or ""
    if not buffer_text:
        return f"buffer '{buffer_name}' is empty"

    # Embed the buffer text into a vector using the index configuration.
    embed_result = _embed(rager, rag_index, buffer_text)
    if not embed_result["ok"]:
        return embed_result["error"]

    # Upsert the vector with the buffer name as the label.
    upsert_result = _upsert_vector(rager, rag_index, embed_result["vector"], buffer_name)
    if isinstance(upsert_result, str):
        return upsert_result

    return upsert_result


def _remove_buffer(
    rager: RAGer,
    rag_index: RAGIndex,
    buffer_name: str,
) -> dict[str, Any] | str:
    """Remove the buffer's vector from the index.

    The buffer name is used as the label to look up the vector in the index.

    Returns {{"ok": True, "faiss_id": ..., "vector": [...]}} on success.
    Returns an error string on failure.
    """
    # The buffer name is the label; delegate directly to the vector removal helper.
    return _remove_vector(rager, rag_index, buffer_name)


async def _sync_index(
    rager: RAGer,
    index_name: str,
) -> str | None:
    """Re-scan all buffers and sync the index with the current buffer set.

    Filters all buffers by the index configuration's buffer_name_pattern.
    Computes the diff between the expected set and the currently indexed labels.
    Removes labels that are no longer in the expected set;
    adds labels that are new.

    Returns an error string on failure, None on success.
    """
    # Load the RAGIndex from live memory via rager._rag_indexes.
    rag_index = rager._rag_indexes.get(index_name)
    if rag_index is None:
        msg = f"RAGIndex '{index_name}' is not active"
        _logger.error(f"[RAGer] {msg}")
        return msg

    # Load the index configuration to get the buffer_name_pattern.
    config = _get_index_config(rager, index_name)
    if config is None:
        msg = f"config buffer for index '{index_name}' not found"
        _logger.error(f"[RAGer] {msg}")
        return msg

    # Read the list of all buffer names from the status buffer.
    status_buffer = rager.get_buffer(_STATUS_BUFFER)
    if status_buffer is None:
        msg = f"buffer list '{_STATUS_BUFFER}' is not available"
        _logger.error(f"[RAGer] {msg}")
        return msg
    try:
        buffer_names: list[str] = [b["name"] for b in json.loads(status_buffer.get_text() or "[]")]
    except json.JSONDecodeError:
        msg = f"buffer list '{_STATUS_BUFFER}' is corrupted"
        _logger.error(f"[RAGer] {msg}")
        return msg

    # Read the buffer selection pattern from the index configuration.
    # Buffers matching this pattern are candidates for inclusion in the index.
    buffer_pattern = config.get("buffer_pattern")
    if not buffer_pattern:
        msg = f"config for index '{index_name}' has no 'buffer_pattern'"
        _logger.error(f"[RAGer] {msg}")
        return msg

    # Filter the buffer name list to those matching the index's buffer_pattern regex.
    # Compiling the pattern once avoids repeated recompilation in the loop.
    pattern_re = re.compile(buffer_pattern)
    expected_labels = [name for name in buffer_names if pattern_re.match(name)]

    # Compute the symmetric diff between what is expected and what is currently indexed.
    # Labels in expected but not indexed are new -- add them.
    # Labels indexed but not expected are stale -- remove them.
    indexed = set(rag_index.labels_to_ids.keys())
    expected = set(expected_labels)
    removed_labels = indexed - expected
    added_labels = expected - indexed

    # Remove each stale label from the index.
    for label in removed_labels:
        remove_err = _remove_buffer(rager, rag_index, label)
        if isinstance(remove_err, str):
            _logger.warning(f"[RAGer] failed to remove '{label}' from index: {remove_err}")

    # Add each new label to the index.
    for label in added_labels:
        upsert_err = _upsert_buffer(rager, rag_index, label)
        if isinstance(upsert_err, str):
            _logger.warning(f"[RAGer] failed to add '{label}' to index: {upsert_err}")

    _logger.debug(f"[RAGer] rescan complete for index '{index_name}'")


# -------------------------------------------------------------------------------------------------------------------- #
# Hooks                                                                                                            #
# -------------------------------------------------------------------------------------------------------------------- #


async def _before_update_config(
    rager: RAGer,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Called before the config buffer of an index is updated.

    Validates that the new config is a sound JSON dictionary with
    all required fields: dim, model, embedding (with via: api|mcp).

    Returns an error string on failure, None to allow the update.
    """
    # Parse the new buffer text as JSON.
    new_text = new_buffer.get_text() or ""
    try:
        config = json.loads(new_text)
    except json.JSONDecodeError:
        # Heuristic: check whether the text looks like JSON but is malformed.
        stripped = new_text.strip()
        if len(stripped) > 0 and stripped[0] == "{" and stripped[-1] == "}":
            return (
                f"config buffer is not valid JSON: "
                f"{_RAGER_SCOPE}:{new_buffer.name} contains malformed JSON"
            )
        # Not JSON and not JSON-shaped -- reject.
        return (
            f"config buffer must be a JSON object: "
            f"{_RAGER_SCOPE}:{new_buffer.name}"
        )

    # Reject if the config is not a dictionary.
    if not isinstance(config, dict):
        return f"config must be a JSON object, got {type(config).__name__}"

    # Validate that the required 'embedding' field is a dict with a valid 'via'.
    embedding = config.get("embedding")
    if not isinstance(embedding, dict):
        return "config is missing or has an invalid 'embedding' (dict)"

    embed_via = embedding.get("via")
    if embed_via not in ("api", "mcp"):
        return f"config 'embedding.via' must be 'api' or 'mcp', got {embed_via!r}"

    # Validate that 'embedding.model' is a non-empty string.
    embed_model = embedding.get("model")
    if not isinstance(embed_model, str) or not embed_model:
        return f"config 'embedding.model' must be a non-empty string, got {embed_model!r}"

    # Validate that the required 'buffer_pattern' field is present.
    if "buffer_pattern" not in config:
        return (
            f"config is missing required field 'buffer_pattern': "
            f"{_RAGER_SCOPE}:{new_buffer.name}"
        )

    # Reject any change to buffer_pattern after index creation; a resync cannot safely refresh result buffers.
    if old_buffer is not None:
        # Read the old config to extract its buffer_pattern.
        old_text = old_buffer.get_text() or ""
        try:
            old_config = json.loads(old_text)
        except json.JSONDecodeError:
            old_config = {}

        # Compare the old pattern against the new one; allow if they are the same.
        old_pattern = old_config.get("buffer_pattern")
        new_pattern = config["buffer_pattern"]
        if old_pattern is not None and old_pattern != new_pattern:
            return (
                f"changing 'buffer_pattern' after index creation is not allowed: "
                f"old={old_pattern!r}, new={new_pattern!r}"
            )

        # Reject any change to embedding.model after index creation; existing vectors were
        # built with the old model and cannot safely be replaced without a full re-embed.
        old_model = old_config.get("embedding", {}).get("model")
        new_model = config.get("embedding", {}).get("model")
        if old_model is not None and old_model != new_model:
            return (
                f"changing 'embedding.model' after index creation is not allowed: "
                f"old={old_model!r}, new={new_model!r}"
            )

        # Reject any change to embedding.url after index creation; the endpoint may serve
        # a different model and produce incompatible vectors.
        old_url = old_config.get("embedding", {}).get("url")
        new_url = config.get("embedding", {}).get("url")
        if old_url is not None and old_url != new_url:
            return (
                f"changing 'embedding.url' after index creation is not allowed: "
                f"old={old_url!r}, new={new_url!r}"
            )

        # Reject any change to embedding.via after index creation; switching the backend
        # changes how vectors are built and invalidates all existing index content.
        old_via = old_config.get("embedding", {}).get("via")
        new_via = config.get("embedding", {}).get("via")
        if old_via is not None and old_via != new_via:
            return (
                f"changing 'embedding.via' after index creation is not allowed: "
                f"old={old_via!r}, new={new_via!r}"
            )

    return None


async def _after_update_config(
    rager: RAGer,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after the config buffer of an index is updated.

    The update already happened; this hook reacts to the change.
    On index creation, initialises and stores the RAGIndex.
    When the buffer_pattern changes, triggers a rescan to re-sync the index.
    """
    # Parse the new config from the updated buffer.
    new_config_text = new_buffer.get_text() or "{}"
    try:
        new_config: dict[str, Any] = json.loads(new_config_text)
    except json.JSONDecodeError:
        # Validation already ran in _before_update_config; this is a safeguard.
        _logger.error(f"[RAGer] config buffer has corrupted JSON: {new_buffer.name}")
        return

    # Guard: a buffer_pattern is required to know which buffers to track.
    # The BEFORE_UPDATE_CONFIG hook enforces this, but we guard here as a safeguard.
    if "buffer_pattern" not in new_config:
        _logger.error(f"[RAGer] new config for '{new_buffer.name}' has no 'buffer_pattern'")
        return

    # Parse the old config; an empty dict is used when there was no prior buffer.
    if old_buffer is None:
        old_config: dict[str, Any] = {}
    else:
        old_config_text = old_buffer.get_text() or "{}"
        try:
            old_config = json.loads(old_config_text)
        except json.JSONDecodeError:
            old_config = {}

    # Extract the index name from the config buffer name.
    index_name = _index_name_from_config(new_buffer.name)
    if index_name is None:
        return

    # If this is a fresh index creation, initialise the RAGIndex and store it in live memory.
    if old_buffer is None:
        # Auto-discover the embedding dimension and store it in the config for index creation.
        new_config["dim"] = _discover_embedding_dimension(rager, new_config)
        if new_config["dim"] is None:
            _logger.error(
                f"[RAGer] could not auto-discover embedding dimension "
                f"for index '{index_name}'"
            )
            return
        _logger.debug(
            f"[RAGer] auto-discovered embedding dimension {new_config["dim"]} "
            f"for index '{index_name}'"
        )

        # Delegate creation or reload to the helper; store the result in live memory.
        rag_index: RAGIndex | str = _create_or_load_index(rager, index_name, new_config)
        if isinstance(rag_index, str):
            _logger.error(f"[RAGer] failed to create index '{index_name}': {rag_index}")
            return
        rager._rag_indexes[index_name] = rag_index

        # Register the hooks for this index; the buffer_pattern drives which buffers are tracked.
        # The AFTER_UPDATE hook re-indexes the changed buffer; the AFTER_DROP hook removes it.
        update_hook = _make_after_update_buffer(rag_index)
        rager.register_buffer_pattern_hook(
            HookPoint.AFTER_UPDATE_BUFFER,
            f"rager_after_update_buffer_{index_name}",
            new_config["buffer_pattern"],
            update_hook,
            apply_to_existing=True,
        )

        # Also register for AFTER_CREATE_BUFFER so brand-new source buffers are indexed immediately.
        rager.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            f"rager_after_create_buffer_{index_name}",
            new_config["buffer_pattern"],
            lambda bm, new_buffer, args: update_hook(bm, None, new_buffer, args),
            apply_to_existing=True,
        )

        drop_hook = _make_after_drop_buffer(rag_index)
        rager.register_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            f"rager_after_drop_buffer_{index_name}",
            new_config["buffer_pattern"],
            drop_hook,
            apply_to_existing=True,
        )

    # On index creation, scan all existing buffers and backfill them into the new index.
    rescan_err = await _sync_index(rager, index_name)
    if rescan_err is not None:
        _logger.warning(f"[RAGer] rescan after config update failed: {rescan_err}")

    # Retrieve the live index from memory for the flush.
    index = rager._rag_indexes.get(index_name)
    if index is None:
        _logger.error(f"[RAGer] RAGIndex '{index_name}' is not active")
        return

    # Persist the live index to disk; a no-op config touch safely triggers a flush.
    flush_result = _flush_index(rager, index)
    if not flush_result.get("ok"):
        _logger.warning(f"[RAGer] index flush failed: {flush_result.get('error')}")


async def _after_drop_config(
    rager: RAGer,
    buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after the config buffer of an index is dropped.

    The drop already happened; this hook reacts to the change.
    Destroys the index and removes it from live memory.
    """
    # Extract the index name from the dropped config buffer name.
    config_buffer_name = buffer.name
    index_name = _index_name_from_config(config_buffer_name)
    if index_name is None:
        return None

    # Abort if the index is not in live memory; the config drop is orphaned.
    if index_name not in rager._rag_indexes:
        _logger.error(
            f"[RAGer] index '{index_name}' not found in live memory -- "
            f"inconsistent state after config drop for '{config_buffer_name}'"
        )
        return None

    # Load the live RAGIndex to retrieve the buffer_pattern for hook deregistration.
    rag_index = rager._rag_indexes[index_name]
    buffer_pattern = rag_index.config.get("buffer_pattern", "")

    # Deregister the hooks registered for this index's buffer_pattern.
    # remove_from_existing=True is required to also remove the hooks from all
    # buffers that already have them attached (not just prevent new registrations).
    if buffer_pattern:
        rager.unregister_buffer_pattern_hook(
            HookPoint.AFTER_UPDATE_BUFFER,
            f"rager_after_update_buffer_{index_name}",
            buffer_pattern,
            remove_from_existing=True,
        )
        rager.unregister_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            f"rager_after_create_buffer_{index_name}",
            buffer_pattern,
            remove_from_existing=True,
        )
        rager.unregister_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            f"rager_after_drop_buffer_{index_name}",
            buffer_pattern,
            remove_from_existing=True,
        )

    # Destroy the index files and side-indexes, then remove from live memory.
    _destroy_index(rager, rag_index)
    del rager._rag_indexes[index_name]
    _logger.debug(f"[RAGer] removed index '{index_name}' from live memory")


async def _before_update_query(
    rager: RAGer,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Called before the query buffer of an index is updated.

    Parses the new buffer as JSON and validates required fields.
    If the buffer is not valid JSON, it is treated as a raw search query
    and allowed through.

    Returns an error string on failure, None to allow the update.
    """
    # Read the text content from the updated buffer.
    new_text = new_buffer.get_text() or ""

    # Attempt to parse the new buffer as JSON.
    try:
        query_data = json.loads(new_text)
    except json.JSONDecodeError:
        # Heuristic: check whether the text looks like JSON but is malformed.
        stripped = new_text.strip()
        if len(stripped) > 0 and stripped[0] == "{" and stripped[-1] == "}":
            return (
                f"query buffer is not valid JSON: "
                f"{_RAGER_SCOPE}:{new_buffer.name} contains malformed JSON"
            )
        # Not JSON and not JSON-shaped -- treat as a raw search query.
        return None

    # Validate that the required 'search' field is present in the JSON.
    if "search" not in query_data:
        return (
            f"query buffer is missing the required 'search' field: "
            f"{_RAGER_SCOPE}:{new_buffer.name}"
        )

    # Validate that 'search' is a string.
    if not isinstance(query_data["search"], str):
        return (
            f"'search' must be a string, got {type(query_data['search']).__name__}: "
            f"{_RAGER_SCOPE}:{new_buffer.name}"
        )

    # Validate that 'threshold' in query_args is a non-negative number if present.
    # Null (JSON null -> Python None) is not a valid value for threshold.
    if "query_args" in query_data and "threshold" in query_data["query_args"]:
        threshold = query_data["query_args"]["threshold"]
        if threshold is not None and not isinstance(threshold, (int, float)):
            return (
                f"'threshold' must be a non-negative number, got {type(threshold).__name__}: "
                f"{_RAGER_SCOPE}:{new_buffer.name}"
            )
        if isinstance(threshold, (int, float)) and threshold < 0:
            return (
                f"'threshold' must be non-negative, got {threshold}: "
                f"{_RAGER_SCOPE}:{new_buffer.name}"
            )

    return None


async def _after_update_query(
    rager: RAGer,
    old_buffer: Buffer | None,
    new_buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after the query buffer of an index is updated.
    """
    # The updated buffer is the query buffer; derive the result buffer name from it.
    query_buffer_name = new_buffer.name
    result_buffer_name = _result_buffer_name_from_query(query_buffer_name)

    # Load the query dictionary from the query buffer and config.
    query_dict = _get_query_dict(rager, query_buffer_name)
    if query_dict is None:
        _logger.error(f"[RAGer] failed to load query for '{query_buffer_name}'")
        return

    # Get the RAGIndex object from the RAGer.
    index_name = _index_name_from_query(query_buffer_name)
    rag_index = rager._rag_indexes.get(index_name)
    if not rag_index:
        _logger.error(f"[RAGer] RAGIndex '{index_name}' is not active")
        return

    # Only pre-populate results if timing is eager.
    timing = query_dict["query_args"]["timing"]
    mode = query_dict["query_args"]["mode"]
    if timing != "eager":
        # Ensure, the query buffer is not in the eager view list.
        rag_index.eager_views.discard(query_buffer_name)

        # Refresh the result buffer.
        create_result = await rager.create_buffer(result_buffer_name, "", overwrite=True)
        if not create_result.get("ok", False):
            _logger.error(
                f"[RAGer] failed to create result buffer '{result_buffer_name}': "
                f"{create_result.get('error', 'unknown')}"
            )

        return

    # If the eager query is a view, add it to the eager view set.
    # set.add is idempotent -- no need to guard for duplicates.
    if mode == "view":
        rag_index.eager_views.add(query_buffer_name)

    # An update of an eager query forces to create fresh results.
    query_result = _get_results_for_query(rager, rag_index, query_buffer_name)
    if not query_result.get("ok"):
        _logger.error(
            f"[RAGer] query execution failed for '{query_buffer_name}': "
            f"{query_result.get('error', 'unknown')}"
        )
        return

    # Render the results as JSON and write them into the result buffer.
    results = query_result.get("results", [])
    results_str = json.dumps(results, indent=2, ensure_ascii=False)

    # Refresh the result buffer
    create_result = await rager.create_buffer(result_buffer_name, results_str, overwrite=True)
    if not create_result.get("ok", False):
        _logger.error(
            f"[RAGer] failed to create result buffer '{result_buffer_name}': "
            f"{create_result.get('error', 'unknown')}"
        )


async def _after_drop_query(
    rager: RAGer,
    buffer: Buffer,
    args: dict[str, Any],
) -> None:
    """Called after the query buffer of an index is dropped.

    Also drops the corresponding result buffer to keep the index consistent.
    """
    # Derive the result buffer name from the dropped query buffer name.
    query_buffer_name = buffer.name
    result_buffer_name = _result_buffer_name_from_query(query_buffer_name)

    # Remove the query buffer from the eager view list of its index.
    index_name = _index_name_from_query(query_buffer_name)
    if index_name is not None and index_name in rager._rag_indexes:
        rager._rag_indexes[index_name].eager_views.discard(query_buffer_name)

    # Drop the result buffer if it exists.
    drop_result = await rager.drop_buffer(result_buffer_name)
    if not drop_result.get("ok"):
        _logger.error(
            f"[RAGer] failed to drop result buffer '{result_buffer_name}': "
            f"{drop_result.get('error', 'unknown')}"
        )


def _make_after_update_buffer(
    rag_index: RAGIndex,
) -> AfterUpdateBuffer:
    """Factory that returns an AFTER_UPDATE_BUFFER hook for a specific RAGIndex.

    The returned hook captures the RAGIndex so it can upsert buffer changes
    into the correct index without an extra index lookup.
    """
    async def _after_update_buffer(
        rager: RAGer,
        old_buffer: Buffer | None,
        new_buffer: Buffer,
        args: dict[str, Any],
    ) -> None:
        """Called after a source buffer is updated.

        Upserts the new buffer content into the RAGIndex captured by the factory.
        """
        # The new buffer is already in the buffer manager; upsert it into the index.
        upsert_result = _upsert_buffer(rager, rag_index, new_buffer.name)
        if isinstance(upsert_result, str):
            _logger.error(
                f"[RAGer] failed to upsert buffer '{new_buffer.name}' "
                f"into index '{rag_index.name}': {upsert_result}"
            )
            return

        # Loop over all eager view queries
        for query_buffer_name in rag_index.eager_views:
            # Get current results from result buffer.
            current_results = _read_results_for_query(rager, query_buffer_name)
            if current_results is None:
                _logger.warning(f"[RAGer] could not read eager view results for '{query_buffer_name}'")
                continue

            # Get the query's embedding.
            query_dict = _get_query_dict(rager, query_buffer_name)
            if query_dict is None:
                _logger.warning(f"[RAGer] could not load query dict for '{query_buffer_name}'")
                continue

            # Embed the search string into a query vector for distance computation.
            embed_result = _embed(rager, rag_index, query_dict.get("search", ""))
            if not embed_result["ok"]:
                _logger.warning(
                    f"[RAGer] could not embed query for '{query_buffer_name}': "
                    f"{embed_result.get('error', 'unknown')}"
                )
                continue
            query_vector = embed_result["vector"]

            # Compute the distance between the new element's and the querie's embedding.
            new_distance = _l2_squared(upsert_result["vector"], query_vector)

            # Compare the new distance with results to find out, if it is a new candidate.
            # Results are sorted ascending by distance; the last entry is the worst.
            # Skip if the result list is full and the new distance does not beat the current worst result.
            threshold = query_dict["query_args"]["hreshold"]
            if threshold < float("inf") and new_distance > threshold:
                continue
            limit = query_dict["query_args"]["limit"]
            if current_results and len(current_results) >= limit and new_distance > current_results[-1]["distance"]:
                continue

            # Remove any stale entry for this buffer first so it is replaced, not duplicated.
            current_results = [r for r in current_results if r["buffer"] != new_buffer.name]

            # Insert the new element in sorted order and drop the last (worst) entry.
            current_results.append({"buffer": new_buffer.name, "distance": _format_distance(new_distance)})
            current_results.sort(key=lambda r: r["distance"])

            # Drop the last (worst) entry only if the list exceeds limit results.
            if len(current_results) > limit:
                current_results.pop()

            # Persist the updated results back into the result buffer.
            result_buffer_name = _result_buffer_name_from_query(query_buffer_name)
            create_result = await rager.create_buffer(
                result_buffer_name,
                text=json.dumps(current_results, indent=2, ensure_ascii=False),
                overwrite=True,
            )
            if not create_result.get("ok", False):
                _logger.error(
                    f"[RAGer] failed to update result buffer '{result_buffer_name}': "
                    f"{create_result.get('error', 'unknown')}"
                )

    return _after_update_buffer


def _make_after_drop_buffer(
    rag_index: RAGIndex,
) -> AfterBufferDropped:
    """Factory that returns an AFTER_DROP_BUFFER hook for a specific RAGIndex.

    The returned hook captures the RAGIndex so it can remove the buffer
    from the correct index without an extra index lookup.
    """
    async def _after_drop_buffer(
        rager: RAGer,
        buffer: Buffer,
        args: dict[str, Any],
    ) -> None:
        """Called after a source buffer is dropped.

        Removes the dropped buffer from the RAGIndex captured by the factory.
        """
        # Remove the buffer from the index using the buffer name as the label.
        remove_result = _remove_buffer(rager, rag_index, buffer.name)
        if isinstance(remove_result, str):
            _logger.warning(
                f"[RAGer] failed to remove buffer '{buffer.name}' "
                f"from index '{rag_index.name}': {remove_result}"
            )
            return

        # Loop over all eager view queries
        for query_buffer_name in rag_index.eager_views:
            # Get current results from result buffer.
            current_results = _read_results_for_query(rager, query_buffer_name)
            if current_results is None:
                _logger.warning(f"[RAGer] could not read eager view results for '{query_buffer_name}'")
                continue

            # If the buffer that we remove is not in the current results, no change is necessary.
            buffer_names = [r["buffer"] for r in current_results]
            if buffer.name not in buffer_names:
                continue

            # The removed buffer was in the result set; a full re-search fills the vacated slot.
            query_result = _get_results_for_query(rager, rag_index, query_buffer_name)
            if not query_result.get("ok"):
                _logger.error(
                    f"[RAGer] query execution failed for '{query_buffer_name}': "
                    f"{query_result.get('error', 'unknown')}"
                )
                return

            # Render the results as JSON and write them into the result buffer.
            results = query_result.get("results", [])
            results_str = json.dumps(results, indent=2, ensure_ascii=False)

            # Derive the result buffer name from the query buffer name.
            result_buffer_name = _result_buffer_name_from_query(query_buffer_name)

            # Refresh the result buffer
            create_result = await rager.create_buffer(result_buffer_name, results_str, overwrite=True)
            if not create_result.get("ok", False):
                _logger.error(
                    f"[RAGer] failed to create result buffer '{result_buffer_name}': "
                    f"{create_result.get('error', 'unknown')}"
                )

    return _after_drop_buffer


async def _before_read_result(
    rager: RAGer,
    buffer: Buffer,
    args: dict[str, Any],
) -> str | None:
    """Called before or after the result buffer of an index is read.

    If the result buffer already has content, return None and allow the read.
    Otherwise, for non-dynamic modes, pre-populate the result buffer.

    Returns an error string on failure, None to allow the read.
    """
    # Derive the query buffer name from the result buffer name.
    result_buffer_name = buffer.name
    query_buffer_name = _query_buffer_name_from_result(result_buffer_name)

    # Load the query dictionary from the query buffer and config.
    query_dict = _get_query_dict(rager, query_buffer_name)
    if query_dict is None:
        return f"failed to load query for '{query_buffer_name}'"

    # For lazy view, return None and let the after-read hook freshen results.
    timing = query_dict["query_args"]["timing"]
    mode = query_dict["query_args"]["mode"]

    # The result is never updated lazyily, when the query is in eager mode.
    if timing == "eager":
        return

    # In lazy snapshot mode, we only update, when the buffer is empty.
    if not buffer.empty() and mode == "snapshot":
        return

    # Derive the query buffer name from the result buffer name.
    result_buffer_name = buffer.name
    query_buffer_name = _query_buffer_name_from_result(result_buffer_name)

    # Execute the query and write the results to the result buffer.
    # Extract the index name from the query buffer name.
    index_name = _index_name_from_query(query_buffer_name)
    if index_name is None:
        _logger.error(f"[RAGer] failed to extract index name from '{query_buffer_name}'")
        return None

    # Look up the live RAGIndex for this index.
    rag_index = rager._rag_indexes.get(index_name)
    if rag_index is None:
        _logger.error(f"[RAGer] RAGIndex '{index_name}' is not active")
        return None

    # Execute the query and get the results.
    query_result = _get_results_for_query(rager, rag_index, query_buffer_name)
    if not query_result["ok"]:
        _logger.error(
            f"[RAGer] query execution failed for '{query_buffer_name}': "
            f"{query_result.get('error', 'unknown')}"
        )
        return None

    # Write the results JSON to the result buffer.
    write_result = await rager.create_buffer(
        result_buffer_name,
        text=json.dumps(query_result.get("results", []), indent=2, ensure_ascii=False),
        overwrite=True,
    )
    if not write_result.get("ok"):
        _logger.error(
            f"[RAGer] failed to write results to '{result_buffer_name}': "
            f"{write_result.get('error', 'unknown')}"
        )
        return None


# -------------------------------------------------------------------------------------------------------------------- #
# RAGer                                                                                                            #
# -------------------------------------------------------------------------------------------------------------------- #


class RAGer(BufferManager, AgenticObject):
    """I can index buffers and search them via a RAG system.
    `doc:rager` holds details.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Seed the doc buffer with the module documentation.
        self._create_buffer(_RAGER_DOC_BUFFER, text=_RAGER_DOC)

        # In-memory storage for live RAGIndex objects, keyed by index name.
        self._rag_indexes: dict[str, RAGIndex] = {}

        # Register hooks on all index config buffers.
        config_pattern = rf"^{_RAGER_SCOPE}:.+:config$"

        # Register hooks on config buffers to validate config changes.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_UPDATE_BUFFER,
            "rager_before_update_config",
            config_pattern,
            _before_update_config,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "rager_before_create_config",
            config_pattern,
            lambda bm, new_buffer, args: _before_update_config(bm, None, new_buffer, args),
            apply_to_existing=True,
        )

        # Register hooks on config buffers to react to config changes.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_UPDATE_BUFFER,
            "rager_after_update_config",
            config_pattern,
            _after_update_config,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "rager_after_create_config",
            config_pattern,
            lambda bm, new_buffer, args: _after_update_config(bm, None, new_buffer, args),
            apply_to_existing=True,
        )

        # Register the AFTER_DROP hook on config buffers to handle index removal.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            "rager_after_drop_config",
            config_pattern,
            _after_drop_config,
            apply_to_existing=True,
        )

        # Register hooks on all index query buffers.
        query_pattern = rf"^{_RAGER_SCOPE}:.+:query:.+$"

        # Register hooks on query buffers to validate query changes.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_UPDATE_BUFFER,
            "rager_before_update_query",
            query_pattern,
            _before_update_query,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_CREATE_BUFFER,
            "rager_before_create_query",
            query_pattern,
            lambda bm, new_buffer, args: _before_update_query(bm, None, new_buffer, args),
            apply_to_existing=True,
        )

        # Register hooks on query buffers to react to query changes.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_UPDATE_BUFFER,
            "rager_after_update_query",
            query_pattern,
            _after_update_query,
            apply_to_existing=True,
        )
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_CREATE_BUFFER,
            "rager_after_create_query",
            query_pattern,
            lambda bm, new_buffer, args: _after_update_query(bm, None, new_buffer, args),
            apply_to_existing=True,
        )

        # Register the AFTER_DROP hook on query buffers to handle query removal.
        self.register_buffer_pattern_hook(
            HookPoint.AFTER_DROP_BUFFER,
            "rager_after_drop_query",
            query_pattern,
            _after_drop_query,
            apply_to_existing=True,
        )

        # Register hooks on all index result buffers.
        result_pattern = rf"^{_RAGER_SCOPE}:.+:result:.+$"

        # Register the BEFORE_READ hook on result buffers to pre-populate results.
        self.register_buffer_pattern_hook(
            HookPoint.BEFORE_READ_BUFFER,
            "rager_before_read_result",
            result_pattern,
            _before_read_result,
            apply_to_existing=True,
        )
