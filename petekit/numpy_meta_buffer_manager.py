"""NumPy meta buffer manager -- mirrors numpy buffers as JSON metadata text buffers.

For every numpy buffer, a text buffer with the same name holds the metadata
(shape, created_at, modified_at) as JSON. The text hook system therefore
observes the full numpy lifecycle: storing, copying, and dropping fire the
text buffer hooks.

Note -- text buffers persist and auto-restore from disk while numpy arrays
do not. After a restart the restored metadata buffers are stale until the
numpy buffers are stored again.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from peteos.oap.agentic_object import AgenticObject
from peteos.utils import get_logger

from petekit.numpy_buffer_manager import NumPyBuffer, NumPyBufferManager
from petekit.text.buffer_manager import BufferManager

_logger = get_logger(__name__)


# -------------------------------------------------------------------------------------------------------------------- #
# Constants                                                                                                            #
# -------------------------------------------------------------------------------------------------------------------- #


_NUMPY_META_DOC_BUFFER = "doc:numpy_meta_buffers"

_NUMPY_META_DOC = """\
## NumPy Meta Buffers

The NumPyMetaBufferManager holds NumPy buffers and mirrors their lifecycle
as JSON metadata text buffers. For every NumPy buffer there is a text buffer
with the same name holding the metadata.

### Metadata Format

The metadata text buffer holds a JSON object:

{
  "shape": [3, 4],
  "created_at": 1731861234.123456,
  "modified_at": 1731861234.123456
}

- shape  --  the dimensions of the NumPy array
- created_at  --  unix timestamp of the first store
- modified_at  --  unix timestamp of the last store

### Lifecycle

- Storing a NumPy buffer creates or updates the metadata buffer.
- Copying a NumPy buffer creates the metadata buffer for the target.
- Dropping a NumPy buffer drops the metadata buffer with it.

The text hook system observes this lifecycle: register pattern hooks on the
buffer manager to get notified when metadata buffers are created, updated,
or dropped.

### Staleness

Text buffers persist across restarts while NumPy buffers do not. After a
restart the restored metadata buffers are stale until the NumPy buffers are
stored again.
"""


# -------------------------------------------------------------------------------------------------------------------- #
# NumPyMetaBufferManager                                                                                               #
# -------------------------------------------------------------------------------------------------------------------- #


def _metadata_text(stored: NumPyBuffer) -> str:
    """Build the JSON metadata text for a stored numpy buffer."""
    # Serialize shape and timestamps -- the observable lifecycle of the array.
    return json.dumps({
        "shape": list(stored.array.shape),
        "created_at": stored.created_at,
        "modified_at": stored.modified_at,
    }, indent=2)


class NumPyMetaBufferManager(NumPyBufferManager, BufferManager, AgenticObject):

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Seed the doc buffer with the module documentation.
        self._create_buffer(_NUMPY_META_DOC_BUFFER, text=_NUMPY_META_DOC)

    async def _store_np_buffer(self, name: str, array: np.ndarray) -> None:
        """Store the array and mirror its metadata as a JSON text buffer."""
        # Remember whether the numpy buffer already exists -- new stores must
        # not silently overwrite an unrelated text buffer with the same name.
        existed = name in self._numpy_buffers

        # Store the array in the numpy namespace first -- the array is the source of truth.
        super()._store_np_buffer(name, array)

        # Mirror the metadata -- overwrite only on updates of the same numpy buffer.
        result = await self.create_buffer(name, text=_metadata_text(self._numpy_buffers[name]), overwrite=existed)
        if not result.get("ok"):
            _logger.error(f"[NumPyMetaBufferManager] failed to write metadata for '{name}': {result.get('error', 'unknown')}")

    async def copy_np_buffer(self, source: str, target: str) -> str:
        """Copy the array and mirror the target's metadata as a JSON text buffer."""
        # Copy the array in the numpy namespace.
        result = super().copy_np_buffer(source, target)

        # Guard: the base copy failed -- keep the metadata namespace untouched.
        if target not in self._numpy_buffers:
            return result

        # Mirror the metadata for the new target -- an unrelated text buffer with this name blocks it.
        meta_result = await self.create_buffer(target, text=_metadata_text(self._numpy_buffers[target]))
        if not meta_result.get("ok"):
            _logger.error(f"[NumPyMetaBufferManager] failed to write metadata for '{target}': {meta_result.get('error', 'unknown')}")

        return result

    async def drop_np_buffer(self, name: str) -> str:
        """Drop the array and drop its metadata text buffer with it."""
        # Drop the array in the numpy namespace.
        result = super().drop_np_buffer(name)

        # Guard: the base drop failed -- keep the metadata buffer.
        if name in self._numpy_buffers:
            return result

        # Drop the metadata buffer -- the text hook system observes the drop.
        drop_result = await self.drop_buffer(name)
        if not drop_result.get("ok"):
            _logger.error(f"[NumPyMetaBufferManager] failed to drop metadata for '{name}': {drop_result.get('error', 'unknown')}")

        return result
