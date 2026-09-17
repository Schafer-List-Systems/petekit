from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from peteos import AgenticObject, tool


@dataclass
class NumPyBuffer:
    """A buffer holding a NumPy array with timestamps."""
    array: np.ndarray
    created_at: float
    modified_at: float


class NumPyBufferManager(AgenticObject):
    """You are a NumPy buffer manager. You hold multiple named buffers, each a NumPy array in memory.

    - You can list, copy, and drop any named buffer.
    - Use list_np_buffers to see what exists.
    - All buffers are stored as true NumPy arrays with no copy-by-reference risk.
    """

    _MAX_PIXELS = 1920 * 1080

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._numpy_buffers: dict[str, NumPyBuffer] = {}

    @tool(description="List all existing NumPy buffers by name, showing shape for each.")
    def list_np_buffers(self) -> dict[str, tuple[int, ...]]:
        """List all buffers and their shapes."""
        return {name: buf.array.shape for name, buf in self._numpy_buffers.items()}

    @tool(description="Drop (delete) a NumPy buffer by name. The buffer and its array are discarded.")
    def drop_np_buffer(self, name: str) -> str:
        """Delete a named buffer."""
        if name not in self._numpy_buffers:
            return f"Error: no buffer named '{name}'."
        del self._numpy_buffers[name]
        return f"Buffer '{name}' dropped."

    def _get_np_buffer(self, name: str) -> np.ndarray | None:
        """Read-only access. Does not update modified_at."""
        if name not in self._numpy_buffers:
            return None
        array = self._numpy_buffers[name].array
        array.flags.writeable = False
        return array

    def _store_np_buffer(self, name: str, array: np.ndarray) -> None:
        """Write a new array into a buffer, stamping created_at and modified_at."""
        now = time.time()
        self._numpy_buffers[name] = NumPyBuffer(array=array.copy(), created_at=now, modified_at=now)

    @tool(description="Copy a NumPy buffer to a new name. Creates a by-value copy of the array.")
    def copy_np_buffer(self, source: str, target: str) -> str:
        """Create a by-value copy of a buffer under a new name."""
        if source not in self._numpy_buffers:
            return f"Error: no buffer named '{source}'."
        if target in self._numpy_buffers:
            return f"Error: a buffer named '{target}' already exists. Drop it first or use a different name."
        src = self._numpy_buffers[source].array
        now = time.time()
        self._numpy_buffers[target] = NumPyBuffer(array=src.copy(), created_at=now, modified_at=now)
        return f"Copied buffer '{source}' to '{target}' (shape {src.shape})."
