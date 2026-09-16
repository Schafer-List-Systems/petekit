"""DiskImageLoader - load and save image files to/from the ImageBufferManager."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from peteos import AgenticObject, tool

from .image_buffer_manager import ImageBufferManager


class DiskImageLoader(ImageBufferManager, AgenticObject):
    """You are a disk image loader. You load images from disk into the ImageBufferManager and save buffers to disk.

    Only use this when you need to read from disk or write to disk.
    """

    _SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".gif"}

    @tool(description="Load an image file from disk into the ImageBufferManager as a named buffer. Pass the full path to the image file. The image is stored under the given buffer name.")
    def load_image(self, path: str, buffer_name: str) -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: file not found: {path}"
        if p.suffix.lower() not in self._SUPPORTED_EXTENSIONS:
            return f"Error: unsupported format '{p.suffix}'. Supported: {', '.join(sorted(self._SUPPORTED_EXTENSIONS))}"
        array = cv2.imread(str(p))
        if array is None or array.size == 0:
            return f"Error: failed to read image from {path}."
        self._store_np_buffer(buffer_name, array)
        return f"OK: loaded '{path}' ({array.shape[1]}x{array.shape[0]}) into buffer '{buffer_name}'."

    @tool(description="Save a named image buffer from the ImageBufferManager to a disk file. Pass the buffer name and the full destination path. Overwrites the file if it exists.")
    def save_image(self, buffer_name: str, path: str) -> str:
        if buffer_name not in self._buffers:
            return f"Error: no buffer named '{buffer_name}'. Use list_np_buffers to see available buffers."
        array = self._buffers[buffer_name].array
        p = Path(path)
        if p.suffix.lower() not in self._SUPPORTED_EXTENSIONS:
            return f"Error: unsupported format '{p.suffix}'. Supported: {', '.join(sorted(self._SUPPORTED_EXTENSIONS))}"
        ok = cv2.imwrite(str(p), array)
        if not ok:
            return f"Error: cv2.imwrite failed for '{path}'."
        return f"OK: buffer '{buffer_name}' ({array.shape[1]}x{array.shape[0]}) saved to '{path}'."

    @tool(description="Load an image from a URL into the ImageBufferManager as a named buffer. Fetches the image via HTTP, then stores it.")
    async def load_url_image(self, url: str, buffer_name: str, runner) -> str:
        try:
            import httpx
        except ImportError:
            return "Error: httpx not available. Install it with: pip install httpx"
        try:
            resp = httpx.get(url, timeout=10.0)
            resp.raise_for_status()
        except Exception as e:
            return f"Error: failed to fetch '{url}': {e}"
        nparr = np.frombuffer(resp.bytes(), dtype=np.uint8)
        arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if arr is None or arr.size == 0:
            return f"Error: failed to decode image from '{url}'."
        self._store_np_buffer(buffer_name, arr)
        return f"OK: loaded '{url}' ({arr.shape[1]}x{arr.shape[0]}) into buffer '{buffer_name}'."
