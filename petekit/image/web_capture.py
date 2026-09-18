"""WebCapture - capture web pages as images into image buffers."""

from __future__ import annotations

import cv2
import numpy as np
import subprocess
import tempfile
import time as time_mod
from pathlib import Path

from peteos.oap.agentic_object import AgenticObject
from peteos.oap.decorators import tool

from petekit.image.image_buffer_manager import ImageBufferManager
from petekit.utils.web_utils import _CHROME_BIN


class WebCapture(ImageBufferManager, AgenticObject):
    """You are a web capture agent. You capture web pages as images and load web images into image buffers.
    Use web_snapshot to take a screenshot of a URL.
    Use web_load_image to load an image from a URL into an image buffer.
    """

    def __init__(self) -> None:
        super().__init__()
        self._temp_dir: Path | None = None

    @property
    def temp_dir(self) -> Path:
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="web_capture_"))
        return self._temp_dir

    @tool(description="Capture a screenshot of a web page using headless Chrome and store it in an image buffer. Pass width and height to override the default 1280x800 viewport size (useful for testing responsive layouts).")
    def web_snapshot(self, url: str, width: int = 1280, height: int = 800, timeout: int = 30) -> str:
        """Capture a screenshot of a web page and store it in an image buffer."""
        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with http:// or https://."

        output_path = self.temp_dir / f"Snapshot_{Path(url).name}_{time_mod.time_ns()}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            proc = subprocess.run(
                (
                    _CHROME_BIN,
                    "--headless=new",
                    "--no-sandbox",
                    f"--screenshot={output_path}",
                    f"--window-size={width},{height}",
                    "--disable-gpu",
                    f"--timeout-ms={timeout * 1000}",
                    url,
                ),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if proc.returncode != 0 or not output_path.is_file():
                error_detail = (proc.stderr or "Chrome exited unexpectedly").strip()
                return f"Error capturing screenshot: {error_detail}"
            image_array = cv2.imread(str(output_path))
            output_path.unlink()
            if image_array is None:
                return f"Error: failed to read screenshot for '{url}'."
            buffer_name = f"image:snapshot:{url}"
            self._store_np_buffer(buffer_name, image_array)
            return f"Stored screenshot for '{url}' in image buffer '{buffer_name}'. Use read_np_buffer to view it."
        except subprocess.TimeoutExpired:
            return f"Error: Screenshot timed out after {timeout} seconds."
        except Exception as e:
            return f"Error capturing screenshot: {type(e).__name__}: {e}"

    @tool(description="Web load an image from a URL into the ImageBufferManager. Fetches the image via HTTP and stores it in an image buffer (keyed by image:url:{url}).")
    async def web_load_image(self, url: str, runner) -> str:
        if not url.startswith(("http://", "https://")):
            return "Error: URL must start with http:// or https://."
        try:
            import httpx
        except ImportError:
            return "Error: httpx not available."
        try:
            resp = httpx.get(url, timeout=10.0)
            resp.raise_for_status()
        except Exception as e:
            return f"Error: failed to fetch '{url}': {e}"
        nparr = np.frombuffer(resp.content, dtype=np.uint8)
        arr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if arr is None or arr.size == 0:
            return f"Error: failed to decode image from '{url}'."
        buffer_name = f"image:url:{url}"
        self._store_np_buffer(buffer_name, arr)
        return f"Loaded image from '{url}' ({arr.shape[1]}x{arr.shape[0]}) into buffer '{buffer_name}'. Use read_np_buffer to view it."
