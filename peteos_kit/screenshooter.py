"""Screenshooter - capture the screen and store images in the ImageBufferManager."""

from __future__ import annotations

import cv2
import mss
import numpy as np

from peteos import AgenticObject, tool

from .image_buffer_manager import ImageBufferManager, ImageRegion


class Screenshooter(ImageBufferManager, AgenticObject):
    """
    You are a screen capture tool. You take screenshots of the full screen or a specific
    region and store them as named image buffers. Use list_np_buffers to see stored buffers
    and read_np_buffer to send a captured image to the LLM for reasoning.
    """

    @tool(description="Capture the full screen or a specific pixel region and store it as a named image buffer. Pass an ImageRegion with x1, y1 (top-left) and x2, y2 (bottom-right) to capture only that region. Omit the region to capture the full screen. The image is stored under the given name and can be read with read_np_buffer.")
    def take_screenshot(self, name: str, region: ImageRegion | None = None) -> str:
        """Capture screen (full or region) and store in the image buffer."""
        with mss.MSS() as s:
            if region is None:
                monitor = s.monitors[0]
            else:
                monitor = (region.x1, region.y1, region.x2, region.y2)
            screenshot = s.grab(monitor)

        array_bgr = cv2.cvtColor(np.asarray(screenshot), cv2.COLOR_BGRA2BGR)

        buffer_name = name if name.startswith("image:") else f"image:{name}"
        self._store_np_buffer(buffer_name, array_bgr)

        h, w = array_bgr.shape[:2]
        return f"Screenshot stored as '{buffer_name}' ({w}x{h} pixels). Use read_np_buffer to send it to the LLM."
