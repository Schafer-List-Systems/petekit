"""Screenshooter - capture the screen and store images in the ImageBufferManager."""

from __future__ import annotations

import cv2
import mss
import numpy as np

from peteos import AgenticObject, tool

from .image_buffer_manager import ImageBufferManager, ImageRegion


class Screenshooter(ImageBufferManager, AgenticObject):
    """You can call the take_screenshoot tool to take a screenshoot and store it into a numby buffer.
    """

    @tool
    def take_screenshot(self, name: str, region: ImageRegion | None = None) -> str:
        """
        Take a screenshot and store it into a numby buffer.
        Pass region as a dict with keys x1, y1 (top-left corner) and x2, y2 (bottom-right corner) to capture only that pixel region.
        Omit region entirely to capture the full screen.
        When calling read_np_buffer, use the exact buffer name from the return message.
        """
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
        return f"Screenshot stored in buffer '{buffer_name}' ({w}x{h} pixels). Use read_np_buffer to send it to the LLM."
