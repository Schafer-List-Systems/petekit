from __future__ import annotations

import cv2

from peteos import AgenticObject, tool

from .numpy_buffer_manager import NumPyBufferManager


class ImageBufferManager(NumPyBufferManager, AgenticObject):
    """
    You are an image buffer manager.
    - Only call read_np_buffer for buffers you know to contain image data.
    - Never call it for arbitrary NumPy arrays.
    """

    @tool(description="Read a NumPy buffer as an image and send it to the LLM for reasoning. Pass xy1/xy2 for pixel-based region cropping (0-based, inclusive). xy1 is top-left corner, xy2 is bottom-right corner. Both default to the full image extents.")
    async def read_np_buffer(self, name: str, runner, xy1: tuple[int, int] | None = None, xy2: tuple[int, int] | None = None) -> str:
        """Read a buffer (or region) and send it to the LLM. Returns a hint; the LLM sees the image directly."""
        if name not in self._buffers:
            return f"Error: no buffer named '{name}'. Use copy_np_buffer or an inheriting agent to create it."
        buf = self._buffers[name]
        array = buf.array

        if xy1 is None and xy2 is None:
            region = array
        else:
            x1 = 0 if xy1 is None else xy1[0]
            y1 = 0 if xy1 is None else xy1[1]
            x2 = array.shape[1] - 1 if xy2 is None else xy2[0]
            y2 = array.shape[0] - 1 if xy2 is None else xy2[1]
            if x1 > x2 or y1 > y2:
                return f"Error: resolved region x1={x1} > x2={x2} or y1={y1} > y2={y2}. Buffer shape is {array.shape}."
            if x2 >= array.shape[1] or y2 >= array.shape[0]:
                return f"Error: resolved x2={x2} or y2={y2} exceeds buffer shape {array.shape}."
            region = array[y1:y2+1, x1:x2+1]

        total_pixels = region.shape[0] * region.shape[1]
        if total_pixels > NumPyBufferManager._MAX_PIXELS:
            actual_w, actual_h = array.shape[1], array.shape[0]
            return (
                f"Buffer '{name}' has {total_pixels} pixels, exceeding the {NumPyBufferManager._MAX_PIXELS}-pixel limit "
                f"(image is {actual_w}x{actual_h}={actual_w*actual_h} pixels). "
                f"Use a smaller xy1/xy2 region to stay under the limit."
            )

        encode_ok, png_bytes = cv2.imencode(".png", region)
        if not encode_ok:
            return f"Error: cv2.imencode failed for buffer '{name}'."
        image_data = png_bytes.tobytes()

        await self._send_media(
            data=image_data,
            mime_type="image/png",
            runner=runner,
        )
        return f"Buffer '{name}' ({region.shape}) sent to LLM for reasoning."
