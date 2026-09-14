from __future__ import annotations

from dataclasses import dataclass

import cv2

from peteos import AgenticObject, tool

from .numpy_buffer_manager import NumPyBufferManager


@dataclass
class ImageRegion:
    """Pixel-based region of an image. Coordinates are 0-based and inclusive. (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner."""
    x1: int
    y1: int
    x2: int
    y2: int


class ImageBufferManager(NumPyBufferManager, AgenticObject):
    """
    - For image-like numpy buffers, you can call read_np_buffer to have a look at them.
    - The read_np_buffer tool will transform image coordinates into numpy coordinates for you.
    - Image-like buffer names are usually prefixed with "image:"
    """

    @tool(description="Read a NumPy buffer as an image and send it to the LLM for reasoning. Pass an ImageRegion for pixel-based region cropping (0-based, inclusive): (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner. Omit the region to read the full image.")
    async def read_np_buffer(self, name: str, runner, region: ImageRegion | None = None) -> str:
        """Read a buffer (or region) and send it to the LLM. Returns a hint; the LLM sees the image directly."""
        if name not in self._buffers:
            return f"Error: no buffer named '{name}'. Use copy_np_buffer or an inheriting agent to create it."
        buf = self._buffers[name]
        array = buf.array

        if region is None:
            region_array = array
        else:
            x1, y1, x2, y2 = region.x1, region.y1, region.x2, region.y2
            if x1 > x2 or y1 > y2:
                return f"Error: region x1={x1} > x2={x2} or y1={y1} > y2={y2}. Buffer shape is {array.shape}."
            if x2 >= array.shape[1] or y2 >= array.shape[0]:
                return f"Error: x2={x2} or y2={y2} exceeds buffer shape {array.shape}."
            region_array = array[y1:y2+1, x1:x2+1]

        total_pixels = region_array.shape[0] * region_array.shape[1]
        if total_pixels > NumPyBufferManager._MAX_PIXELS:
            actual_w, actual_h = array.shape[1], array.shape[0]
            return (
                f"Requested region for buffer '{name}' has {total_pixels} pixels, exceeding the {NumPyBufferManager._MAX_PIXELS}-pixel limit "
                f"(image is {actual_w}x{actual_h}={actual_w*actual_h} pixels). "
                f"Use a smaller ImageRegion to stay under the limit."
            )

        encode_ok, png_bytes = cv2.imencode(".png", region_array)
        if not encode_ok:
            return f"Error: cv2.imencode failed for buffer '{name}'."
        image_data = png_bytes.tobytes()

        await self._send_media(
            data=image_data,
            mime_type="image/png",
            runner=runner,
        )
        return f"Buffer '{name}' ({region_array.shape}) sent to LLM for reasoning."
