from __future__ import annotations

from dataclasses import dataclass

import cv2

from peteos import AgenticObject, tool

from petekit.numpy_buffer_manager import NumPyBufferManager


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
    - Auto-zoom is enabled by default: if the image/region exceeds the pixel limit, it is automatically downscaled
      and the zoom factor is reported in the return value. For sharper regions.
    - Once an image is in a numpy buffer, you can read it multiple times for analysis.
      - The auto-zoom allows you to get an overview. To get sharper results, you can try to look at smaller regions.
    """

    def _compute_zoom_factor(self, pixels: int, max_pixels: int) -> float:
        if pixels <= max_pixels:
            return 1.0
        return (max_pixels / pixels) ** 0.5

    def _apply_zoom(self, array, zoom_factor: float):
        if zoom_factor >= 1.0:
            return array
        h, w = array.shape[:2]
        new_h, new_w = max(1, int(h * zoom_factor)), max(1, int(w * zoom_factor))
        return cv2.resize(array, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @tool(description="Read a NumPy buffer as an image and send it to the LLM for reasoning. Pass an ImageRegion for pixel-based region cropping (0-based, inclusive): (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner. Omit the region to read the full image. Set auto_zoom to False to disable automatic downscaling on large images.")
    async def read_np_buffer(
        self, name: str, runner, region: ImageRegion | None = None, auto_zoom: bool = True
    ) -> str:
        """Read a buffer (or region) and send it to the LLM. Returns a hint; the LLM sees the image directly."""
        if name not in self._numpy_buffers:
            return f"Error: no buffer named '{name}'. Use copy_np_buffer or an inheriting agent to create it."
        buf = self._numpy_buffers[name]
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
        zoom_factor = 1.0
        if total_pixels > NumPyBufferManager._MAX_PIXELS:
            if auto_zoom:
                zoom_factor = self._compute_zoom_factor(total_pixels, NumPyBufferManager._MAX_PIXELS)
                region_array = self._apply_zoom(region_array, zoom_factor)
            else:
                actual_w, actual_h = array.shape[1], array.shape[0]
                return (
                    f"Requested region for buffer '{name}' has {total_pixels} pixels, exceeding the {NumPyBufferManager._MAX_PIXELS}-pixel limit "
                    f"(image is {actual_w}x{actual_h}={actual_w*actual_h} pixels). "
                    f"Use a smaller ImageRegion or enable auto_zoom."
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
        zoom_hint = f", zoom={zoom_factor:.2f}x" if zoom_factor < 1.0 else ", no zoom applied"
        return f"Buffer '{name}' ({region_array.shape[1]}x{region_array.shape[0]}{zoom_hint}) sent to LLM for reasoning."
