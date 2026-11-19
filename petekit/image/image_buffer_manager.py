"""Image buffer manager -- reads image-like numpy buffers and sends them to the LLM."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from peteos import AgenticObject, tool
from peteos.engine import Runner

from petekit.numpy_buffer_manager import NumPyBufferManager


# -------------------------------------------------------------------------------------------------------------------- #
# Data Classes                                                                                                         #
# -------------------------------------------------------------------------------------------------------------------- #


@dataclass
class ImageRegion:
    """Pixel-based region of an image. Coordinates are 0-based and inclusive. (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner."""
    x1: int
    y1: int
    x2: int
    y2: int


def _compute_zoom_factor(pixels: int, max_pixels: int) -> float:
    """Compute the downscaling factor that fits the pixel budget.

    Returns 1.0 when the pixels already fit. Otherwise the square root of
    the budget ratio -- scaling both dimensions by it reduces the pixel
    count to the budget.
    """
    # Within budget -- no scaling needed.
    if pixels <= max_pixels:
        return 1.0

    # Scale both dimensions by the square root of the budget ratio.
    return (max_pixels / pixels) ** 0.5


def _apply_zoom(array: np.ndarray, zoom_factor: float) -> np.ndarray:
    """Downscale the array by the given zoom factor.

    Returns the array unchanged when the factor is 1.0 or greater. The
    scaled dimensions are clamped to at least one pixel per dimension.
    """
    # No scaling needed -- return the array unchanged.
    if zoom_factor >= 1.0:
        return array

    # Compute the scaled dimensions, clamped to at least one pixel.
    h, w = array.shape[:2]
    new_h, new_w = max(1, int(h * zoom_factor)), max(1, int(w * zoom_factor))

    # INTER_AREA is the correct interpolation for shrinking.
    return cv2.resize(array, (new_w, new_h), interpolation=cv2.INTER_AREA)


# -------------------------------------------------------------------------------------------------------------------- #
# ImageBufferManager                                                                                                   #
# -------------------------------------------------------------------------------------------------------------------- #


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

    @tool(description="Read a NumPy buffer as an image and send it to the LLM for reasoning. Pass an ImageRegion for pixel-based region cropping (0-based, inclusive): (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner. Omit the region to read the full image. Set auto_zoom to False to disable automatic downscaling on large images.")
    async def read_np_buffer(
        self, name: str, runner, region: ImageRegion | None = None, auto_zoom: bool = True
    ) -> str:
        """Read a buffer (or region) and send it to the LLM for reasoning.

        Returns a hint naming the buffer, the delivered dimensions, and the
        applied zoom factor. The image itself reaches the LLM through the
        media channel, not through this return value.
        """
        # Look up the named buffer and resolve its array.
        if name not in self._numpy_buffers:
            return f"Error: no buffer named '{name}'. Use copy_np_buffer or an inheriting agent to create it."
        buf = self._numpy_buffers[name]
        array = buf.array

        # Guard that the buffer is image-like -- other dimensionalities cannot
        # be read as images and would crash on the shape unpacking below.
        if array.ndim not in (2, 3):
            return f"Error: buffer '{name}' has {array.ndim} dimensions -- only 2-D (grayscale) or 3-D (channel-last) arrays can be read as images."

        # Crop the requested region, or read the full array when no region is given.
        if region is None:
            region_array = array
        else:
            # Unpack the region corners for validation.
            x1, y1, x2, y2 = region.x1, region.y1, region.x2, region.y2

            # Reject an inverted region -- the corners must be ordered.
            if x1 > x2 or y1 > y2:
                return f"Error: region x1={x1} > x2={x2} or y1={y1} > y2={y2}. Buffer shape is {array.shape}."

            # Reject negative coordinates -- Python negative indexing would silently read from the wrong end.
            if x1 < 0 or y1 < 0:
                return f"Error: region coordinates must be non-negative -- got x1={x1}, y1={y1}."

            # Reject a region reaching past the image bounds.
            if x2 >= array.shape[1] or y2 >= array.shape[0]:
                return f"Error: x2={x2} or y2={y2} exceeds buffer shape {array.shape}."

            # Slice the region -- the inclusive corners become an exclusive end.
            region_array = array[y1:y2+1, x1:x2+1]

        # Count the pixels under inspection to compare against the pixel budget.
        total_pixels = region_array.shape[0] * region_array.shape[1]
        zoom_factor = 1.0

        # Over-budget path: downscale when auto_zoom is on, reject otherwise.
        if total_pixels > NumPyBufferManager._MAX_PIXELS:
            if auto_zoom:
                # Compute the zoom factor and apply it to the region.
                zoom_factor = _compute_zoom_factor(total_pixels, NumPyBufferManager._MAX_PIXELS)
                region_array = _apply_zoom(region_array, zoom_factor)
            else:
                # Report the full image dimensions so a smaller region can be picked.
                actual_w, actual_h = array.shape[1], array.shape[0]
                return (
                    f"Requested region for buffer '{name}' has {total_pixels} pixels, exceeding the {NumPyBufferManager._MAX_PIXELS}-pixel limit "
                    f"(image is {actual_w}x{actual_h}={actual_w*actual_h} pixels). "
                    f"Use a smaller ImageRegion or enable auto_zoom."
                )

        # Encode the region as PNG -- the media channel carries the image to the LLM.
        encode_ok, png_bytes = cv2.imencode(".png", region_array)
        if not encode_ok:
            return f"Error: cv2.imencode failed for buffer '{name}'."
        image_data = png_bytes.tobytes()

        # Send the image to the runner's session -- the return value is only a hint.
        await self._send_media(
            data=image_data,
            mime_type="image/png",
            runner=runner,
        )

        # Report the applied zoom so the agent knows the read was downscaled.
        zoom_hint = f", zoom={zoom_factor:.2f}x" if zoom_factor < 1.0 else ", no zoom applied"
        return f"Buffer '{name}' ({region_array.shape[1]}x{region_array.shape[0]}{zoom_hint}) sent to LLM for reasoning."
