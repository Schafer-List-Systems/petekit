"""CameraObserver - webcam access via a CameraDriver, backed by ImageBufferManager."""

from __future__ import annotations

import cv2

from peteos import AgenticObject
from peteos.oap.decorators import tool

from .camera_driver import V4L2CameraDriver
from .image_buffer_manager import ImageBufferManager


class CameraObserver(ImageBufferManager, AgenticObject):
    """You are an observer with access to cameras.

    Only one camera can be open at a time. grab_image stores frames in ImageBufferManager.
    Use read_np_buffer to send a captured frame to the LLM for reasoning.
    """

    def __init__(
        self,
        driver: CameraDriver | None = None,
        scaling: int | None = None,
    ) -> None:
        super().__init__()
        self._driver: CameraDriver = driver or V4L2CameraDriver()
        self._scaling: int | None = scaling
        self._active_camera_id: int | None = None

    def _resize_if_needed(self, frame: cv2.Mat) -> cv2.Mat:
        """Resize frame if scaling is configured."""
        if self._scaling is None or self._scaling <= 0:
            return frame
        h, w = frame.shape[:2]
        if w > h:
            new_w = self._scaling
            new_h = round(h * self._scaling / w)
        else:
            new_h = self._scaling
            new_w = round(w * self._scaling / h)
        return cv2.resize(frame, (new_w, new_h))

    @tool(description="List all available cameras.")
    def list_cameras(self) -> str:
        cameras = self._driver.list_cameras()
        if not cameras:
            return "No cameras found."
        return "Available cameras:\n" + "\n".join(
            f"  Camera ID: {cid}, Driver: {self._driver.driver_name}, Description: {name}"
            for cid, name in cameras
        )

    @tool(description="Open a camera by ID. Must be called before grab_image.")
    def open_camera(self, camera_id: int) -> str:
        if self._active_camera_id is not None:
            return "ERROR: A camera is already open. Call close_camera first."

        ok = self._driver.open(camera_id)
        if not ok:
            return f"ERROR: Camera {camera_id} could not be opened."

        self._active_camera_id = camera_id

        ok, frame, _ = self._driver.grab_frame()
        if not ok or frame is None:
            self._driver.close()
            self._active_camera_id = None
            return f"ERROR: Camera {camera_id} warm-up failed."

        return f"OK: Camera {camera_id} (driver: {self._driver.driver_name}) opened."

    @tool(description="Close the currently open camera.")
    def close_camera(self) -> str:
        if self._active_camera_id is None:
            return "No camera is currently open."

        self._driver.close()
        self._active_camera_id = None
        return "OK: Camera closed."

    @tool(description="Capture a single image from the open camera and store it in the ImageBufferManager. Pass buffer_name to override the default name 'image:camera:<id>'.")
    def grab_image(self, buffer_name: str | None = None) -> str:
        if self._active_camera_id is None:
            return "ERROR: No camera is open."

        ok, frame, _ = self._driver.grab_frame()
        if not ok or frame is None:
            return "ERROR: Failed to capture image."

        frame = self._resize_if_needed(frame)

        name = buffer_name if buffer_name is not None else f"image:camera:{self._active_camera_id}"
        self._store_np_buffer(name, frame)

        return f"OK: Image captured ({frame.shape[1]}x{frame.shape[0]}), stored as '{name}'."
