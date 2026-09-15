"""CV2CameraDriver - portable V4L2 camera driver via OpenCV.

Unlike OpenCVCameraDriver, this driver does not shell out to
v4l2-ctl. It simply tests indices 0-9 with OpenCV, making it
portable across Linux, Windows, and macOS.
"""

from __future__ import annotations

import logging
import re
import subprocess
from typing import TYPE_CHECKING

import cv2
import numpy as np

from petekit.utils.camera_driver.camera_driver import CameraDriver

_logger = logging.getLogger(__name__)
if not _logger.handlers and not logging.getLogger().handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logging.getLogger().addHandler(_handler)
    logging.getLogger().setLevel(logging.INFO)

if TYPE_CHECKING:
    pass


class CV2CameraDriver(CameraDriver):
    """Portable V4L2 driver backed by OpenCV only."""

    driver_name = "opencv-cv2"

    def list_cameras(self) -> list[tuple[int, str]]:
        cameras: list[tuple[int, str]] = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append((i, ""))
                cap.release()
        return cameras
        # NOTE: Probing only 0-9 is a heuristic. Device indices can have gaps
        # or exceed 9 on some systems. This is the common convention in
        # OpenCV's ecosystem for quick device discovery.

    def open(self, camera_id: int) -> bool:
        modes = self._get_available_modes(camera_id)
        best_w, best_h, best_fmt = 1920, 1080, "MJPG"
        if modes:
            best_w, best_h = max(modes, key=lambda s: s[0] * s[1])
        pipeline = (
            f"v4l2src device=/dev/video{camera_id} ! "
            f"image/jpeg, width={best_w}, height={best_h} ! "
            f"jpegdec ! appsink"
        )
        try:
            cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        except Exception:
            _logger.warning("GStreamer pipeline open failed for camera %d — falling back to raw V4L2.", camera_id)
            cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return False
        actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        _logger.info("Camera %d: GStreamer pipeline, selected %dx%d %s, actual %dx%d.", camera_id, best_w, best_h, best_fmt, int(actual_w), int(actual_h))

        self._cap = cap
        return True

    def close(self) -> None:
        if hasattr(self, "_cap"):
            self._cap.release()
            del self._cap

    def grab_frame(self) -> tuple[bool, np.ndarray, tuple[int, int]]:
        if not hasattr(self, "_cap"):
            return False, np.empty((0, 0, 3)), (0, 0)
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return False, np.empty((0, 0, 3)), (0, 0)
        return True, frame, (frame.shape[1], frame.shape[0])

    def _apply_device_format(self, camera_id: int, width: int, height: int, pixel_format: str) -> bool:
        try:
            result = subprocess.run(
                ["v4l2-ctl", "-d", str(camera_id), "--set-fmt-video", f"width={width},height={height},pixelformat={pixel_format}"],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired):
            _logger.warning("v4l2-ctl not available for format set on camera %d.", camera_id)
            return False
        if result.returncode != 0:
            _logger.warning("v4l2-ctl format set failed for camera %d: %s.", camera_id, result.stderr.strip())
            return False
        return True

    def _get_available_modes(self, camera_id: int) -> list[tuple[int, int]]:
        try:
            formats_result = subprocess.run(
                ["v4l2-ctl", "-d", str(camera_id), "--list-formats"],
                capture_output=True,
                text=True,
                timeout=2,
            )
        except OSError:
            _logger.warning("v4l2-ctl not available — resolution negotiation unavailable.")
            return []
        except subprocess.TimeoutExpired:
            _logger.warning("v4l2-ctl timed out for camera %d — resolution negotiation unavailable.", camera_id)
            return []
        if formats_result.returncode != 0:
            return []

        format_names: list[str] = []
        for line in formats_result.stdout.splitlines():
            m = re.search(r"\[\d+\]:\s*'(\w+)'", line)
            if m:
                format_names.append(m.group(1))

        if not format_names:
            return []

        modes: list[tuple[int, int]] = []
        for fmt in format_names:
            try:
                sizes_result = subprocess.run(
                    ["v4l2-ctl", "-d", str(camera_id), f"--list-framesizes={fmt}"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            if sizes_result.returncode != 0:
                continue
            for line in sizes_result.stdout.splitlines():
                sm = re.search(r"Size:\s*Discrete\s*(\d+)x(\d+)", line)
                if sm:
                    modes.append((int(sm.group(1)), int(sm.group(2))))

        return modes