"""V4L2CameraDriver - V4L2 camera capture via linuxpy.video library.

Pure Python V4L2 access via linuxpy.video. Negotiates 1920x1080 MJPG format
directly with the V4L2 device, bypassing OpenCV's V4L2 backend.
"""

from __future__ import annotations

import logging
import mmap
import select
from typing import TYPE_CHECKING

import cv2
import numpy as np

from peteos_kit.utils.camera_driver.camera_driver import CameraDriver

if TYPE_CHECKING:
    pass

_logger = logging.getLogger(__name__)
if not _logger.handlers and not logging.getLogger().handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logging.getLogger().addHandler(_handler)
    logging.getLogger().setLevel(logging.INFO)


class V4L2CameraDriver(CameraDriver):
    """V4L2 driver via linuxpy.video, bypassing OpenCV's V4L2 backend."""

    driver_name = "linuxpy-v4l2"

    def __init__(self) -> None:
        super().__init__()
        self._device = None
        self._width = 1920
        self._height = 1080

    def list_cameras(self) -> list[tuple[int, str]]:
        try:
            from linuxpy.video import device as vd
            cameras = []
            for d in vd.iter_video_capture_devices():
                cameras.append((d.index, d.filename))
            return cameras
        except Exception:
            return []

    def open(self, camera_id: int, width: int = 1920, height: int = 1080) -> bool:
        try:
            from linuxpy.video import device as vd_module
            from linuxpy.video.device import BufferType, ImageFormat, Memory
        except Exception as e:
            _logger.error("V4L2CameraDriver: linuxpy.video not available: %s", e)
            return False

        self._device = vd_module.Device.from_id(camera_id)
        self._device.open()

        self._width = width
        self._height = height

        try:
            self._device.set_format(
                BufferType.VIDEO_CAPTURE,
                ImageFormat.MJPG,
                width=width,
                height=height,
            )
        except Exception:
            pass

        fmt = self._device.get_format(BufferType.VIDEO_CAPTURE)
        self._width = fmt.width
        self._height = fmt.height

        self._buffers = self._device.create_buffers(
            BufferType.VIDEO_CAPTURE, Memory.MMAP, count=4
        )
        for b in self._buffers:
            self._device.enqueue_buffer(
                BufferType.VIDEO_CAPTURE, Memory.MMAP, b.length, b.index
            )

        self._device.stream_on(BufferType.VIDEO_CAPTURE)

        _logger.info(
            "V4L2CameraDriver: camera %d open at %dx%d.",
            camera_id, self._width, self._height,
        )
        return True

    def close(self) -> None:
        if self._device is None:
            return
        try:
            from linuxpy.video.device import BufferType
            self._device.stream_off(BufferType.VIDEO_CAPTURE)
            self._device.close()
        except Exception:
            pass
        self._device = None

    def grab_frame(self) -> tuple[bool, np.ndarray, tuple[int, int]]:
        if self._device is None:
            return False, np.empty((0, 0, 3)), (0, 0)

        fd = self._device.fileno()
        for _ in range(10):
            ready, _, _ = select.select([fd], [], [], 2.0)
            if not ready:
                return False, np.empty((0, 0, 3)), (0, 0)
            try:
                from linuxpy.video.device import BufferType, Memory
                buf = self._device.dequeue_buffer(BufferType.VIDEO_CAPTURE, Memory.MMAP)
            except Exception:
                return False, np.empty((0, 0, 3)), (0, 0)
            if buf.bytesused > 0 and buf.m.offset >= 0:
                try:
                    with mmap.mmap(fd, buf.length, offset=0) as mm:
                        data = bytes(mm[buf.m.offset:buf.m.offset + buf.bytesused])
                    if len(data) >= 2 and data[0] == 0xFF and data[1] == 0xD8:
                        arr = cv2.imdecode(
                            np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR
                        )
                        if arr is not None and arr.size > 0:
                            try:
                                self._device.enqueue_buffer(
                                    BufferType.VIDEO_CAPTURE,
                                    Memory.MMAP,
                                    buf.length,
                                    buf.index,
                                )
                            except Exception:
                                pass
                            return True, arr, (self._width, self._height)
                except Exception:
                    pass
            try:
                self._device.enqueue_buffer(
                    BufferType.VIDEO_CAPTURE, Memory.MMAP, buf.length, buf.index
                )
            except Exception:
                pass
        return False, np.empty((0, 0, 3)), (0, 0)
