import threading

import cv2
import numpy as np
import pytest

from dms.capture import CameraSource, VideoSource, preprocess

CFG = {"width": 640, "height": 480, "driver_roi": None}


@pytest.fixture
def video(tmp_path):
    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for i in range(6):
        writer.write(np.full((48, 64, 3), i * 40, np.uint8))
    writer.release()
    return path


def timestamps(source):
    out = []
    while (frame := source.read()) is not None:
        out.append(frame.timestamp_ms)
    return out


def test_video_timestamps_follow_frame_index(video):
    assert timestamps(VideoSource(video)) == [0, 100, 200, 300, 400, 500]


def test_video_step_downsamples_without_changing_clock(video):
    assert timestamps(VideoSource(video, step=2)) == [0, 200, 400]


def test_video_loop_keeps_timestamps_monotonic(video):
    source = VideoSource(video, loop=True)
    stamps = [source.read().timestamp_ms for _ in range(10)]
    assert stamps == sorted(set(stamps))
    assert stamps[-1] == 900


def test_video_loop_with_step_keeps_timestamps_monotonic(video):
    source = VideoSource(video, step=4, loop=True)
    stamps = [source.read().timestamp_ms for _ in range(5)]
    assert stamps == sorted(set(stamps))


def test_preprocess_converts_to_rgb():
    bgr = np.zeros((48, 64, 3), np.uint8)
    bgr[..., 0] = 255
    rgb = preprocess(bgr, CFG)
    assert rgb.shape == (48, 64, 3)
    assert rgb[0, 0].tolist() == [0, 0, 255]


def test_preprocess_never_upscales_and_keeps_aspect():
    assert preprocess(np.zeros((240, 320, 3), np.uint8), CFG).shape == (240, 320, 3)
    assert preprocess(np.zeros((720, 1280, 3), np.uint8), CFG).shape == (360, 640, 3)


def test_preprocess_crops_driver_roi():
    bgr = np.zeros((100, 200, 3), np.uint8)
    bgr[:, 100:] = 255
    rgb = preprocess(bgr, {**CFG, "driver_roi": [0.5, 0.0, 0.5, 1.0]})
    assert rgb.shape == (100, 100, 3)
    assert rgb.min() == 255


class FakeCapture:
    """Entrega quadros numerados sob comando, como uma câmera mais rápida que o consumidor."""

    def __init__(self):
        self.ready = threading.Semaphore(0)
        self.value = 0

    def read(self):
        self.ready.acquire()
        self.value += 1
        return self.value <= 3, np.full((2, 2, 3), self.value, np.uint8)

    def release(self):
        pass


def test_camera_delivers_only_latest_frame():
    source = CameraSource.__new__(CameraSource)
    fake = FakeCapture()
    source._cap = fake
    source._cond = threading.Condition()
    source._latest = None
    source._running = True
    source.captured = source.delivered = 0
    source._thread = threading.Thread(target=source._loop, daemon=True)

    # Três quadros chegam antes da primeira leitura: os dois antigos são descartados.
    source._thread.start()
    fake.ready.release(4)
    source._thread.join(timeout=2.0)

    frame = source.read()
    assert frame.bgr[0, 0, 0] == 3
    assert source.read() is None
    assert source.captured - source.delivered == 2
