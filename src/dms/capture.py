"""Fontes de vídeo. Câmera descarta quadros antigos; arquivo entrega todos."""

import threading
import time
from typing import NamedTuple

import cv2
import numpy as np


class Frame(NamedTuple):
    bgr: np.ndarray
    timestamp_ms: int


class VideoSource:
    """Replay determinístico: o timestamp vem do índice do quadro, não do relógio."""

    def __init__(self, path: str, step: int = 1, loop: bool = False):
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise OSError(f"não foi possível abrir o vídeo: {path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._step = step
        self._loop = loop
        self._index = 0

    def read(self) -> Frame | None:
        ok, bgr = self._cap.read()
        if not ok and self._loop and self._index > 0:
            # O índice segue crescendo: o MediaPipe exige timestamps monotônicos.
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, bgr = self._cap.read()
        if not ok:
            return None
        timestamp_ms = int(self._index * 1000 / self._fps)
        self._index += self._step
        for _ in range(self._step - 1):
            self._cap.grab()
        return Frame(bgr, timestamp_ms)

    def close(self) -> None:
        self._cap.release()


class CameraSource:
    """Uma thread lê a câmera sem parar e guarda só o quadro mais recente."""

    def __init__(self, index: int, width: int, height: int, fps: int):
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise OSError(f"não foi possível abrir a câmera: {index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cond = threading.Condition()
        self._latest: Frame | None = None
        self._running = True
        self.captured = 0
        self.delivered = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, bgr = self._cap.read()
            timestamp_ms = time.monotonic_ns() // 1_000_000
            with self._cond:
                if not ok:
                    self._running = False
                else:
                    self._latest = Frame(bgr, timestamp_ms)
                    self.captured += 1
                self._cond.notify_all()

    def read(self) -> Frame | None:
        with self._cond:
            while self._latest is None and self._running:
                self._cond.wait(timeout=1.0)
            frame, self._latest = self._latest, None
            if frame is not None:
                self.delivered += 1
            return frame

    def close(self) -> None:
        with self._cond:
            self._running = False
            self._cond.notify_all()
        self._thread.join(timeout=2.0)
        self._cap.release()


def open_source(source: str, capture_cfg: dict, step: int = 1, loop: bool = False):
    """`source` numérico abre a câmera com esse índice; o resto é caminho de vídeo."""
    if source.isdigit():
        return CameraSource(
            int(source),
            capture_cfg["width"],
            capture_cfg["height"],
            capture_cfg["camera_fps"],
        )
    return VideoSource(source, step=step, loop=loop)


def preprocess(bgr: np.ndarray, capture_cfg: dict) -> np.ndarray:
    """Recorta o lado do motorista, limita a resolução e converte para RGB."""
    roi = capture_cfg.get("driver_roi")
    if roi is not None:
        height, width = bgr.shape[:2]
        x, y, w, h = roi
        bgr = bgr[
            int(y * height) : int((y + h) * height),
            int(x * width) : int((x + w) * width),
        ]
    scale = min(
        capture_cfg["width"] / bgr.shape[1],
        capture_cfg["height"] / bgr.shape[0],
    )
    if scale < 1.0:
        size = (round(bgr.shape[1] * scale), round(bgr.shape[0] * scale))
        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
