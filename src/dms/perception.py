"""Wrapper do MediaPipe FaceLandmarker."""

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision


class Landmarker:
    """Modo VIDEO: bloqueante e determinístico, então replay e ao vivo se comportam igual."""

    def __init__(self, perception_cfg: dict):
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=perception_cfg["model_path"]),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=perception_cfg["min_face_detection_confidence"],
            min_face_presence_confidence=perception_cfg["min_face_presence_confidence"],
            min_tracking_confidence=perception_cfg["min_tracking_confidence"],
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def infer(self, rgb: np.ndarray, timestamp_ms: int) -> vision.FaceLandmarkerResult:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._landmarker.detect_for_video(image, timestamp_ms)

    @staticmethod
    def to_array(result: vision.FaceLandmarkerResult) -> np.ndarray | None:
        """Retorna (478, 3) em coordenadas normalizadas, ou None sem face."""
        if not result.face_landmarks:
            return None
        return np.array(
            [(p.x, p.y, p.z) for p in result.face_landmarks[0]], dtype=np.float32
        )

    def close(self) -> None:
        self._landmarker.close()
