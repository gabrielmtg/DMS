"""Crop do classificador de celular.

A regra é contrato com o runtime em C++: qualquer mudança aqui precisa ser repetida
lá e os golden files (scripts/export_crop_golden.py) precisam ser gerados de novo.
"""

import math

import cv2
import numpy as np


def _round(value: float) -> int:
    # Nem round() do Python (empate vai para o par) nem std::round (empate se afasta
    # do zero): floor(x + 0.5) é fácil de escrever igual nas duas linguagens.
    return math.floor(value + 0.5)


def crop_box(
    landmarks: np.ndarray, width: int, height: int, scale: float, shift: float
) -> tuple[int, int, int]:
    """Quadrado (left, top, side) em pixels a partir dos landmarks normalizados.

    s é o maior lado da caixa dos landmarks; o quadrado tem lado scale * s e fica
    centrado na caixa, descido shift * s. Contas em double.
    """
    xy = landmarks[:, :2].astype(np.float64) * (width, height)
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    s = max(x1 - x0, y1 - y0)
    side = scale * s
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2 + shift * s
    return _round(cx - side / 2), _round(cy - side / 2), max(1, _round(side))


def phone_crop(
    rgb: np.ndarray, landmarks: np.ndarray, scale: float, shift: float, size: int
) -> np.ndarray:
    """Recorta o quadrado e redimensiona para size x size. Fora da imagem fica preto."""
    height, width = rgb.shape[:2]
    left, top, side = crop_box(landmarks, width, height, scale, shift)
    crop = np.zeros((side, side, 3), np.uint8)
    x0, y0 = max(left, 0), max(top, 0)
    x1, y1 = min(left + side, width), min(top + side, height)
    if x0 < x1 and y0 < y1:
        crop[y0 - top : y1 - top, x0 - left : x1 - left] = rgb[y0:y1, x0:x1]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
