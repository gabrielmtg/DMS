import json
from pathlib import Path

import cv2
import numpy as np

from dms.crop import crop_box, phone_crop

GOLDEN = Path(__file__).parent / "golden" / "crop"


def landmarks_for(box, width=640, height=480):
    x0, y0, x1, y1 = box
    pixels = np.array([(x0, y0), (x1, y1), ((x0 + x1) / 2, (y0 + y1) / 2)])
    return pixels / (width, height)


def test_box_uses_largest_side_and_shifts_down():
    # Caixa 100 x 80: s = 100, lado 200, centro (150, 140) descido 10.
    assert crop_box(landmarks_for((100, 100, 200, 180)), 640, 480, 2.0, 0.1) == (50, 50, 200)


def test_rounding_ties_go_up_even_when_negative():
    assert crop_box(landmarks_for((102.5, 150, 202.5, 225)), 640, 480, 2.0, 0.0) == (53, 88, 200)
    assert crop_box(landmarks_for((-47.5, 150, 52.5, 225)), 640, 480, 2.0, 0.0) == (-97, 88, 200)


def test_outside_the_image_is_black():
    rgb = np.full((480, 640, 3), 255, np.uint8)
    # Lado 160 = tamanho de saída, então o resize não mistura os pixels da borda.
    crop = phone_crop(rgb, landmarks_for((-40, 100, 40, 180)), 2.0, 0.0, 160)
    assert crop.shape == (160, 160, 3)
    assert crop[:, :80].max() == 0
    assert crop[:, 80:].min() == 255


def test_small_face_is_upscaled_to_output_size():
    rgb = np.zeros((480, 640, 3), np.uint8)
    assert phone_crop(rgb, landmarks_for((300, 200, 320, 220)), 2.2, 0.1, 160).shape == (160, 160, 3)


def test_matches_golden_files():
    golden = json.loads((GOLDEN / "cases.json").read_text())
    rgb = cv2.cvtColor(cv2.imread(str(GOLDEN / golden["image"])), cv2.COLOR_BGR2RGB)
    # O resize pode variar 1 nível de cinza entre versões do OpenCV.
    tolerance = 0 if cv2.__version__ == golden["opencv"] else golden["tolerance"]
    for case in golden["cases"]:
        landmarks = np.array(case["landmarks"], np.float32)
        box = crop_box(landmarks, golden["width"], golden["height"], case["scale"], case["shift"])
        assert list(box) == case["box"], case["name"]
        crop = phone_crop(rgb, landmarks, case["scale"], case["shift"], golden["size"])
        expected = cv2.cvtColor(cv2.imread(str(GOLDEN / case["crop"])), cv2.COLOR_BGR2RGB)
        diff = np.abs(crop.astype(np.int16) - expected.astype(np.int16)).max()
        assert diff <= tolerance, case["name"]
