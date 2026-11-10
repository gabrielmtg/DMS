"""Gera os casos de teste do crop (tests/golden/crop/) para conferir a versão em C++.

Imagem e landmarks são sintéticos: o crop não depende de haver um rosto. Rode de
novo sempre que src/dms/crop.py mudar.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from dms.crop import crop_box, phone_crop

OUT = Path(__file__).resolve().parent.parent / "tests" / "golden" / "crop"
WIDTH, HEIGHT, SIZE = 640, 480, 160

# (nome, caixa dos landmarks em pixels (x0, y0, x1, y1), scale, shift)
CASES = [
    ("centro", (220, 140, 400, 340), 2.2, 0.1),
    ("rosto_pequeno", (300, 200, 350, 260), 2.2, 0.1),
    ("borda_esquerda", (-30, 150, 120, 330), 2.2, 0.1),
    ("canto_inferior_direito", (520, 360, 660, 520), 2.2, 0.1),
    ("rosto_grande", (120, 40, 520, 470), 2.2, 0.1),
    # Empates exatos em .5: floor(x + 0.5) dá left 53 e -97. O round() do Python
    # daria 52 no primeiro, e o std::round do C++ daria -98 no segundo.
    ("empate_positivo", (102.5, 150, 202.5, 225), 2.0, 0.0),
    ("empate_negativo", (-47.5, 150, 52.5, 225), 2.0, 0.0),
]


def synthetic_image() -> np.ndarray:
    """Gradientes e formas, para que um crop ou resize errado apareça na comparação."""
    x = np.linspace(0, 255, WIDTH, dtype=np.float32)
    y = np.linspace(0, 255, HEIGHT, dtype=np.float32)
    rgb = np.empty((HEIGHT, WIDTH, 3), np.uint8)
    rgb[..., 0] = x[None, :]
    rgb[..., 1] = y[:, None]
    rgb[..., 2] = 128
    for i in range(0, WIDTH, 40):
        cv2.line(rgb, (i, 0), (i, HEIGHT - 1), (255, 255, 255), 1)
    for i in range(0, HEIGHT, 40):
        cv2.line(rgb, (0, i), (WIDTH - 1, i), (0, 0, 0), 1)
    rng = np.random.default_rng(0)
    for _ in range(25):
        center = (int(rng.integers(0, WIDTH)), int(rng.integers(0, HEIGHT)))
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        cv2.circle(rgb, center, int(rng.integers(5, 40)), color, -1)
    return rgb


def synthetic_landmarks(box, rng) -> np.ndarray:
    """478 pontos normalizados em float32, como o MediaPipe: os 4 extremos e o resto dentro."""
    x0, y0, x1, y1 = box
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
    extremes = [(x0, ym), (x1, ym), (xm, y0), (xm, y1)]
    inside = np.column_stack(
        [rng.uniform(x0 + 1, x1 - 1, 474), rng.uniform(y0 + 1, y1 - 1, 474)]
    )
    pixels = np.vstack([extremes, inside])
    return (pixels / (WIDTH, HEIGHT)).astype(np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rgb = synthetic_image()
    cv2.imwrite(str(OUT / "input.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    rng = np.random.default_rng(1)
    cases = []
    for name, box, scale, shift in CASES:
        landmarks = synthetic_landmarks(box, rng)
        crop = phone_crop(rgb, landmarks, scale, shift, SIZE)
        cv2.imwrite(str(OUT / f"crop_{name}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        cases.append(
            {
                "name": name,
                "scale": scale,
                "shift": shift,
                "box": list(crop_box(landmarks, WIDTH, HEIGHT, scale, shift)),
                "crop": f"crop_{name}.png",
                # float() de um float32 é exato; ler o texto como float ou double dá o mesmo valor.
                "landmarks": [[float(x), float(y)] for x, y in landmarks],
            }
        )
    golden = {
        "image": "input.png",
        "width": WIDTH,
        "height": HEIGHT,
        "size": SIZE,
        "opencv": cv2.__version__,
        "tolerance": 1,
        "cases": cases,
    }
    (OUT / "cases.json").write_text(json.dumps(golden, indent=1))
    for case in cases:
        print(f"{case['name']:<24} box {case['box']}")


if __name__ == "__main__":
    main()
