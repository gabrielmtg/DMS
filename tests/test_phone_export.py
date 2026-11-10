"""Treino, exportação e avaliação de ponta a ponta com crops sintéticos.

Roda só no ambiente de treino. Os números não dizem nada sobre o modelo real:
o teste garante que o caminho até o .tflite funciona e que o arquivo bate com o PyTorch.
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("litert_torch")

from ai_edge_litert.interpreter import Interpreter  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CLASSES = ["normal", "phone_call", "drinking", "hair_makeup"]
COLORS = [(40, 40, 40), (220, 40, 40), (40, 220, 40), (40, 40, 220)]


def make_dataset(data_dir: Path) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for subject in ("a", "b", "c", "d"):
        for name, color in zip(CLASSES, COLORS):
            for i in range(3):
                noise = rng.integers(-20, 21, (160, 160, 3))
                crop = np.clip(np.array(color) + noise, 0, 255).astype(np.uint8)
                rel = f"crops/{name}/{subject}_{i}.png"
                (data_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(data_dir / rel), crop)
                rows.append({"crop": rel, "class": name, "label": name, "subject": subject, "video": "v", "frame": i})
    with open(data_dir / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    build = {"classes": CLASSES, "input_size": 160, "crop": {"scale": 2.2, "shift": 0.1}}
    (data_dir / "build.json").write_text(json.dumps(build))


def run(script: str, *args) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, args)], check=True, cwd=ROOT)


def test_train_export_and_evaluate(tmp_path):
    data = tmp_path / "phone"
    make_dataset(data)
    runs = tmp_path / "runs"
    run(
        "train_phone.py", "--data", data, "--out", runs, "--no-pretrained",
        "--head-epochs", 1, "--finetune-epochs", 1, "--workers", 0, "--threads", 2,
    )
    (run_dir,) = runs.iterdir()
    model = tmp_path / "models" / "phone_classifier.tflite"
    run("export_tflite.py", "--run", run_dir, "--out", model)

    metadata = json.loads(model.with_suffix(".json").read_text())
    assert metadata["classes"] == CLASSES
    assert metadata["input"]["shape"] == [1, 160, 160, 3]
    assert metadata["checks"]["max_diff_vs_pytorch"] < 1e-4
    assert metadata["checks"]["transpose_ops"] == 0

    interpreter = Interpreter(model_path=str(model))
    interpreter.allocate_tensors()
    assert list(interpreter.get_input_details()[0]["shape"]) == [1, 160, 160, 3]
    assert list(interpreter.get_output_details()[0]["shape"]) == [1, 4]

    report = tmp_path / "eval.json"
    run("evaluate_phone.py", "--model", model, "--data", data, "--out", report)
    assert json.loads(report.read_text())["crops"] > 0
