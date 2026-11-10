"""Exporta o melhor modelo de um treino para .tflite (float32) e confere o resultado no LiteRT.

Grava também um JSON de metadados ao lado do modelo: é o contrato com o runtime em C++.
Se alguma verificação falhar, nada é gravado.
"""

import argparse
import collections
import json
from datetime import datetime
from pathlib import Path

import cv2
import litert_torch
import numpy as np
import torch
from ai_edge_litert.interpreter import Interpreter

from dms_train.phone_data import read_index, rows_for_split
from dms_train.phone_model import ExportWrapper, PhoneNet

TOLERANCE = 1e-4


def convert(wrapper: ExportWrapper, input_size: int, path: Path) -> None:
    sample = torch.zeros(1, input_size, input_size, 3)
    litert_torch.convert(wrapper.eval(), (sample,)).export(str(path))


def load_samples(meta: dict, input_size: int, count: int = 32) -> list[np.ndarray]:
    """Crops reais da validação, se ainda existirem, mais entradas aleatórias e extremas."""
    rng = np.random.default_rng(0)
    samples = [rng.uniform(0, 255, (1, input_size, input_size, 3)).astype(np.float32) for _ in range(8)]
    samples += [np.full((1, input_size, input_size, 3), v, np.float32) for v in (0, 255)]
    data, splits = Path(meta["data"]), Path(meta["splits"])
    if (data / "index.csv").exists() and splits.exists():
        rows = rows_for_split(read_index(data), json.loads(splits.read_text()), "val")
        for row in rows[:count]:
            rgb = cv2.cvtColor(cv2.imread(str(data / row["crop"])), cv2.COLOR_BGR2RGB)
            samples.append(rgb[None].astype(np.float32))
    return samples


def verify(path: Path, wrapper: ExportWrapper, input_size: int, num_classes: int, samples) -> dict:
    """Confere formato, ops e se o .tflite responde o mesmo que o PyTorch."""
    interpreter = Interpreter(model_path=str(path))
    interpreter.allocate_tensors()
    inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
    problems = []
    expected_in, expected_out = [1, input_size, input_size, 3], [1, num_classes]
    if len(inputs) != 1 or list(inputs[0]["shape"]) != expected_in or inputs[0]["dtype"] != np.float32:
        problems.append(f"entrada deveria ser float32 {expected_in}")
    if len(outputs) != 1 or list(outputs[0]["shape"]) != expected_out or outputs[0]["dtype"] != np.float32:
        problems.append(f"saída deveria ser float32 {expected_out}")
    # API privada, mas é a única que lista as ops do grafo.
    ops = collections.Counter(op["op_name"] for op in interpreter._get_ops_details())
    foreign = sorted(name for name in ops if name.startswith("Flex") or name == "CUSTOM")
    if foreign:
        problems.append(f"ops fora do TFLite padrão: {foreign}")
    report = {"ops": dict(ops), "transpose_ops": ops.get("TRANSPOSE", 0), "problems": problems}
    if problems:
        return report

    max_diff = max_sum_error = 0.0
    with torch.no_grad():
        for x in samples:
            interpreter.set_tensor(inputs[0]["index"], x)
            interpreter.invoke()
            y = interpreter.get_tensor(outputs[0]["index"])
            reference = wrapper(torch.from_numpy(x)).numpy()
            max_diff = max(max_diff, float(np.abs(y - reference).max()))
            max_sum_error = max(max_sum_error, abs(float(y.sum()) - 1.0))
    if max_diff > TOLERANCE:
        problems.append(f"diferença para o PyTorch {max_diff:.2e} > {TOLERANCE:.0e}")
    if max_sum_error > 1e-5:
        problems.append(f"saída não soma 1 (erro {max_sum_error:.2e})")
    return {**report, "max_diff": max_diff, "samples": len(samples)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="pasta do treino em runs/phone")
    parser.add_argument("--out", default="models/phone_classifier.tflite", type=Path)
    args = parser.parse_args()

    meta = json.loads((args.run / "meta.json").read_text())
    names, size = meta["classes"], meta["input_size"]
    model = PhoneNet(len(names), pretrained=False)
    model.load_state_dict(torch.load(args.run / "best.pt", map_location="cpu"))
    wrapper = ExportWrapper(model).eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp.tflite")
    convert(wrapper, size, tmp)
    report = verify(tmp, wrapper, size, len(names), load_samples(meta, size))
    print(f"\nops: {report['ops']}")
    if report["problems"]:
        tmp.unlink()
        raise SystemExit("exportação reprovada: " + "; ".join(report["problems"]))
    tmp.replace(args.out)

    metadata = {
        "model": args.out.name,
        "classes": names,
        "input": {
            "shape": [1, size, size, 3],
            "dtype": "float32",
            "layout": "NHWC",
            "color": "RGB",
            "range": [0, 255],
            "normalization": "dentro do grafo",
        },
        "output": {"shape": [1, len(names)], "dtype": "float32", "activation": "softmax"},
        "crop": {
            **meta["crop"],
            "size": size,
            "s": "maior lado da caixa dos 478 landmarks, em pixels",
            "cx, cy": "centro da caixa dos landmarks",
            "left": "floor(cx - scale*s/2 + 0.5)",
            "top": "floor(cy + shift*s - scale*s/2 + 0.5)",
            "side": "floor(scale*s + 0.5)",
            "math": "double",
            "border": "preto (0)",
            "interpolation": "INTER_AREA",
            "reference": "src/dms/crop.py",
            "golden": "tests/golden/crop/",
        },
        "validation": meta["best"],
        "checks": {"max_diff_vs_pytorch": report["max_diff"], "transpose_ops": report["transpose_ops"]},
        "run": str(args.run),
        "commit": meta["commit"],
        "exported": datetime.now().isoformat(timespec="seconds"),
    }
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(
        f"ok: {args.out} ({args.out.stat().st_size / 1e6:.1f} MB), diferença máx. para o PyTorch "
        f"{report['max_diff']:.1e} em {report['samples']} entradas, {report['transpose_ops']} TRANSPOSE"
    )


if __name__ == "__main__":
    main()
