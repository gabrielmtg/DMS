"""Avalia o .tflite exportado, que é o que vai para a Pi, em um split dos crops."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

from dms_train.phone_data import (
    confusion_matrix,
    format_report,
    macro_f1,
    per_class_scores,
    read_index,
    rows_for_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/phone_classifier.tflite", type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=("val", "test"))
    parser.add_argument("--splits", type=Path, help="padrão: <data>/splits.json")
    parser.add_argument("--out", type=Path, help="salva o relatório em JSON")
    args = parser.parse_args()

    names = json.loads(args.model.with_suffix(".json").read_text())["classes"]
    splits = json.loads((args.splits or args.data / "splits.json").read_text())
    rows = rows_for_split(read_index(args.data), splits, args.split)
    if not rows:
        raise SystemExit(f"nenhum crop no split {args.split}")
    stray = {row["class"] for row in rows} - set(names)
    if stray:
        raise SystemExit(f"classes nos dados que o modelo não conhece: {sorted(stray)}")
    if args.split == "test":
        print("Atenção: o teste é para a avaliação final; ajustar algo depois de olhar invalida o resultado.\n")

    interpreter = Interpreter(model_path=str(args.model))
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    y_true, y_pred, times = [], [], []
    for row in rows:
        rgb = cv2.cvtColor(cv2.imread(str(args.data / row["crop"])), cv2.COLOR_BGR2RGB)
        interpreter.set_tensor(input_index, rgb[None].astype(np.float32))
        start = time.perf_counter()
        interpreter.invoke()
        times.append(time.perf_counter() - start)
        y_true.append(names.index(row["class"]))
        y_pred.append(int(interpreter.get_tensor(output_index).argmax()))

    cm = confusion_matrix(y_true, y_pred, len(names))
    print(format_report(cm, names))
    print(f"\n{len(rows)} crops, inferência média de {np.mean(times) * 1000:.2f} ms neste computador")
    if args.out:
        scores = per_class_scores(cm)
        report = {
            "model": str(args.model),
            "split": args.split,
            "crops": len(rows),
            "macro_f1": macro_f1(cm),
            "per_class": {
                name: {key: float(scores[key][i]) for key in ("precision", "recall", "f1")}
                for i, name in enumerate(names)
            },
            "confusion": cm.tolist(),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
