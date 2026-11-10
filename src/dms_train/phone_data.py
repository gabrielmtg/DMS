"""Configuração, dados e métricas do classificador de celular.

Só NumPy: os testes rodam no ambiente do runtime, sem PyTorch.
"""

import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

MANIFEST_COLUMNS = ("video", "frame", "label", "subject")
INDEX_COLUMNS = ("crop", "class", "label", "subject", "video", "frame")


def load_config(path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def class_names(cfg: dict) -> list[str]:
    return list(cfg["classes"])


def label_mapping(cfg: dict) -> dict[str, str | None]:
    """Rótulo do DMD -> nome da classe; None para os rótulos ignorados."""
    entries = [
        (label, name) for name, labels in cfg["classes"].items() for label in labels
    ]
    entries += [(label, None) for label in cfg.get("ignore", [])]
    mapping: dict[str, str | None] = {}
    for label, name in entries:
        if label in mapping:
            raise ValueError(f"rótulo repetido na configuração: {label!r}")
        mapping[label] = name
    return mapping


def read_manifest(path) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = set(MANIFEST_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"faltam colunas no manifesto: {sorted(missing)}")
        return [{**row, "frame": int(row["frame"])} for row in reader]


def select_rows(rows: list[dict], mapping: dict, every_n_frames: int) -> list[dict]:
    """Troca o rótulo pela classe e mantém 1 a cada N quadros (quadros vizinhos são quase iguais)."""
    unknown = Counter(row["label"] for row in rows if row["label"] not in mapping)
    if unknown:
        listed = ", ".join(f"{label!r} ({n})" for label, n in unknown.most_common())
        raise ValueError(f"rótulos fora da configuração: {listed}")
    return [
        {**row, "class": mapping[row["label"]]}
        for row in rows
        if mapping[row["label"]] is not None and row["frame"] % every_n_frames == 0
    ]


def make_splits(subjects, fractions: dict, seed: int) -> dict[str, list[str]]:
    """Divide por motorista, para o modelo nunca ser avaliado em quem viu no treino."""
    subjects = sorted(set(subjects))
    n_test = max(1, round(len(subjects) * fractions["test"]))
    n_val = max(1, round(len(subjects) * fractions["val"]))
    if n_test + n_val >= len(subjects):
        raise ValueError(
            f"motoristas insuficientes para dividir em treino, validação e teste: {len(subjects)}"
        )
    random.Random(seed).shuffle(subjects)
    return {
        "train": sorted(subjects[n_test + n_val :]),
        "val": sorted(subjects[n_test : n_test + n_val]),
        "test": sorted(subjects[:n_test]),
    }


def load_or_make_splits(path, subjects, split_cfg: dict) -> dict[str, list[str]]:
    """Reusa o split salvo: o conjunto de teste não pode mudar entre experimentos."""
    path = Path(path)
    if path.exists():
        splits = json.loads(path.read_text())
        known = {subject for names in splits.values() for subject in names}
        new = sorted(set(subjects) - known)
        if new:
            raise ValueError(
                f"motoristas fora de {path}: {new}. Apague o arquivo para gerar outro split."
            )
        return splits
    splits = make_splits(subjects, split_cfg["fractions"], split_cfg["seed"])
    path.write_text(json.dumps(splits, indent=2))
    return splits


def read_index(data_dir) -> list[dict]:
    with open(Path(data_dir) / "index.csv", newline="") as f:
        return list(csv.DictReader(f))


def rows_for_split(rows: list[dict], splits: dict, split: str) -> list[dict]:
    subjects = set(splits[split])
    return [row for row in rows if row["subject"] in subjects]


def confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    """Linha = classe verdadeira, coluna = classe prevista."""
    cm = np.zeros((num_classes, num_classes), np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def per_class_scores(cm: np.ndarray) -> dict[str, np.ndarray]:
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    total = precision + recall
    f1 = np.divide(2 * precision * recall, total, out=np.zeros_like(tp), where=total > 0)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support}


def macro_f1(cm: np.ndarray) -> float:
    """Média do F1 nas classes que aparecem; classe sem exemplos não puxa a média."""
    scores = per_class_scores(cm)
    present = scores["support"] > 0
    return float(scores["f1"][present].mean()) if present.any() else 0.0


def class_weights(counts) -> np.ndarray:
    """Inverso da frequência, com peso médio 1 por exemplo (o "balanced" do scikit-learn)."""
    counts = np.asarray(counts, np.float64)
    weights = np.zeros_like(counts)
    present = counts > 0
    weights[present] = counts.sum() / (present.sum() * counts[present])
    return weights


def format_report(cm: np.ndarray, names: list[str]) -> str:
    scores = per_class_scores(cm)
    width = max(len(name) for name in names) + 2
    lines = [f"{'classe':<{width}}{'precisão':>10}{'recall':>8}{'F1':>8}{'exemplos':>10}"]
    for i, name in enumerate(names):
        lines.append(
            f"{name:<{width}}{scores['precision'][i]:>10.3f}{scores['recall'][i]:>8.3f}"
            f"{scores['f1'][i]:>8.3f}{scores['support'][i]:>10d}"
        )
    lines.append(f"\nmacro-F1: {macro_f1(cm):.3f}")
    lines.append("\nconfusão (linha = verdade, coluna = previsto):")
    lines.append(" " * width + "".join(f"{name[:10]:>12}" for name in names))
    for i, name in enumerate(names):
        lines.append(f"{name:<{width}}" + "".join(f"{v:>12d}" for v in cm[i]))
    return "\n".join(lines)
