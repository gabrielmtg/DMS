"""Treina o classificador de celular a partir dos crops do build_phone_dataset.py.

Duas fases: só a cabeça com o backbone congelado, depois fine-tune com learning rate
menor. Guarda a época de maior macro-F1 na validação.
"""

import argparse
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dms_train.phone_data import (
    class_names,
    class_weights,
    confusion_matrix,
    format_report,
    load_config,
    load_or_make_splits,
    macro_f1,
    per_class_scores,
    read_index,
    rows_for_split,
)
from dms_train.phone_model import CropDataset, PhoneNet, eval_transform, train_transform


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


def git_commit() -> str:
    try:
        return _git("rev-parse", "--short", "HEAD") + ("-dirty" if _git("status", "--porcelain") else "")
    except (OSError, subprocess.CalledProcessError):
        return "desconhecido"


def make_loader(rows, data_dir, names, transform, batch_size, shuffle, workers):
    paths = [Path(data_dir) / row["crop"] for row in rows]
    labels = [names.index(row["class"]) for row in rows]
    return DataLoader(
        CropDataset(paths, labels, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
    )


def train_epoch(model, loader, criterion, optimizer) -> float:
    model.train()
    total, count = 0.0, 0
    for x, y in loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)
        count += len(y)
    return total / count


def evaluate(model, loader, num_classes, criterion) -> tuple[np.ndarray, float]:
    model.eval()
    preds, targets, total = [], [], 0.0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            total += criterion(logits, y).item() * len(y)
            preds.append(logits.argmax(dim=1))
            targets.append(y)
    y_true = torch.cat(targets).numpy()
    y_pred = torch.cat(preds).numpy()
    return confusion_matrix(y_true, y_pred, num_classes), total / len(y_true)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="saída do build")
    parser.add_argument("--config", default="configs/phone.yaml")
    parser.add_argument("--splits", type=Path, help="padrão: <data>/splits.json")
    parser.add_argument("--out", default="runs/phone", type=Path)
    parser.add_argument("--head-epochs", type=int)
    parser.add_argument("--finetune-epochs", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    parser.add_argument(
        "--no-pretrained", action="store_true", help="sem pesos do ImageNet (só para testes)"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["train"]
    build = json.loads((args.data / "build.json").read_text())
    if build["input_size"] != cfg["input_size"]:
        raise SystemExit("input_size do build difere da configuração; rode o build de novo")
    names = class_names(cfg)
    rows = read_index(args.data)
    stray = {row["class"] for row in rows} - set(names)
    if stray:
        raise SystemExit(f"classes do índice fora de {args.config}: {sorted(stray)}; rode o build de novo")

    splits_path = args.splits or args.data / "splits.json"
    try:
        splits = load_or_make_splits(splits_path, [row["subject"] for row in rows], cfg["split"])
    except ValueError as error:
        raise SystemExit(str(error))
    train_rows = rows_for_split(rows, splits, "train")
    val_rows = rows_for_split(rows, splits, "val")
    if not train_rows or not val_rows:
        raise SystemExit("treino ou validação ficou sem exemplos")
    counts = [sum(row["class"] == name for row in train_rows) for name in names]
    print(f"motoristas: {', '.join(f'{k} {len(v)}' for k, v in splits.items())}")
    print(f"exemplos de treino: {dict(zip(names, counts))}  validação: {len(val_rows)}")

    torch.manual_seed(cfg["split"]["seed"])
    torch.set_num_threads(args.threads)
    workers = train_cfg["num_workers"] if args.workers is None else args.workers
    batch = train_cfg["batch_size"]
    train_loader = make_loader(train_rows, args.data, names, train_transform(), batch, True, workers)
    val_loader = make_loader(val_rows, args.data, names, eval_transform(), batch, False, workers)
    model = PhoneNet(len(names), pretrained=not args.no_pretrained)
    weights = torch.tensor(class_weights(counts), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)

    run_dir = args.out / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    phases = [
        ("cabeça", False, args.head_epochs, train_cfg["head_epochs"], train_cfg["head_lr"]),
        ("fine-tune", True, args.finetune_epochs, train_cfg["finetune_epochs"], train_cfg["finetune_lr"]),
    ]
    history, best = [], {"macro_f1": -1.0}
    for phase, backbone, override, default_epochs, lr in phases:
        epochs = default_epochs if override is None else override
        if epochs == 0:
            continue
        model.set_trainable(backbone)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=train_cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        stale = 0
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss = train_epoch(model, train_loader, criterion, optimizer)
            cm, val_loss = evaluate(model, val_loader, len(names), criterion)
            scheduler.step()
            score = macro_f1(cm)
            history.append(
                {"phase": phase, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_macro_f1": score}
            )
            print(
                f"[{phase} {epoch}/{epochs}] loss {train_loss:.4f}  val loss {val_loss:.4f}"
                f"  val macro-F1 {score:.3f}  ({time.time() - start:.0f} s)"
            )
            if score > best["macro_f1"]:
                best = {"macro_f1": score, "phase": phase, "epoch": epoch, "confusion": cm.tolist()}
                torch.save(model.state_dict(), run_dir / "best.pt")
                stale = 0
            else:
                stale += 1
                if stale >= train_cfg["patience"]:
                    print(f"sem melhora em {stale} épocas, encerrando a fase")
                    break

    best_cm = np.array(best["confusion"])
    f1 = per_class_scores(best_cm)["f1"]
    best["per_class_f1"] = dict(zip(names, np.round(f1, 4).tolist()))
    meta = {
        "classes": names,
        "input_size": build["input_size"],
        "crop": build["crop"],
        "pretrained": not args.no_pretrained,
        "best": best,
        "train_counts": dict(zip(names, counts)),
        "data": str(args.data),
        "splits": str(splits_path),
        "commit": git_commit(),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    shutil.copy(args.config, run_dir / "phone.yaml")
    print(f"\nmelhor: {best['phase']} época {best['epoch']}\n")
    print(format_report(best_cm, names))
    print(f"\nmodelo em {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
