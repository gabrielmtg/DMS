"""Gera os crops de treino do classificador de celular a partir do manifesto.

O manifesto é um CSV com video,frame,label,subject (ver README). Para cada quadro
amostrado, roda o FaceLandmarker, aplica o crop de referência (src/dms/crop.py) na
mesma resolução do runtime e grava um PNG.
"""

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from dms.capture import preprocess
from dms.crop import phone_crop
from dms.perception import Landmarker
from dms_train.phone_data import (
    INDEX_COLUMNS,
    class_names,
    label_mapping,
    load_config,
    read_manifest,
    select_rows,
)


def create_landmarker(perception_cfg: dict) -> vision.FaceLandmarker:
    """Modo IMAGE: os quadros amostrados não são vizinhos, então não há o que rastrear."""
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=perception_cfg["model_path"]),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=perception_cfg["min_face_detection_confidence"],
        min_face_presence_confidence=perception_cfg["min_face_presence_confidence"],
    )
    return vision.FaceLandmarker.create_from_options(options)


def read_frames(video: str, wanted):
    """Lê o vídeo em ordem e entrega só os quadros pedidos; buscar por índice é lento e impreciso."""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise OSError(f"não foi possível abrir o vídeo: {video}")
    index = 0
    try:
        for target in sorted(wanted):
            while index < target:
                if not cap.grab():
                    return
                index += 1
            ok, bgr = cap.read()
            if not ok:
                return
            index += 1
            yield target, bgr
    finally:
        cap.release()


def print_summary(names, requested, saved, no_face, out_of_video) -> None:
    print(f"\n{'classe':<14}{'pedidos':>9}{'salvos':>9}{'sem rosto':>11}")
    for name in names:
        missing = no_face[name] / requested[name] if requested[name] else 0.0
        print(f"{name:<14}{requested[name]:>9}{saved[name]:>9}{missing:>10.1%}")
    if out_of_video:
        print(f"\n{out_of_video} quadros pedidos estão além do fim do vídeo")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", default="configs/phone.yaml")
    parser.add_argument("--runtime-config", default="configs/default.yaml")
    parser.add_argument("--every", type=int, help="usa 1 a cada N quadros")
    parser.add_argument("--overwrite", action="store_true", help="apaga os crops anteriores")
    args = parser.parse_args()

    cfg = load_config(args.config)
    runtime = load_config(args.runtime_config)
    index_path = args.out / "index.csv"
    if index_path.exists() and not args.overwrite:
        raise SystemExit(f"{index_path} já existe; use --overwrite para refazer")
    if args.overwrite:
        shutil.rmtree(args.out / "crops", ignore_errors=True)

    every = args.every or cfg["dataset"]["every_n_frames"]
    try:
        rows = select_rows(read_manifest(args.manifest), label_mapping(cfg), every)
    except ValueError as error:
        raise SystemExit(str(error))
    by_video = defaultdict(dict)
    for row in rows:
        by_video[row["video"]].setdefault(row["frame"], row)

    names = class_names(cfg)
    size, crop_cfg = cfg["input_size"], cfg["crop"]
    requested = Counter(row["class"] for frames in by_video.values() for row in frames.values())
    saved, no_face = Counter(), Counter()
    out_of_video = 0
    index_rows = []
    landmarker = create_landmarker(runtime["perception"])
    try:
        for video, frames in by_video.items():
            # O caminho entra no nome do arquivo para vídeos de pastas diferentes não colidirem.
            tag = hashlib.sha1(video.encode()).hexdigest()[:8]
            found = 0
            for frame, bgr in read_frames(video, frames):
                found += 1
                row = frames[frame]
                rgb = preprocess(bgr, runtime["capture"])
                result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                landmarks = Landmarker.to_array(result)
                if landmarks is None:
                    no_face[row["class"]] += 1
                    continue
                crop = phone_crop(rgb, landmarks, crop_cfg["scale"], crop_cfg["shift"], size)
                rel = Path("crops") / row["class"] / f"{row['subject']}_{tag}_{frame:06d}.png"
                (args.out / rel).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(args.out / rel), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
                saved[row["class"]] += 1
                index_rows.append({**row, "crop": rel.as_posix()})
            out_of_video += len(frames) - found
    finally:
        landmarker.close()

    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(index_rows)
    build = {
        "classes": names,
        "input_size": size,
        "crop": {"scale": crop_cfg["scale"], "shift": crop_cfg["shift"]},
        "capture": {k: runtime["capture"][k] for k in ("width", "height", "driver_roi")},
        "every_n_frames": every,
        "manifest": str(args.manifest),
        "counts": {
            name: {"requested": requested[name], "saved": saved[name], "no_face": no_face[name]}
            for name in names
        },
        "out_of_video": out_of_video,
    }
    (args.out / "build.json").write_text(json.dumps(build, indent=2, ensure_ascii=False))
    print_summary(names, requested, saved, no_face, out_of_video)
    print(f"\n{len(index_rows)} crops em {args.out}")


if __name__ == "__main__":
    main()
