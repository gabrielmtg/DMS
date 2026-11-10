"""Mede o custo do pipeline por etapa e compara com as metas do Gate 1.

Rode em cima de vídeo para comparar execuções com a mesma entrada. --fps limita a
taxa; sem ele, mede a vazão máxima. O Gate 1 vale com --fps: entre um quadro e outro
a CPU fica ociosa, e a inferência sai mais lenta do que em vazão máxima.
"""

import argparse
import json
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml

from dms.capture import CameraSource, open_source, preprocess
from dms.perception import Landmarker

STAGES = ("capture", "preprocess", "infer", "to_array")
THERMAL_ZONE = Path("/sys/class/thermal/thermal_zone0/temp")


def read_temp_c() -> float | None:
    try:
        return int(THERMAL_ZONE.read_text()) / 1000
    except OSError:
        return None


def read_throttled() -> int | None:
    """Máscara do `vcgencmd get_throttled`; None fora da Raspberry Pi."""
    try:
        out = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2
        )
        return int(out.stdout.strip().split("=")[1], 16)
    except (OSError, IndexError, ValueError, subprocess.SubprocessError):
        return None


def new_throttling(start: int | None, end: int | None) -> bool | None:
    """Bits 0-3 valem agora; 16-19 ficam acesos desde o boot, então comparamos com o início."""
    if start is None or end is None:
        return None
    return bool(end & 0xF or end & ~start & 0xF0000)


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def percentiles(values_ms: list[float]) -> dict:
    p50, p95, p99 = np.percentile(values_ms, [50, 95, 99])
    return {"p50": float(p50), "p95": float(p95), "p99": float(p99)}


def run(args: argparse.Namespace, cfg: dict) -> dict:
    source = open_source(args.source, cfg["capture"], step=args.step, loop=args.loop)
    landmarker = Landmarker(cfg["perception"])
    times = {stage: [] for stage in (*STAGES, "frame")}
    faces = frames = 0
    period = 1 / args.fps if args.fps else 0.0
    temp_start, throttled_start = read_temp_c(), read_throttled()
    wall_start = next_tick = time.perf_counter()
    cpu_start = cpu_seconds()
    try:
        while time.perf_counter() - wall_start < args.seconds:
            t0 = time.perf_counter()
            frame = source.read()
            if frame is None:
                break
            t1 = time.perf_counter()
            rgb = preprocess(frame.bgr, cfg["capture"])
            t2 = time.perf_counter()
            result = landmarker.infer(rgb, frame.timestamp_ms)
            t3 = time.perf_counter()
            landmarks = landmarker.to_array(result)
            t4 = time.perf_counter()

            # O aquecimento fica fora das estatísticas de latência.
            if frames >= args.warmup:
                for stage, start, end in zip(
                    STAGES, (t0, t1, t2, t3), (t1, t2, t3, t4)
                ):
                    times[stage].append((end - start) * 1000)
                times["frame"].append((t4 - t1) * 1000)
                faces += landmarks is not None
            frames += 1

            if period:
                next_tick += period
                time.sleep(max(0.0, next_tick - time.perf_counter()))
        wall = time.perf_counter() - wall_start
        cpu = cpu_seconds() - cpu_start
    finally:
        camera = source if isinstance(source, CameraSource) else None
        dropped = camera.captured - camera.delivered if camera else None
        source.close()
        landmarker.close()

    measured = len(times["frame"])
    if measured == 0:
        raise SystemExit("nenhum quadro medido: aumente --seconds ou reduza --warmup")
    return {
        "source": args.source,
        "paced_fps": args.fps,
        "frames": measured,
        "face_rate": faces / measured,
        "dropped_frames": dropped,
        "fps": frames / wall,
        "cpu_cores": cpu / wall,
        "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "temp_c": {"start": temp_start, "end": read_temp_c()},
        "throttled": new_throttling(throttled_start, read_throttled()),
        "latency_ms": {stage: percentiles(v) for stage, v in times.items()},
    }


def evaluate(report: dict, targets: dict) -> dict:
    return {
        f"fps >= {targets['fps']}": report["fps"] >= targets["fps"] * 0.99,
        f"frame p95 <= {targets['frame_p95_ms']} ms": (
            report["latency_ms"]["frame"]["p95"] <= targets["frame_p95_ms"]
        ),
        f"cpu <= {targets['cpu_cores']} núcleos": (
            report["cpu_cores"] <= targets["cpu_cores"]
        ),
        f"rss <= {targets['rss_mb']} MB": report["rss_mb"] <= targets["rss_mb"],
        # Fora da Raspberry Pi não há como medir: None não pode contar como aprovado.
        "sem throttling": (
            None if report["throttled"] is None else not report["throttled"]
        ),
    }


def print_report(report: dict, checks: dict) -> None:
    print(f"\n{'etapa':<12}{'p50':>9}{'p95':>9}{'p99':>9}  (ms)")
    for stage, values in report["latency_ms"].items():
        print(
            f"{stage:<12}{values['p50']:>9.2f}{values['p95']:>9.2f}{values['p99']:>9.2f}"
        )
    temp = report["temp_c"]
    print(
        f"\nquadros: {report['frames']}  face: {report['face_rate']:.1%}"
        f"  fps: {report['fps']:.1f}  cpu: {report['cpu_cores']:.2f} núcleos"
        f"  rss: {report['rss_mb']:.0f} MB  temp: {temp['start']} -> {temp['end']} C"
        f"  throttling: {report['throttled']}"
    )
    if report["dropped_frames"] is not None:
        print(f"quadros descartados pela captura: {report['dropped_frames']}")
    print()
    labels = {True: "ok", False: "FALHOU", None: "não medido"}
    for name, ok in checks.items():
        print(f"[{labels[ok]}] {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="vídeo ou índice da câmera")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--warmup", type=int, default=30, help="quadros ignorados")
    parser.add_argument("--fps", type=float, default=0.0, help="limita a taxa do laço")
    parser.add_argument("--step", type=int, default=1, help="usa 1 a cada N quadros")
    parser.add_argument("--loop", action="store_true", help="repete o vídeo")
    parser.add_argument("--out", type=Path, help="salva o relatório em JSON")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    report = run(args, cfg)
    checks = evaluate(report, cfg["targets"])
    print_report(report, checks)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({**report, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
