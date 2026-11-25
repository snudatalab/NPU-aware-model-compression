#!/usr/bin/env python3
"""Dense CliffPrune candidate scan on an RBLN NPU.

Every pruned point uses the activation-aware unit score. Dense widths inside the
same NPU efficiency plateau are intentionally measured so the scan can
distinguish an accuracy cost from a real NPU cost reduction.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from cliffprune_core import (
    build_resnet_activation_candidate,
    build_vit_activation_candidate,
    collect_resnet_activation_stats,
    collect_vit_activation_stats,
    imagenette_split,
    make_loader,
    model_stats,
    resnet_activation_scores,
    source_model,
    transform_for,
    vit_activation_scores,
)


DEFAULT_OUT = Path("../out/cliffprune_scan")
WIDTHS = {
    "resnet50": [64, 62, 60, 58, 57, 56, 54, 52, 49, 48],
    "vit_large": [4096, 4032, 3968, 3952, 3936, 3920, 3904, 3840, 3776, 3712, 3648, 3584, 3072, 2560, 2048, 1920, 1792, 1664, 1536, 1408, 1280, 1152, 1024],
}


def benchmark(runtime, x: np.ndarray, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        runtime.run(x)
    values = []
    for _ in range(iterations):
        start = time.perf_counter()
        runtime.run(x)
        values.append((time.perf_counter() - start) * 1000)
    median = statistics.median(values)
    return {
        "npu_latency_ms": median,
        "npu_latency_p95_ms": float(np.percentile(values, 95)),
        "throughput_images_per_sec": len(x) * 1000 / median,
    }


@torch.inference_mode()
def evaluate_imagenette(rebel, rbln_path: Path, loader, batch_size: int, device: int) -> dict[str, float | int]:
    runtime = rebel.Runtime(str(rbln_path), device=device)
    imagenette_ids = np.asarray(sorted({0, 217, 482, 491, 497, 566, 569, 571, 574, 701}))
    correct = total = 0
    for images, targets in loader:
        valid = len(images)
        if valid < batch_size:
            images = torch.cat([images, torch.zeros(batch_size - valid, *images.shape[1:])])
        output = runtime.run(images.numpy())
        logits = np.asarray(output[0] if isinstance(output, (tuple, list)) else output)[:valid]
        prediction = imagenette_ids[logits[:, imagenette_ids].argmax(axis=1)]
        correct += int((prediction == targets.numpy()).sum())
        total += valid
    del runtime
    return {"primary_accuracy_percent": 100 * correct / total, "primary_accuracy_samples": total}


def measure_memory(rbln_path: Path, batch_size: int, out_dir: Path, label: str, device: int) -> float:
    csv_path = out_dir / "runtime_memory.csv"
    command = [
        sys.executable,
        str(Path(__file__).with_name("measure_rbln_runtime_memory.py")),
        "--rbln_path", str(rbln_path),
        "--batch_size", str(batch_size),
        "--samples", "3",
        "--interval", "0.5",
        "--csv", str(csv_path),
        "--label", label,
        "--device", str(device),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = next(row for row in reversed(rows) if row["label"] == label)
    value = row["memory_mib_last"]
    if not value:
        raise RuntimeError(f"Could not read memory measurement: {completed.stdout}")
    return float(value)


def write_rows(path: Path, rows: list[dict]) -> None:
    fields = [
        "model", "method", "internal_width", "pruning_ratio_percent",
        "primary_accuracy_percent", "primary_accuracy_samples", "npu_latency_ms",
        "npu_latency_p95_ms", "runtime_memory_mib", "compiled_model_size_mib",
        "throughput_images_per_sec", "rbln_path", "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_rows(out_dir: Path, rows: list[dict]) -> None:
    """Persist completed candidates so an interrupted scan can resume safely."""
    write_rows(
        out_dir / "cliffprune_results.csv",
        sorted(rows, key=lambda row: (row["model"], -row["internal_width"])),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="resnet50,vit_large")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("../data/imagenette2-160"),
        help="Imagenette root containing train/ and val/ directories.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(48, torch.get_num_threads()))

    import rebel

    rows: list[dict] = []
    for model_name in [name.strip() for name in args.models.split(",") if name.strip()]:
        if model_name not in WIDTHS:
            raise ValueError(f"Unsupported model: {model_name}")
        model_dir = args.out_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        transform = transform_for(model_name)
        calibration_set, _, validation_set, split = imagenette_split(args.data_root, transform, 16)
        calibration_loader = make_loader(calibration_set, 4 if model_name == "vit_large" else 16, args.workers)
        validation_loader = make_loader(validation_set, args.batch_size, args.workers)
        source = source_model(model_name).eval()
        source_parameters = model_stats(source)["parameters"]

        # Calibrate once. Every candidate then uses the identical label-free score.
        if model_name == "resnet50":
            means, mean_squares = collect_resnet_activation_stats(source, calibration_loader)
            scores = resnet_activation_scores(source, mean_squares)
            build = lambda width: build_resnet_activation_candidate(source, width, means, scores)
        else:
            means, mean_squares = collect_vit_activation_stats(source, calibration_loader)
            scores = vit_activation_scores(source, mean_squares)
            build = lambda width: build_vit_activation_candidate(source, width, means, scores)

        for width in WIDTHS[model_name]:
            label = f"cliffprune_w{width}"
            result_path = model_dir / label / "result.json"
            if result_path.exists():
                row = json.loads(result_path.read_text(encoding="utf-8"))
                rows.append(row)
                checkpoint_rows(args.out_dir, rows)
                print(json.dumps({"model": model_name, "width": width, "status": "skipped_existing"}), flush=True)
                continue

            candidate_dir = result_path.parent
            candidate_dir.mkdir(parents=True, exist_ok=True)
            candidate = source if width == WIDTHS[model_name][0] else build(width)
            candidate.eval()
            rbln_path = candidate_dir / f"{label}_b{args.batch_size}.rbln"
            started = time.perf_counter()
            rebel.compile_from_torch(
                candidate,
                [("x", [args.batch_size, 3, 224, 224], torch.float32)],
            ).save(str(rbln_path))
            compile_seconds = time.perf_counter() - started
            runtime = rebel.Runtime(str(rbln_path), device=args.device)
            timing_input = np.random.default_rng(args.seed).standard_normal(
                (args.batch_size, 3, 224, 224), dtype=np.float32
            )
            timing = benchmark(runtime, timing_input, args.warmup, args.iterations)
            del runtime
            accuracy = evaluate_imagenette(rebel, rbln_path, validation_loader, args.batch_size, args.device)
            runtime_memory = measure_memory(
                rbln_path, args.batch_size, args.out_dir, f"{model_name}_{label}", args.device
            )
            stats = model_stats(candidate)
            row = {
                "model": model_name,
                "method": "CliffPrune" if width != WIDTHS[model_name][0] else "Original",
                "internal_width": width,
                "pruning_ratio_percent": 100 * (1 - stats["parameters"] / source_parameters),
                **accuracy,
                **timing,
                "runtime_memory_mib": runtime_memory,
                "compiled_model_size_mib": rbln_path.stat().st_size / 2**20,
                "rbln_path": str(rbln_path),
                "status": "success",
                "compile_seconds": compile_seconds,
                "calibration_samples": split["calibration_samples"],
                "calibration_labels_used": False,
                "score": "E[activation^2] * outgoing_weight_energy",
            }
            result_path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            rows.append(row)
            checkpoint_rows(args.out_dir, rows)
            print(json.dumps({
                "model": model_name,
                "width": width,
                "pruning_ratio": round(row["pruning_ratio_percent"], 4),
                "accuracy": round(row["primary_accuracy_percent"], 4),
                "latency_ms": round(row["npu_latency_ms"], 4),
                "memory_mib": row["runtime_memory_mib"],
            }), flush=True)
            if candidate is not source:
                del candidate
            gc.collect()

        del source, means, mean_squares, scores
        gc.collect()

    checkpoint_rows(args.out_dir, rows)
    protocol = {
        "method": "CliffPrune dense candidate scan",
        "score": "E[activation^2] * outgoing_weight_energy",
        "calibration": "160 fixed Imagenette training images; labels not used",
        "evaluation": "Imagenette validation 3,925 images; 10-class restricted ImageNet-1K Top-1",
        "hardware": "ATOM PLUS / RBLN-CA12",
        "device": args.device,
        "compiler": "RBLN SDK 0.9.3.post1",
        "timing": {"batch_size": args.batch_size, "warmup": args.warmup, "iterations": args.iterations},
        "widths": WIDTHS,
    }
    (args.out_dir / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.out_dir / "cliffprune_results.csv", flush=True)


if __name__ == "__main__":
    main()
