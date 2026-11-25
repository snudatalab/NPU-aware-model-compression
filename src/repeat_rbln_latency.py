#!/usr/bin/env python3
"""Repeat runtime latency measurements for already compiled RBLN candidates."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import numpy as np


def one_round(runtime, input_array: np.ndarray, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        runtime.run(input_array)
    values = []
    for _ in range(iterations):
        started = time.perf_counter()
        runtime.run(input_array)
        values.append((time.perf_counter() - started) * 1000)
    return statistics.median(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["model"] == args.model]
    rows.sort(key=lambda row: float(row["pruning_ratio_percent"]))

    import rebel

    inputs = np.random.default_rng(20261124).standard_normal(
        (args.batch_size, 3, 224, 224), dtype=np.float32
    )
    output_rows = []
    for row in rows:
        runtime = rebel.Runtime(row["rbln_path"], device=args.device)
        measurements = [
            one_round(runtime, inputs, args.warmup, args.iterations)
            for _ in range(args.repeats)
        ]
        del runtime
        result = {
            "model": args.model,
            "internal_width": row["internal_width"],
            "pruning_ratio_percent": row["pruning_ratio_percent"],
            "device": args.device,
            "repeats": args.repeats,
            "iterations_per_repeat": args.iterations,
            "latency_median_ms": statistics.median(measurements),
            "latency_mean_ms": statistics.mean(measurements),
            "latency_std_ms": statistics.stdev(measurements) if len(measurements) > 1 else 0.0,
            "round_medians_ms": ";".join(f"{value:.6f}" for value in measurements),
        }
        output_rows.append(result)
        print(result, flush=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
