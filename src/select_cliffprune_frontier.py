#!/usr/bin/env python3
"""Select the lowest-pruning candidate at each measured NPU efficiency boundary."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRICS = (
    "npu_latency_ms",
    "runtime_memory_mib",
    "compiled_model_size_mib",
)


def relative_gain(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference if reference else 0.0


def is_new_efficiency_boundary(reference: dict, candidate: dict, threshold: float) -> bool:
    """Return True only when a candidate materially improves an NPU cost metric."""
    gains = [
        relative_gain(float(reference[metric]), float(candidate[metric]))
        for metric in METRICS
    ]
    return max(gains) >= threshold


def select_frontier(rows: list[dict], threshold: float) -> list[dict]:
    """Keep the least-pruned model in each measured efficiency region.

    Rows are ordered by pruning ratio. A candidate is retained only when it
    crosses a new NPU efficiency boundary. Candidates in between have the same
    practical efficiency but retain fewer parameters, so CliffPrune discards
    them in favour of the preceding, less-pruned model.
    """
    ordered = sorted(rows, key=lambda row: float(row["pruning_ratio_percent"]))
    selected = [ordered[0]]
    for candidate in ordered[1:]:
        if is_new_efficiency_boundary(selected[-1], candidate, threshold):
            selected.append(candidate)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--model", choices=["resnet50", "vit_large"], required=True)
    parser.add_argument(
        "--efficiency-gain-threshold",
        type=float,
        default=0.01,
        help="Minimum relative improvement in latency, memory, or model size to create a new boundary.",
    )
    args = parser.parse_args()

    with args.input_csv.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["model"] == args.model]
    if not rows:
        raise ValueError(f"No rows found for {args.model}")

    selected = select_frontier(rows, args.efficiency_gain_threshold)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        fields = list(selected[0]) + ["cliffprune_selected"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow({**row, "cliffprune_selected": True})

    print(
        f"Selected {len(selected)}/{len(rows)} {args.model} candidates: "
        + ", ".join(row["pruning_ratio_percent"] for row in selected)
    )


if __name__ == "__main__":
    main()
