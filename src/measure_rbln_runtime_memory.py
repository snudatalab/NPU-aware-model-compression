import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path


def parse_mib(value):
    value = value.strip()
    if value.endswith("MiB"):
        return float(value[:-3])
    if value.endswith("GiB"):
        return float(value[:-3]) * 1024
    if value.endswith("B"):
        return float(value[:-1]) / 1024 / 1024
    return None


def poll_memory(samples, interval, device):
    memory_re = re.compile(
        rf"^\|\s*{device}\s+\|\s+RBLN-CA12\s+\|\s+rbln{device}\s+\|.*?\|\s*([^|]+?)\s*/",
        re.MULTILINE,
    )
    values = []
    for _ in range(samples):
        out = subprocess.check_output(["rbln-smi"], text=True)
        match = memory_re.search(out)
        if match:
            mib = parse_mib(match.group(1))
            if mib is not None:
                values.append(mib)
        time.sleep(interval)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rbln_path", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()

    hold_script = Path(__file__).with_name("hold_rbln_runtime.py")
    hold = subprocess.Popen(
        [
            sys.executable,
            str(hold_script),
            args.rbln_path,
            "--batch-size",
            str(args.batch_size),
            "--height",
            str(args.height),
            "--width",
            str(args.width),
            "--device",
            str(args.device),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        time.sleep(2.0)
        values = poll_memory(args.samples, args.interval, args.device)
    finally:
        try:
            hold.terminate()
            hold.wait(timeout=5)
        except subprocess.TimeoutExpired:
            hold.kill()
            hold.wait()

    row = {
        "label": args.label,
        "rbln_path": args.rbln_path,
        "batch_size": args.batch_size,
        "device": args.device,
        "samples": len(values),
        "memory_mib_min": min(values) if values else "",
        "memory_mib_max": max(values) if values else "",
        "memory_mib_last": values[-1] if values else "",
    }
    csv_path = Path(args.csv)
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(row)


if __name__ == "__main__":
    main()
