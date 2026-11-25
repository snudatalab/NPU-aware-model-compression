#!/usr/bin/env python3
"""Keep an RBLN runtime active while rbln-smi samples device memory."""

import argparse
import time

import numpy as np
import rebel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("rbln_path")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    runtime = rebel.Runtime(args.rbln_path, device=args.device)
    x = np.random.randn(args.batch_size, 3, args.height, args.width).astype("float32")
    end = time.time() + args.seconds
    while time.time() < end:
        runtime.run(x)


if __name__ == "__main__":
    main()
