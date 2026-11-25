import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class TinyImageClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal PyTorch to RBLN NPU inference example")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--force-compile", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    import rebel

    torch.manual_seed(0)
    np.random.seed(0)

    workdir = Path.cwd()
    model_path = workdir / f"tiny_image_classifier_b{args.batch_size}.rbln"

    model = TinyImageClassifier(num_classes=10).eval()
    x = torch.randn(args.batch_size, 3, args.height, args.width, dtype=torch.float32)

    if args.force_compile and model_path.exists():
        model_path.unlink()

    if not model_path.exists():
        input_info = [("x", list(x.shape), torch.float32)]
        compiled = rebel.compile_from_torch(model, input_info)
        compiled.save(str(model_path))
        print(f"compiled_model_saved={model_path}")
    else:
        print(f"reuse_compiled_model={model_path}")

    runtime = rebel.Runtime(str(model_path))

    input_np = x.numpy()
    for _ in range(args.warmup):
        runtime.run(input_np)

    latencies_ms = []
    output = None
    for _ in range(args.runs):
        start = time.perf_counter()
        result = runtime.run(input_np)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        output = result[0] if isinstance(result, (list, tuple)) else result

    npu_output = np.asarray(output)
    with torch.no_grad():
        cpu_output = model(x).numpy()

    print("--- RBLN NPU inference completed ---")
    print(f"input_shape={tuple(x.shape)}")
    print(f"npu_output_shape={npu_output.shape}")
    print(f"cpu_output_shape={cpu_output.shape}")
    print(f"max_abs_diff={np.max(np.abs(npu_output - cpu_output)):.6f}")
    print(f"latency_ms_mean={np.mean(latencies_ms):.3f}")
    print(f"latency_ms_median={np.median(latencies_ms):.3f}")


if __name__ == "__main__":
    main()
