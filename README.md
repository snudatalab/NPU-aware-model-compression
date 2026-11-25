# Reproducibility Package

This repository contains the official research implementation of **CliffPrune**,
an NPU-aware structural pruning method that selects the least-pruned model in
each measured NPU efficiency region. The experiments use ImageNet-pretrained
ResNet50 and ViT-Large/16, Imagenette, and an RBLN NPU runtime.

## 1) Directory layout

- `src/`: CliffPrune implementation, NPU measurement utilities, and experiment entrypoints.
- `data/`: Imagenette dataset directory; not included in this repository.
- `out/`: compiled models, per-candidate records, and CSV results; created automatically.
- `DATASET.md`: dataset layout and evaluation protocol.
- `DEPENDENCIES.md`: Python, PyTorch, and RBLN SDK setup details.
- `environment.rbln_env.yml`: Conda environment specification for public dependencies.
- `requirements.txt`: minimal pip requirements.

## 2) Environment setup

Recommended:

```bash
conda env create -f environment.rbln_env.yml
conda activate cliffprune-rbln
```

Manual installation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The RBLN compiler is a private package. Install the compiler version compatible
with the target server driver as described in `DEPENDENCIES.md`.

## 3) Important run rule

Run experiment entrypoints from `src/`. Relative paths are defined from that
directory: `../data/` for datasets and `../out/` for generated artifacts.
The candidate scan compiles and executes models on the selected NPU, so it
requires an RBLN runtime environment and access to a supported device.

## 4) Example commands

Smoke test:

```bash
cd src
python basic_rbln_inference.py --batch-size 1 --force-compile
```

Main candidate scan:

```bash
cd src
python run_cliffprune_scan.py \
  --data-root ../data/imagenette2-160 \
  --models resnet50,vit_large \
  --out-dir ../out/cliffprune_scan \
  --batch-size 32 \
  --device 0
```

Select the measured efficiency frontier:

```bash
cd src
python select_cliffprune_frontier.py \
  --input-csv ../out/cliffprune_scan/cliffprune_results.csv \
  --output-csv ../out/cliffprune_scan/vit_large_frontier.csv \
  --model vit_large
```

Repeat latency measurement for selected compiled candidates:

```bash
cd src
python repeat_rbln_latency.py \
  --input-csv ../out/cliffprune_scan/cliffprune_results.csv \
  --output-csv ../out/cliffprune_scan/resnet50_latency_repeats.csv \
  --model resnet50 --device 0 --repeats 5
```

## 5) Dataset

The submitted experiments use Imagenette. See `DATASET.md` for the required
directory layout, label mapping, calibration split, and evaluation metric.

## 6) Outputs

The candidate scan writes resumable results under `out/cliffprune_scan/`:

- `cliffprune_results.csv`: candidate-level accuracy, latency, memory, and compiled-model size.
- `<model>/<candidate>/result.json`: per-candidate result record.
- `<model>/<candidate>/*.rbln`: compiled NPU artifact.

The `data/` and `out/` directories are ignored by Git.
