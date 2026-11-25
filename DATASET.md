# Dataset

## Imagenette

The experiments use the Imagenette2-160 image classification dataset in the
standard `torchvision.datasets.ImageFolder` layout:

```text
data/imagenette2-160/
  train/
    n01440764/
    ...
  val/
    n01440764/
    ...
```

The dataset is not distributed with this repository. Obtain it separately and
place it under `data/imagenette2-160/` before running the experiment scripts.

## Protocol

- Calibration inputs: the first 16 training images from each of the 10 classes
  (160 images total). Labels are not used to compute pruning importance.
- Evaluation inputs: all 3,925 validation images.
- Metric: restricted 10-class ImageNet Top-1 accuracy. Model logits are
  restricted to Imagenette's corresponding ImageNet-1K class indices before
  selecting the prediction.
- Input preprocessing: the official torchvision ImageNet transforms associated
  with the selected ResNet50 or ViT-Large/16 pretrained weights.
