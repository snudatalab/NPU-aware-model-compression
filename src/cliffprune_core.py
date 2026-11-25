#!/usr/bin/env python3
"""Model, data, and activation-aware structural pruning helpers for CliffPrune."""

from __future__ import annotations

import copy
from typing import Callable, Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision.models import (
    ResNet50_Weights,
    ViT_L_16_Weights,
    VisionTransformer,
    resnet50,
    vit_l_16,
)
from torchvision.models.resnet import Bottleneck
from torchvision.models.vision_transformer import MLPBlock

CALIBRATION_PER_CLASS = 16

IMAGENETTE_TO_IMAGENET = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}


@torch.no_grad()
def copy_bn_selected(target: nn.BatchNorm2d, source: nn.BatchNorm2d, keep_idx: torch.Tensor) -> None:
    """Copy the selected BatchNorm channels into a structurally smaller layer."""
    target.weight.copy_(source.weight[keep_idx])
    target.bias.copy_(source.bias[keep_idx])
    target.running_mean.copy_(source.running_mean[keep_idx])
    target.running_var.copy_(source.running_var[keep_idx])
    target.num_batches_tracked.copy_(source.num_batches_tracked)


def source_model(model_name: str) -> nn.Module:
    if model_name == "resnet50":
        return resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    if model_name == "vit_large":
        return vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1).eval()
    raise ValueError(f"Unsupported model: {model_name}")


def empty_candidate_architecture(model_name: str, width: int) -> nn.Module:
    """Build only a candidate's shape, without downloads or RBLN dependencies.

    This is used on the GPU server when a checkpoint already contains every
    selected channel/neuron.  It intentionally does not repeat calibration.
    """

    if model_name == "resnet50":
        return resnet50(weights=None, width_per_group=width).eval()
    if model_name == "vit_large":
        return VisionTransformer(
            image_size=224,
            patch_size=16,
            num_layers=24,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=width,
            num_classes=1000,
        ).eval()
    raise ValueError(f"Unsupported model: {model_name}")


def transform_for(model_name: str):
    if model_name == "resnet50":
        return ResNet50_Weights.IMAGENET1K_V2.transforms()
    if model_name == "vit_large":
        return ViT_L_16_Weights.IMAGENET1K_V1.transforms()
    raise ValueError(f"Unsupported model: {model_name}")


def model_stats(model: nn.Module) -> dict[str, int]:
    parameters = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    buffer_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    return {
        "parameters": parameters,
        "pytorch_bytes": parameter_bytes + buffer_bytes,
    }


def named_bottlenecks(model: nn.Module):
    blocks = []
    for stage_name in ("layer1", "layer2", "layer3", "layer4"):
        for index, block in enumerate(getattr(model, stage_name)):
            blocks.append((stage_name, index, block))
    return blocks


class ImageNetLabelSubset(Dataset):
    """ImageFolder subset that returns original ImageNet-1K target ids."""

    def __init__(self, dataset: ImageFolder, indices: Iterable[int]):
        self.dataset = dataset
        self.indices = list(indices)
        self.local_to_imagenet = {
            local: IMAGENETTE_TO_IMAGENET[wnid]
            for wnid, local in dataset.class_to_idx.items()
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, local_target = self.dataset[self.indices[index]]
        return image, self.local_to_imagenet[int(local_target)]


def imagenette_split(
    root: Path,
    transform: Callable,
    calibration_per_class: int = CALIBRATION_PER_CLASS,
):
    """Return disjoint deterministic train-calibration, fine-tune, and val sets.

    ImageFolder orders files by class/path, so this selection is deterministic
    and can be written to the experiment artifacts for exact replay.
    """

    train = ImageFolder(root / "train", transform=transform)
    val = ImageFolder(root / "val", transform=transform)
    selected = []
    counts = {index: 0 for index in range(len(train.classes))}
    for index, (_path, label) in enumerate(train.samples):
        if counts[label] < calibration_per_class:
            selected.append(index)
            counts[label] += 1
    calibration_set = set(selected)
    fine_tune_indices = [index for index in range(len(train)) if index not in calibration_set]
    return (
        ImageNetLabelSubset(train, selected),
        ImageNetLabelSubset(train, fine_tune_indices),
        ImageNetLabelSubset(val, range(len(val))),
        {
            "calibration_per_class": calibration_per_class,
            "calibration_samples": len(selected),
            "fine_tune_train_samples": len(fine_tune_indices),
            "validation_samples": len(val),
            "calibration_paths": [str(train.samples[index][0]) for index in selected],
        },
    )


def make_loader(dataset: Dataset, batch_size: int, workers: int, shuffle: bool = False):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=False,
    )


@torch.inference_mode()
def collect_resnet_activation_stats(model: nn.Module, loader: DataLoader):
    blocks = named_bottlenecks(model)
    sums = [torch.zeros(block.conv2.out_channels) for _, _, block in blocks]
    square_sums = [torch.zeros(block.conv2.out_channels) for _, _, block in blocks]
    counts = [0 for _ in blocks]
    handles = []

    for block_index, (_stage, _index, block) in enumerate(blocks):
        def hook(_module, _inputs, output, index=block_index):
            values = F.relu(output.detach().float())
            sums[index].add_(values.sum(dim=(0, 2, 3)))
            square_sums[index].add_((values * values).sum(dim=(0, 2, 3)))
            counts[index] += values.shape[0] * values.shape[2] * values.shape[3]

        handles.append(block.bn2.register_forward_hook(hook))

    for images, _targets in loader:
        model(images)
    for handle in handles:
        handle.remove()
    means = [value / count for value, count in zip(sums, counts)]
    mean_squares = [value / count for value, count in zip(square_sums, counts)]
    return means, mean_squares


@torch.no_grad()
def resnet_activation_scores(source: nn.Module, mean_squares):
    return [
        activation_square * block.conv3.weight.square().sum(dim=(0, 2, 3))
        for (_stage, _index, block), activation_square in zip(
            named_bottlenecks(source), mean_squares
        )
    ]


@torch.no_grad()
def build_resnet_activation_candidate(source: nn.Module, width_per_group: int, means, scores):
    candidate = copy.deepcopy(source)
    for block_index, ((stage, index, old), mean, score) in enumerate(
        zip(named_bottlenecks(source), means, scores)
    ):
        planes = old.conv3.out_channels // Bottleneck.expansion
        new_block = Bottleneck(
            inplanes=old.conv1.in_channels,
            planes=planes,
            stride=old.stride,
            downsample=copy.deepcopy(old.downsample),
            groups=old.conv2.groups,
            base_width=width_per_group,
            dilation=old.conv2.dilation[0],
            norm_layer=nn.BatchNorm2d,
        ).eval()
        keep_count = new_block.conv1.out_channels
        keep_idx = torch.topk(score, keep_count, largest=True).indices.sort().values
        keep_mask = torch.zeros(old.conv1.out_channels, dtype=torch.bool)
        keep_mask[keep_idx] = True
        drop_idx = (~keep_mask).nonzero(as_tuple=False).flatten()

        new_block.conv1.weight.copy_(old.conv1.weight[keep_idx])
        copy_bn_selected(new_block.bn1, old.bn1, keep_idx)
        new_block.conv2.weight.copy_(old.conv2.weight[keep_idx][:, keep_idx])
        copy_bn_selected(new_block.bn2, old.bn2, keep_idx)
        new_block.conv3.weight.copy_(old.conv3.weight[:, keep_idx])
        new_block.bn3.load_state_dict(old.bn3.state_dict())
        # Keep the expected residual contribution of discarded channels.
        new_block.bn3.running_mean.sub_(old.conv3.weight[:, drop_idx, 0, 0] @ mean[drop_idx])
        if old.downsample is not None:
            new_block.downsample.load_state_dict(old.downsample.state_dict())
        getattr(candidate, stage)[index] = new_block
    return candidate.eval()


@torch.inference_mode()
def collect_vit_activation_stats(model: nn.Module, loader: DataLoader):
    width = model.encoder.layers[0].mlp[0].out_features
    sums = [torch.zeros(width) for _ in model.encoder.layers]
    square_sums = [torch.zeros(width) for _ in model.encoder.layers]
    counts = [0 for _ in model.encoder.layers]
    handles = []
    for layer_index, block in enumerate(model.encoder.layers):
        def hook(_module, _inputs, output, index=layer_index):
            values = output.detach().float()
            sums[index].add_(values.sum(dim=(0, 1)))
            square_sums[index].add_((values * values).sum(dim=(0, 1)))
            counts[index] += values.shape[0] * values.shape[1]

        handles.append(block.mlp[1].register_forward_hook(hook))
    for images, _targets in loader:
        model(images)
    for handle in handles:
        handle.remove()
    means = [value / count for value, count in zip(sums, counts)]
    mean_squares = [value / count for value, count in zip(square_sums, counts)]
    return means, mean_squares


@torch.no_grad()
def vit_activation_scores(source: nn.Module, mean_squares):
    return [
        activation_square * block.mlp[3].weight.square().sum(dim=0)
        for block, activation_square in zip(source.encoder.layers, mean_squares)
    ]


@torch.no_grad()
def build_vit_activation_candidate(source: nn.Module, mlp_width: int, means, scores):
    candidate = copy.deepcopy(source)
    for old_block, new_block, mean, score in zip(
        source.encoder.layers, candidate.encoder.layers, means, scores
    ):
        old_fc1, old_fc2 = old_block.mlp[0], old_block.mlp[3]
        keep_idx = torch.topk(score, mlp_width, largest=True).indices.sort().values
        keep_mask = torch.zeros(old_fc1.out_features, dtype=torch.bool)
        keep_mask[keep_idx] = True
        drop_idx = (~keep_mask).nonzero(as_tuple=False).flatten()
        new_mlp = MLPBlock(1024, mlp_width, 0.0).eval()
        new_mlp[0].weight.copy_(old_fc1.weight[keep_idx])
        new_mlp[0].bias.copy_(old_fc1.bias[keep_idx])
        new_mlp[3].weight.copy_(old_fc2.weight[:, keep_idx])
        # Preserve the average removed MLP contribution before fine-tuning.
        new_mlp[3].bias.copy_(old_fc2.bias + old_fc2.weight[:, drop_idx] @ mean[drop_idx])
        new_block.mlp = new_mlp
    return candidate.eval()

