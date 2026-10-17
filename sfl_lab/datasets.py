"""Deterministic disjoint data splits and nonempty federated partitions."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .config import ExperimentConfig


@dataclass
class DataBundle:
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    client_indices: list[np.ndarray]
    num_classes: int
    metadata: dict


def _sample_stratified(indices: np.ndarray, labels: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Largest-remainder stratification also works when a split is smaller than K."""
    if count >= len(indices):
        return rng.permutation(indices)
    classes, sizes = np.unique(labels[indices], return_counts=True)
    expected = count * sizes / len(indices)
    allocation = np.floor(expected).astype(int)
    tie = rng.uniform(0, 1e-9, len(classes))
    order = np.argsort(-(expected - allocation + tie))
    allocation[order[:count - int(allocation.sum())]] += 1
    chunks = [rng.permutation(indices[labels[indices] == c])[:n] for c, n in zip(classes, allocation)]
    return rng.permutation(np.concatenate(chunks))


def partition_labels(labels: np.ndarray, clients: int, method: str, alpha: float, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(labels)
    if n < clients:
        raise ValueError(f"Only {n} training examples are available for {clients} clients.")
    if method == "iid":
        partitions = [list(p) for p in np.array_split(rng.permutation(n), clients)]
    elif method == "quantity":
        # Reserve one example each, then allocate the remainder by a Dirichlet draw.
        counts = rng.multinomial(n - clients, rng.dirichlet(np.full(clients, alpha))) + 1
        partitions = [list(p) for p in np.split(rng.permutation(n), np.cumsum(counts)[:-1])]
    elif method == "label_skew":
        # Two sorted shards per client: common label-shard non-IID protocol.
        order = np.lexsort((rng.random(n), labels))
        shards = np.array_split(order, min(2 * clients, n))
        rng.shuffle(shards)
        partitions = [[] for _ in range(clients)]
        for i, shard in enumerate(shards):
            partitions[i % clients].extend(shard.tolist())
    elif method == "dirichlet":
        partitions = [[] for _ in range(clients)]
        for label in np.unique(labels):
            indices = rng.permutation(np.flatnonzero(labels == label))
            counts = rng.multinomial(len(indices), rng.dirichlet(np.full(clients, alpha)))
            for k, part in enumerate(np.split(indices, np.cumsum(counts)[:-1])):
                partitions[k].extend(part.tolist())
    else:
        raise ValueError(f"Unknown partition: {method}")
    # Deterministic repair preserves disjointness and every sample, unlike retry loops.
    for empty in [i for i, p in enumerate(partitions) if not p]:
        donor = max(range(clients), key=lambda i: len(partitions[i]))
        partitions[empty].append(partitions[donor].pop())
    return [rng.permutation(np.asarray(p, dtype=np.int64)) for p in partitions]


def _digest(indices: np.ndarray) -> str:
    return sha256(np.sort(indices).astype("<i8").tobytes()).hexdigest()


def prepare_data(config: ExperimentConfig, seed: int) -> DataBundle:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 101]))
    source_name = config.dataset
    image_dataset = None
    if source_name == "digits":
        from sklearn.datasets import load_digits
        source = load_digits()
        labels = np.asarray(source.target, dtype=np.int64)
        images = torch.as_tensor(source.images, dtype=torch.float32).unsqueeze(1) / 16.0
        images = F.interpolate(images, size=(32, 32), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1)
        class_names = [str(i) for i in range(10)]
    elif source_name == "cifar10":
        from torchvision.datasets import CIFAR10
        root = Path(config.dataset_path or Path(__file__).resolve().parent.parent / "data" / "cifar10")
        train_source = CIFAR10(str(root), train=True, download=True)
        test_source = CIFAR10(str(root), train=False, download=True)
        # Pool official folds then create one explicit research split. No canonical test claims.
        labels = np.asarray(train_source.targets + test_source.targets, dtype=np.int64)
        images = np.concatenate((train_source.data, test_source.data))
        class_names = train_source.classes
    else:
        from torchvision.datasets import ImageFolder
        from torchvision.transforms import Compose, Resize, ToTensor
        root = Path(config.dataset_path or "").expanduser().resolve()
        if not root.is_dir():
            raise ValueError("dataset_path must identify an existing image folder with class subdirectories.")
        image_dataset = ImageFolder(str(root), transform=Compose([Resize((32, 32)), ToTensor()]))
        labels = np.asarray(image_dataset.targets, dtype=np.int64)
        images = None
        class_names = image_dataset.classes
    if len(np.unique(labels)) < 2:
        raise ValueError("The dataset must contain at least two classes.")
    count = min(config.samples, len(labels))
    if count < 40:
        raise ValueError("At least 40 images are required for separate train, validation and test splits.")
    selected = _sample_stratified(np.arange(len(labels)), labels, count, rng)
    train_indices = _sample_stratified(selected, labels, int(count * .7), rng)
    remaining = np.setdiff1d(selected, train_indices)
    val_indices = _sample_stratified(remaining, labels, int(count * .1), rng)
    test_indices = np.setdiff1d(remaining, val_indices)
    partitions = partition_labels(labels[train_indices], config.clients, config.partition, config.dirichlet_alpha, seed)

    def materialize(indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        if image_dataset is not None:
            x = torch.stack([image_dataset[int(i)][0] for i in indices])
        elif source_name == "cifar10":
            x = torch.as_tensor(images[indices]).permute(0, 3, 1, 2).float() / 255.0
        else:
            x = images[indices].clone()
        return x.contiguous(), torch.as_tensor(labels[indices], dtype=torch.long)

    train_x, train_y = materialize(train_indices)
    val_x, val_y = materialize(val_indices)
    test_x, test_y = materialize(test_indices)
    metadata = {
        "name": source_name, "available": len(labels), "requested": config.samples,
        "samples": count, "train": len(train_indices), "validation": len(val_indices), "test": len(test_indices),
        "partition": config.partition, "dirichlet_alpha": config.dirichlet_alpha,
        "class_names": class_names, "input_shape": [3, 32, 32], "pixel_range": [0, 1],
        "split_hashes": {"train": _digest(train_indices), "validation": _digest(val_indices), "test": _digest(test_indices)},
        "indices": {"train": train_indices.tolist(), "validation": val_indices.tolist(), "test": test_indices.tolist()},
        "client_sizes": [len(p) for p in partitions],
        "client_label_counts": [{str(c): int(np.sum(train_y[p].numpy() == c)) for c in range(len(class_names))} for p in partitions],
        "split_policy": "70/10/20, deterministic proportional stratification; small splits may omit classes",
        "cifar_policy": "official train/test pooled for explicit research repartition" if source_name == "cifar10" else None,
    }
    return DataBundle(train_x, train_y, val_x, val_y, test_x, test_y, partitions, len(class_names), metadata)
