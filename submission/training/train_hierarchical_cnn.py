"""Train the hierarchical CQT72 CNN from cached song features."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    from scripts.environments import restart_in_application_environment

    restart_in_application_environment(Path(__file__).resolve())

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.hierarchical.hierarchical_cnn import HierarchicalCNN72, rotate_to_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the hierarchical root-quality-seventh CNN."
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=Path("data/features_cqt72_v1"),
        help="Folder containing RWC_PNNN.npz feature files.",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("models/split_manifest.json"),
    )
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        default=Path("models/best_model.pt"),
        help="Checkpoint used for vocabulary and architecture validation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/training_runs/hierarchical_cnn"),
    )
    parser.add_argument(
        "--extension-key",
        default=None,
        help="NPZ key for multi-hot extension targets (auto-detected if omitted).",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cpu, mps, or cuda")
    return parser


def choose_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_song_arrays(
    feature_dir: Path,
    song: int,
    extension_count: int,
    extension_key: str | None,
) -> tuple[np.ndarray, ...]:
    path = feature_dir / f"RWC_P{int(song):03d}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing feature file: {path}")
    with np.load(path, allow_pickle=False) as stored:
        features = stored["X"].astype(np.float32)
        root = stored["root"].astype(np.int64)
        quality = stored["quality"].astype(np.int64)
        seventh = stored["seventh"].astype(np.int64)
        extension_candidates = (
            [extension_key]
            if extension_key
            else ["extensions", "extension", "ext", "extension_targets", "ext_targets"]
        )
        chosen_key = next(
            (key for key in extension_candidates if key in stored.files), None
        )
        if chosen_key is None:
            raise KeyError(
                f"Missing extension targets in {path.name}; expected one of "
                f"{extension_candidates}, available keys: {stored.files}"
            )
        extensions = np.asarray(stored[chosen_key])

    count = len(features)
    if features.shape[1:] != (72, 32):
        raise ValueError(f"Expected {path.name} X shape (N,72,32), got {features.shape}")
    if root.shape != (count,) or quality.shape != (count,) or seventh.shape != (count,):
        raise ValueError(f"Target lengths do not match X in {path.name}")
    if not np.all((root >= 0) & (root < 12)):
        raise ValueError(f"Root targets must be between 0 and 11 in {path.name}")
    if extensions.shape == (count,) and np.issubdtype(extensions.dtype, np.integer):
        bit_positions = np.arange(extension_count)
        extensions = (extensions.astype(np.int64)[:, None] >> bit_positions) & 1
    if extensions.shape != (count, extension_count):
        raise ValueError(
            f"Expected extension shape {(count, extension_count)}, "
            f"got {extensions.shape} in {path.name}"
        )
    if not np.all((extensions == 0) | (extensions == 1)):
        raise ValueError(f"Extension targets must be binary in {path.name}")
    return features, root, quality, seventh, extensions.astype(np.float32)


def load_split(
    feature_dir: Path,
    songs: list[int],
    extension_count: int,
    extension_key: str | None,
) -> tuple[np.ndarray, ...]:
    loaded = [
        load_song_arrays(feature_dir, song, extension_count, extension_key)
        for song in songs
    ]
    return tuple(
        np.concatenate([song_arrays[index] for song_arrays in loaded], axis=0)
        for index in range(5)
    )


def transpose_cqt(features: torch.Tensor, semitones: int) -> torch.Tensor:
    return torch.roll(
        features.reshape(6, 12, 32), shifts=semitones, dims=1
    ).reshape(72, 32)


class ChordDataset(Dataset):
    def __init__(self, arrays: tuple[np.ndarray, ...], augment: bool = False) -> None:
        self.features, self.root, self.quality, self.seventh, self.extensions = arrays
        self.augment = augment

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        features = torch.from_numpy(self.features[index].copy())
        root = int(self.root[index])
        if self.augment:
            shift = random.randrange(12)
            features = transpose_cqt(features, shift)
            root = (root + shift) % 12
            if random.random() < 0.5:
                features *= random.uniform(0.90, 1.10)
        return (
            features,
            torch.tensor(root, dtype=torch.long),
            torch.tensor(self.quality[index], dtype=torch.long),
            torch.tensor(self.seventh[index], dtype=torch.long),
            torch.from_numpy(self.extensions[index].copy()),
        )


def run_epoch(
    model: HierarchicalCNN72,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    extension_loss: nn.Module | None,
    mean: float,
    std: float,
    device: torch.device,
    training: bool,
) -> dict[str, float]:
    model.train(training)
    totals: Counter[str] = Counter()
    example_count = 0
    for features, root, quality, seventh, extensions in loader:
        features, root, quality, seventh, extensions = (
            value.to(device)
            for value in (features, root, quality, seventh, extensions)
        )
        relative = rotate_to_root(features, root)
        with torch.set_grad_enabled(training):
            root_logits, quality_logits, seventh_logits, extension_logits = model(
                (features - mean) / std,
                (relative - mean) / std,
                quality,
                seventh,
            )
            losses = {
                "root": F.cross_entropy(root_logits, root),
                "quality": F.cross_entropy(quality_logits, quality),
                "seventh": F.cross_entropy(seventh_logits, seventh),
            }
            loss = losses["root"] + losses["quality"] + losses["seventh"]
            if extension_loss is not None:
                losses["extensions"] = extension_loss(extension_logits, extensions)
                loss += 0.25 * losses["extensions"]
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        batch_size = len(features)
        example_count += batch_size
        totals["loss"] += float(loss.item()) * batch_size
        for name, value in losses.items():
            totals[name] += float(value.item()) * batch_size
    return {name: total / example_count for name, total in totals.items()}


@torch.no_grad()
def validate_joint_rqs(
    model: HierarchicalCNN72,
    loader: DataLoader,
    mean: float,
    std: float,
    quality_count: int,
    seventh_count: int,
    device: torch.device,
) -> float:
    model.eval()
    correct = total = 0
    for features, root, quality, seventh, _ in loader:
        features = features.to(device)
        batch_size = len(features)
        root_probability = torch.softmax(
            model.root_logits((features - mean) / std), dim=1
        )
        joint_parts = []
        for root_index in range(12):
            roots = torch.full(
                (batch_size,), root_index, dtype=torch.long, device=device
            )
            relative = rotate_to_root(features, roots)
            encoded = model.rel_features((relative - mean) / std)
            quality_probability = torch.softmax(model.quality_logits(encoded), dim=1)
            for quality_index in range(quality_count):
                qualities = torch.full(
                    (batch_size,), quality_index, dtype=torch.long, device=device
                )
                seventh_probability = torch.softmax(
                    model.seventh_logits(encoded, qualities), dim=1
                )
                joint_parts.append(
                    root_probability[:, root_index : root_index + 1]
                    * quality_probability[:, quality_index : quality_index + 1]
                    * seventh_probability
                )
        prediction = torch.cat(joint_parts, dim=1).argmax(dim=1)
        target = (
            root.to(device) * quality_count * seventh_count
            + quality.to(device) * seventh_count
            + seventh.to(device)
        )
        correct += int((prediction == target).sum().item())
        total += batch_size
    return correct / total


def write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def train(args: argparse.Namespace) -> Path:
    if args.epochs <= 0 or args.patience <= 0 or args.batch_size <= 0:
        raise ValueError("epochs, patience, and batch-size must be positive")
    seed_everything(args.seed)
    device = choose_device(args.device)
    reference = torch.load(
        args.reference_checkpoint, map_location="cpu", weights_only=False
    )
    quality = list(reference["quality"])
    seventh = list(reference["seventh"])
    extensions = list(reference["extensions"])
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    splits = manifest["splits"]
    sizes = tuple(len(splits[name]) for name in ("train", "val", "test"))
    if sizes != (70, 15, 15):
        raise ValueError(f"Expected a 70/15/15 split, found {sizes}")

    print(f"Device: {device}")
    print("Loading training and validation features...")
    training_arrays = load_split(
        args.feature_dir, splits["train"], len(extensions), args.extension_key
    )
    validation_arrays = load_split(
        args.feature_dir, splits["val"], len(extensions), args.extension_key
    )
    mean = float(training_arrays[0].mean(dtype=np.float64))
    std = float(training_arrays[0].std(dtype=np.float64))
    if std <= 0:
        raise ValueError("Training feature standard deviation must be positive")
    if not np.all((training_arrays[2] >= 0) & (training_arrays[2] < len(quality))):
        raise ValueError("Training quality targets do not match the checkpoint vocabulary")
    if not np.all((training_arrays[3] >= 0) & (training_arrays[3] < len(seventh))):
        raise ValueError("Training seventh targets do not match the checkpoint vocabulary")
    print(
        f"Segments: {len(training_arrays[0])} train, "
        f"{len(validation_arrays[0])} validation"
    )
    print(f"Normalization: mean={mean:.10f}, std={std:.10f}")
    positives = training_arrays[4].sum(axis=0).astype(int)
    print("Extension positives:", dict(zip(extensions, positives)))

    train_loader = DataLoader(
        ChordDataset(training_arrays, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        ChordDataset(validation_arrays),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = HierarchicalCNN72(
        len(quality), len(seventh), len(extensions)
    ).to(device)
    HierarchicalCNN72(
        len(quality), len(seventh), len(extensions)
    ).load_state_dict(reference["model"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )
    positive = torch.tensor(
        training_arrays[4].sum(axis=0), dtype=torch.float32, device=device
    )
    negative = len(training_arrays[4]) - positive
    weights = (negative / positive.clamp_min(1)).clamp(min=1.0, max=15.0)
    extension_loss = nn.BCEWithLogitsLoss(pos_weight=weights)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_model.pt"
    history_path = args.output_dir / "training_history.csv"
    config = vars(args).copy()
    config.update(
        {
            "feature_dir": str(args.feature_dir),
            "split_manifest": str(args.split_manifest),
            "reference_checkpoint": str(args.reference_checkpoint),
            "output_dir": str(args.output_dir),
            "device": str(device),
        }
    )
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    best_score = -np.inf
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            extension_loss,
            mean,
            std,
            device,
            True,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            optimizer,
            extension_loss,
            mean,
            std,
            device,
            False,
        )
        validation_rqs = validate_joint_rqs(
            model,
            validation_loader,
            mean,
            std,
            len(quality),
            len(seventh),
            device,
        )
        scheduler.step(validation_rqs)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": validation_metrics["loss"],
            "val_rqs": validation_rqs,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        write_history(history_path, history)
        improved = validation_rqs > best_score
        if improved:
            best_score = validation_rqs
            best_epoch = epoch
            torch.save(
                {
                    "model": copy.deepcopy(model.state_dict()),
                    "epoch": epoch,
                    "score": float(validation_rqs),
                    "feature": "CQT72_v1",
                    "mean": mean,
                    "std": std,
                    "seed": args.seed,
                    "quality": quality,
                    "seventh": seventh,
                    "extensions": extensions,
                    "source_split_manifest": manifest,
                },
                checkpoint_path,
            )
        marker = " *" if improved else ""
        print(
            f"Epoch {epoch:03d} | train {train_metrics['loss']:.4f} | "
            f"val {validation_metrics['loss']:.4f} | "
            f"RQS {validation_rqs:.4f} | best {best_score:.4f}{marker}"
        )
        if epoch - best_epoch >= args.patience:
            print("Early stopping")
            break
    print(f"Best epoch: {best_epoch}; validation RQS: {best_score:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    return checkpoint_path


def main() -> int:
    train(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
