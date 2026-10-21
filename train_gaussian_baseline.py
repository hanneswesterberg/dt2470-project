"""Train the legacy Gaussian-HMM baseline.

The hierarchical CNN is the repository's default inference model, but its
historical training loop was not included in the handoff and must not be
guessed. This script is intentionally retained for reproducible Gaussian
baseline experiments.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __name__ == "__main__":
    from scripts.environments import restart_in_application_environment

    restart_in_application_environment(Path(__file__).resolve())

import numpy as np

from data.load_data import SongFiles, find_songs
from src.cli import ask_choice, ask_integer, ask_text, ask_yes_no
from src.features import CQTChromaExtractor
from src.models import component_path, resolve_model_path, save_component
from src.training import (
    train_emission_from_songs,
    train_model,
    train_transition_from_songs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the legacy Gaussian-HMM baseline."
    )
    parser.add_argument("--target", choices=("full", "emission", "hmm"), default=None)
    parser.add_argument(
        "--model", default=None, help="Name for the saved model or component."
    )
    parser.add_argument("--training-songs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--audio-dataset", type=Path, default=None)
    parser.add_argument("--annotation-dataset", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Require choices from flags instead of asking terminal questions.",
    )
    return parser


def _select_training_songs(
    available: list[SongFiles], count: int, seed: int
) -> list[SongFiles]:
    indices = np.random.default_rng(seed).permutation(len(available))[:count]
    return [available[index] for index in indices]


def _artifact_path(target: str, name: str, models_dir: Path) -> Path:
    if target == "full":
        return resolve_model_path(name, models_dir)
    kind = "emission" if target == "emission" else "transition"
    return component_path(models_dir, kind, name)


def _collect_choices(
    args: argparse.Namespace, available_count: int
) -> tuple[str, int, str]:
    interactive = not args.non_interactive
    target = args.target
    if target is None:
        if not interactive:
            raise ValueError("--target is required with --non-interactive.")
        target = ask_choice(
            "What would you like to train?",
            (
                ("full", "Complete model: emission + HMM + validation tuning"),
                ("emission", "Emission component only"),
                ("hmm", "HMM transition component only (no audio loading)"),
            ),
        )

    minimum = 2 if target == "full" else 1
    if available_count < minimum:
        raise RuntimeError(
            f"Training target {target!r} needs at least {minimum} available dataset song(s)."
        )
    default_count = min(80, available_count)
    count = args.training_songs
    if count is None:
        if not interactive:
            raise ValueError("--training-songs is required with --non-interactive.")
        count = ask_integer(
            "How many dataset songs should be used?",
            max(minimum, default_count),
            minimum,
            available_count,
        )
    if not minimum <= count <= available_count:
        raise ValueError(
            f"--training-songs must be between {minimum} and {available_count} for {target}."
        )

    default_name = {
        "full": f"gaussian_hmm_{count}",
        "emission": f"gaussian_emission_{count}",
        "hmm": f"relative_transition_{count}",
    }[target]
    name = args.model
    if name is None:
        if not interactive:
            raise ValueError("--model is required with --non-interactive.")
        name = ask_text("What should the artifact be named?", default_name)
    return target, count, name


def run(args: argparse.Namespace) -> int:
    songs_by_id = find_songs(args.audio_dataset, args.annotation_dataset)
    available = list(songs_by_id.values())
    if not available:
        raise RuntimeError("No paired AAM dataset songs were found.")
    print(f"Found {len(available)} paired AAM songs in the mounted dataset.")

    target, count, name = _collect_choices(args, len(available))
    path = _artifact_path(target, name, args.models_dir)
    if (
        path.exists()
        and not args.overwrite
        and (
            args.non_interactive
            or not ask_yes_no(f"{path} already exists. Replace it?")
        )
    ):
        raise FileExistsError(f"Artifact already exists: {path}")

    selected = _select_training_songs(available, count, args.seed)
    if not args.non_interactive:
        print("\nTraining configuration:")
        print(f"  Target: {target}")
        print(f"  Dataset songs: {count}")
        print(f"  Seed: {args.seed}")
        print(f"  Output: {path}")
        if not ask_yes_no("Start training?", default=True):
            print("Training cancelled.")
            return 0

    metadata: dict[str, object] = {
        "dataset_song_ids": [song.song_id for song in selected],
        "seed": args.seed,
        "target": target,
    }
    if target == "full":
        extractor = CQTChromaExtractor()
        model = train_model(
            selected, extractor=extractor, seed=args.seed, progress=print
        )
        model.metadata.update(metadata)
        model.metadata["feature_extractor"] = "CQTChromaExtractor"
        model.metadata["sample_rate"] = extractor.config.sample_rate
        model.metadata["hop_length"] = extractor.config.hop_length
        model.save(path)
    elif target == "emission":
        component = train_emission_from_songs(selected, progress=print)
        save_component(component, "emission", path, metadata)
    else:
        component = train_transition_from_songs(selected, progress=print)
        save_component(component, "transition", path, metadata)

    print(f"\nSaved {target} artifact: {path}")
    if target != "full":
        print(
            "Component-only artifacts are for experiments and cannot be run by main.py alone."
        )
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
