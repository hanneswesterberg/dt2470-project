"""Interactive terminal program for inference with already-trained models.

Run ``python3 main.py`` for a guided selection of local/Drive songs, one or
more complete model checkpoints, a beat tracker, and optional ground-truth
comparison. Each run writes CSV timelines and a four-beat-per-bar PNG chart.
This script never trains models; use ``python3 train.py`` when the required
checkpoint does not exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

if __name__ == "__main__":
    from scripts.environments import restart_in_application_environment

    restart_in_application_environment(Path(__file__).resolve())

import numpy as np

from data.annotations import load_annotations
from data.load_data import (
    SongFiles,
    find_input_annotation,
    find_input_audio,
    find_songs,
)
from src.beat_net import create_beat_tracker
from src.cli import ask_choice, ask_many, ask_text, ask_yes_no
from src.evaluation import evaluate_timeline
from src.features import CQTChromaExtractor
from src.inference import ChordTimeline, prepare_audio
from src.models import ChordRecognitionModel, list_models, resolve_model_path
from src.visualization import write_chord_chart


@dataclass(frozen=True)
class InferenceSong:
    """One selected audio file and its optional reference annotation."""

    name: str
    audio: Path
    annotations: Path | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run trained chord models on selected songs."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("input"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Complete model name/path; repeat the option to compare models.",
    )
    parser.add_argument("--source", choices=("local", "cloud", "both"), default=None)
    parser.add_argument(
        "--song",
        action="append",
        default=None,
        help="Local filename/stem or Drive song ID; repeat to select multiple songs.",
    )
    parser.add_argument(
        "--beat-tracker", choices=("auto", "beatnet", "librosa"), default=None
    )
    parser.add_argument(
        "--seed", type=int, default=2026, help="Seed for random cloud selection."
    )
    parser.add_argument("--audio-dataset", type=Path, default=None)
    parser.add_argument("--annotation-dataset", type=Path, default=None)
    parser.add_argument(
        "--compare-ground-truth",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Evaluate and include available annotations in charts.",
    )
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Require selections from flags instead of asking terminal questions.",
    )
    return parser


def _print_models(models_dir: Path) -> list[Path]:
    models = list_models(models_dir)
    if models:
        print("Available complete models:")
        for path in models:
            print(f"  {path.stem}")
    else:
        print(f"No complete model checkpoints found in {models_dir}.")
        print("Train one with: python3 train.py")
    return models


def _select_models(args: argparse.Namespace, available: list[Path]) -> list[Path]:
    if args.model:
        paths = [resolve_model_path(name, args.models_dir) for name in args.model]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            names = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                f"Model checkpoint(s) not found: {names}. Run python3 train.py."
            )
        return paths
    if args.non_interactive:
        raise ValueError("At least one --model is required with --non-interactive.")
    if not available:
        raise FileNotFoundError(
            "No complete models are available. Run python3 train.py first."
        )
    return ask_many(
        "Which complete model(s) should run?",
        [(path, path.stem) for path in available],
    )


def _load_cloud_songs(
    args: argparse.Namespace,
) -> tuple[dict[str, SongFiles], str | None]:
    try:
        return find_songs(args.audio_dataset, args.annotation_dataset), None
    except (FileNotFoundError, RuntimeError) as exc:
        return {}, str(exc)


def _select_source(
    args: argparse.Namespace,
    local: list[Path],
    cloud: dict[str, SongFiles],
    cloud_error: str | None,
) -> str:
    if args.source:
        return args.source
    if args.non_interactive:
        raise ValueError("--source is required with --non-interactive.")
    options: list[tuple[str, str]] = []
    if local:
        options.append(("local", f"Local input/ songs ({len(local)} available)"))
    if cloud:
        options.append(("cloud", f"Google Drive AAM songs ({len(cloud)} available)"))
    if local and cloud:
        options.append(("both", "Songs from both local input/ and Google Drive"))
    if not options:
        detail = f" Drive error: {cloud_error}" if cloud_error else ""
        raise FileNotFoundError(f"No local or cloud songs are available.{detail}")
    return ask_choice("Where should inference songs come from?", options)


def _matches_local(request: str, path: Path) -> bool:
    return request in {path.name, path.stem}


def _select_local_songs(
    local: list[Path], requested: list[str] | None, interactive: bool
) -> list[InferenceSong]:
    if not local:
        raise FileNotFoundError("No supported audio files were found in input/.")
    if requested:
        selected = [
            path
            for path in local
            if any(_matches_local(item, path) for item in requested)
        ]
        unresolved = [
            item
            for item in requested
            if not any(_matches_local(item, path) for path in local)
        ]
        if unresolved and not interactive:
            raise FileNotFoundError(f"Local song(s) not found: {', '.join(unresolved)}")
    elif interactive:
        selected = ask_many(
            "Which local song(s) should run?",
            [(path, path.name) for path in local],
        )
    else:
        selected = local
    return [
        InferenceSong(path.stem, path, find_input_annotation(path)) for path in selected
    ]


def _parse_cloud_request(
    answer: str, cloud: dict[str, SongFiles], seed: int
) -> list[SongFiles]:
    normalized = answer.strip().lower()
    if normalized == "all":
        return list(cloud.values())
    if normalized.startswith("random "):
        try:
            count = int(normalized.split(maxsplit=1)[1])
        except ValueError as exc:
            raise ValueError("Use 'random N', where N is a whole number.") from exc
        if not 1 <= count <= len(cloud):
            raise ValueError(f"Random count must be between 1 and {len(cloud)}.")
        values = list(cloud.values())
        indices = np.random.default_rng(seed).choice(len(values), count, replace=False)
        return [values[index] for index in indices]
    ids = [item.strip() for item in answer.split(",") if item.strip()]
    missing = [song_id for song_id in ids if song_id not in cloud]
    if not ids or missing:
        detail = f" Unknown IDs: {', '.join(missing)}." if missing else ""
        raise ValueError(f"Enter valid comma-separated song IDs.{detail}")
    return [cloud[song_id] for song_id in ids]


def _select_cloud_songs(
    cloud: dict[str, SongFiles],
    requested: list[str] | None,
    interactive: bool,
    seed: int,
) -> list[InferenceSong]:
    if not cloud:
        raise FileNotFoundError("The Google Drive AAM dataset is unavailable.")
    if requested:
        selected = _parse_cloud_request(",".join(requested), cloud, seed)
    elif not interactive:
        raise ValueError("At least one --song ID is required for cloud inference.")
    else:
        preview = ", ".join(list(cloud)[:12])
        print(f"\nAvailable Drive IDs begin: {preview}, ...")
        print("Enter comma-separated IDs, 'random N', or 'all'.")
        while True:
            answer = ask_text("Which Drive songs should run?", "random 1")
            try:
                selected = _parse_cloud_request(answer, cloud, seed)
                if answer.strip().lower() == "all" and not ask_yes_no(
                    f"Run all {len(selected)} cloud songs?"
                ):
                    continue
                break
            except ValueError as exc:
                print(exc)
    return [
        InferenceSong(song.song_id, song.audio, song.annotations) for song in selected
    ]


def _select_songs(
    args: argparse.Namespace,
    source: str,
    local: list[Path],
    cloud: dict[str, SongFiles],
) -> list[InferenceSong]:
    interactive = not args.non_interactive
    if source == "both" and not interactive and args.song:
        raise ValueError(
            "Non-interactive --source both is ambiguous with --song. Run the wizard, "
            "or run separate local and cloud commands."
        )
    selected: list[InferenceSong] = []
    if source in {"local", "both"}:
        selected.extend(_select_local_songs(local, args.song, interactive))
    if source in {"cloud", "both"}:
        selected.extend(_select_cloud_songs(cloud, args.song, interactive, args.seed))
    return selected


def _select_ground_truth_comparison(
    args: argparse.Namespace, songs: list[InferenceSong]
) -> bool:
    """Resolve whether available annotations should be loaded and compared."""

    annotations_available = any(song.annotations is not None for song in songs)
    if args.compare_ground_truth is not None:
        if args.compare_ground_truth and not annotations_available:
            print(
                "No selected song has ground truth; charts will show predictions only."
            )
        return args.compare_ground_truth
    if args.non_interactive or not annotations_available:
        return False
    return ask_yes_no(
        "Compare predictions with ground truth when available?", default=True
    )


def run(args: argparse.Namespace) -> int:
    available_models = _print_models(args.models_dir)
    if args.list_models:
        return 0

    local = find_input_audio(args.input_dir)
    needs_cloud = not args.non_interactive or args.source in {"cloud", "both"}
    cloud, cloud_error = _load_cloud_songs(args) if needs_cloud else ({}, None)
    source = _select_source(args, local, cloud, cloud_error)
    songs = _select_songs(args, source, local, cloud)
    model_paths = _select_models(args, available_models)
    compare_ground_truth = _select_ground_truth_comparison(args, songs)

    tracker_name = args.beat_tracker
    if tracker_name is None:
        if args.non_interactive:
            tracker_name = "auto"
        else:
            tracker_name = ask_choice(
                "Which beat tracker should be used?",
                (
                    ("auto", "Auto: BeatNet if installed, otherwise librosa"),
                    ("librosa", "librosa"),
                    ("beatnet", "BeatNet"),
                ),
            )

    models = [(path.stem, ChordRecognitionModel.load(path)) for path in model_paths]
    tracker = create_beat_tracker(tracker_name)
    extractor = CQTChromaExtractor()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, object]] = []

    for song in songs:
        print(f"\nPreparing audio once: {song.name}")
        prepared = prepare_audio(song.audio, tracker, extractor)
        reference: ChordTimeline | None = None
        if compare_ground_truth and song.annotations is not None:
            annotations = load_annotations(song.annotations)
            starts, ends, labels = annotations.intervals(prepared.duration)
            reference = ChordTimeline(starts, ends, labels)

        predictions: dict[str, ChordTimeline] = {}
        song_results: list[dict[str, object]] = []
        for model_name, model in models:
            print(f"  Running model: {model_name}")
            timeline = prepared.predict(model)
            predictions[model_name] = timeline
            output_path = args.output_dir / model_name / f"{song.name}_chords.csv"
            timeline.write_csv(output_path)
            result: dict[str, object] = {
                "model": model_name,
                "song": song.name,
                "audio": str(song.audio),
                "prediction": str(output_path),
                "intervals": len(timeline.labels),
            }
            if reference is not None:
                metrics = evaluate_timeline(
                    reference.starts,
                    reference.ends,
                    reference.labels,
                    timeline.starts,
                    timeline.ends,
                    timeline.labels,
                )
                result["ground_truth"] = str(song.annotations)
                result["metrics"] = metrics.to_dict()
                print(
                    f"    accuracy={metrics.chord_accuracy:.1%}, "
                    f"beat F1={metrics.beat_f1:.1%}, coverage={metrics.prediction_coverage:.1%}"
                )
            elif compare_ground_truth:
                print("    no ground truth; comparison skipped")
            print(f"    wrote {output_path}")
            song_results.append(result)

        chart_path = write_chord_chart(
            song.name,
            predictions,
            args.output_dir / "charts" / f"{song.name}_chord_chart.png",
            reference,
        )
        for result in song_results:
            result["chart"] = str(chart_path)
        summary.extend(song_results)
        print(f"  Wrote chart: {chart_path}")

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nFinished {len(songs)} song(s) with {len(models)} model(s).")
    print(f"Summary: {summary_path}")
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (EOFError, FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
