"""Run hierarchical chord recognition over an annotated audio collection."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from data.annotations import load_annotations
from data.load_data import find_songs
from src.evaluation import evaluate_timeline
from src.hierarchical.artifacts import HierarchicalArtifacts
from src.hierarchical.inference_pipeline import HierarchicalRecognitionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the hierarchical CNN/HSMM on annotated songs."
    )
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--annotation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/evaluation"))
    parser.add_argument("--models-dir", type=Path, default=Path("models/hierarchical"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/chordrec"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-roots", type=int, default=4, choices=range(1, 13))
    parser.add_argument("--hsmm-order", type=int, default=2, choices=(1, 2))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--songs",
        nargs="+",
        type=int,
        default=None,
        help="Specific RWC song numbers to evaluate.",
    )
    return parser


def triad_label(label: str) -> str:
    """Remove seventh/extension suffixes for comparison with AAM triad labels."""

    match = re.match(r"^([A-G](?:#|b)?)(maj|m|dim|aug|sus2|sus4)?", label)
    if not match:
        return "N"
    root, quality = match.groups()
    if quality == "m":
        return f"{root}min"
    if quality in {"dim", "aug", "sus2", "sus4"}:
        return "N"
    return f"{root}maj"


def evaluate_song(
    pipeline: HierarchicalRecognitionPipeline, song, output_dir: Path
) -> dict[str, object]:
    timeline, cache_hit = pipeline.process(song.audio)
    reference = load_annotations(song.annotations)
    starts, ends, labels = reference.intervals(float(timeline.ends[-1]))
    predicted_labels = np.asarray([triad_label(label) for label in timeline.labels])
    metrics = evaluate_timeline(
        starts,
        ends,
        labels,
        timeline.starts,
        timeline.ends,
        predicted_labels,
    )
    return {
        "song": song.song_id,
        "audio": str(song.audio),
        "annotation": str(song.annotations),
        "cache_hit": cache_hit,
        "metrics": metrics.to_dict(),
    }


def write_results(output_dir: Path, results: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    fields = ["song", "chord_accuracy", "prediction_coverage", "beat_precision", "beat_recall", "beat_f1"]
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({"song": result["song"], **result["metrics"]})

    metric_names = fields[1:]
    aggregate = {
        name: float(np.mean([result["metrics"][name] for result in results]))
        for name in metric_names
    }
    aggregate["songs"] = len(results)
    (output_dir / "aggregate.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = build_parser().parse_args()
    songs = list(find_songs(args.audio_dir, args.annotation_dir).values())
    if args.songs:
        wanted = {f"{song:04d}" for song in args.songs}
        songs = [song for song in songs if song.song_id in wanted]
    if args.limit is not None:
        songs = songs[: args.limit]
    if not songs:
        raise SystemExit("No matching audio and annotation files were found.")
    artifacts = HierarchicalArtifacts.resolve(local_dir=args.models_dir)
    pipeline = HierarchicalRecognitionPipeline(
        artifacts,
        device=args.device,
        top_roots=args.top_roots,
        hsmm_order=args.hsmm_order,
        cache_dir=args.cache_dir,
    )
    results = []
    for index, song in enumerate(songs, start=1):
        print(f"[{index}/{len(songs)}] {song.song_id}")
        result = evaluate_song(pipeline, song, args.output_dir)
        results.append(result)
        metrics = result["metrics"]
        print(
            f"  accuracy={metrics['chord_accuracy']:.1%} "
            f"beat_f1={metrics['beat_f1']:.1%} "
            f"coverage={metrics['prediction_coverage']:.1%}"
        )
    write_results(args.output_dir, results)
    print(f"Wrote evaluation results to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
