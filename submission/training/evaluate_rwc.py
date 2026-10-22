"""Evaluate CNN emissions and HSMM predictions on the RWC feature files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from data.annotations import load_annotations
from src.hierarchical import HierarchicalArtifacts, HierarchicalRecognitionPipeline
from src.hierarchical.hierarchical_cnn import HierarchicalCNNRecognizer

TEST_SONGS = [37, 34, 90, 89, 3, 60, 72, 17, 35, 12, 69, 75, 40, 47, 39]
OUTPUT_PATH = Path("output/rwc_test_evaluation.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the RWC test songs.")
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/hierarchical/best_model.pt"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--top-roots", type=int, default=12, choices=range(1, 13))
    parser.add_argument("--hsmm-order", type=int, default=2, choices=(1, 2))
    parser.add_argument("--audio-dir", type=Path, default=Path("audio"))
    parser.add_argument("--annotation-dir", type=Path, default=Path("annotations"))
    parser.add_argument(
        "--language-model", type=Path, default=Path("models/harmonic_model.pkl")
    )
    parser.add_argument(
        "--split-manifest", type=Path, default=Path("models/split_manifest.json")
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/chordrec"))
    parser.add_argument(
        "--posterior-dir",
        type=Path,
        default=Path("features/continuous_posteriors"),
        help="Reusable continuous CNN posteriors for HSMM evaluation.",
    )
    return parser


def state_parts(states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return states // 24, (states // 4) % 6, states % 4


def scores(target: np.ndarray, prediction: np.ndarray, durations: np.ndarray) -> dict[str, float]:
    target_root, target_quality, target_seventh = state_parts(target)
    pred_root, pred_quality, pred_seventh = state_parts(prediction)
    root = pred_root == target_root
    root_quality = root & (pred_quality == target_quality)
    root_seventh = root_quality & (pred_seventh == target_seventh)
    return {
        "root_accuracy": float(np.mean(root)),
        "root_quality_accuracy": float(np.mean(root_quality)),
        "root_quality_seventh_accuracy": float(np.mean(root_seventh)),
        "song_root_percent": float(np.sum(durations[root]) / np.sum(durations)),
        "song_root_quality_percent": float(np.sum(durations[root_quality]) / np.sum(durations)),
        "song_root_quality_seventh_percent": float(np.sum(durations[root_seventh]) / np.sum(durations)),
        "duration_seconds": float(np.sum(durations)),
    }


def evaluate_song(song: int, feature_dir: Path, recognizer) -> dict[str, object]:
    path = feature_dir / f"RWC_P{song:03d}.npz"
    with np.load(path, allow_pickle=False) as data:
        features = data["X"].astype(np.float32)
        target = data["root"] * 24 + data["quality"] * 4 + data["seventh"]
        starts, ends = data["t_start"], data["t_end"]
    durations = np.maximum(ends - starts, 1e-6)
    posterior = recognizer.predict_posteriors(features, top_roots=12)
    cnn_only_prediction = posterior.argmax(axis=1)
    comparison_dir = OUTPUT_PATH.parent / "rwc_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    with (comparison_dir / f"RWC_P{song:03d}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["start_seconds", "end_seconds", "ground_truth_state", "predicted_state", "root_correct", "root_quality_correct", "root_quality_seventh_correct"])
        for start, end, truth, pred in zip(starts, ends, target, cnn_only_prediction):
            tr, tq, ts = state_parts(np.asarray([truth]))
            pr, pq, ps = state_parts(np.asarray([pred]))
            writer.writerow([f"{start:.3f}", f"{end:.3f}", int(truth), int(pred), bool(tr[0] == pr[0]), bool(tr[0] == pr[0] and tq[0] == pq[0]), bool(truth == pred)])
    cnn_only_scores = scores(target, cnn_only_prediction, durations)
    return {"song": song, "frames": len(target), "cnn_only": cnn_only_scores}


def aggregate(results: list[dict[str, object]], key: str) -> dict[str, float]:
    metric_names = [name for name in results[0][key] if name != "duration_seconds"]
    return {
        name: float(np.mean([result[key][name] for result in results]))
        for name in metric_names
    }


def _states_at_times(
    times: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    labels: np.ndarray,
    name_to_state: dict[str, int],
) -> np.ndarray:
    """Return vocabulary states at query times; unsupported spans become -1."""

    output = np.full(len(times), -1, dtype=np.int64)
    indices = np.searchsorted(starts, times, side="right") - 1
    valid = indices >= 0
    clipped = np.clip(indices, 0, max(0, len(starts) - 1))
    valid &= times < ends[clipped]
    for row in np.flatnonzero(valid):
        output[row] = name_to_state.get(str(labels[clipped[row]]), -1)
    return output


def evaluate_playable_song(
    song: int,
    audio_path: Path | None,
    annotation_path: Path,
    pipeline: HierarchicalRecognitionPipeline,
    hsmm_order: int,
    posterior_dir: Path,
) -> dict[str, object]:
    """Compare raw CNN and CNN+HSMM on the same continuous audio frames."""

    posterior_path = posterior_dir / f"RWC_P{song:03d}.npz"
    if posterior_path.is_file():
        with np.load(posterior_path, allow_pickle=False) as saved:
            centers = saved["centers"]
            posterior = saved["probabilities"]
            duration = float(saved["duration"])
        posterior_source = "saved continuous posterior"
        frontend_cache_hit = True
    else:
        if audio_path is None:
            raise FileNotFoundError(
                f"RWC_P{song:03d} needs audio or {posterior_path}"
            )
        centers, posterior, duration, frontend_cache_hit = pipeline.posteriors(
            audio_path
        )
        posterior_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            posterior_path,
            centers=centers,
            probabilities=posterior,
            duration=np.asarray(duration),
            step=np.asarray(pipeline.decoder_config.step),
            top_roots=np.asarray(pipeline.top_roots),
        )
        posterior_source = "generated from audio"
    cnn_only_prediction = posterior.argmax(axis=1)
    segments = pipeline.language.decode(
        posterior, order=hsmm_order, config=pipeline.decoder_config
    )
    cnn_hsmm_prediction = np.empty(len(centers), dtype=np.int64)
    for segment in segments:
        cnn_hsmm_prediction[segment.start_step : segment.end_step] = segment.state

    annotations = load_annotations(annotation_path, hierarchical=True)
    starts, ends, labels = annotations.intervals(duration)
    vocabulary = pipeline.recognizer.vocabulary
    name_to_state = {
        vocabulary.chord_name(state): state for state in range(vocabulary.state_count)
    }
    target = _states_at_times(centers, starts, ends, labels, name_to_state)
    valid = target >= 0
    if not np.any(valid):
        raise ValueError(f"No scorable RQS annotations found in {annotation_path}")
    step = pipeline.decoder_config.step
    frame_durations = np.minimum(step, np.maximum(0.0, duration - centers + step / 2))
    return {
        "song": f"RWC_P{song:03d}",
        "frames": int(np.sum(valid)),
        "posterior_source": posterior_source,
        "frontend_cache_hit": frontend_cache_hit,
        "cnn_only": scores(
            target[valid], cnn_only_prediction[valid], frame_durations[valid]
        ),
        "cnn_hsmm": scores(
            target[valid], cnn_hsmm_prediction[valid], frame_durations[valid]
        ),
    }


def continuous_comparisons(
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    """Evaluate continuous posteriors, generating them from audio when needed."""

    artifacts = HierarchicalArtifacts(
        cnn_checkpoint=args.checkpoint,
        language_model=args.language_model,
        split_manifest=args.split_manifest,
    )
    pipeline = HierarchicalRecognitionPipeline(
        artifacts,
        device=args.device,
        top_roots=args.top_roots,
        hsmm_order=args.hsmm_order,
        cache_dir=args.cache_dir,
    )
    results = []
    for song in TEST_SONGS:
        stem = f"RWC_P{song:03d}"
        annotation_path = args.annotation_dir / f"{stem}.csv"
        audio_path = next(
            (
                candidate
                for candidate in sorted(args.audio_dir.glob(f"{stem}.*"))
                if candidate.is_file() and candidate.stat().st_size
            ),
            None,
        )
        posterior_path = args.posterior_dir / f"{stem}.npz"
        if annotation_path.is_file() and (audio_path is not None or posterior_path.is_file()):
            results.append(
                evaluate_playable_song(
                    song,
                    audio_path,
                    annotation_path,
                    pipeline,
                    args.hsmm_order,
                    args.posterior_dir,
                )
            )
    return results


def main() -> int:
    args = build_parser().parse_args()
    recognizer = HierarchicalCNNRecognizer(args.checkpoint, args.device)
    results = [evaluate_song(song, args.feature_dir, recognizer) for song in TEST_SONGS]
    continuous = continuous_comparisons(args)
    cnn_only_macro = aggregate(results, "cnn_only")
    report = {
        "songs": TEST_SONGS,
        "cnn_only_macro_average": cnn_only_macro,
        "per_song": results,
        "continuous_pipeline_comparison": continuous,
        "continuous_pipeline_macro_average": {
            "cnn_only": aggregate(continuous, "cnn_only") if continuous else {},
            "cnn_hsmm": aggregate(continuous, "cnn_hsmm") if continuous else {},
        },
        "notes": {
            "cnn_only_test_set": "15-song macro average on saved annotation-aligned CQT windows.",
            "continuous_pipeline_comparison": "CNN-only and CNN+HSMM evaluated on identical continuous 0.2-second frames. Saved posteriors were generated by the full audio frontend and CNN.",
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("CNN-only · 15-song test split")
    print(json.dumps(cnn_only_macro, indent=2))
    if continuous:
        print(f"CNN-only vs CNN+HSMM · {len(continuous)} continuous test songs")
        print(json.dumps(report["continuous_pipeline_macro_average"], indent=2))
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote readable per-frame comparisons to {OUTPUT_PATH.parent / 'rwc_comparison'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
