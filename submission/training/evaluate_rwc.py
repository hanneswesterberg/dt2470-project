"""Evaluate CNN emissions and HSMM predictions on the RWC feature files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

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
    cnn_prediction = posterior.argmax(axis=1)
    comparison_dir = OUTPUT_PATH.parent / "rwc_comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    with (comparison_dir / f"RWC_P{song:03d}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["start_seconds", "end_seconds", "ground_truth_state", "predicted_state", "root_correct", "root_quality_correct", "root_quality_seventh_correct"])
        for start, end, truth, pred in zip(starts, ends, target, cnn_prediction):
            tr, tq, ts = state_parts(np.asarray([truth]))
            pr, pq, ps = state_parts(np.asarray([pred]))
            writer.writerow([f"{start:.3f}", f"{end:.3f}", int(truth), int(pred), bool(tr[0] == pr[0]), bool(tr[0] == pr[0] and tq[0] == pq[0]), bool(truth == pred)])
    cnn_scores = scores(target, cnn_prediction, durations)
    return {"song": song, "frames": len(target), "cnn": cnn_scores}


def aggregate(results: list[dict[str, object]], key: str) -> dict[str, float]:
    metric_names = [name for name in results[0][key] if name != "duration_seconds"]
    return {
        name: float(np.mean([result[key][name] for result in results]))
        for name in metric_names
    }


def main() -> int:
    args = build_parser().parse_args()
    recognizer = HierarchicalCNNRecognizer(args.checkpoint, args.device)
    results = [evaluate_song(song, args.feature_dir, recognizer) for song in TEST_SONGS]
    report = {
        "songs": TEST_SONGS,
        "cnn_macro_average": aggregate(results, "cnn"),
        "per_song": results,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["cnn_macro_average"], indent=2))
    print(f"Wrote {OUTPUT_PATH}")
    print(f"Wrote readable per-frame comparisons to {OUTPUT_PATH.parent / 'rwc_comparison'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
