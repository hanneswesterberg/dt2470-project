"""Second-order HSMM with beam-pruned segmental Viterbi decoding."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.stats import lognorm

from src.hierarchical.vocabulary import ChordVocabulary


@dataclass(frozen=True)
class DecoderConfig:
    step: float = 0.2
    acoustic_weight: float = 0.28
    language_weight: float = 0.35
    duration_weight: float = 0.50
    min_duration: float = 0.4
    max_duration: float = 6.0
    beam_size: int = 30
    segment_top_k: int = 6


@dataclass(frozen=True)
class Segment:
    start_step: int
    end_step: int
    state: int


class HarmonicLanguageModel:
    """Saved 0/1/2-order event transitions plus a log-normal duration prior."""

    def __init__(self, path: Path | str, vocabulary: ChordVocabulary) -> None:
        self.path = Path(path)
        with self.path.open("rb") as stream:
            payload = pickle.load(stream)
        if (
            tuple(payload["quality"]) != vocabulary.quality
            or tuple(payload["seventh"]) != vocabulary.seventh
        ):
            raise ValueError("CNN and harmonic-model vocabularies differ.")
        self.vocabulary = vocabulary
        self.counts_global = payload["counts_global"]
        self.counts_first = payload["counts_first_order"]
        self.counts_second = payload["counts_second_order"]
        self.global_prior = payload["global_prior"]
        self.tau1 = float(payload["tau1"])
        self.tau2 = float(payload["tau2"])
        durations = np.asarray(payload["durations_seconds"], dtype=float)
        durations = durations[np.isfinite(durations) & (durations > 0)]
        if not len(durations):
            raise ValueError("Harmonic model has no valid duration samples.")
        self.duration_shape, _, self.duration_scale = lognorm.fit(durations, floc=0)

    @lru_cache(maxsize=200_000)
    def transition_log_probability(
        self, older: int, previous: int, nxt: int, order: int = 2
    ) -> float:
        previous_chord = self.vocabulary.decode(previous)
        candidate = self.vocabulary.decode(nxt)
        outcome = (
            (candidate[0] - previous_chord[0]) % 12,
            candidate[1],
            candidate[2],
        )
        probability = float(self.global_prior[outcome])
        first = self.counts_first.get((previous_chord[1], previous_chord[2]))
        if first:
            count = sum(first.values())
            weight = count / (count + self.tau1)
            probability = weight * first[outcome] / count + (1 - weight) * probability
        if order >= 2 and older >= 0:
            older_chord = self.vocabulary.decode(older)
            context = (
                (previous_chord[0] - older_chord[0]) % 12,
                older_chord[1],
                older_chord[2],
                previous_chord[1],
                previous_chord[2],
            )
            second = self.counts_second.get(context)
            if second:
                count = sum(second.values())
                weight = count / (count + self.tau2)
                probability = weight * second[outcome] / count + (1 - weight) * probability
        return float(np.log(max(probability, 1e-12)))

    @lru_cache(maxsize=1024)
    def duration_log_probability(self, steps: int, step_seconds: float) -> float:
        duration = steps * step_seconds
        lower = max(0, duration - step_seconds / 2)
        upper = duration + step_seconds / 2
        probability = lognorm.cdf(
            upper, self.duration_shape, scale=self.duration_scale
        ) - lognorm.cdf(lower, self.duration_shape, scale=self.duration_scale)
        return float(np.log(max(probability, 1e-12)))

    def decode(
        self,
        probabilities: NDArray[np.floating],
        *,
        order: int = 2,
        config: DecoderConfig | None = None,
    ) -> list[Segment]:
        """Decode a contiguous event path with the notebook's beam approximation."""

        if order not in (1, 2):
            raise ValueError("HSMM order must be 1 or 2.")
        cfg = config or DecoderConfig()
        values = np.asarray(probabilities, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.vocabulary.state_count:
            raise ValueError(
                f"Expected posterior shape (N, {self.vocabulary.state_count}), got {values.shape}."
            )
        length = len(values)
        if not length:
            return []
        log_probability = np.log(np.maximum(values, 1e-8)) + np.log(
            self.vocabulary.state_count
        )
        prefix = np.vstack(
            [
                np.zeros((1, self.vocabulary.state_count)),
                np.cumsum(log_probability * cfg.acoustic_weight, axis=0),
            ]
        )
        minimum_steps = max(1, round(cfg.min_duration / cfg.step))
        maximum_steps = max(minimum_steps, round(cfg.max_duration / cfg.step))
        # score, prior endpoint, prior history key, state
        Hypothesis = tuple[
            float, int | None, tuple[int, ...] | None, int | None
        ]
        dynamic: list[dict[tuple[int, ...], Hypothesis]] = [
            {} for _ in range(length + 1)
        ]
        dynamic[0][()] = (0.0, None, None, None)
        for end in range(1, length + 1):
            possibilities: dict[
                tuple[int, ...], tuple[float, int, tuple[int, ...], int]
            ] = {}
            durations = list(range(minimum_steps, min(maximum_steps, end) + 1))
            if end <= maximum_steps and end not in durations:
                durations.append(end)  # allow a short opening segment
            for steps in durations:
                start = end - steps
                if start < 0 or not dynamic[start]:
                    continue
                segment_scores = prefix[end] - prefix[start]
                candidates = np.argsort(segment_scores)[-cfg.segment_top_k :][::-1]
                duration_score = cfg.duration_weight * self.duration_log_probability(
                    steps, cfg.step
                )
                for old_key, (old_score, _, _, _) in dynamic[start].items():
                    previous = old_key[-1] if old_key else None
                    older = old_key[-2] if len(old_key) >= 2 else -1
                    for candidate_value in candidates:
                        candidate = int(candidate_value)
                        if previous is not None and candidate == previous:
                            continue
                        score = (
                            old_score
                            + float(segment_scores[candidate])
                            + duration_score
                        )
                        if previous is not None:
                            score += cfg.language_weight * self.transition_log_probability(
                                older, previous, candidate, order
                            )
                        key = (
                            (candidate,)
                            if order == 1
                            else (
                                (previous, candidate)
                                if previous is not None
                                else (candidate,)
                            )
                        )
                        existing = possibilities.get(key)
                        if existing is None or score > existing[0]:
                            possibilities[key] = (score, start, old_key, candidate)
            if possibilities:
                dynamic[end] = dict(
                    sorted(
                        possibilities.items(),
                        key=lambda item: item[1][0],
                        reverse=True,
                    )[: cfg.beam_size]
                )
        if not dynamic[length]:
            raise RuntimeError(
                "No complete HSMM path; check duration/candidate settings."
            )
        key = max(
            dynamic[length], key=lambda candidate: dynamic[length][candidate][0]
        )
        segments: list[Segment] = []
        end = length
        while end > 0:
            _, start, previous_key, state = dynamic[end][key]
            assert (
                start is not None and previous_key is not None and state is not None
            )
            segments.append(Segment(start, end, state))
            end, key = start, previous_key
        segments.reverse()
        return segments
