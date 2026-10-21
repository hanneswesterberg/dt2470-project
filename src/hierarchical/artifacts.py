"""Resolve the trained CNN, harmonic model, and frozen split manifest."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HierarchicalArtifacts:
    cnn_checkpoint: Path
    language_model: Path
    split_manifest: Path

    def validate(self) -> "HierarchicalArtifacts":
        missing = [path for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing hierarchical artifact(s): " + ", ".join(map(str, missing))
            )
        return self

    @property
    def paths(self) -> tuple[Path, Path, Path]:
        return self.cnn_checkpoint, self.language_model, self.split_manifest

    @classmethod
    def resolve(
        cls,
        *,
        project_dir: Path | str | None = None,
        cnn_checkpoint: Path | str | None = None,
        language_model: Path | str | None = None,
        split_manifest: Path | str | None = None,
        local_dir: Path | str = "models/hierarchical",
    ) -> "HierarchicalArtifacts":
        """Resolve explicit paths, local artifacts, or a mounted Drive project."""

        explicit = (cnn_checkpoint, language_model, split_manifest)
        if any(item is not None for item in explicit):
            if not all(item is not None for item in explicit):
                raise ValueError(
                    "--cnn-checkpoint, --language-model, and --split-manifest "
                    "must be supplied together."
                )
            return cls(*(Path(item).expanduser() for item in explicit)).validate()

        local_candidates = (Path(local_dir), Path("artifacts"))
        for local in local_candidates:
            local_set = cls(
                local / "best_model.pt",
                local / "harmonic_model.pkl",
                local / "split_manifest.json",
            )
            if all(path.is_file() for path in local_set.paths):
                return local_set

        configured = os.environ.get("HIERARCHICAL_PROJECT_DIR") or _dotenv_value(
            Path(".env"), "HIERARCHICAL_PROJECT_DIR"
        )
        root = Path(project_dir or configured or _discover_drive_project()).expanduser()
        return cls(
            root / "seed42_cqt72_hierarchical_aug_v1" / "best_model.pt",
            root / "seed42_second_order_hsmm_v1" / "harmonic_model.pkl",
            root / "seed42_songs100" / "split_manifest.json",
        ).validate()


def _discover_drive_project() -> Path:
    cloud = Path.home() / "Library" / "CloudStorage"
    candidates = sorted(
        cloud.glob("GoogleDrive-*/My Drive/RWC_multitask_CNN_v1")
    )
    existing = [path for path in candidates if path.is_dir()]
    if len(existing) == 1:
        return existing[0]
    raise FileNotFoundError(
        "Could not uniquely locate RWC_multitask_CNN_v1. Set "
        "HIERARCHICAL_PROJECT_DIR, pass --hierarchical-project-dir, or place "
        "best_model.pt, harmonic_model.pkl, and split_manifest.json in "
        "models/hierarchical/."
    )


def _dotenv_value(path: Path, key: str) -> str | None:
    if not path.is_file():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            return value.strip().strip("'\"") or None
    return None
