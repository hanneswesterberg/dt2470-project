"""Save standalone emission or transition components for experiments.

Complete inference models use :class:`src.models.ChordRecognitionModel` and are
stored directly under ``models/``. Component-only training uses this module to
store typed artifacts under ``models/emissions/`` or ``models/transitions/``.
These component files are not directly runnable by ``main.py``.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

from src.hmm import GaussianEmissionModel, RelativeTransitionModel

ComponentKind = Literal["emission", "transition"]
COMPONENT_FORMAT_VERSION = 1


def component_path(models_dir: Path, kind: ComponentKind, name: str) -> Path:
    """Return the conventional checkpoint path for a named component."""

    folder = "emissions" if kind == "emission" else "transitions"
    filename = name if name.endswith(".pkl") else f"{name}.pkl"
    return models_dir / folder / filename


def save_component(
    component: GaussianEmissionModel | RelativeTransitionModel,
    kind: ComponentKind,
    path: Path,
    metadata: dict[str, object] | None = None,
) -> None:
    """Atomically save a fitted component with type and training metadata."""

    expected_type = (
        GaussianEmissionModel if kind == "emission" else RelativeTransitionModel
    )
    if not isinstance(component, expected_type):
        raise TypeError(f"Expected {expected_type.__name__} for a {kind} checkpoint.")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": COMPONENT_FORMAT_VERSION,
        "kind": kind,
        "component": component,
        "metadata": metadata or {},
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream)
    temporary.replace(path)
