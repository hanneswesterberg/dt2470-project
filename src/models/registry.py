"""Resolve and discover named model checkpoints in the local model registry.

Both ``train_gaussian_baseline.py`` and ``main.py`` use these helpers so a
baseline model name always maps to the same file. Plain names live under
``models/``; explicit relative or absolute paths remain supported.

Example::

    path = resolve_model_path("gaussian_hmm_80", Path("models"))
    # models/gaussian_hmm_80.pkl
"""

from pathlib import Path


def resolve_model_path(model: str, models_dir: Path) -> Path:
    """Resolve a friendly model name or explicit path to a ``.pkl`` file."""

    requested = Path(model)
    if requested.is_absolute() or requested.parent != Path("."):
        return requested if requested.suffix else requested.with_suffix(".pkl")
    filename = (
        requested if requested.suffix == ".pkl" else Path(f"{requested.name}.pkl")
    )
    return models_dir / filename


def list_models(models_dir: Path) -> list[Path]:
    """Return the named ``.pkl`` checkpoints available for inference."""

    return sorted(models_dir.glob("*.pkl")) if models_dir.is_dir() else []
