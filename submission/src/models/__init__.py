"""Complete recognition models, component artifacts, and model discovery.

Use :class:`ChordRecognitionModel` to combine fitted emission and transition
components, decode feature sequences, or load/save named model checkpoints.
The registry helpers map terminal model names to files, while artifact helpers
store emission-only and transition-only experiments separately.
"""

from src.models.artifacts import component_path, save_component
from src.models.chord_model import ChordRecognitionModel
from src.models.registry import list_models, resolve_model_path

__all__ = [
    "ChordRecognitionModel",
    "component_path",
    "list_models",
    "resolve_model_path",
    "save_component",
]
