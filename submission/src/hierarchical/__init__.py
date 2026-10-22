"""Hierarchical CQT72 chord recognition and second-order HSMM decoding.

Exports are lazy so data/decoder utilities remain importable on machines where
the optional runtime dependency PyTorch has not been installed yet.
"""

from importlib import import_module

__all__ = [
    "CQT72Config",
    "ChordVocabulary",
    "ContinuousCQT72Extractor",
    "DecoderConfig",
    "HarmonicLanguageModel",
    "HierarchicalArtifacts",
    "HierarchicalCNN72",
    "HierarchicalCNNRecognizer",
    "HierarchicalRecognitionPipeline",
    "Segment",
]

_MODULE_BY_NAME = {
    "HierarchicalArtifacts": "artifacts",
    "CQT72Config": "cqt_frontend",
    "ContinuousCQT72Extractor": "cqt_frontend",
    "DecoderConfig": "hsmm_decoder",
    "HarmonicLanguageModel": "hsmm_decoder",
    "Segment": "hsmm_decoder",
    "HierarchicalCNN72": "hierarchical_cnn",
    "HierarchicalCNNRecognizer": "hierarchical_cnn",
    "HierarchicalRecognitionPipeline": "inference_pipeline",
    "ChordVocabulary": "vocabulary",
}


def __getattr__(name: str):
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(name)
    return getattr(import_module(f"src.hierarchical.{module_name}"), name)
