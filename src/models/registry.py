"""
Lightweight registry for available image editing models.
"""

from __future__ import annotations

import importlib
from typing import Callable, Dict, List, Type


MODEL_REGISTRY: Dict[str, Type] = {}
_DEFAULT_MODEL_MODULES = [
    "src.models.ddim",
    "src.models.null_text",
    # Additional models can be listed here as they are added
]


def register_model(name: str) -> Callable:
    """
    Decorator to register a model class under a string name.
    """

    def decorator(cls):
        MODEL_REGISTRY[name.lower()] = cls
        return cls

    return decorator


def _ensure_defaults_loaded():
    for module_path in _DEFAULT_MODEL_MODULES:
        try:
            importlib.import_module(module_path)
        except ImportError:
            continue


def available_models() -> List[str]:
    _ensure_defaults_loaded()
    return sorted(MODEL_REGISTRY.keys())


def get_model_builder(name: str):
    _ensure_defaults_loaded()
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available models: {available_models()}")
    return MODEL_REGISTRY[key]

