"""
Paquete base del proyecto TDSP.

Exponemos accesos de conveniencia al pipeline de entrenamiento y tracking para
usar desde notebooks o servicios sin depender de los scripts CLI.
"""

from scripts.training.data_utils import build_fast_subset, load_data  # noqa: F401
from scripts.training.model_registry import ModelSpec, available_specs  # noqa: F401
from scripts.training.training_pipeline import run_model_search  # noqa: F401
from scripts.mlflow_utils.tracking import run_model_search_with_mlflow, setup_mlflow  # noqa: F401

__all__ = [
    "ModelSpec",
    "available_specs",
    "load_data",
    "build_fast_subset",
    "run_model_search",
    "run_model_search_with_mlflow",
    "setup_mlflow",
]
