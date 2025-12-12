"""
Accesos de conveniencia al pipeline de entrenamiento.
"""

from scripts.training.model_registry import ModelSpec, available_specs  # noqa: F401
from scripts.training.training_pipeline import run_model_search  # noqa: F401
from scripts.mlflow_utils.tracking import run_model_search_with_mlflow  # noqa: F401

__all__ = ["ModelSpec", "available_specs", "run_model_search", "run_model_search_with_mlflow"]
