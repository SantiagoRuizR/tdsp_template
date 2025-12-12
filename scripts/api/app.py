"""
App factory de FastAPI para servir el modelo entrenado con su preprocesamiento.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI

# Compatibilidad con entornos sin pydantic-settings (pydantic v2)
try:  # pragma: no cover
    from pydantic_settings import BaseSettings as _BaseSettings
except ImportError:  # pragma: no cover
    try:
        from pydantic import BaseSettings as _BaseSettings  # pydantic v1
    except Exception:
        _BaseSettings = None  # Forzamos uso de dataclass manual

from scripts.api.artifacts import InferenceArtifacts, load_artifacts
from scripts.api.routes import get_routes
from scripts.api.schemas import RAW_FEATURES


ENV_PREFIX = "API_"


if _BaseSettings:
    class Settings(_BaseSettings):
        model_path: str = "models_fast/gradient_boosting.joblib"
        preprocessor_path: str = "data/processed/preprocessor.joblib"
        selection_metadata_path: Optional[str] = "data/selected/metadata_selection.json"
        title: str = "Modulo 3 – Dipl. ML & Data Science UNAL – Weather API"
        description: str = (
            "Servicio de predicción de temperatura (Tlog) con preprocesamiento y selección de "
            "características alineados al pipeline TDSP del Diplomado en Machine Learning y "
            "Data Science Avanzado (UNAL), Módulo 3."
        )
        version: str = "0.1.0"

        class Config:
            env_prefix = ENV_PREFIX
            case_sensitive = False
else:
    @dataclass
    class Settings:  # pragma: no cover - fallback sin pydantic
        model_path: str = "models_fast/gradient_boosting.joblib"
        preprocessor_path: str = "data/processed/preprocessor.joblib"
        selection_metadata_path: Optional[str] = "data/selected/metadata_selection.json"
        title: str = "Modulo 3 – Dipl. ML & Data Science UNAL – Weather API"
        description: str = (
            "Servicio de predicción de temperatura (Tlog) con preprocesamiento y selección de "
            "características alineados al pipeline TDSP del Diplomado en Machine Learning y "
            "Data Science Avanzado (UNAL), Módulo 3."
        )
        version: str = "0.1.0"
        _env_prefix: str = field(default=ENV_PREFIX, repr=False)

        def __post_init__(self):
            for attr in ["model_path", "preprocessor_path", "selection_metadata_path", "title", "description", "version"]:
                env_key = f"{self._env_prefix}{attr}".upper()
                val = os.getenv(env_key)
                if val is not None:
                    setattr(self, attr, val)


@lru_cache()
def get_settings() -> Settings:
    return Settings()


def create_app(settings: Settings = None) -> FastAPI:
    settings = settings or get_settings()
    artifacts: InferenceArtifacts = load_artifacts(
        model_path=settings.model_path,
        preprocessor_path=settings.preprocessor_path,
        selection_metadata_path=settings.selection_metadata_path,
    )

    app = FastAPI(
        title=settings.title,
        description=settings.description,
        version=settings.version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.include_router(get_routes(artifacts))

    @app.get("/", tags=["status"])
    def root():
        return {
            "message": "API de predicción operativa",
            "model": artifacts.model_name,
            "required_features": RAW_FEATURES,
        }

    return app
