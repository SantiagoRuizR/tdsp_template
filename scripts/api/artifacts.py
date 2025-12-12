"""
Carga y ejecución del pipeline de inferencia: preprocesador + selección + modelo.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

from scripts.api.schemas import RAW_FEATURES


@dataclass
class InferenceArtifacts:
    model: object
    preprocessor: object
    selected_features: Optional[List[str]]
    feature_names: List[str]
    model_name: str


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def load_artifacts(
    model_path: str = "models_fast/gradient_boosting.joblib",
    preprocessor_path: str = "data/processed/preprocessor.joblib",
    selection_metadata_path: Optional[str] = "data/selected/metadata_selection.json",
) -> InferenceArtifacts:
    """Carga modelo, preprocesador y metadatos de selección de features (si existen)."""
    model_path_p = Path(model_path)
    preproc_path_p = Path(preprocessor_path)
    if not model_path_p.exists():
        raise FileNotFoundError(f"Modelo no encontrado en {model_path_p}")
    if not preproc_path_p.exists():
        raise FileNotFoundError(f"Preprocesador no encontrado en {preproc_path_p}")

    model = joblib.load(model_path_p)
    preprocessor = joblib.load(preproc_path_p)

    selected_features = None
    feature_names: List[str] = []
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        pass

    if selection_metadata_path:
        sel_path = Path(selection_metadata_path)
        if sel_path.exists():
            metadata = _load_json(sel_path)
            selected_features = metadata.get("selected_features") or metadata.get("selected_feats")

    return InferenceArtifacts(
        model=model,
        preprocessor=preprocessor,
        selected_features=selected_features,
        feature_names=feature_names,
        model_name=model_path_p.stem,
    )


def _validate_columns(df: pd.DataFrame):
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")


def preprocess_records(df_raw: pd.DataFrame, artifacts: InferenceArtifacts) -> pd.DataFrame:
    """Aplica el preprocesador y (opcionalmente) la selección de features."""
    _validate_columns(df_raw)
    # Ensure datetime parsing matches training behavior
    if "date" in df_raw.columns:
        df_raw = df_raw.copy()
        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")

    transformed = artifacts.preprocessor.transform(df_raw)
    try:
        cols = artifacts.preprocessor.get_feature_names_out()
    except Exception:
        cols = [f"f{i}" for i in range(transformed.shape[1])]
    X_df = pd.DataFrame(transformed, columns=cols)

    if artifacts.selected_features:
        missing_sel = [c for c in artifacts.selected_features if c not in X_df.columns]
        if missing_sel:
            raise ValueError(f"El preprocesador no generó columnas esperadas por la selección: {missing_sel}")
        X_df = X_df[artifacts.selected_features]
    return X_df


def predict(df_raw: pd.DataFrame, artifacts: InferenceArtifacts) -> np.ndarray:
    """Pipeline completo de scoring."""
    X = preprocess_records(df_raw, artifacts)
    preds = artifacts.model.predict(X)
    return np.asarray(preds)
