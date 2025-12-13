"""
Carga de artefactos de inferencia y pipeline de scoring.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

from scripts.app.schemas import RAW_FEATURES


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
    model_path: str,
    preprocessor_path: str = "data/processed/preprocessor.joblib",
    selection_metadata_path: Optional[str] = "data/selected/metadata_selection.json",
) -> InferenceArtifacts:
    model_path_p = Path(model_path)
    preproc_path_p = Path(preprocessor_path)
    if not model_path_p.exists():
        raise FileNotFoundError(f"No se encontro el modelo en {model_path_p}")
    if not preproc_path_p.exists():
        raise FileNotFoundError(f"No se encontro el preprocesador en {preproc_path_p}")

    model = joblib.load(model_path_p)
    preprocessor = joblib.load(preproc_path_p)

    selected_features: Optional[List[str]] = None
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


def validate_raw_columns(df: pd.DataFrame) -> List[str]:
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    return missing


def preprocess_records(df_raw: pd.DataFrame, artifacts: InferenceArtifacts) -> pd.DataFrame:
    missing = validate_raw_columns(df_raw)
    if missing:
        raise ValueError(f"Faltan columnas requeridas: {missing}")

    df_raw = df_raw.copy()
    if "date" in df_raw.columns:
        df_raw["date"] = pd.to_datetime(df_raw["date"], errors="coerce")

    transformed = artifacts.preprocessor.transform(df_raw)
    try:
        cols = artifacts.preprocessor.get_feature_names_out()
    except Exception:
        cols = [f"f{i}" for i in range(transformed.shape[1])]
    X_df = pd.DataFrame(transformed, columns=cols, index=df_raw.index)

    if artifacts.selected_features:
        missing_sel = [c for c in artifacts.selected_features if c not in X_df.columns]
        if missing_sel:
            raise ValueError(f"El preprocesador no genero columnas esperadas por la seleccion: {missing_sel}")
        X_df = X_df[artifacts.selected_features]
    return X_df


def predict(df_raw: pd.DataFrame, artifacts: InferenceArtifacts) -> np.ndarray:
    X = preprocess_records(df_raw, artifacts)
    preds = artifacts.model.predict(X)
    return np.asarray(preds)
