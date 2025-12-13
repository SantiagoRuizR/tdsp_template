"""
API FastAPI para servir predicciones con el preprocesador y modelo entrenado.
"""

from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException

from scripts.app.artifacts import InferenceArtifacts, load_artifacts, predict
from scripts.app.schemas import RAW_FEATURES, PredictRequest, PredictResponse


@lru_cache()
def get_artifacts() -> InferenceArtifacts:
    # Rutas por defecto: modelo en models_fast y preprocesador/seleccion en data/
    return load_artifacts(
        model_path="models_fast/gradient_boosting.joblib",
        preprocessor_path="data/processed/preprocessor.joblib",
        selection_metadata_path="data/selected/metadata_selection.json",)


app = FastAPI(
    title="Weather TDSP API",
    description=(
        "Servicio de prediccion de Tlog con preprocesamiento y seleccion de features "
        "alineados al pipeline TDSP (Diplomado ML & Data Science UNAL)."
    ),
    version="0.1.0",)


@app.get("/")
def root():
    artifacts = get_artifacts()
    return {
        "message": "API de prediccion operativa",
        "model": artifacts.model_name,
        "required_features": RAW_FEATURES,
        "has_feature_selection": bool(artifacts.selected_features),}


@app.get("/health")
def health():
    artifacts = get_artifacts()
    return {
        "status": "ok",
        "model": artifacts.model_name,
        "n_features": len(artifacts.selected_features or artifacts.feature_names),}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: PredictRequest):
    if not payload.records:
        raise HTTPException(status_code=400, detail="No se enviaron registros.")
    df_raw = payload.model_dump()["records"]
    df = pd.DataFrame([rec for rec in df_raw])
    try:
        preds = predict(df, get_artifacts())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error en prediccion: {exc}") from exc

    artifacts = get_artifacts()
    return PredictResponse(
        model_name=artifacts.model_name,
        n_records=len(preds),
        predictions=[float(p) for p in preds],
        selected_features=artifacts.selected_features,)


# Ejecutar con: uvicorn main_api:app --reload --port 8000
