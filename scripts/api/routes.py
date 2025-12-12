"""
Rutas FastAPI para scoring, evaluación y chequeo de drift.
"""

from fastapi import APIRouter, HTTPException
import pandas as pd
from typing import Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scripts.api.artifacts import InferenceArtifacts, predict
from scripts.api.schemas import (
    DriftRequest,
    EvaluateRequest,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    RAW_FEATURES,
)
from scripts.evaluation.main import detect_drift

router = APIRouter()


def _records_to_df(records) -> pd.DataFrame:
    data = [rec.to_raw() for rec in records]
    return pd.DataFrame(data)


def build_health_response(artifacts: InferenceArtifacts) -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_name=artifacts.model_name,
        n_features=len(artifacts.selected_features or artifacts.feature_names),
        has_feature_selection=bool(artifacts.selected_features),
        required_features=list(RAW_FEATURES),
    )


def get_routes(artifacts: InferenceArtifacts) -> APIRouter:
    @router.get("/health", response_model=HealthResponse, tags=["status"])
    def health():
        return build_health_response(artifacts)

    @router.post("/predict", response_model=PredictResponse, tags=["prediction"])
    def predict_endpoint(payload: PredictRequest):
        if not payload.records:
            raise HTTPException(status_code=400, detail="No se enviaron registros.")
        df_raw = _records_to_df(payload.records)
        try:
            preds = predict(df_raw, artifacts)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Error en preprocesamiento/predicción: {exc}") from exc

        return PredictResponse(
            model_name=artifacts.model_name,
            n_records=len(preds),
            predictions=[float(p) for p in preds],
            selected_features=artifacts.selected_features,
        )

    @router.post("/evaluate", tags=["evaluation"])
    def evaluate_endpoint(payload: EvaluateRequest) -> Dict:
        if len(payload.records) != len(payload.y_true):
            raise HTTPException(status_code=400, detail="records y y_true deben tener la misma longitud.")
        df_raw = _records_to_df(payload.records)
        preds = predict(df_raw, artifacts)
        y_true = pd.Series(payload.y_true)
        metrics = {
            "r2": float(r2_score(y_true, preds)),
            "mae": float(mean_absolute_error(y_true, preds)),
            "mse": float(mean_squared_error(y_true, preds)),
        }
        return {"metrics": metrics, "n_records": len(preds)}

    @router.post("/drift", tags=["monitoring"])
    def drift_endpoint(payload: DriftRequest):
        ref_df = _records_to_df(payload.reference)
        cur_df = _records_to_df(payload.current)
        ref_df = ref_df.select_dtypes(include="number")
        cur_df = cur_df.select_dtypes(include="number")
        # Se calcula drift sobre las columnas numéricas en crudo
        try:
            drift_df = detect_drift(ref_df, cur_df, alpha=payload.alpha)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"No se pudo calcular drift: {exc}") from exc
        return {"alpha": payload.alpha, "n_features": len(drift_df), "drift": drift_df.to_dict(orient="records")}

    return router
