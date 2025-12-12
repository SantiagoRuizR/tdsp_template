"""
Pydantic schemas para la API de scoring y evaluación.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


RAW_FEATURES = [
    "date",
    "p",
    "T",
    "Tpot",
    "Tdew",
    "rh",
    "VPmax",
    "VPact",
    "VPdef",
    "sh",
    "H2OC",
    "rho",
    "wv",
    "max. wv",
    "wd",
    "rain",
    "raining",
    "SWDR",
    "PAR",
    "max. PAR",
]


class WeatherRecord(BaseModel):
    """Registro en crudo que replica las columnas del dataset original."""

    date: Optional[datetime] = Field(None, description="Fecha/hora de la observación (se ignora en el modelo).")
    p: float
    T: float
    Tpot: float
    Tdew: float
    rh: float
    VPmax: float
    VPact: float
    VPdef: float
    sh: float
    H2OC: float
    rho: float
    wv: float
    max_wv: float = Field(..., alias="max. wv")
    wd: float
    rain: float
    raining: float
    SWDR: float
    PAR: float
    max_PAR: float = Field(..., alias="max. PAR")

    class Config:
        populate_by_name = True
        json_encoders = {datetime: lambda v: v.isoformat() if v else None}

    def to_raw(self) -> dict:
        """Devuelve el diccionario con los alias originales (incluyendo nombres con espacios/puntos)."""
        return self.model_dump(by_alias=True)


class PredictRequest(BaseModel):
    records: List[WeatherRecord]


class PredictResponse(BaseModel):
    model_name: str
    n_records: int
    predictions: List[float]
    selected_features: Optional[List[str]] = None
    tracking: Optional[dict] = None


class EvaluateRequest(BaseModel):
    records: List[WeatherRecord]
    y_true: List[float]


class DriftRequest(BaseModel):
    reference: List[WeatherRecord]
    current: List[WeatherRecord]
    alpha: float = 0.05


class HealthResponse(BaseModel):
    status: str
    model_name: str
    n_features: int
    has_feature_selection: bool
    required_features: List[str]
