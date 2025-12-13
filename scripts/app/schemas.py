"""
Schemas y constantes para la API y la app de Streamlit.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

# Features crudos esperados (dataset original)
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
    date: Optional[datetime] = Field(None, description="Fecha/hora de la observacion.")
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
        return self.model_dump(by_alias=True)


class PredictRequest(BaseModel):
    records: List[WeatherRecord]


class PredictResponse(BaseModel):
    model_name: str
    n_records: int
    predictions: List[float]
    selected_features: Optional[List[str]] = None
