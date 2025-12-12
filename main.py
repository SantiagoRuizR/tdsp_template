from fastapi import FastAPI
from pydantic import BaseModel
from prophet.serialize import model_from_json
from prophet import Prophet
import json
import pandas as pd

with open("models_fast/prophet_model.json", "r") as f:
    m: Prophet = model_from_json(json.load(f))

app = FastAPI(title="API de pronóstico con Prophet")


class ForecastRequest(BaseModel):
    periods: int = 30
    freq: str = "D"


@app.get("/")
def root():
    return {"message": "API Prophet funcionando"}


@app.post("/forecast")
def forecast(req: ForecastRequest):
    future = m.make_future_dataframe(periods=req.periods, freq=req.freq)

    forecast_df = m.predict(future)

    future_only = forecast_df.tail(req.periods)

    result = future_only[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    result["ds"] = result["ds"].astype(str)

    return {
        "periods": req.periods,
        "freq": req.freq,
        "forecast": result.to_dict(orient="records"),
    }
