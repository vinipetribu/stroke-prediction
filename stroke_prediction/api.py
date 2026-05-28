from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
import mlflow
import pandas as pd
from pydantic import BaseModel, Field

from stroke_prediction.config import REPORTS_DIR
from stroke_prediction.features import build_feature_matrix

DEFAULT_MODEL_NAME = "logistic_regression"
CHAMPION_PATH = Path(os.environ.get("CHAMPION_PATH", REPORTS_DIR / "champion.json"))
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5002")
MLFLOW_MODEL_URI = os.environ.get("MLFLOW_MODEL_URI")


class PatientFeatures(BaseModel):
    gender: Literal["Male", "Female"]
    age: float = Field(..., ge=0, le=120)
    hypertension: Literal[0, 1]
    heart_disease: Literal[0, 1]
    ever_married: Literal["Yes", "No"]
    work_type: Literal["Private", "Self-employed", "Govt_job", "children", "Never_worked"]
    Residence_type: Literal["Urban", "Rural"]
    avg_glucose_level: float = Field(..., ge=0)
    bmi: float | None = Field(None, ge=0)
    smoking_status: Literal["formerly smoked", "never smoked", "smokes", "Unknown"]


class PredictionResponse(BaseModel):
    prediction: int
    stroke_risk: bool
    probability_stroke: float | None
    model_name: str
    model_uri: str


app = FastAPI(
    title="Stroke Prediction API",
    version="0.1.0",
    description="API FastAPI para servir o modelo champion de predicao de AVC.",
)


def _read_champion_metadata() -> dict:
    if not CHAMPION_PATH.exists():
        return {
            "model_name": DEFAULT_MODEL_NAME,
            "reason": "Fallback: reports/champion.json nao encontrado.",
        }
    with open(CHAMPION_PATH) as f:
        return json.load(f)


def _resolve_model_uri() -> str:
    if MLFLOW_MODEL_URI:
        return MLFLOW_MODEL_URI

    champion = _read_champion_metadata()
    model_uri = champion.get("mlflow_model_uri")
    if model_uri:
        return model_uri

    raise FileNotFoundError(
        "URI do modelo no MLflow nao encontrado. Rode o treino com "
        "`MLFLOW_TRACKING_URI=http://localhost:5002 dvc repro --force train`."
    )


@lru_cache(maxsize=1)
def _load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(_resolve_model_uri())


def _prepare_features(payload: PatientFeatures, model) -> pd.DataFrame:
    raw = pd.DataFrame([payload.model_dump()])
    features = build_feature_matrix(raw)

    expected_columns = getattr(model, "feature_names_in_", None)
    if expected_columns is not None:
        features = features.reindex(columns=list(expected_columns), fill_value=0)

    return features


@app.get("/health")
def health() -> dict:
    model_uri = None
    model_available = False
    try:
        model_uri = _resolve_model_uri()
        model_available = True
    except FileNotFoundError:
        pass

    return {
        "status": "ok" if model_available else "model_missing",
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "model_uri": model_uri,
    }


@app.get("/model-info")
def model_info() -> dict:
    metadata = _read_champion_metadata()
    metadata["mlflow_tracking_uri"] = MLFLOW_TRACKING_URI
    metadata["model_available"] = "mlflow_model_uri" in metadata or MLFLOW_MODEL_URI is not None
    return metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PatientFeatures) -> PredictionResponse:
    try:
        model = _load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    features = _prepare_features(payload, model)
    prediction = int(model.predict(features)[0])

    probability_stroke = None
    if hasattr(model, "predict_proba"):
        probability_stroke = float(model.predict_proba(features)[0][1])

    champion = _read_champion_metadata()
    return PredictionResponse(
        prediction=prediction,
        stroke_risk=bool(prediction),
        probability_stroke=probability_stroke,
        model_name=champion.get("model_name", DEFAULT_MODEL_NAME),
        model_uri=_resolve_model_uri(),
    )
