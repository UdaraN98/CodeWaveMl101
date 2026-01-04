import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None

import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


@dataclass(frozen=True)
class ModelBundle:
    model_path: str
    model: Any
    preprocessor_path: str
    preprocessor: Any


def _project_dir() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_experiment_name() -> str:
    return os.getenv("MLFLOW_EXPERIMENT_NAME", "customer_churn_optimization")


def _default_metric_name() -> str:
    return os.getenv("BEST_MODEL_METRIC", "f1")


def _parse_export_folder_metric(folder_name: str, metric_name: str) -> Optional[float]:
    # Expected pattern: {metric}_{value}_{run_id}
    # Example: f1_0.999970_205cc...
    pattern = rf"^{re.escape(metric_name)}_([0-9]+\.[0-9]+)_[0-9a-f]+$"
    match = re.match(pattern, folder_name)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _find_best_exported_model_dir() -> Path:
    explicit = os.getenv("BEST_MODEL_DIR")
    if explicit:
        return Path(explicit).expanduser().resolve()

    base = _project_dir() / "mlruns" / "best_model_artifacts" / _default_experiment_name()
    if not base.exists():
        raise FileNotFoundError(
            f"Best-model export folder not found: {base}. "
            "Run Scripts/export_best_artifacts.py first."
        )

    metric_name = _default_metric_name()
    best_dir: Optional[Path] = None
    best_value: float = float("-inf")

    for child in base.iterdir():
        if not child.is_dir():
            continue
        value = _parse_export_folder_metric(child.name, metric_name)
        if value is None:
            continue
        if value > best_value:
            best_value = value
            best_dir = child

    if best_dir is None:
        raise FileNotFoundError(
            f"No exported model folders found under {base} matching metric '{metric_name}'."
        )

    return best_dir


def _resolve_model_path() -> Path:
    explicit = os.getenv("BEST_MODEL_PATH")
    if explicit:
        return Path(explicit).expanduser().resolve()

    model_dir = _find_best_exported_model_dir()
    model_path = model_dir / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"model.pkl not found in best model dir: {model_dir}")

    return model_path


def _resolve_preprocessor_path() -> Path:
    explicit = os.getenv("PREPROCESSOR_PATH")
    if explicit:
        return Path(explicit).expanduser().resolve()

    default_path = _project_dir() / "Data" / "preprocessor.pkl"
    if not default_path.exists():
        raise FileNotFoundError(
            f"Preprocessor not found at {default_path}. "
            "Run Scripts/data_preprocessing.py or the training script to generate it."
        )
    return default_path


def load_bundle() -> ModelBundle:
    model_path = _resolve_model_path()
    preprocessor_path = _resolve_preprocessor_path()

    if joblib is not None:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
    else:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)

    return ModelBundle(
        model_path=str(model_path),
        model=model,
        preprocessor_path=str(preprocessor_path),
        preprocessor=preprocessor,
    )


class PredictRequest(BaseModel):
    # Raw (unprocessed) feature dict. Keys must match the columns used to fit the preprocessor.
    features: dict[str, Any]


class PredictResponse(BaseModel):
    prediction: int
    probability: Optional[float] = None
    model_path: str


app = FastAPI(title="Customer Churn Prediction API")
_bundle: Optional[ModelBundle] = None


@app.on_event("startup")
def _startup() -> None:
    global _bundle
    _bundle = load_bundle()


@app.get("/health")
def health() -> dict[str, str]:
    if _bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.get("/schema")
def schema() -> dict[str, Any]:
    if _bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    pre = _bundle.preprocessor
    cols = None
    if hasattr(pre, "feature_names_in_"):
        cols = list(getattr(pre, "feature_names_in_"))

    return {
        "expects": "PredictRequest.features contains raw feature names/values",
        "required_columns": cols,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    pre = _bundle.preprocessor
    model = _bundle.model

    # Build a 1-row DataFrame with the same input columns the preprocessor was fit on.
    if hasattr(pre, "feature_names_in_"):
        columns = list(getattr(pre, "feature_names_in_"))
        row = {col: req.features.get(col, None) for col in columns}
        df = pd.DataFrame([row], columns=columns)
    else:
        # Fallback: use provided keys.
        df = pd.DataFrame([req.features])

    try:
        X = pre.transform(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e}")

    try:
        pred = model.predict(X)
        pred_int = int(pred[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    prob: Optional[float] = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            prob = float(proba[0][1])
        except Exception:
            prob = None

    return PredictResponse(prediction=pred_int, probability=prob, model_path=_bundle.model_path)
