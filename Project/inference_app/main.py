import os
from typing import List

import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model")

app = FastAPI(title="Churn Inference API", version="0.1.0")


class PredictRequest(BaseModel):
    rows: List[dict]


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise RuntimeError(f"Model path not found: {model_path}")
    return mlflow.pyfunc.load_model(model_path)


model = load_model(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: PredictRequest):
    if not payload.rows:
        raise HTTPException(status_code=400, detail="rows cannot be empty")
    df = pd.DataFrame(payload.rows)
    preds = model.predict(df)
    return {"predictions": preds.tolist()}


@app.get("/")
def root():
    return {"service": "churn-inference", "model_path": MODEL_PATH}
