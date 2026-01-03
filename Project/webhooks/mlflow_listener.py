import hashlib
import hmac
import os

import requests
from fastapi import FastAPI, Header, HTTPException, Request

app = FastAPI(title="MLflow Webhook -> GitHub Dispatch")

GITHUB_REPO = os.getenv("GITHUB_REPO")  # e.g., owner/repo
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
DISPATCH_EVENT = os.getenv("DISPATCH_EVENT", "mlflow-model-change")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")


@app.get("/health")
def health():
    return {"status": "ok"}


def _verify_signature(raw_body: bytes, provided: str | None):
    if not WEBHOOK_SECRET:
        return True
    if not provided:
        return False
    digest = hmac.new(WEBHOOK_SECRET.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(digest, provided)


@app.post("/mlflow-webhook")
async def mlflow_webhook(request: Request, x_mlflow_signature: str | None = Header(None)):
    raw_body = await request.body()
    if not _verify_signature(raw_body, x_mlflow_signature):
        raise HTTPException(status_code=401, detail="Invalid signature")

    payload = await request.json()

    if not GITHUB_REPO or not GITHUB_TOKEN:
        raise HTTPException(status_code=500, detail="GITHUB_REPO or GITHUB_TOKEN not set")

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"token {GITHUB_TOKEN}",
    }
    data = {
        "event_type": DISPATCH_EVENT,
        "client_payload": payload,
    }
    resp = requests.post(
        f"https://api.github.com/repos/{GITHUB_REPO}/dispatches",
        headers=headers,
        json=data,
        timeout=10,
    )
    if resp.status_code >= 300:
        raise HTTPException(status_code=resp.status_code, detail=f"Dispatch failed: {resp.text}")

    return {"status": "forwarded", "event_type": DISPATCH_EVENT}
