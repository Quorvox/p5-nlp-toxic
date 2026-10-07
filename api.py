import json
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict

app = FastAPI(title="Toxic classifier")


class TextIn(BaseModel):
    text: str = Field(min_length=1, max_length=500)


@app.post("/api/predict")
def api_predict(body: TextIn):
    label, score = predict(body.text)
    return {"toxic": bool(label), "score": score}


@app.get("/api/health")
def health():
    m = json.loads((Path(__file__).resolve().parent / "metrics.json").read_text(encoding="utf-8"))
    return {"ok": True, **m}
