from fastapi.testclient import TestClient

from api import app
from src.predict import predict
from src.train import main as train_main

api = TestClient(app)


def test_train_metrics():
    acc, f1 = train_main()
    assert f1 >= 0.8, f1
    assert acc >= 0.8, acc


def test_predict_toxic():
    label, score = predict("ты дурак и идиот")
    assert label == 1
    assert score > 0.5


def test_predict_clean():
    label, score = predict("спасибо, отличный сервис")
    assert label == 0
    assert score < 0.5


def test_api_toxic():
    r = api.post("/api/predict", json={"text": "ненавижу, полный отстой"})
    assert r.json()["toxic"] is True


def test_api_clean():
    r = api.post("/api/predict", json={"text": "супер, рекомендую всем"})
    assert r.json()["toxic"] is False


def test_api_empty():
    assert api.post("/api/predict", json={"text": ""}).status_code == 422


def test_health():
    assert api.get("/api/health").json()["ok"] is True
