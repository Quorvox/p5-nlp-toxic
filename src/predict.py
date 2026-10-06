from pathlib import Path

import joblib

root = Path(__file__).resolve().parents[1]
bundle = None


def _load():
    # ленивая загрузка, один раз
    global bundle
    if bundle is None:
        bundle = joblib.load(root / "model.joblib")
    return bundle


def predict(text):
    b = _load()
    proba = b["clf"].predict_proba(b["vec"].transform([text]))[0]
    label = int(proba[1] >= 0.5)
    return label, round(float(proba[1]), 3)
