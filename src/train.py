import csv
import json
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

root = Path(__file__).resolve().parents[1]


def load(name):
    # читаем csv в два списка
    texts, labels = [], []
    with open(root / "data" / name, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            texts.append(row["text"])
            labels.append(int(row["label"]))
    return texts, labels


def main():
    xtr, ytr = load("train.csv")
    xte, yte = load("test.csv")
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
    clf = LogisticRegression(max_iter=1000).fit(vec.fit_transform(xtr), ytr)
    pred = clf.predict(vec.transform(xte))
    acc, f1 = accuracy_score(yte, pred), f1_score(yte, pred)
    joblib.dump({"vec": vec, "clf": clf}, root / "model.joblib")
    (root / "metrics.json").write_text(json.dumps({"acc": round(acc, 3), "f1": round(f1, 3)},
                                                  ensure_ascii=False), encoding="utf-8")
    print(f"acc={acc:.3f} f1={f1:.3f}")
    return acc, f1


if __name__ == "__main__":
    main()
