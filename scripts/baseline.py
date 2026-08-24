"""TF-IDF + logistic regression baseline for topic classification.

Usage:
    python scripts/baseline.py
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline

DATA_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
SEED = 42


def load(split: str) -> pd.DataFrame:
    return pd.read_json(DATA_DIR / f"{split}.jsonl", lines=True)


def combined(df: pd.DataFrame) -> pd.Series:
    """Title carries strong topical signal; give the model both."""
    return df["title"].fillna("") + "\n\n" + df["text"]


def main() -> None:
    train, val, test = load("train"), load("val"), load("test")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            sublinear_tf=True,
            min_df=3,
            max_df=0.6,
            ngram_range=(1, 2),
            max_features=100_000,
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            C=5.0,
            class_weight="balanced",
            random_state=SEED,
        )),
    ])

    pipeline.fit(combined(train), train["category"])

    val_pred = pipeline.predict(combined(val))
    print(f"val accuracy: {accuracy_score(val['category'], val_pred):.4f}")

    test_pred = pipeline.predict(combined(test))
    labels = sorted(train["category"].unique())

    acc = accuracy_score(test["category"], test_pred)
    macro_f1 = f1_score(test["category"], test_pred, average="macro")

    print(f"\ntest accuracy : {acc:.4f}")
    print(f"test macro-F1 : {macro_f1:.4f}\n")
    print(classification_report(test["category"], test_pred, digits=3))

    cm = confusion_matrix(test["category"], test_pred, labels=labels)
    print("confusion matrix (rows = true, cols = predicted)")
    print(f"{'':14}" + "".join(f"{l[:9]:>11}" for l in labels))
    for label, row in zip(labels, cm):
        print(f"{label:14}" + "".join(f"{v:>11}" for v in row))

    RESULTS_DIR.mkdir(exist_ok=True)
    with (RESULTS_DIR / "baseline.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "model": "tfidf+logreg",
            "accuracy": acc,
            "macro_f1": macro_f1,
            "labels": labels,
            "confusion_matrix": cm.tolist(),
            "n_train": len(train),
            "n_test": len(test),
        }, fh, indent=2)

    print(f"\nsaved -> {RESULTS_DIR / 'baseline.json'}")


if __name__ == "__main__":
    main()