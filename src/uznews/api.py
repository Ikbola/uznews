"""Topic classification API for Uzbek news articles.

Serves the TF-IDF + logistic regression baseline, which outperformed
a fine-tuned XLM-RoBERTa on this dataset (78.6% vs 77.9% accuracy).
See README for the comparison.
"""

from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path("results/baseline_model.joblib")

app = FastAPI(
    title="uznews",
    description="Topic classification for Uzbek news articles",
    version="0.1.0",
)

_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Model not found. Run scripts/baseline.py first.",
            )
        _model = joblib.load(MODEL_PATH)
    return _model


class Article(BaseModel):
    text: str = Field(..., min_length=20, description="Article text, optionally with the title first")


class Prediction(BaseModel):
    label: str
    confidence: float
    scores: dict[str, float]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/classify", response_model=Prediction)
def classify(article: Article) -> Prediction:
    model = get_model()
    probs = model.predict_proba([article.text])[0]
    labels = list(model.classes_)
    scores = {l: round(float(p), 4) for l, p in zip(labels, probs)}
    best = max(scores, key=scores.get)
    return Prediction(label=best, confidence=scores[best], scores=scores)