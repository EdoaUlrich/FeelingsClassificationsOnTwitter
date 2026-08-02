"""
main.py
-------
API FastAPI pour l'analyse de sentiment.

Endpoints :
  - GET  /health   -> vérifie que l'API et le modèle sont opérationnels
  - POST /predict   -> prédit le sentiment d'un texte (positive / negative)
  - POST /explain   -> retourne les mots qui ont le plus influencé la prédiction

Lancement local :
    uvicorn main:app --reload --port 8000
"""

from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ARTIFACTS_DIR = Path("api_artifacts")
MODEL_PATH = ARTIFACTS_DIR / "sentiment_model.joblib"
VECTORIZER_PATH = ARTIFACTS_DIR / "tfidf_vectorizer.joblib"

ml_models = {}

# Correspondance entre les labels bruts du modèle (0/1) et le sentiment lisible
LABEL_TO_SENTIMENT = {0: "negative", 1: "positive", "0": "negative", "1": "positive"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Chargement des artifacts ML au démarrage de l'API
    try:
        ml_models["model"] = joblib.load(MODEL_PATH)
        ml_models["vectorizer"] = joblib.load(VECTORIZER_PATH)
    except FileNotFoundError:
        # L'API doit pouvoir démarrer même si les artifacts sont absents,
        # /health signalera alors model_loaded = False
        ml_models["model"] = None
        ml_models["vectorizer"] = None
    yield
    ml_models.clear()


app = FastAPI(
    title="Sentiment Analysis API",
    description="API de prédiction de sentiment (positive / negative) à partir d'un texte.",
    version="1.0.0",
    lifespan=lifespan,
)

# Autorise les appels depuis l'interface Streamlit (à restreindre en prod si besoin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texte à analyser")


class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float


class ExplainRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texte à analyser")
    top_n: int = Field(5, ge=1, le=20, description="Nombre de mots à retourner")


class WordContribution(BaseModel):
    word: str
    weight: float


class ExplainResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    top_words: list[WordContribution]


def _get_model_and_vectorizer():
    model = ml_models.get("model")
    vectorizer = ml_models.get("vectorizer")
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503,
            detail="Modèle non disponible. Vérifiez que les artifacts ML sont bien présents.",
        )
    return model, vectorizer


@app.get("/health")
def health():
    model_loaded = ml_models.get("model") is not None
    return {
        "status": "ok" if model_loaded else "degraded",
        "model_loaded": model_loaded,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    model, vectorizer = _get_model_and_vectorizer()

    vec = vectorizer.transform([payload.text])
    proba = model.predict_proba(vec)[0]
    classes = model.classes_
    best_idx = int(np.argmax(proba))

    return PredictResponse(
        text=payload.text,
        sentiment=LABEL_TO_SENTIMENT[classes[best_idx]],
        confidence=round(float(proba[best_idx]), 4),
    )


@app.post("/explain", response_model=ExplainResponse)
def explain(payload: ExplainRequest):
    model, vectorizer = _get_model_and_vectorizer()

    vec = vectorizer.transform([payload.text])
    proba = model.predict_proba(vec)[0]
    classes = model.classes_
    best_idx = int(np.argmax(proba))
    predicted_class_idx = list(model.classes_).index(classes[best_idx])

    # Contribution = poids du modèle (coef) * valeur tf-idf du mot dans le texte
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0] if len(model.classes_) == 2 else model.coef_[predicted_class_idx]

    row = vec.toarray()[0]
    nonzero_idx = np.nonzero(row)[0]

    contributions = coefs[nonzero_idx] * row[nonzero_idx]
    words = feature_names[nonzero_idx]

    order = np.argsort(-np.abs(contributions))[: payload.top_n]

    top_words = [
        WordContribution(word=str(words[i]), weight=round(float(contributions[i]), 4))
        for i in order
    ]

    return ExplainResponse(
        text=payload.text,
        sentiment=LABEL_TO_SENTIMENT[classes[best_idx]],
        confidence=round(float(proba[best_idx]), 4),
        top_words=top_words,
    )
