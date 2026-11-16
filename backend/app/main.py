import logging
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from .models import (
    BertNewsClassifier,
    FakeNewsDetector,
    LogisticNewsClassifier,
)

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGISTIC_MODEL_PATH = Path(
    os.getenv("LOGISTIC_MODEL_PATH", PROJECT_ROOT / "logistic_regression.pkl")
)
VECTORIZER_PATH = Path(
    os.getenv("VECTORIZER_PATH", PROJECT_ROOT / "tfidf_vectorizer.pkl")
)
BERT_MODEL_PATH = Path(os.getenv("BERT_MODEL_PATH", PROJECT_ROOT / "bert_finetuned"))
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
    if origin.strip()
]

app = FastAPI(title="Fake News Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Article text must not be blank.")
        return value


class PredictionResponse(BaseModel):
    predictions: dict[str, str]


@lru_cache(maxsize=1)
def load_detector() -> FakeNewsDetector:
    try:
        logistic_classifier = LogisticNewsClassifier(
            vectorizer_path=str(VECTORIZER_PATH),
            model_path=str(LOGISTIC_MODEL_PATH),
        )
        bert_classifier = BertNewsClassifier(model_path=str(BERT_MODEL_PATH))
        return FakeNewsDetector(logistic_classifier, bert_classifier)
    except Exception as error:
        logger.exception("Could not load the fake news prediction models.")
        raise HTTPException(
            status_code=503,
            detail=(
                "Prediction models are unavailable. Restore valid model artifacts "
                "and restart the API."
            ),
        ) from error


@app.get("/api/v1/health")
def health() -> dict[str, object]:
    logistic_artifacts_exist = (
        LOGISTIC_MODEL_PATH.is_file()
        and VECTORIZER_PATH.is_file()
        and LOGISTIC_MODEL_PATH.stat().st_size > 1
        and VECTORIZER_PATH.stat().st_size > 1
    )
    bert_artifacts_exist = (
        (BERT_MODEL_PATH / "config.json").is_file()
        and any(
            (BERT_MODEL_PATH / filename).is_file()
            for filename in ("pytorch_model.bin", "model.safetensors")
        )
    )
    prediction_ready = logistic_artifacts_exist and bert_artifacts_exist
    return {
        "status": "ready" if prediction_ready else "degraded",
        "prediction_ready": prediction_ready,
    }


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    detector = load_detector()
    return PredictionResponse(predictions=detector.predict(request.text))
