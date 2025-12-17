"""FastAPI REST API for ArXiv classification."""

import pickle
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import DistilBertTokenizer

from arxiv_classifier.models.distilbert_classifier import DistilBertClassifier
from arxiv_classifier.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="ArXiv Paper Category Classifier",
    description="Classify ArXiv papers by category using DistilBERT",
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    """Request model for predictions."""

    title: str = Field(..., max_length=200, description="Paper title")
    summary: str = Field(..., max_length=2000, description="Paper abstract")


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    category: str = Field(..., description="Predicted category (e.g. cs.LG)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool


# Global model instance
_model: Optional[DistilBertClassifier] = None
_tokenizer: Optional[DistilBertTokenizer] = None
_label_encoder = None


def load_model(checkpoint_path: str = "train_artifacts/model.pt"):
    """Load model from checkpoint."""
    global _model, _tokenizer, _label_encoder

    logger.info(f"Loading model from {checkpoint_path}...")
    _model = DistilBertClassifier.load_from_checkpoint(checkpoint_path)
    _model.eval()
    logger.info("Model loaded and set to evaluation mode")

    logger.info("Loading tokenizer...")
    _tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    logger.info("Tokenizer loaded")

    # Load label encoder (should be saved during training)
    encoder_path = Path("train_artifacts/label_encoder.pkl")
    if encoder_path.exists():
        with open(encoder_path, "rb") as f:
            _label_encoder = pickle.load(f)
        logger.info(f"Label encoder loaded: {len(_label_encoder.classes_)} classes")
    else:
        logger.warning("Label encoder not found, using class indices")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("API server starting up...")
    load_model()
    logger.info("API server ready")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict category for a paper.

    Args:
        request: Prediction request with title and summary

    Returns:
        Predicted category and confidence score
    """
    if _model is None or _tokenizer is None:
        logger.error("Model not loaded, returning 503")
        raise HTTPException(status_code=503, detail="Model not loaded")

    logger.debug(f"Processing prediction request: title='{request.title[:50]}...'")
    text = f"{request.title} {request.summary}"
    encoding = _tokenizer(
        text,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = _model(encoding["input_ids"], encoding["attention_mask"])
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    # Get category name
    if _label_encoder:
        category = _label_encoder.inverse_transform([pred_idx.item()])[0]
    else:
        category = f"class_{pred_idx.item()}"

    logger.info(f"Prediction: category={category}, confidence={confidence.item():.4f}")

    return PredictionResponse(
        category=category,
        confidence=confidence.item(),
    )


if __name__ == "__main__":
    import uvicorn

    from arxiv_classifier.utils.logger import setup_logger

    setup_logger()
    uvicorn.run(app, host="0.0.0.0", port=8000)
