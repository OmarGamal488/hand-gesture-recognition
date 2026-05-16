"""FastAPI service for hand-gesture classification.

Endpoints
- GET  /health              → liveness + loaded-model info
- GET  /classes             → list of gesture labels
- POST /predict             → JSON body with 21x3 landmarks → label + confidence
- POST /predict_image       → multipart image upload (single hand)
- POST /predict_image_all   → multipart image upload (up to 2 hands, returns list)

Run:
    uv run uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference import (  # noqa: E402
    DEFAULT_CONFIDENCE_THRESHOLD,
    Prediction,
    detect_all_landmarks,
    detect_landmarks,
    get_model,
    normalize_landmarks_array,
)

app = FastAPI(
    title="Hand Gesture Classifier",
    description="Classifies 18 hand gestures from MediaPipe landmarks.",
    version="1.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class LandmarkPoint(BaseModel):
    x: float
    y: float
    z: float = 0.0


class LandmarksPayload(BaseModel):
    landmarks: list[LandmarkPoint] = Field(..., min_length=21, max_length=21)
    already_normalized: bool = False
    threshold: float = Field(
        default=DEFAULT_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0,
        description="Below this top-1 confidence the response label becomes 'unknown'.",
    )


class TopK(BaseModel):
    label: str
    confidence: float


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    is_unknown: bool
    handedness: str | None = None
    top_k: list[TopK]


class MultiHandResponse(BaseModel):
    n_hands: int
    hands: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    n_features: int
    n_classes: int
    default_threshold: float


def _to_response(p: Prediction) -> PredictionResponse:
    return PredictionResponse(
        label=p.label,
        confidence=p.confidence,
        is_unknown=p.is_unknown,
        handedness=p.handedness,
        top_k=[TopK(label=l, confidence=c) for l, c in p.top_k],
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    m = get_model()
    return HealthResponse(
        status="ok",
        n_features=m.n_features,
        n_classes=int(len(m.label_encoder.classes_)),
        default_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
    )


@app.get("/classes")
def classes() -> dict[str, list[str]]:
    return {"classes": [str(c) for c in get_model().label_encoder.classes_]}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: LandmarksPayload) -> PredictionResponse:
    coords = np.array([[p.x, p.y, p.z] for p in payload.landmarks], dtype=np.float64)
    if not payload.already_normalized:
        coords = normalize_landmarks_array(coords)
    try:
        return _to_response(get_model().predict(coords, threshold=payload.threshold))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _decode_image(data: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"cannot decode image: {e}")
    return np.array(image)


@app.post("/predict_image", response_model=PredictionResponse)
async def predict_image(
    file: UploadFile = File(...),
    threshold: float = Query(DEFAULT_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0),
) -> PredictionResponse:
    rgb = _decode_image(await file.read())
    raw = detect_landmarks(rgb)
    if raw is None:
        raise HTTPException(status_code=422, detail="no hand detected in image")
    coords = normalize_landmarks_array(raw)
    return _to_response(get_model().predict(coords, threshold=threshold))


@app.post("/predict_image_all", response_model=MultiHandResponse)
async def predict_image_all(
    file: UploadFile = File(...),
    threshold: float = Query(DEFAULT_CONFIDENCE_THRESHOLD, ge=0.0, le=1.0),
    num_hands: int = Query(2, ge=1, le=2),
) -> MultiHandResponse:
    rgb = _decode_image(await file.read())
    detections = detect_all_landmarks(rgb, num_hands=num_hands)
    model = get_model()
    hands = []
    for raw, hand_label in detections:
        coords = normalize_landmarks_array(raw)
        hands.append(_to_response(model.predict(coords, threshold=threshold,
                                                handedness=hand_label)))
    return MultiHandResponse(n_hands=len(hands), hands=hands)
