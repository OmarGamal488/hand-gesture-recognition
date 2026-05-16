from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parents[1]
_MODEL = _ROOT / "models" / "best_gesture_model.pkl"
_DATA = _ROOT / "data" / "hand_landmarks_data.csv"
_missing = [p for p in (_MODEL, _DATA) if not p.exists()]
if _missing:
    pytest.skip(
        f"trained artifacts not in this checkout: {[str(p.relative_to(_ROOT)) for p in _missing]}",
        allow_module_level=True,
    )

from api.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def sample_landmarks():
    from pathlib import Path
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "hand_landmarks_data.csv").iloc[0]
    landmarks = [
        {"x": float(df[f"x{i}"]), "y": float(df[f"y{i}"]), "z": float(df[f"z{i}"])}
        for i in range(1, 22)
    ]
    return landmarks, df["label"]


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["n_classes"] == 18


def test_classes(client):
    r = client.get("/classes")
    assert r.status_code == 200
    assert len(r.json()["classes"]) == 18


def test_predict_correct_label(client, sample_landmarks):
    landmarks, true_label = sample_landmarks
    r = client.post("/predict", json={"landmarks": landmarks})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == true_label
    assert body["confidence"] > 0.5
    assert body["is_unknown"] is False
    assert len(body["top_k"]) == 3


def test_predict_threshold_triggers_unknown(client, sample_landmarks):
    landmarks, _ = sample_landmarks
    r = client.post("/predict", json={"landmarks": landmarks, "threshold": 1.0})
    assert r.status_code == 200
    body = r.json()
    assert body["is_unknown"] is True
    assert body["label"] == "unknown"


def test_predict_rejects_wrong_landmark_count(client, sample_landmarks):
    landmarks, _ = sample_landmarks
    r = client.post("/predict", json={"landmarks": landmarks[:20]})
    assert r.status_code == 422  # pydantic min_length


def test_predict_image_rejects_non_image(client):
    r = client.post(
        "/predict_image",
        files={"file": ("note.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 400
