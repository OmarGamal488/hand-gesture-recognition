from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# The model + dataset are gitignored (large binaries), so CI won't have them.
# Skip the whole module gracefully when running against a fresh checkout.
_ROOT = Path(__file__).resolve().parents[1]
_MODEL = _ROOT / "models" / "best_gesture_model.pkl"
_DATA = _ROOT / "data" / "hand_landmarks_data.csv"
_missing = [p for p in (_MODEL, _DATA) if not p.exists()]
if _missing:
    pytest.skip(
        f"trained artifacts not in this checkout: {[str(p.relative_to(_ROOT)) for p in _missing]}",
        allow_module_level=True,
    )

from inference import (  # noqa: E402
    DEFAULT_CONFIDENCE_THRESHOLD,
    UNKNOWN_LABEL,
    get_model,
    normalize_landmarks_array,
)


@pytest.fixture(scope="module")
def real_sample():
    from pathlib import Path
    df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "hand_landmarks_data.csv").iloc[0]
    coords = np.array(
        [[df[f"x{i}"], df[f"y{i}"], df[f"z{i}"]] for i in range(1, 22)],
        dtype=np.float64,
    )
    return coords, df["label"]


def test_model_loads():
    m = get_model()
    assert m.n_features in (63, 288)
    assert len(m.label_encoder.classes_) == 18


def test_predict_real_row_is_high_confidence(real_sample):
    coords, true_label = real_sample
    norm = normalize_landmarks_array(coords)
    pred = get_model().predict(norm)
    assert pred.label == true_label
    assert pred.confidence > 0.5
    assert not pred.is_unknown


def test_threshold_marks_unknown(real_sample):
    coords, _ = real_sample
    norm = normalize_landmarks_array(coords)
    pred = get_model().predict(norm, threshold=1.01)  # impossible to clear
    assert pred.is_unknown is True
    assert pred.label == UNKNOWN_LABEL


def test_default_threshold_constant():
    assert 0.0 <= DEFAULT_CONFIDENCE_THRESHOLD <= 1.0


def test_normalize_is_wrist_centered():
    coords = np.random.default_rng(0).standard_normal((21, 3))
    n = normalize_landmarks_array(coords)
    assert abs(n[0, 0]) < 1e-12
    assert abs(n[0, 1]) < 1e-12
