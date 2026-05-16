"""Personalization: collect user-recorded landmark samples and train a tuned model.

Storage
-------
`personal_landmarks.csv` — same column layout as `hand_landmarks_data.csv`
(63 coord columns x{i}/y{i}/z{i} for i=1..21, plus a `label` column). Raw
unnormalized coords, exactly like the base CSV. Rows are appended over time.

Training
--------
`train_personal_model(...)` loads the user's CSV plus the base CSV,
normalizes, featurizes (288 features), and fits an SVM-RBF with
`sample_weight` boosting the personal rows by `personal_weight` (default 8x).
Saves `personal_gesture_model.pkl` + `personal_label_encoder.pkl` next to the
default model files.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from features import featurize_coords
from model import df_to_coords, normalize_landmarks

ROOT = Path(__file__).resolve().parents[1]
BASE_CSV = ROOT / "data" / "hand_landmarks_data.csv"
PERSONAL_CSV = ROOT / "data" / "personal_landmarks.csv"
PERSONAL_MODEL = ROOT / "models" / "personal_gesture_model.pkl"
PERSONAL_LE = ROOT / "models" / "personal_label_encoder.pkl"

COLUMNS = [f"{c}{i}" for i in range(1, 22) for c in ("x", "y", "z")] + ["label"]


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=COLUMNS)


def load_personal() -> pd.DataFrame:
    if not PERSONAL_CSV.exists():
        return _empty_df()
    return pd.read_csv(PERSONAL_CSV)


def append_samples(samples: list[np.ndarray], label: str) -> int:
    """Append (21, 3) sample arrays to personal_landmarks.csv. Returns new total."""
    if not samples:
        return len(load_personal())
    rows = []
    for s in samples:
        flat = s.reshape(-1).tolist()
        rows.append(dict(zip(COLUMNS[:-1], flat, strict=True), label=label))
    new_df = pd.DataFrame(rows, columns=COLUMNS)
    existing = load_personal()
    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.to_csv(PERSONAL_CSV, index=False)
    return len(combined)


def personal_summary() -> dict[str, int]:
    df = load_personal()
    if df.empty:
        return {}
    return df["label"].value_counts().to_dict()


def clear_label(label: str) -> int:
    df = load_personal()
    before = len(df)
    df = df[df["label"] != label]
    df.to_csv(PERSONAL_CSV, index=False)
    return before - len(df)


def clear_all() -> None:
    PERSONAL_CSV.unlink(missing_ok=True)


def _coords_to_features(coords: np.ndarray) -> np.ndarray:
    out = np.empty((len(coords), 63 + 210 + 15), dtype=np.float64)
    for i in range(len(coords)):
        out[i] = featurize_coords(coords[i])
    return out


def train_personal_model(
    personal_weight: float = 8.0,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict:
    """Train SVM-RBF on base + personal data, up-weighting personal rows.

    Returns a result dict with test F1, sample counts per source, and the
    output paths.
    """
    personal_df = load_personal()
    if personal_df.empty:
        raise RuntimeError(
            "no personal samples recorded yet — record some first."
        )

    base_df = pd.read_csv(BASE_CSV)
    base_df["_source"] = "base"
    personal_df = personal_df.copy()
    personal_df["_source"] = "personal"

    df = pd.concat([base_df, personal_df], ignore_index=True)
    df = normalize_landmarks(df.drop(columns=["_source"])).join(df[["_source"]])
    df = df.dropna().reset_index(drop=True)

    coords = df_to_coords(df)
    X = _coords_to_features(coords)
    le = LabelEncoder()
    y = le.fit_transform(df["label"].values)
    sources = df["_source"].values
    weights = np.where(sources == "personal", personal_weight, 1.0)

    X_tr, X_te, y_tr, y_te, w_tr, _, src_tr, src_te = train_test_split(
        X, y, weights, sources,
        test_size=test_size, random_state=random_state, stratify=y,
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=20, gamma="scale",
                    probability=True, random_state=random_state)),
    ])
    pipe.fit(X_tr, y_tr, clf__sample_weight=w_tr)
    y_pred = pipe.predict(X_te)
    f1_all = float(f1_score(y_te, y_pred, average="weighted", zero_division=0))

    # Personal-only metric on the held-out portion of personal rows.
    personal_mask = src_te == "personal"
    if personal_mask.sum() > 0:
        f1_personal = float(f1_score(
            y_te[personal_mask], y_pred[personal_mask],
            average="weighted", zero_division=0,
        ))
    else:
        f1_personal = float("nan")

    joblib.dump(pipe, PERSONAL_MODEL)
    joblib.dump(le, PERSONAL_LE)

    return {
        "f1_overall": f1_all,
        "f1_personal": f1_personal,
        "n_base": int((sources == "base").sum()),
        "n_personal": int((sources == "personal").sum()),
        "personal_per_label": load_personal()["label"].value_counts().to_dict(),
        "personal_weight": float(personal_weight),
        "model_path": str(PERSONAL_MODEL),
    }
