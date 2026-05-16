"""Engineered-feature + optional-augmentation training run.

Adds pairwise distances + finger-joint angles (see features.py); optionally
augments the train split with rotated + jittered copies (augmentations.py).
Logs every candidate to MLflow under the `hand-gesture-classification`
experiment and registers the best as `champion`.

Flags
-----
--augment            include rotation + jitter augmentation
--n-copies INT       number of augmented copies per training row (default 2)
--no-mlflow          skip MLflow logging (e.g. for fast local iteration)

The script saves:
- improved_gesture_model.pkl   — always (the best candidate from this run)
- best_gesture_model.pkl       — only if F1 beats the prior baseline
- improved_metrics.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from augmentations import augment_batch
from features import featurize_coords

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "hand_landmarks_data.csv"
BASELINE_MODEL = ROOT / "models" / "best_gesture_model.pkl"
LABEL_ENCODER = ROOT / "models" / "label_encoder.pkl"
IMPROVED_MODEL = ROOT / "models" / "improved_gesture_model.pkl"
METRICS_JSON = ROOT / "outputs" / "improved_metrics.json"

BASELINE_F1 = 0.979


def normalize_landmarks(df: pd.DataFrame) -> pd.DataFrame:
    df_norm = df.copy()
    wrist_x = df_norm["x1"].values
    wrist_y = df_norm["y1"].values
    for i in range(1, 22):
        df_norm[f"x{i}"] = df_norm[f"x{i}"] - wrist_x
        df_norm[f"y{i}"] = df_norm[f"y{i}"] - wrist_y
    mid_x = df_norm["x13"].values
    mid_y = df_norm["y13"].values
    scale = np.sqrt(mid_x ** 2 + mid_y ** 2)
    scale = np.where(scale == 0, 1e-6, scale)
    for i in range(1, 22):
        df_norm[f"x{i}"] = df_norm[f"x{i}"] / scale
        df_norm[f"y{i}"] = df_norm[f"y{i}"] / scale
    return df_norm


def df_to_coords(df_norm: pd.DataFrame) -> np.ndarray:
    """Stack rows into an (N, 21, 3) array."""
    rows = []
    for i in range(21):
        rows.append(df_norm[[f'x{i+1}', f'y{i+1}', f'z{i+1}']].values)
    return np.stack(rows, axis=1).astype(np.float64)


def coords_to_features(coords: np.ndarray) -> np.ndarray:
    out = np.empty((len(coords), 63 + 210 + 15), dtype=np.float64)
    for i in range(len(coords)):
        out[i] = featurize_coords(coords[i])
    return out


def evaluate(name: str, pipe: Pipeline, X_train, X_test, y_train, y_test) -> dict:
    t0 = time.time()
    pipe.fit(X_train, y_train)
    train_s = time.time() - t0
    y_pred = pipe.predict(X_test)
    return {
        "name": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "train_seconds": round(train_s, 1),
    }


def build_candidates() -> dict[str, Pipeline]:
    return {
        "RandomForest+features": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=400, max_features="sqrt",
                min_samples_leaf=1, random_state=42, n_jobs=-1,
            )),
        ]),
        "ExtraTrees+features": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(
                n_estimators=400, max_features="sqrt",
                min_samples_leaf=1, random_state=42, n_jobs=-1,
            )),
        ]),
        "SVM-RBF+features": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=20, gamma="scale",
                        probability=True, random_state=42)),
        ]),
    }


def log_to_mlflow(results, trained, df_norm, X_train, X_test, y_train, y_test,
                  label_classes, augmented: bool):
    """Convert results dict into the shape mlflow_utils expects and log."""
    from mlflow_utils import run_mlflow_tracking

    metric_map_results = {
        name: {
            "Accuracy":  r["accuracy"],
            "Precision": r["precision"],
            "Recall":    r["recall"],
            "F1-Score":  r["f1"],
        }
        for name, r in zip([m["name"] for m in results], results)
    }
    models = {name: trained[name] for name in trained}
    run_ids = run_mlflow_tracking(
        models=models,
        results=metric_map_results,
        trained_models=trained,
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        df_norm=df_norm,
        label_classes=label_classes,
        experiment_name="hand-gesture-classification",
    )
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    for name, rid in run_ids.items():
        client.set_tag(rid, "feature_set", "raw63+dist210+ang15")
        client.set_tag(rid, "augmented", str(augmented))
    return run_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--n-copies", type=int, default=2)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    print(f"loading {CSV_PATH.name}...")
    df = pd.read_csv(CSV_PATH)
    df_norm = normalize_landmarks(df).dropna()
    print(f"  rows: {len(df_norm)}, classes: {df_norm['label'].nunique()}")

    print("building coord arrays...")
    coords = df_to_coords(df_norm)
    le = LabelEncoder()
    y = le.fit_transform(df_norm["label"].values)

    coords_train, coords_test, y_train, y_test = train_test_split(
        coords, y, test_size=0.20, random_state=42, stratify=y
    )

    if args.augment:
        print(f"augmenting train set ({args.n_copies} extra copies, ±15° rot, σ=0.01)...")
        coords_train, y_train = augment_batch(
            coords_train, y_train, n_copies=args.n_copies, seed=42,
        )
        print(f"  augmented train size: {coords_train.shape[0]}")

    print("featurizing (288 features)...")
    X_train = coords_to_features(coords_train)
    X_test = coords_to_features(coords_test)
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    candidates = build_candidates()
    results = []
    trained = {}
    for name, pipe in candidates.items():
        print(f"training {name}...")
        r = evaluate(name, pipe, X_train, X_test, y_train, y_test)
        print(f"  acc={r['accuracy']:.4f}  f1={r['f1']:.4f}  ({r['train_seconds']}s)")
        results.append(r)
        trained[name] = pipe

    best = max(results, key=lambda r: r["f1"])
    print("\n=== summary ===")
    print(f"  baseline f1 (raw 63 feats, RF):       {BASELINE_F1:.4f}")
    for r in sorted(results, key=lambda r: -r["f1"]):
        print(f"  {r['name']:<28} acc={r['accuracy']:.4f}  f1={r['f1']:.4f}")
    print(f"\n  best: {best['name']} (f1={best['f1']:.4f})")

    improved = best["f1"] > BASELINE_F1
    print(f"  beats baseline? {'YES' if improved else 'no'} "
          f"(delta f1 = {best['f1'] - BASELINE_F1:+.4f})")

    print("\nclassification report (best model):")
    y_pred = trained[best["name"]].predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(trained[best["name"]], IMPROVED_MODEL)
    joblib.dump(le, LABEL_ENCODER)
    print(f"saved: {IMPROVED_MODEL.name}, {LABEL_ENCODER.name}")
    if improved:
        joblib.dump(trained[best["name"]], BASELINE_MODEL)
        print(f"also overwrote {BASELINE_MODEL.name} (new champion).")

    METRICS_JSON.write_text(json.dumps({
        "baseline_f1": BASELINE_F1,
        "augmented": bool(args.augment),
        "n_copies": int(args.n_copies) if args.augment else 0,
        "results": results,
        "best": best,
        "n_features": int(X_train.shape[1]),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }, indent=2))
    print(f"saved: {METRICS_JSON.name}")

    if not args.no_mlflow:
        print("\nlogging to MLflow...")
        df_norm_aug = df_norm  # log against original (post-augmentation set
                                # would be misleading for a dataset summary).
        log_to_mlflow(
            results, trained, df_norm_aug,
            X_train, X_test, y_train, y_test,
            list(le.classes_), augmented=args.augment,
        )
        print("mlflow logging complete.")


if __name__ == "__main__":
    main()
