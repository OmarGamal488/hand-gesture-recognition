"""Error analysis on the current champion model.

Writes:
- errors.csv             — every misclassified test row with true/pred/conf
- confused_pairs.png     — heatmap of confused (true → pred) pairs (errors only)
- per_class_summary.csv  — per-class error count + worst confused-with class

Run:
    uv run python analyze_errors.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from features import featurize_dataframe
from model import normalize_landmarks

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "hand_landmarks_data.csv"
MODEL_PATH = ROOT / "models" / "best_gesture_model.pkl"
LE_PATH = ROOT / "models" / "label_encoder.pkl"
ERRORS_CSV = ROOT / "outputs" / "errors.csv"
PAIRS_PNG = ROOT / "outputs" / "confused_pairs.png"
SUMMARY_CSV = ROOT / "outputs" / "per_class_summary.csv"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    df_norm = normalize_landmarks(df).dropna().reset_index(drop=True)
    le = joblib.load(LE_PATH)
    model = joblib.load(MODEL_PATH)

    n_features = getattr(getattr(model, "named_steps", {}).get("clf", model),
                         "n_features_in_", 63)
    X_full = featurize_dataframe(df_norm)[:, :n_features]
    y_full = le.transform(df_norm["label"].values)

    _, X_test, _, y_test, _, idx_test = train_test_split(
        X_full, y_full, np.arange(len(df_norm)),
        test_size=0.20, random_state=42, stratify=y_full,
    )

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        y_pred = proba.argmax(axis=1)
        conf = proba.max(axis=1)
    else:
        y_pred = model.predict(X_test)
        conf = np.ones(len(y_pred))

    wrong = y_pred != y_test
    n_err = int(wrong.sum())
    print(f"test size: {len(y_test)}  errors: {n_err}  acc: {1 - n_err/len(y_test):.4f}")

    err_df = pd.DataFrame({
        "row_index": idx_test[wrong],
        "true": le.inverse_transform(y_test[wrong]),
        "pred": le.inverse_transform(y_pred[wrong]),
        "confidence": conf[wrong],
    }).sort_values("confidence", ascending=False)
    err_df.to_csv(ERRORS_CSV, index=False)
    print(f"saved: {ERRORS_CSV.name}")

    pair_matrix = pd.crosstab(
        pd.Series(le.inverse_transform(y_test[wrong]), name="true"),
        pd.Series(le.inverse_transform(y_pred[wrong]), name="pred"),
    )
    pair_matrix = pair_matrix.reindex(index=le.classes_, columns=le.classes_, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(pair_matrix, annot=True, fmt="d", cmap="Reds",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title("Confused pairs (errors only) — rows: true, cols: predicted",
                 fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(PAIRS_PNG, dpi=130)
    plt.close(fig)
    print(f"saved: {PAIRS_PNG.name}")

    rows = []
    for cls in le.classes_:
        mask = err_df["true"] == cls
        n = int(mask.sum())
        if n == 0:
            worst = ""
        else:
            worst = Counter(err_df.loc[mask, "pred"]).most_common(1)[0][0]
        rows.append({
            "class": cls,
            "errors": n,
            "most_confused_with": worst,
        })
    summary = pd.DataFrame(rows).sort_values("errors", ascending=False)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"saved: {SUMMARY_CSV.name}")
    print("\ntop confused classes:")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
