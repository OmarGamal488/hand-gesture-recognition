"""Feature engineering for hand-landmark gesture classification.

Adds geometric features on top of the raw 63 normalized coords:
- 210 pairwise (x, y) distances between all 21 landmarks
- 15 finger-joint angles (3 per finger)

Result: 63 + 210 + 15 = 288 features.

Both the training pipeline and the live FastAPI/Gradio services use this
module so train-time and inference-time features stay identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_LANDMARKS = 21

FINGER_JOINTS: list[tuple[int, int, int]] = [
    (1, 2, 3), (2, 3, 4),
    (5, 6, 7), (6, 7, 8),
    (9, 10, 11), (10, 11, 12),
    (13, 14, 15), (14, 15, 16),
    (17, 18, 19), (18, 19, 20),
    (0, 5, 9), (0, 9, 13), (0, 13, 17),
    (5, 0, 17),
    (4, 0, 20),
]


def _pairwise_distances_xy(coords: np.ndarray) -> np.ndarray:
    """Flat upper-triangle of pairwise (x, y) distances. Returns 210 floats."""
    xy = coords[:, :2]
    diff = xy[:, None, :] - xy[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    iu = np.triu_indices(N_LANDMARKS, k=1)
    return d[iu]


def _joint_angles(coords: np.ndarray) -> np.ndarray:
    """Cosine of angle at the middle joint for each triplet. Returns 15 floats."""
    xy = coords[:, :2]
    out = np.empty(len(FINGER_JOINTS), dtype=np.float64)
    for i, (a, b, c) in enumerate(FINGER_JOINTS):
        v1 = xy[a] - xy[b]
        v2 = xy[c] - xy[b]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            out[i] = 1.0
        else:
            out[i] = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return out


def featurize_coords(coords: np.ndarray) -> np.ndarray:
    """Build the full 288-feature vector from a (21, 3) normalized landmark array."""
    if coords.shape != (N_LANDMARKS, 3):
        raise ValueError(f"expected (21, 3), got {coords.shape}")
    return np.concatenate([
        coords.flatten(),
        _pairwise_distances_xy(coords),
        _joint_angles(coords),
    ])


def featurize_dataframe(df_norm: pd.DataFrame) -> np.ndarray:
    """Vectorized featurization of a normalized landmark dataframe."""
    rows = []
    for i in range(N_LANDMARKS):
        rows.append(df_norm[[f'x{i+1}', f'y{i+1}', f'z{i+1}']].values)
    stacked = np.stack(rows, axis=1)
    out = np.empty((len(df_norm), 63 + 210 + 15), dtype=np.float64)
    for i in range(len(df_norm)):
        out[i] = featurize_coords(stacked[i])
    return out


FEATURE_NAMES: list[str] = (
    [f'{c}{i+1}' for i in range(N_LANDMARKS) for c in ('x', 'y', 'z')]
    + [f'd_{a}_{b}' for a in range(N_LANDMARKS) for b in range(a + 1, N_LANDMARKS)]
    + [f'ang_{a}_{b}_{c}' for (a, b, c) in FINGER_JOINTS]
)
assert len(FEATURE_NAMES) == 63 + 210 + 15
