"""Landmark-space augmentations for training.

These operate on raw (N, 21, 3) coordinate arrays *after* wrist-centering
normalization, so the wrist sits at the origin.

- `rotate2d(angle_deg)` — rotate around the wrist; defensible because
  classifying a gesture should be view-rotation invariant within reasonable
  limits.
- `jitter(sigma)` — add gaussian noise to (x, y) to simulate MediaPipe
  detection noise.

Flipping is intentionally not included: many classes have explicit `_inverted`
variants (`peace`/`peace_inverted`, etc.), so a generic mirror would corrupt labels.
"""

from __future__ import annotations

import numpy as np


def rotate2d(coords: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate (N, 21, 3) or (21, 3) around the origin in the xy-plane."""
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=coords.dtype)
    out = coords.copy()
    out[..., :2] = out[..., :2] @ R.T
    return out


def jitter(coords: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    out = coords.copy()
    noise = rng.normal(0.0, sigma, size=out[..., :2].shape).astype(out.dtype)
    out[..., :2] += noise
    return out


def augment_batch(
    coords: np.ndarray,
    y: np.ndarray,
    *,
    n_copies: int = 2,
    max_rot_deg: float = 15.0,
    jitter_sigma: float = 0.01,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_aug, y_aug) including the original + n_copies augmented versions.

    coords: (N, 21, 3), y: (N,).
    """
    rng = np.random.default_rng(seed)
    pieces_x = [coords]
    pieces_y = [y]
    for _ in range(n_copies):
        angles = rng.uniform(-max_rot_deg, max_rot_deg, size=len(coords))
        aug = np.empty_like(coords)
        for i in range(len(coords)):
            aug[i] = rotate2d(coords[i], angles[i])
        aug = jitter(aug, jitter_sigma, rng)
        pieces_x.append(aug)
        pieces_y.append(y.copy())
    return np.concatenate(pieces_x, axis=0), np.concatenate(pieces_y, axis=0)
