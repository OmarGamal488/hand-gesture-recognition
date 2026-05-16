"""Shared inference utilities for FastAPI and Gradio.

Three predict paths:
- `GestureModel.predict(coords)` — caller has a (21, 3) array.
- `detect_landmarks(rgb)` — single-hand convenience (returns first hand).
- `detect_all_landmarks(rgb)` — multi-hand, returns list of (coords, handedness).

Hand tracking quality
---------------------
- MediaPipe is configured in `RunningMode.VIDEO` with monotonic timestamps,
  so it can reuse the previous frame's tracking instead of running a full
  palm detection on every frame. This is much smoother and faster than
  `IMAGE` mode for continuous webcam streams.
- Detection / presence / tracking confidence thresholds are tuned for
  permissive recovery (re-acquires the hand quickly after occlusion).
- Returned landmarks are passed through a vectorized OneEuro filter per
  hand. OneEuro adapts to motion speed — heavy smoothing when the hand is
  still, snappy response when it moves fast — which kills the per-frame
  jitter that survives MediaPipe's own internal tracking.
"""

from __future__ import annotations

import threading
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from features import featurize_coords

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_gesture_model.pkl"
PERSONAL_MODEL_PATH = MODELS_DIR / "personal_gesture_model.pkl"
PERSONAL_LE_PATH = MODELS_DIR / "personal_label_encoder.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
LANDMARKER_PATH = MODELS_DIR / "hand_landmarker.task"
LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

DEFAULT_CONFIDENCE_THRESHOLD = 0.6
UNKNOWN_LABEL = "unknown"


@dataclass
class Prediction:
    label: str
    confidence: float
    top_k: list[tuple[str, float]]
    handedness: str | None = None
    is_unknown: bool = False


@dataclass
class MultiHandResult:
    hands: list[Prediction] = field(default_factory=list)


class GestureModel:
    """Wraps the sklearn pipeline and label encoder; thread-safe."""

    def __init__(self, model_path: Path = MODEL_PATH,
                 label_encoder_path: Path = LABEL_ENCODER_PATH) -> None:
        self.model_path = model_path
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(label_encoder_path)
        self.n_features = self._infer_n_features()
        self._lock = threading.Lock()

    def _infer_n_features(self) -> int:
        clf = getattr(self.model, "named_steps", {}).get("clf", self.model)
        return int(getattr(clf, "n_features_in_", 63))

    def predict(
        self,
        coords: np.ndarray,
        top_k: int = 3,
        threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        handedness: str | None = None,
    ) -> Prediction:
        if coords.shape != (21, 3):
            raise ValueError(f"expected (21, 3) landmarks, got {coords.shape}")
        full = featurize_coords(coords)
        x = full[: self.n_features].reshape(1, -1)
        with self._lock:
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(x)[0]
                order = np.argsort(proba)[::-1][:top_k]
                top = [
                    (str(self.label_encoder.inverse_transform([int(i)])[0]), float(proba[i]))
                    for i in order
                ]
                top_label, top_conf = top[0]
            else:
                idx = int(self.model.predict(x)[0])
                top_label = str(self.label_encoder.inverse_transform([idx])[0])
                top_conf = 1.0
                top = [(top_label, 1.0)]

        is_unknown = top_conf < threshold
        return Prediction(
            label=UNKNOWN_LABEL if is_unknown else top_label,
            confidence=top_conf,
            top_k=top,
            handedness=handedness,
            is_unknown=is_unknown,
        )


def normalize_landmarks_array(landmarks) -> np.ndarray:
    """Wrist-origin + mid-finger-tip scale (matches training normalization).

    Accepts a MediaPipe NormalizedLandmark list or a (21, 3) array.
    """
    if isinstance(landmarks, np.ndarray):
        coords = landmarks.astype(np.float64).copy()
    else:
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
    coords[:, 0] -= coords[0, 0]
    coords[:, 1] -= coords[0, 1]
    scale = float(np.sqrt(coords[12, 0] ** 2 + coords[12, 1] ** 2))
    if scale > 0:
        coords[:, 0] /= scale
        coords[:, 1] /= scale
    return coords


def ensure_landmarker_task() -> None:
    if LANDMARKER_PATH.exists():
        return
    print(f"downloading {LANDMARKER_PATH.name}...")
    urllib.request.urlretrieve(LANDMARKER_URL, LANDMARKER_PATH)


# ────────────────────────────────────────────────────────────────────
# OneEuro filter — adaptive landmark smoother
# Casiez et al. https://gery.casiez.net/1euro/
# ────────────────────────────────────────────────────────────────────
class OneEuroLandmarkSmoother:
    """Vectorized OneEuro filter for (21, 3) landmark arrays.

    - `mincutoff` — lower = more aggressive smoothing when still.
    - `beta` — higher = snappier response when motion is detected.
    Defaults are tuned for fingertip-scale tracking.
    """

    def __init__(self, mincutoff: float = 1.5, beta: float = 0.7,
                 dcutoff: float = 1.0) -> None:
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.reset()

    def reset(self) -> None:
        self._x_prev: np.ndarray | None = None
        self._dx_prev: np.ndarray | None = None
        self._t_prev_s: float | None = None

    @staticmethod
    def _alpha(cutoff, freq):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def smooth(self, x: np.ndarray, t_s: float) -> np.ndarray:
        if self._x_prev is None:
            self._x_prev = x.copy()
            self._dx_prev = np.zeros_like(x)
            self._t_prev_s = t_s
            return x
        te = max(t_s - self._t_prev_s, 1e-6)
        freq = 1.0 / te
        dx = (x - self._x_prev) * freq
        ed = self._alpha(self.dcutoff, freq)
        self._dx_prev = ed * dx + (1 - ed) * self._dx_prev
        cutoff = self.mincutoff + self.beta * np.abs(self._dx_prev)
        a = self._alpha(cutoff, freq)
        x_hat = a * x + (1 - a) * self._x_prev
        self._x_prev = x_hat
        self._t_prev_s = t_s
        return x_hat


_smoothers: dict[str, OneEuroLandmarkSmoother] = {}
_smoother_last_seen: dict[str, float] = {}
SMOOTHER_RESET_GAP_S = 0.4  # reset if hand disappears for this long


def _get_smoother(key: str, now_s: float) -> OneEuroLandmarkSmoother:
    sm = _smoothers.get(key)
    last = _smoother_last_seen.get(key, 0.0)
    if sm is None or (now_s - last) > SMOOTHER_RESET_GAP_S:
        sm = OneEuroLandmarkSmoother()
        _smoothers[key] = sm
    _smoother_last_seen[key] = now_s
    return sm


# ────────────────────────────────────────────────────────────────────
# MediaPipe HandLandmarker — VIDEO mode with monotonic timestamps
# ────────────────────────────────────────────────────────────────────
_landmarker = None
_landmarker_lock = threading.Lock()
_landmarker_num_hands = 0
_last_ts_ms = 0


def _next_timestamp_ms() -> int:
    """Monotonically-increasing millisecond timestamps for detect_for_video."""
    global _last_ts_ms
    ts = int(time.monotonic_ns() // 1_000_000)
    if ts <= _last_ts_ms:
        ts = _last_ts_ms + 1
    _last_ts_ms = ts
    return ts


def get_landmarker(num_hands: int = 2):
    """Lazy-load MediaPipe HandLandmarker in VIDEO mode. Rebuild if num_hands changes."""
    global _landmarker, _landmarker_num_hands
    if _landmarker is not None and _landmarker_num_hands == num_hands:
        return _landmarker
    with _landmarker_lock:
        if _landmarker is not None and _landmarker_num_hands == num_hands:
            return _landmarker
        ensure_landmarker_task()
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(LANDMARKER_PATH)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=num_hands,
            # More permissive than defaults — recovers faster from occlusion.
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.5,
        )
        _landmarker = mp_vision.HandLandmarker.create_from_options(options)
        _landmarker_num_hands = num_hands
        return _landmarker


def _ensure_mp_image_format(image: np.ndarray) -> np.ndarray:
    """Coerce an input array into the exact format MediaPipe + the trained
    model expect: 3-channel RGB, uint8, C-contiguous.

    The training landmarks were generated by running MediaPipe on
    HaGRID's RGB JPEGs — to reproduce that distribution at inference
    time the format must match exactly. RGBA gets its alpha dropped; a
    non-contiguous slice (e.g. from `image[:, ::-1, :]`) is materialized
    into a fresh contiguous buffer because MediaPipe binds its C++ side
    directly to the array's memory.
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"expected numpy array, got {type(image).__name__}")
    if image.ndim != 3:
        raise ValueError(f"expected (H, W, 3) array, got shape {image.shape}")
    if image.shape[2] == 4:           # drop alpha if present
        image = image[..., :3]
    if image.shape[2] != 3:
        raise ValueError(f"expected 3 channels, got {image.shape[2]}")
    if image.dtype != np.uint8:       # webcams give uint8; coerce stragglers
        image = np.clip(image, 0, 255).astype(np.uint8)
    if not image.flags["C_CONTIGUOUS"]:
        image = np.ascontiguousarray(image)
    return image


def _run_landmarker(image_rgb: np.ndarray, num_hands: int):
    import mediapipe as mp

    image_rgb = _ensure_mp_image_format(image_rgb)
    landmarker = get_landmarker(num_hands=num_hands)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    # detect_for_video requires monotonically-increasing timestamps. The
    # lock around the actual call serializes detection so concurrent
    # callers can't interleave timestamps out-of-order.
    with _landmarker_lock:
        ts = _next_timestamp_ms()
        return landmarker.detect_for_video(mp_image, ts)


def _coords_from(landmarks) -> np.ndarray:
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)


def detect_landmarks(image_rgb: np.ndarray, *, smooth: bool = True) -> np.ndarray | None:
    """Single-hand convenience. Returns first hand's raw (21, 3) coords or None."""
    result = _run_landmarker(image_rgb, num_hands=1)
    if not result.hand_landmarks:
        return None
    coords = _coords_from(result.hand_landmarks[0])
    if smooth:
        coords = _get_smoother("single", time.monotonic()).smooth(coords, time.monotonic())
    return coords


def detect_all_landmarks(image_rgb: np.ndarray, num_hands: int = 2,
                         *, smooth: bool = True):
    """Multi-hand. Returns list of (raw_coords (21,3), handedness str or None).

    Smoothing is per-handedness so swapping left/right doesn't mix filters.
    """
    result = _run_landmarker(image_rgb, num_hands=num_hands)
    if not result.hand_landmarks:
        return []
    now_s = time.monotonic()
    out = []
    for i, landmarks in enumerate(result.hand_landmarks):
        coords = _coords_from(landmarks)
        hand_label = None
        if result.handedness and i < len(result.handedness):
            cat = result.handedness[i]
            if cat:
                hand_label = cat[0].category_name
        if smooth:
            key = f"multi:{hand_label or f'hand{i}'}"
            coords = _get_smoother(key, now_s).smooth(coords, now_s)
        out.append((coords, hand_label))
    return out


_model: GestureModel | None = None
_personal_model: GestureModel | None = None
_use_personal: bool = False


def has_personal_model() -> bool:
    return PERSONAL_MODEL_PATH.exists() and PERSONAL_LE_PATH.exists()


def set_use_personal(value: bool) -> bool:
    """Toggle which model `get_model()` returns. Returns the active flag."""
    global _use_personal
    _use_personal = bool(value) and has_personal_model()
    return _use_personal


def reload_personal_model() -> None:
    """Drop the cached personal model so the next call reloads it from disk."""
    global _personal_model
    _personal_model = None


def get_model() -> GestureModel:
    """Return the active model — personal if toggled & available, else default."""
    global _model, _personal_model
    if _use_personal and has_personal_model():
        if _personal_model is None:
            _personal_model = GestureModel(
                model_path=PERSONAL_MODEL_PATH,
                label_encoder_path=PERSONAL_LE_PATH,
            )
        return _personal_model
    if _model is None:
        _model = GestureModel()
    return _model
