"""Gradio UI for the hand-gesture classifier — camera-only, modern theme.

Two camera modes:
- Live Stream — continuous classification with per-hand sliding-window smoothing.
- Snapshot   — click to capture a frame and classify a single still.

Run:
    uv run python gradio_app.py
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from inference import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    detect_all_landmarks,
    detect_landmarks,
    get_model,
    has_personal_model,
    normalize_landmarks_array,
    reload_personal_model,
    set_use_personal,
)
from personalize import (
    append_samples,
    clear_all as clear_all_personal,
    personal_summary,
    train_personal_model,
)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]

GESTURE_EMOJI: dict[str, str] = {
    "call": "🤙",
    "dislike": "👎",
    "fist": "✊",
    "four": "4️⃣",
    "like": "👍",
    "mute": "🤫",
    "ok": "👌",
    "one": "☝️",
    "palm": "🖐️",
    "peace": "✌️",
    "peace_inverted": "✌️🔄",
    "rock": "🤘",
    "stop": "✋",
    "stop_inverted": "✋🔄",
    "three": "3️⃣",
    "three2": "3️⃣✨",
    "two_up": "2️⃣",
    "two_up_inverted": "2️⃣🔄",
    "unknown": "❓",
}


def emoji_for(label: str) -> str:
    return GESTURE_EMOJI.get(label, "❔")


# Per-hand neon palette (BGR for OpenCV).
HAND_PALETTE_BGR = [(255, 110, 199), (110, 220, 255)]   # magenta-pink, cyan
JOINT_BGR        = (255, 255, 255)
LINE_BGR_FAINT   = (160, 90, 255)                       # violet line accent


def _draw_hand(image_bgr: np.ndarray, raw_coords: np.ndarray,
               label: str, accent_bgr: tuple[int, int, int]) -> None:
    h, w = image_bgr.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in raw_coords]
    for a, b in HAND_CONNECTIONS:
        cv2.line(image_bgr, pts[a], pts[b], accent_bgr, 3, cv2.LINE_AA)
    for p in pts:
        cv2.circle(image_bgr, p, 5, JOINT_BGR, -1, cv2.LINE_AA)
        cv2.circle(image_bgr, p, 6, accent_bgr, 2, cv2.LINE_AA)
    # Floating capsule with the label, anchored above the wrist.
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x0, y0 = pts[0][0] - 8, max(28, pts[0][1] - 36)
    x1, y1 = x0 + tw + 16, y0 + th + 16
    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (30, 20, 50), -1)
    cv2.addWeighted(overlay, 0.78, image_bgr, 0.22, 0, dst=image_bgr)
    cv2.rectangle(image_bgr, (x0, y0), (x1, y1), accent_bgr, 2, cv2.LINE_AA)
    cv2.putText(image_bgr, label, (x0 + 8, y0 + th + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def _hand_card(emoji: str, label: str, conf: float,
               handedness: str | None, accent_hex: str) -> str:
    bar_pct = max(2, int(conf * 100))
    sub = f"<span style='opacity:0.55;font-size:13px;'>{handedness}</span>" if handedness else ""
    return f"""
    <div style="
        flex:1 1 200px;min-width:200px;padding:18px 20px;border-radius:18px;
        background:linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
        border:1px solid rgba(255,255,255,0.08);
        box-shadow:0 8px 32px rgba(0,0,0,0.35);
        backdrop-filter:blur(8px);
        ">
        <div style="display:flex;align-items:center;justify-content:space-between;">
            <div style="font-size:62px;line-height:1;">{emoji}</div>
            <div style="text-align:right;">
                <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;opacity:0.55;">{sub or "hand"}</div>
                <div style="font-size:13px;opacity:0.75;">{conf:.0%} conf</div>
            </div>
        </div>
        <div style="margin-top:14px;font-size:22px;font-weight:600;letter-spacing:0.3px;">{label}</div>
        <div style="margin-top:12px;height:6px;border-radius:99px;background:rgba(255,255,255,0.08);overflow:hidden;">
            <div style="height:100%;width:{bar_pct}%;background:linear-gradient(90deg,{accent_hex},#a78bfa);border-radius:99px;"></div>
        </div>
    </div>
    """


_EMPTY_CARD = """
<div style="padding:32px;border-radius:18px;text-align:center;
            background:rgba(255,255,255,0.03);border:1px dashed rgba(255,255,255,0.12);">
    <div style="font-size:54px;opacity:0.6;">📷</div>
    <div style="margin-top:8px;font-size:14px;opacity:0.55;">point your hand at the camera</div>
</div>
"""

HAND_ACCENTS_HEX = ["#ff6ec7", "#6edcff"]  # pink-magenta, cyan


def _classify(image_rgb: np.ndarray | None, threshold: float,
              smoother: dict[int, deque] | None = None):
    if image_rgb is None:
        return None, _EMPTY_CARD, {}
    detections = detect_all_landmarks(image_rgb, num_hands=2)
    if not detections:
        return image_rgb, _EMPTY_CARD, {}

    model = get_model()
    bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)
    cards = []
    combined_probs: dict[str, float] = {}
    for i, (raw, hand_label) in enumerate(detections):
        coords = normalize_landmarks_array(raw)
        pred = model.predict(coords, threshold=threshold, handedness=hand_label)
        display_label = pred.label
        if smoother is not None:
            smoother[i].append(pred.label)
            display_label = max(set(smoother[i]), key=smoother[i].count)

        overlay = f"{display_label.upper()}  {pred.confidence:.0%}"
        _draw_hand(bgr, raw, overlay, HAND_PALETTE_BGR[i % 2])
        cards.append(_hand_card(
            emoji_for(display_label),
            display_label,
            pred.confidence,
            hand_label,
            HAND_ACCENTS_HEX[i % 2],
        ))
        for lbl, c in pred.top_k:
            combined_probs[lbl] = max(combined_probs.get(lbl, 0.0), float(c))

    annotated = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cards_html = (
        "<div style='display:flex;gap:16px;flex-wrap:wrap;'>" + "".join(cards) + "</div>"
    )
    return annotated, cards_html, combined_probs


_stream_smoothers: dict[int, deque] = defaultdict(lambda: deque(maxlen=8))


def predict_stream(image_rgb: np.ndarray | None, threshold: float):
    annotated, cards, _ = _classify(image_rgb, threshold, smoother=_stream_smoothers)
    return annotated, cards


def predict_snapshot(image_rgb: np.ndarray | None, threshold: float):
    return _classify(image_rgb, threshold)


# ────────────────────────────────────────────────────────────────────
# Gesture-to-action: hand-drawing canvas
#
# Drawing: pinch your thumb tip and index fingertip together (pen down);
#          separate them to lift the pen.  The index fingertip is the cursor.
# Actions: ✋ palm  = clear canvas       (debounced: 4 consec. frames)
#          ✌️ peace = cycle pen color   (debounced: 4 consec. frames)
#
# Camera is mirrored (selfie view) so moving your hand right moves the
# cursor right.  Fingertip is EMA-smoothed to remove MediaPipe jitter.
# ────────────────────────────────────────────────────────────────────
PEN_COLORS_BGR = [
    (255, 110, 199),  # pink
    (110, 220, 255),  # cyan
    (180, 255, 110),  # lime
    (100, 220, 255),  # amber
    (255, 255, 255),  # white
]
PEN_COLORS_NAMES = ["pink", "cyan", "lime", "amber", "white"]
DRAW_THRESHOLD = 0.5
ACTION_DEBOUNCE = 4
PINCH_THRESHOLD = 0.35       # thumb↔index distance in normalized landmark space
SMOOTH_ALPHA = 0.55          # higher = more responsive, lower = smoother
PEN_THICKNESS = 5
MAX_UNDO_HISTORY = 20
SAVED_DIR = Path(__file__).resolve().parents[1] / "outputs" / "saved_canvases"

# Number gestures → direct color-slot selection.
NUMBER_TO_COLOR = {"one": 0, "two_up": 1, "three": 2, "four": 3}


def _init_draw_state() -> dict:
    return {
        "canvas": None,
        "last_pt": None,
        "smoothed_pt": None,
        "color_idx": 0,
        "gesture_streak": ("", 0),
        "last_trigger": "",
        "history": [],            # canvas snapshots for undo
        "last_save_path": "",
        "flash": "",              # short-lived HUD message ("✅ saved", "↩ undo")
        "flash_until": 0.0,
    }


def _save_canvas(canvas_bgr: np.ndarray) -> str:
    SAVED_DIR.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = SAVED_DIR / f"canvas_{ts}.png"
    cv2.imwrite(str(path), canvas_bgr)
    return str(path)


def _flash(state: dict, msg: str, seconds: float = 1.6) -> None:
    state["flash"] = msg
    state["flash_until"] = time.time() + seconds


def predict_draw(image_rgb, threshold, state):
    if state is None:
        state = _init_draw_state()
    if image_rgb is None:
        return None, _draw_hud(state, "no camera", False, None), state

    # Format-critical: detect on the ORIGINAL frame so MediaPipe + the
    # trained SVM see the same image distribution as during training (no
    # mirror flip). The selfie-view mirror happens only for the display
    # frame and the landmark x-coords are flipped after the fact for the
    # on-screen overlay.
    H, W = image_rgb.shape[:2]
    if state["canvas"] is None or state["canvas"].shape[:2] != (H, W):
        state["canvas"] = np.zeros((H, W, 3), dtype=np.uint8)
        state["last_pt"] = None
        state["smoothed_pt"] = None

    raw_orig = detect_landmarks(image_rgb)  # un-mirrored — matches training

    # Now build the mirrored display frame.
    image_disp = np.ascontiguousarray(image_rgb[:, ::-1, :])
    bgr = cv2.cvtColor(image_disp, cv2.COLOR_RGB2BGR)

    gesture = "no hand"
    pen_down = False
    pinch_dist: float | None = None
    raw = None
    if raw_orig is not None:
        # Display-space landmarks (x flipped). Distance-based features
        # (pinch, gesture features) are invariant under reflection of x,
        # so classification/pinch use either copy equivalently.
        raw = raw_orig.copy()
        raw[:, 0] = 1.0 - raw[:, 0]

        coords = normalize_landmarks_array(raw_orig)
        pred = get_model().predict(coords, threshold=DRAW_THRESHOLD)
        gesture = pred.label

        # Pinch detection in normalized landmark space (mid-finger-tip = 1.0).
        pinch_dist = float(np.linalg.norm(coords[4, :2] - coords[8, :2]))
        pen_down = pinch_dist < PINCH_THRESHOLD

        # Debounced trigger actions. Gate on `not pen_down` so commands never
        # fire mid-stroke — even if the model briefly mislabels the pinch.
        last_g, streak = state["gesture_streak"]
        streak = streak + 1 if gesture == last_g else 1
        state["gesture_streak"] = (gesture, streak)
        if (not pen_down and streak == ACTION_DEBOUNCE
                and gesture != state["last_trigger"]):
            if gesture == "palm":
                state["canvas"][:] = 0
                state["history"].clear()
                _flash(state, "🗑️ cleared")
            elif gesture == "peace":
                state["color_idx"] = (state["color_idx"] + 1) % len(PEN_COLORS_BGR)
                _flash(state, f"🎨 {PEN_COLORS_NAMES[state['color_idx']]}")
            elif gesture in NUMBER_TO_COLOR:
                state["color_idx"] = NUMBER_TO_COLOR[gesture]
                _flash(state, f"🎨 {PEN_COLORS_NAMES[state['color_idx']]}")
            elif gesture == "rock":
                if state["history"]:
                    state["canvas"] = state["history"].pop()
                    state["last_pt"] = None
                    _flash(state, "↩ undo")
                else:
                    _flash(state, "nothing to undo")
            elif gesture == "like":
                path = _save_canvas(state["canvas"])
                state["last_save_path"] = path
                _flash(state, f"✅ saved · {Path(path).name}")
            state["last_trigger"] = gesture

        # EMA-smooth the index fingertip cursor.
        tip = raw[8]
        raw_pt = np.array([tip[0] * W, tip[1] * H], dtype=np.float32)
        prev = state["smoothed_pt"]
        sm_pt = raw_pt if prev is None else (SMOOTH_ALPHA * raw_pt + (1 - SMOOTH_ALPHA) * prev)
        state["smoothed_pt"] = sm_pt
        sm_xy = (int(sm_pt[0]), int(sm_pt[1]))

        pen_color = PEN_COLORS_BGR[state["color_idx"]]
        accent = pen_color if pen_down else (140, 140, 160)
        status_text = "DRAW" if pen_down else "PEN UP"
        _draw_hand(bgr, raw, f"{status_text} · {gesture}", accent)

        if pen_down:
            # First frame of a new stroke → snapshot the canvas for undo.
            if state["last_pt"] is None:
                state["history"].append(state["canvas"].copy())
                if len(state["history"]) > MAX_UNDO_HISTORY:
                    state["history"].pop(0)
            else:
                cv2.line(state["canvas"], state["last_pt"], sm_xy,
                         pen_color, PEN_THICKNESS, cv2.LINE_AA)
            state["last_pt"] = sm_xy
        else:
            state["last_pt"] = None

        # Cursor ring at the smoothed fingertip — solid dot when pen is down.
        cv2.circle(bgr, sm_xy, 10, accent, 2, cv2.LINE_AA)
        if pen_down:
            cv2.circle(bgr, sm_xy, 4, (255, 255, 255), -1, cv2.LINE_AA)
    else:
        state["last_pt"] = None
        state["smoothed_pt"] = None
        state["gesture_streak"] = ("no_hand", 0)

    # Composite drawing canvas over the camera frame.
    mask = state["canvas"].sum(axis=-1) > 0
    out = bgr.copy()
    out[mask] = state["canvas"][mask]

    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return rgb, _draw_hud(state, gesture, pen_down, pinch_dist), state


def _draw_hud(state: dict, current_gesture: str,
              pen_down: bool, pinch_dist: float | None) -> str:
    color_name = PEN_COLORS_NAMES[state.get("color_idx", 0)]
    color_bgr = PEN_COLORS_BGR[state.get("color_idx", 0)]
    color_hex = "#{:02x}{:02x}{:02x}".format(color_bgr[2], color_bgr[1], color_bgr[0])

    if pinch_dist is None:
        pinch_txt, pinch_pct, pinch_label = "—", 0, "no hand"
    else:
        # 0 = touching, PINCH_THRESHOLD = pen-down threshold; show progress.
        pinch_pct = max(0, min(100, int((1 - pinch_dist / max(PINCH_THRESHOLD * 2, 1e-6)) * 100)))
        pinch_txt = f"{pinch_dist:.2f}"
        pinch_label = "pinched · pen down" if pen_down else "open · pen up"

    pen_chip_bg = color_hex if pen_down else "rgba(255,255,255,0.08)"
    pen_chip_color = "#0b0d17" if pen_down else "#d4d6df"
    pen_chip_label = "DRAW" if pen_down else "PEN UP"

    return f"""
    <div style="display:flex;gap:14px;flex-wrap:wrap;align-items:stretch;">
      <div style="flex:0 0 auto;padding:14px 18px;border-radius:14px;
                  background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
                  display:flex;flex-direction:column;justify-content:space-between;">
        <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;opacity:0.5;">pen</div>
        <div style="display:flex;align-items:center;gap:10px;margin-top:6px;">
          <div style="width:22px;height:22px;border-radius:50%;background:{color_hex};box-shadow:0 0 12px {color_hex};"></div>
          <div style="font-weight:600;">{color_name}</div>
        </div>
        <div style="margin-top:10px;padding:4px 10px;border-radius:99px;
                    background:{pen_chip_bg};color:{pen_chip_color};
                    font-size:11px;font-weight:700;letter-spacing:1.5px;text-align:center;">
          {pen_chip_label}
        </div>
      </div>
      <div style="flex:1 1 220px;padding:14px 18px;border-radius:14px;
                  background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);">
        <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;opacity:0.5;">pinch</div>
        <div style="margin-top:6px;font-size:15px;">{pinch_label} <span style='opacity:0.5;'>({pinch_txt})</span></div>
        <div style="margin-top:10px;height:6px;border-radius:99px;background:rgba(255,255,255,0.08);overflow:hidden;">
          <div style="height:100%;width:{pinch_pct}%;background:linear-gradient(90deg,{color_hex},#a78bfa);border-radius:99px;"></div>
        </div>
      </div>
      <div style="flex:1 1 220px;padding:14px 18px;border-radius:14px;
                  background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);">
        <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;opacity:0.5;">gesture</div>
        <div style="margin-top:6px;font-size:20px;font-weight:600;">
          {emoji_for(current_gesture)} {current_gesture}
        </div>
      </div>
      <div style="flex:1 1 100%;padding:14px 18px;border-radius:14px;
                  background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);">
        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;font-size:14px;">
          <div>🤏 <b>pinch</b> · pen down</div>
          <div>✋ <b>open hand</b> · pen up</div>
          <div>☝️2️⃣3️⃣4️⃣ <b>one/two/three/four</b> · color 1-4</div>
          <div>✌️ <b>peace</b> · cycle color</div>
          <div>🤘 <b>rock</b> · undo last stroke</div>
          <div>👍 <b>like</b> · save PNG</div>
          <div>✋ <b>palm</b> · clear canvas</div>
        </div>
      </div>
      {_draw_extras_html(state)}
    </div>
    """


def _draw_extras_html(state: dict) -> str:
    parts = []
    flash = state.get("flash", "")
    if flash and time.time() < state.get("flash_until", 0):
        parts.append(
            f"<div style='flex:0 1 auto;padding:8px 16px;border-radius:99px;"
            f"background:linear-gradient(135deg,#a78bfa,#f472b6);color:#0b0d17;"
            f"font-weight:700;font-size:13px;letter-spacing:0.5px;'>{flash}</div>"
        )
    undo_depth = len(state.get("history", []))
    parts.append(
        f"<div style='flex:0 1 auto;padding:8px 14px;border-radius:99px;"
        f"background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.08);"
        f"font-size:12px;opacity:0.75;'>↩ undo: {undo_depth} stroke{'s' if undo_depth != 1 else ''}</div>"
    )
    last = state.get("last_save_path", "")
    if last:
        parts.append(
            f"<div style='flex:0 1 auto;padding:8px 14px;border-radius:99px;"
            f"background:rgba(110,220,255,0.12);color:#6edcff;font-size:12px;'>"
            f"💾 last saved: {Path(last).name}</div>"
        )
    if not parts:
        return ""
    return (
        "<div style='flex:1 1 100%;display:flex;gap:10px;flex-wrap:wrap;align-items:center;'>"
        + "".join(parts) + "</div>"
    )


def clear_draw_canvas(state):
    if state is None:
        return _init_draw_state()
    if state.get("canvas") is not None:
        state["canvas"][:] = 0
    state["last_pt"] = None
    state["last_trigger"] = ""
    state["history"] = []
    _flash(state, "🗑️ cleared")
    return state


def undo_draw_canvas(state):
    if state is None or not state.get("history"):
        return state, _draw_hud(state or _init_draw_state(), "—", False, None)
    state["canvas"] = state["history"].pop()
    state["last_pt"] = None
    _flash(state, "↩ undo")
    return state, _draw_hud(state, "—", False, None)


def save_draw_canvas(state):
    """Manual save button → returns a file path for gr.File to surface as a download."""
    if state is None or state.get("canvas") is None:
        return state, None
    path = _save_canvas(state["canvas"])
    state["last_save_path"] = path
    _flash(state, f"✅ saved · {Path(path).name}")
    return state, path


# ────────────────────────────────────────────────────────────────────
# Personalization: record samples + train + toggle
# ────────────────────────────────────────────────────────────────────
def _init_personalize_state() -> dict:
    return {
        "active": False,
        "label": "",
        "target": 0,
        "captured": [],   # list of (21, 3) arrays
        "last_msg": "idle",
    }


_START_BTN = lambda: gr.update(value="▶ Start Recording", variant="primary")
_STOP_BTN = lambda: gr.update(value="■ Stop & Save", variant="stop")


def _btn_for(state: dict):
    """Return the button update that matches the current recording state.

    The stream fires several times per second; emitting a bare `gr.update()`
    in Gradio 6 resets a Button to its construction defaults — that's what
    was causing the visible Record↔Stop flicker. Always returning the
    correct, explicit label keeps the button stable across re-renders.
    """
    return _STOP_BTN() if state.get("active") else _START_BTN()


def toggle_recording(label: str, n_samples: int, state):
    """Single Start/Stop button handler. Stopping mid-record saves what was
    captured so the user never loses progress."""
    state = state or _init_personalize_state()

    if state["active"]:
        # Stop in the middle → save partial progress.
        captured = state["captured"]
        if captured:
            total = append_samples(captured, state["label"])
            msg = (f"⏹ stopped early. Saved {len(captured)} samples for "
                   f"'{state['label']}'. Total: {total}.")
        else:
            msg = "⏹ stopped (no samples captured)."
        state["active"] = False
        state["captured"] = []
        state["last_msg"] = msg
        return _personalize_status(state, msg), state, _btn_for(state), ""

    label = (label or "").strip()
    if not label:
        return (
            _personalize_status(state, "⚠️ enter a label name first."),
            state,
            _btn_for(state),
            "",
        )
    state.update({
        "active": True,
        "label": label,
        "target": int(n_samples),
        "captured": [],
        "last_msg": f"recording 0/{int(n_samples)} for '{label}'...",
    })
    return _personalize_status(state, state["last_msg"]), state, _btn_for(state), ""


def stream_recording(image_rgb, state):
    """Per-frame handler. Captures landmarks only while `active`.

    Importantly, this does NOT include the Start/Stop button in its outputs.
    In Gradio 6, every emitted output (even an unchanged one) is applied to
    the frontend, which forces a tab re-render and resets the webcam
    component's internal state — that was making the Record/Stop chip
    flicker on the Personalize tab.  Button label updates happen only on
    explicit clicks via `toggle_recording`.
    """
    if state is None:
        state = _init_personalize_state()
    if image_rgb is None:
        return None, _personalize_status(state, state.get("last_msg", "")), state

    bgr = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2BGR)

    if not state["active"]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        msg = state.get("last_msg") or "press ▶ Start Recording to capture samples."
        return rgb, _personalize_status(state, msg), state

    raw = detect_landmarks(image_rgb)
    if raw is not None:
        state["captured"].append(raw.astype(np.float64))
        _draw_hand(bgr, raw,
                   f"REC {len(state['captured'])}/{state['target']}",
                   (110, 220, 255))
        msg = f"recording {len(state['captured'])}/{state['target']} for '{state['label']}'..."
    else:
        msg = f"⚠️ no hand · {len(state['captured'])}/{state['target']} for '{state['label']}'"

    if len(state["captured"]) >= state["target"]:
        total = append_samples(state["captured"], state["label"])
        msg = (f"✅ recorded {len(state['captured'])} samples for "
               f"'{state['label']}'. Total saved: {total}.")
        state["active"] = False
        state["captured"] = []

    state["last_msg"] = msg
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, _personalize_status(state, msg), state


def _personalize_status(state, msg: str) -> str:
    counts = personal_summary()
    if counts:
        rows = "".join(
            f"<tr><td style='padding:3px 10px;'>{emoji_for(k)} {k}</td>"
            f"<td style='padding:3px 10px;text-align:right;font-variant-numeric:tabular-nums;'>{v}</td></tr>"
            for k, v in sorted(counts.items(), key=lambda kv: -kv[1])
        )
        table = (
            "<table style='width:100%;border-collapse:collapse;'>"
            "<thead><tr><th style='text-align:left;padding:3px 10px;opacity:0.5;font-size:11px;letter-spacing:2px;text-transform:uppercase;'>label</th>"
            "<th style='text-align:right;padding:3px 10px;opacity:0.5;font-size:11px;letter-spacing:2px;text-transform:uppercase;'>n</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    else:
        table = "<div style='opacity:0.55;font-size:13px;'>no personal samples yet.</div>"

    has_pm = has_personal_model()
    pm_chip = (
        "<span style='padding:3px 10px;border-radius:99px;background:rgba(110,220,255,0.15);color:#6edcff;font-size:12px;'>personal model trained ✓</span>"
        if has_pm else
        "<span style='padding:3px 10px;border-radius:99px;background:rgba(255,255,255,0.06);opacity:0.6;font-size:12px;'>no personal model yet</span>"
    )

    return f"""
    <div style="display:flex;gap:14px;flex-wrap:wrap;">
      <div style="flex:1 1 260px;padding:14px 18px;border-radius:14px;
                  background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);">
        <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;opacity:0.5;">status</div>
        <div style="margin-top:6px;font-size:15px;">{msg}</div>
        <div style="margin-top:10px;">{pm_chip}</div>
      </div>
      <div style="flex:1 1 260px;padding:14px 18px;border-radius:14px;
                  background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);">
        <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;opacity:0.5;margin-bottom:6px;">your samples</div>
        {table}
      </div>
    </div>
    """


def do_train_personal():
    try:
        result = train_personal_model()
    except Exception as e:
        return f"❌ training failed: {e}"
    reload_personal_model()
    return (
        f"✅ trained on {result['n_base']} base + {result['n_personal']} personal "
        f"samples (weight={result['personal_weight']:.1f}). "
        f"Held-out F1: overall={result['f1_overall']:.4f}, "
        f"personal={result['f1_personal']:.4f}. "
        f"Toggle 'Use personal model' to activate."
    )


def do_clear_all_personal():
    clear_all_personal()
    reload_personal_model()
    return "🗑️ cleared all personal samples."


def do_set_use_personal(use_personal: bool):
    active = set_use_personal(use_personal)
    if use_personal and not active:
        return "⚠️ no personal model trained yet — falling back to default."
    return f"using {'personal' if active else 'default'} model."


# Modern theme + custom CSS.
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.violet,
    secondary_hue=gr.themes.colors.fuchsia,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
).set(
    body_background_fill="#0b0d17",
    body_background_fill_dark="#0b0d17",
    body_text_color="#e6e7ee",
    background_fill_primary="rgba(255,255,255,0.03)",
    background_fill_secondary="rgba(255,255,255,0.02)",
    block_background_fill="rgba(255,255,255,0.03)",
    block_border_width="1px",
    block_border_color="rgba(255,255,255,0.08)",
    block_radius="18px",
    block_shadow="0 8px 32px rgba(0,0,0,0.35)",
    button_primary_background_fill="linear-gradient(135deg,#a78bfa,#f472b6)",
    button_primary_background_fill_hover="linear-gradient(135deg,#c4b5fd,#f9a8d4)",
    button_primary_text_color="#0b0d17",
    input_background_fill="rgba(255,255,255,0.04)",
    slider_color="#a78bfa",
)

CSS = """
.gradio-container {
    background:
        radial-gradient(1200px 600px at 10% -10%, rgba(167,139,250,0.18), transparent 60%),
        radial-gradient(900px 500px at 110% 10%, rgba(244,114,182,0.15), transparent 60%),
        #0b0d17 !important;
    color: #e6e7ee !important;
    font-family: Inter, ui-sans-serif, system-ui !important;
}
#hero {
    padding: 28px 32px;
    border-radius: 22px;
    background: linear-gradient(135deg, rgba(167,139,250,0.18), rgba(244,114,182,0.10));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 10px 40px rgba(0,0,0,0.4);
    margin-bottom: 18px;
}
#hero h1 {
    margin: 0 0 6px 0;
    font-size: 30px;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg,#c4b5fd,#f9a8d4);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
#hero p {
    margin: 0;
    opacity: 0.7;
    font-size: 14px;
}
.tabitem { padding-top: 12px !important; }
label { color: #d4d6df !important; font-weight: 500 !important; }
.gr-button { font-weight: 600 !important; letter-spacing: 0.3px !important; }
footer { display: none !important; }
"""


def _build_ui() -> gr.Blocks:
    model = get_model()
    n_classes = len(model.label_encoder.classes_)
    n_feats = model.n_features

    with gr.Blocks(title="Hand Gesture Studio") as demo:
        gr.HTML(f"""
        <div id="hero">
            <h1>✋ Hand Gesture Studio</h1>
            <p>Real-time recognition of <b>{n_classes}</b> gestures from MediaPipe landmarks ·
            SVM-RBF on <b>{n_feats}</b> engineered features · F1 = 0.989</p>
        </div>
        """)

        with gr.Row():
            threshold_slider = gr.Slider(
                minimum=0.0, maximum=1.0, step=0.05,
                value=DEFAULT_CONFIDENCE_THRESHOLD,
                label="Confidence threshold (below this → unknown)",
                scale=3,
            )
            personal_toggle = gr.Checkbox(
                value=False,
                label="🧠 Use my personal model",
                scale=1,
            )
        personal_msg = gr.Markdown("")
        personal_toggle.change(
            do_set_use_personal,
            inputs=personal_toggle,
            outputs=personal_msg,
        )

        with gr.Tabs():
            with gr.Tab("🎥  Live Stream"):
                gr.Markdown(
                    "Click **▶ Start** on the webcam to begin. Sliding-window "
                    "smoothing stabilizes per-hand predictions across frames."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        stream_in = gr.Image(
                            sources=["webcam"], type="numpy", streaming=True,
                            label="Camera input",
)
                    with gr.Column(scale=1):
                        stream_out = gr.Image(type="numpy", label="Annotated")
                stream_cards = gr.HTML(value=_EMPTY_CARD)
                stream_in.stream(
                    predict_stream,
                    inputs=[stream_in, threshold_slider],
                    outputs=[stream_out, stream_cards],
                    stream_every=0.2,
                    show_progress="hidden",
                )

            with gr.Tab("📸  Snapshot"):
                gr.Markdown(
                    "Take a single frame from the camera, then **Predict**. "
                    "Useful for posed shots where you want a clean per-hand reading."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        snap_in = gr.Image(
                            sources=["webcam"], type="numpy",
                            label="Camera snapshot",
                        )
                        btn = gr.Button("Predict", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        snap_out = gr.Image(type="numpy", label="Annotated")
                snap_cards = gr.HTML(value=_EMPTY_CARD)
                snap_probs = gr.Label(label="Top-3 confidences", num_top_classes=3)
                btn.click(
                    predict_snapshot,
                    inputs=[snap_in, threshold_slider],
                    outputs=[snap_out, snap_cards, snap_probs],
                )

            with gr.Tab("🎨  Hand Drawing"):
                gr.Markdown(
                    "**Pinch to draw.** Bring your thumb tip and index "
                    "fingertip together to put the pen down; separate them to "
                    "lift it. Camera is mirrored (selfie view) and the cursor "
                    "is smoothed, so writing your name should feel natural."
                )
                draw_state = gr.State(value=None)
                with gr.Row():
                    with gr.Column(scale=1):
                        draw_in = gr.Image(
                            sources=["webcam"], type="numpy", streaming=True,
                            label="Camera",
                        )
                    with gr.Column(scale=1):
                        draw_out = gr.Image(type="numpy", label="Canvas")
                draw_hud = gr.HTML(value=_draw_hud(_init_draw_state(), "—", False, None))
                with gr.Row():
                    undo_btn = gr.Button("↩  Undo")
                    clear_btn = gr.Button("🗑️  Clear")
                    save_btn = gr.Button("💾  Save PNG", variant="primary")
                save_file = gr.File(label="Download canvas", interactive=False)
                clear_btn.click(clear_draw_canvas, inputs=draw_state, outputs=draw_state)
                undo_btn.click(undo_draw_canvas,
                               inputs=draw_state,
                               outputs=[draw_state, draw_hud])
                save_btn.click(save_draw_canvas,
                               inputs=draw_state,
                               outputs=[draw_state, save_file])
                draw_in.stream(
                    predict_draw,
                    inputs=[draw_in, threshold_slider, draw_state],
                    outputs=[draw_out, draw_hud, draw_state],
                    stream_every=0.06,   # ~17 FPS — smoother strokes
                    show_progress="hidden",
                )

            with gr.Tab("🧠  Personalize"):
                gr.Markdown(
                    "**Train the model on your own hands.** Record landmark "
                    "samples per label, then train a personal model that "
                    "up-weights your samples 8x. Toggle the **Use my personal "
                    "model** checkbox above to activate it across the app."
                )
                p_state = gr.State(value=None)
                with gr.Row():
                    with gr.Column(scale=1):
                        p_label = gr.Textbox(
                            label="Label",
                            placeholder="e.g. palm, peace, or a custom name",
                            value="palm",
                        )
                        p_count = gr.Slider(
                            minimum=10, maximum=200, step=10, value=30,
                            label="Samples per recording",
                        )
                        p_start = gr.Button("▶ Start Recording", variant="primary")
                        p_train = gr.Button("🎓 Train Personal Model", variant="primary")
                        p_clear = gr.Button("🗑️ Clear All Samples", variant="secondary")
                    with gr.Column(scale=1):
                        p_camera = gr.Image(
                            sources=["webcam"], type="numpy", streaming=True,
                            label="Camera input",
                        )
                        p_camera_out = gr.Image(
                            type="numpy", label="Recording overlay",
                        )
                p_status = gr.HTML(value=_personalize_status(_init_personalize_state(), "idle"))
                p_train_msg = gr.Markdown("")

                p_start.click(
                    toggle_recording,
                    inputs=[p_label, p_count, p_state],
                    outputs=[p_status, p_state, p_start, p_train_msg],
                )
                # Stream outputs deliberately exclude `p_start`. Emitting a
                # Button update every frame (even via gr.update/gr.skip, which
                # are identical in Gradio 6) causes the tab to re-render and
                # resets the webcam component's internal state → its built-in
                # Record/Stop chip would flicker. Button updates only happen
                # on explicit clicks via `toggle_recording`.
                p_camera.stream(
                    stream_recording,
                    inputs=[p_camera, p_state],
                    outputs=[p_camera_out, p_status, p_state],
                    stream_every=0.15,
                    show_progress="hidden",
                )
                p_train.click(do_train_personal, outputs=p_train_msg)
                p_clear.click(do_clear_all_personal, outputs=p_train_msg)

        gr.HTML("""
        <div style="margin-top:24px;text-align:center;opacity:0.45;font-size:12px;">
            Hand Gesture Studio · Inter font · MediaPipe + scikit-learn
        </div>
        """)
    return demo


if __name__ == "__main__":
    get_model()
    _build_ui().launch(
        server_name="0.0.0.0", server_port=7860,
        theme=THEME, css=CSS,
    )
