"""
Math Vision — pinch to draw, open hand to erase, write = to auto-solve.
"""

import base64
import math
import re
import threading
import time
import urllib.request
import os

import groq as groq_lib
import requests as req_lib
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH  = "hand_landmarker.task"
MODEL_URL   = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DRAW_COLOR  = (255, 255, 255)
HOVER_CLR   = (80, 180, 255)
DRAW_CLR    = (0, 255, 100)
ANSWER_CLR  = (0, 255, 120)
SOLVE_CLR   = (0, 210, 255)
PEN_W       = 16
PINCH_RATIO = 0.38
SOLVE_DELAY = 0.7
MIN_PIX     = 200
ALPHA       = 0.6

canvas      = None
prev_pt     = None
was_pinched = False
answers     = []     # list of (text, pos) — persistent, never auto-removed
solving     = False
solve_timer = None
smooth_pt   = None
last_lm     = None
timer_start = None

OLLAMA_MODEL = "moondream"
OLLAMA_URL   = "http://localhost:11434/api/generate"

def _load_api_key(var):
    key = os.environ.get(var, "")
    if key:
        return key
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(var):
                    val = line.split("=", 1)[-1].strip().strip('"').strip("'")
                    if val:
                        os.environ[var] = val
                        return val
    return ""

groq_key = _load_api_key("GROQ_API_KEY")
groq_client = groq_lib.Groq(api_key=groq_key) if groq_key else None
print(f"[API] Groq {'ready' if groq_key else 'NO KEY'} | Ollama fallback: {OLLAMA_MODEL}")


def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def pdist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def hand_size(lm):
    """Wrist to middle-MCP — stable hand scale reference."""
    return max(pdist(lm[0], lm[9]), 0.01)

def pinching(lm):
    """Scale-invariant pinch: works close or far from camera."""
    return pdist(lm[4], lm[8]) / hand_size(lm) < PINCH_RATIO

def open_hand(lm):
    return all(lm[t].y < lm[p].y for t, p in zip([8,12,16,20],[6,10,14,18]))

def index_up(lm):
    return lm[8].y < lm[6].y


def _encode_img(img, width=400):
    """Resize and encode image as base64 PNG to reduce token usage."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (width, int(h * width / w)))
    inverted = cv2.bitwise_not(small)   # black-on-white for VLMs
    _, buf = cv2.imencode(".png", inverted)
    return base64.b64encode(buf).decode()

PROMPT_TEXT = (
    "Handwritten math expression. Compute the answer. "
    "Reply with ONLY the number. No words."
)

def _try_groq(b64):
    resp = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        max_tokens=10,
        messages=[
            {"role": "system", "content": "Calculator. Output one number only."},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text",      "text": PROMPT_TEXT},
            ]},
        ],
    )
    return resp.choices[0].message.content.strip()

def _try_ollama(b64):
    resp = req_lib.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL, "stream": False,
        "prompt": PROMPT_TEXT, "images": [b64],
        "options": {"temperature": 0, "num_predict": 12},
    }, timeout=30)
    return resp.json().get("response", "").strip()

def call_claude(img):
    global answers, solving, timer_start
    timer_start = None
    b64 = _encode_img(img)
    raw = ""
    try:
        if groq_client:
            print("[Groq] Sending...")
            raw = _try_groq(b64)
            print(f"[Groq] Raw: {raw}")
    except Exception as e:
        print(f"[Groq] Error: {e} — trying Ollama")
        try:
            raw = _try_ollama(b64)
            print(f"[Ollama] Raw: {raw}")
        except Exception as e2:
            print(f"[Ollama] Error: {e2}")

    if not raw and not groq_client:
        try:
            raw = _try_ollama(b64)
            print(f"[Ollama] Raw: {raw}")
        except Exception as e:
            print(f"[Ollama] Error: {e}")

    nums = re.findall(r"-?\d+(?:\.\d+)?", raw)
    if nums:
        answer_txt = nums[-1]
        rows = np.where(img[:, :, 0] > 0)[0]
        cols = np.where(img[:, :, 0] > 0)[1]
        if len(cols) > 0:
            pos = (int(np.max(cols)) + 30,
                   int((np.min(rows) + np.max(rows)) / 2) + 20)
            answers.append((answer_txt, pos))
            print(f"[Answer] {answer_txt}")
    solving = False


def schedule_solve(img):
    global solving, solve_timer, timer_start
    solve_timer  = None
    timer_start  = None
    if not solving:
        solving = True
        threading.Thread(target=call_claude, args=(img,), daemon=True).start()


def has_equals_sign(canvas):
    """
    Detect = sign: two thin horizontal lines, close together, both wide,
    in the rightmost strip of the drawing.
    """
    all_cols = np.where(canvas[:, :, 0] > 0)[1]
    if len(all_cols) == 0:
        return False

    right_x   = int(np.max(all_cols))
    left_x    = int(np.min(all_cols))
    total_w   = max(right_x - left_x, 1)

    # look only at rightmost 22% of the drawing (where = lives)
    strip_w = max(80, int(total_w * 0.22))
    x1      = max(0, right_x - strip_w)
    region  = canvas[:, x1:right_x + 1, 0]
    rw      = region.shape[1]
    if rw < 25:
        return False

    row_counts = np.sum(region > 0, axis=1)
    # each stroke of = must span at least 35% of the strip
    h_rows = np.where(row_counts > rw * 0.35)[0]
    if len(h_rows) < 3:
        return False

    # split into bands separated by gaps > 4px
    diffs = np.diff(h_rows)
    gap_idx = np.where(diffs > 4)[0]
    if len(gap_idx) == 0:
        return False

    # build bands
    bands, prev = [], 0
    for gi in gap_idx:
        bands.append(h_rows[prev:gi + 1])
        prev = gi + 1
    bands.append(h_rows[prev:])

    if len(bands) < 2:
        return False

    b1, b2  = bands[0], bands[1]
    gap     = int(b2[0]) - int(b1[-1])
    thick1  = int(b1[-1]) - int(b1[0]) + 1
    thick2  = int(b2[-1]) - int(b2[0]) + 1

    # = sign: gap 4–40px, each line thin (≤ 28px), no extra bands above/below
    return 4 <= gap <= 40 and thick1 <= 28 and thick2 <= 28


def draw_smooth(canvas, p1, p2, color, thickness):
    """Interpolate points between p1 and p2 for gapless smooth lines."""
    dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
    steps = max(int(dist / 3), 1)
    for i in range(steps + 1):
        t = i / steps
        x = int(p1[0] + (p2[0]-p1[0]) * t)
        y = int(p1[1] + (p2[1]-p1[1]) * t)
        cv2.circle(canvas, (x, y), thickness // 2, color, -1)


def main():
    global canvas, prev_pt, was_pinched, answers, solving
    global solve_timer, smooth_pt, last_lm, timer_start

    download_model()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    ret, frame = cap.read()
    h, w = frame.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    DW, DH   = w // 2, h // 2
    frame_n  = 0

    with mp_vision.HandLandmarker.create_from_options(opts) as landmarker:
        while True:
            cap.grab()
            ret, frame = cap.retrieve()
            if not ret:
                break

            frame   = cv2.flip(frame, 1)
            frame_n += 1

            # detect every 2nd frame — use last_lm on skipped frames
            if frame_n % 2 == 0:
                small  = cv2.resize(frame, (DW, DH))
                rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_img)
                last_lm = result.hand_landmarks[0] if result.hand_landmarks else None

            lm          = last_lm
            pinched_now = False
            finger_pos  = None

            if lm:
                if open_hand(lm):
                    # duster — erase only where palm moves
                    px = int(lm[9].x * w)
                    py = int(lm[9].y * h)
                    cv2.circle(canvas, (px, py), 60, (0, 0, 0), -1)
                    # visual duster ring on frame
                    cv2.circle(frame, (px, py), 60, (80, 80, 255), 3)
                    cv2.putText(frame, "ERASE", (px - 30, py - 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 255), 2, cv2.LINE_AA)
                    prev_pt   = None
                    smooth_pt = None

                elif index_up(lm):
                    rx = int(lm[8].x * w)
                    ry = int(lm[8].y * h)

                    # smooth cursor position
                    smooth_pt = (rx, ry) if smooth_pt is None else (
                        int(smooth_pt[0] * (1-ALPHA) + rx * ALPHA),
                        int(smooth_pt[1] * (1-ALPHA) + ry * ALPHA),
                    )
                    finger_pos  = smooth_pt
                    pinched_now = pinching(lm)

                    if pinched_now:
                        if solve_timer:
                            solve_timer.cancel()
                            solve_timer = None
                        if prev_pt and prev_pt != finger_pos:
                            draw_smooth(canvas, prev_pt, finger_pos, DRAW_COLOR, PEN_W)
                        elif not prev_pt:
                            cv2.circle(canvas, finger_pos, PEN_W//2, DRAW_COLOR, -1)
                        prev_pt = finger_pos
                        # draw cursor
                        cv2.circle(frame, finger_pos, 14, DRAW_CLR, -1)
                        cv2.circle(frame, finger_pos,  4, (255,255,255), -1)
                    else:
                        prev_pt = None
                        cv2.circle(frame, finger_pos, 20, HOVER_CLR, 2)
                        cv2.circle(frame, finger_pos,  5, HOVER_CLR, -1)
                else:
                    prev_pt = smooth_pt = None
            else:
                prev_pt = smooth_pt = None

            if was_pinched and not pinched_now and not solving:
                if int(np.count_nonzero(canvas[:,:,0])) > MIN_PIX and has_equals_sign(canvas):
                    if solve_timer:
                        solve_timer.cancel()
                    snap = canvas.copy()
                    timer_start = time.time()
                    solve_timer = threading.Timer(SOLVE_DELAY, schedule_solve, args=(snap,))
                    solve_timer.daemon = True
                    solve_timer.start()
                elif solve_timer and not has_equals_sign(canvas):
                    solve_timer.cancel()
                    solve_timer = None
                    timer_start = None

            was_pinched = pinched_now

            # overlay canvas
            output = frame.copy()
            mask   = canvas[:,:,0] > 0
            output[mask] = canvas[mask]

            # draw all persistent answers
            for atxt, apos in answers:
                ax, ay = apos
                (tw, _), _ = cv2.getTextSize(atxt, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3.5, 4)
                ax = max(10, min(ax, w - tw - 20))
                ay = max(80, min(ay, h - 20))
                cv2.putText(output, atxt, (ax, ay),
                            cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3.5, (0, 0, 0), 10, cv2.LINE_AA)
                cv2.putText(output, atxt, (ax, ay),
                            cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3.5, DRAW_COLOR, 4, cv2.LINE_AA)

            if solving:
                cv2.putText(output, "Solving...", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, SOLVE_CLR, 2, cv2.LINE_AA)

            mc = DRAW_CLR if pinched_now else HOVER_CLR
            cv2.putText(output, "DRAW" if pinched_now else "MOVE",
                        (w-110, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, mc, 2, cv2.LINE_AA)

            cv2.putText(output,
                "Pinch=draw  Open hand=erase  Write = to solve  C=clear",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

            cv2.imshow("Math Vision", output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c") or key == ord("C"):
                canvas[:] = 0
                answers.clear()
                if solve_timer:
                    solve_timer.cancel()
                    solve_timer = None
                timer_start = None
            elif key == ord(" ") and not solving:
                if int(np.count_nonzero(canvas[:,:,0])) > MIN_PIX:
                    if solve_timer:
                        solve_timer.cancel()
                        solve_timer = None
                    timer_start = None
                    solving = True
                    threading.Thread(target=call_claude, args=(canvas.copy(),), daemon=True).start()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
