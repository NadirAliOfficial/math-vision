"""
Math Vision — draw math with your index finger, write = to auto-solve.
Open hand to erase.
"""

import base64
import threading
import time

import anthropic
import cv2
import mediapipe as mp
import numpy as np

# ── config ────────────────────────────────────────────────────────────────────
DRAW_COLOR      = (255, 255, 255)
ANSWER_COLOR    = (0, 255, 120)
SOLVING_COLOR   = (0, 200, 255)
PEN_THICKNESS   = 10
MIN_PIXELS      = 300
CHECK_COOLDOWN  = 1.2   # seconds between API calls

# ── globals ───────────────────────────────────────────────────────────────────
canvas      = None
prev_pt     = None
was_drawing = False
answer_txt  = ""
solving     = False
last_check  = 0.0

client = anthropic.Anthropic()

mp_hands = mp.solutions.hands


# ── hand gesture helpers ──────────────────────────────────────────────────────

def index_only(lm):
    return (
        lm[8].y  < lm[6].y  and   # index up
        lm[12].y > lm[10].y and   # middle down
        lm[16].y > lm[14].y and   # ring down
        lm[20].y > lm[18].y       # pinky down
    )


def open_hand(lm):
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return all(lm[t].y < lm[p].y for t, p in zip(tips, pips))


# ── solver ────────────────────────────────────────────────────────────────────

def call_claude(img):
    global answer_txt, solving

    _, buf = cv2.imencode(".png", img)
    b64    = base64.b64encode(buf).decode()

    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": b64},
                    },
                    {
                        "type": "text",
                        "text": (
                            "This is a handwritten math expression drawn with a finger on screen. "
                            "If it contains an equals sign (=), evaluate the expression before it "
                            "and reply ONLY with the numeric answer (e.g. 4, 9, 3.14). "
                            "If there is no = sign yet or the expression is incomplete, reply WAIT. "
                            "Do not explain. Reply with just the number or WAIT."
                        ),
                    },
                ],
            }],
        )
        text = resp.content[0].text.strip()
        if text.upper() != "WAIT" and text:
            answer_txt = text
        else:
            answer_txt = ""
    except Exception as e:
        print(f"[Claude error] {e}")
    finally:
        solving = False


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global canvas, prev_pt, was_drawing, answer_txt, solving, last_check

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, frame = cap.read()
    h, w = frame.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame      = cv2.flip(frame, 1)
            rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result     = hands.process(rgb)
            drawing_now = False

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0].landmark

                if open_hand(lm):
                    canvas[:]  = 0
                    answer_txt = ""
                    prev_pt    = None

                elif index_only(lm):
                    fx = int(lm[8].x * w)
                    fy = int(lm[8].y * h)
                    drawing_now = True
                    if prev_pt:
                        cv2.line(canvas, prev_pt, (fx, fy), DRAW_COLOR, PEN_THICKNESS)
                    prev_pt = (fx, fy)

                    # fingertip dot
                    cv2.circle(frame, (fx, fy), 14, (0, 200, 255), -1)
                    cv2.circle(frame, (fx, fy),  6, (255, 255, 255), -1)
                else:
                    prev_pt = None
            else:
                prev_pt = None

            # stroke just ended → check with Claude
            if was_drawing and not drawing_now and not solving:
                now    = time.time()
                pixels = int(np.count_nonzero(canvas[:, :, 0]))
                if pixels > MIN_PIXELS and now - last_check > CHECK_COOLDOWN:
                    last_check = now
                    solving    = True
                    threading.Thread(
                        target=call_claude,
                        args=(canvas.copy(),),
                        daemon=True,
                    ).start()

            was_drawing = drawing_now

            # blend canvas over frame
            mask           = canvas > 0
            output         = frame.copy()
            output[mask]   = cv2.addWeighted(frame, 0.15, canvas, 0.85, 0)[mask]

            # answer
            if answer_txt:
                # expression from canvas is already visible; show = answer large
                txt = f"= {answer_txt}"
                (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 5)
                cv2.rectangle(output, (20, h - 80), (40 + tw, h - 20), (0, 0, 0), -1)
                cv2.putText(output, txt, (30, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.2, ANSWER_COLOR, 5, cv2.LINE_AA)

            if solving:
                cv2.putText(output, "Solving...", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, SOLVING_COLOR, 2, cv2.LINE_AA)

            # HUD
            cv2.putText(output,
                        "index finger = draw  |  open hand = clear  |  write = to solve",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1, cv2.LINE_AA)

            cv2.imshow("Math Vision", output)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
