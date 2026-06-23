# ✍️ Math Vision — AI Finger Math Solver

Draw math expressions in the air with your index finger. Write `=` and the answer appears instantly. Powered by Claude AI + MediaPipe.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-green)](https://mediapipe.dev)
[![Claude](https://img.shields.io/badge/Claude-Haiku-orange?logo=anthropic)](https://anthropic.com)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## ✨ How it works

| Gesture | Action |
|---|---|
| ☝️ Index finger | Draw on screen |
| 🖐 Open hand | Erase everything |
| ✏️ Write `=` | Auto-solve the expression |

Write any math expression with your finger, add `=` at the end, and Claude AI reads your handwriting and returns the answer in real time.

**Supports:** `2 + 2`, `5 × 3`, `√16`, `2²`, `100 / 4`, `(3 + 5) × 2` — anything Claude can read.

---

## 🚀 Quick Start

```bash
git clone https://github.com/NadirAliOfficial/math-vision.git
cd math-vision
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

Run:

```bash
python main.py
```

---

## 🏗️ Stack

- **[MediaPipe Hands](https://mediapipe.dev)** — real-time finger tracking
- **[Claude Haiku](https://anthropic.com)** — reads handwritten math expressions
- **[OpenCV](https://opencv.org)** — webcam + drawing canvas

---

## 📋 Requirements

- Python 3.10+
- Webcam
- Anthropic API key (get one free at [console.anthropic.com](https://console.anthropic.com))

---

<p align="center">Built with ❤️ · Powered by Claude AI · Real-time finger math</p>
<p align="center">If this is cool, give it a ⭐</p>
