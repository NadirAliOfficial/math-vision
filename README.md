# ✍️ Math Vision — AI Finger Math Solver

Draw math expressions in the air with your index finger. Write `=` and the answer appears instantly — powered by Groq AI + MediaPipe, running in real time.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-green)](https://mediapipe.dev)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_4-orange)](https://groq.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-purple)](https://ollama.com)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## ✨ How it works

| Gesture | Action |
|---|---|
| ☝️ Index finger up | Hover / move cursor |
| 🤌 Pinch (thumb + index) | Draw on screen |
| 🖐 Open hand | Erase where palm moves |
| ✏️ Write `=` at the end | Auto-solve the expression |

Write any math expression with your finger, add `=` at the end, release your pinch — the answer appears right next to the `=` sign in handwriting style.

**Supports:** `2 + 2`, `5 × 3`, `2²`, `√16`, `100 / 4`, `(3 + 5) × 2` and more.

---

## 🚀 Quick Start

```bash
git clone https://github.com/NadirAliOfficial/math-vision.git
cd math-vision
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your Groq API key (free at [console.groq.com](https://console.groq.com)):

```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

Run:

```bash
python main.py
```

---

## 🖥️ Local LLM (No API Key)

You can run fully offline using [Ollama](https://ollama.com):

```bash
ollama pull llava-phi3
```

The app automatically falls back to Ollama if Groq is unavailable.

---

## 🏗️ Stack

- **[MediaPipe Hands](https://mediapipe.dev)** — real-time finger + gesture tracking
- **[Groq](https://groq.com)** (LLaMA 4 Scout) — ultra-fast AI math recognition
- **[Ollama](https://ollama.com)** — local LLM fallback (llava-phi3)
- **[OpenCV](https://opencv.org)** — webcam feed + drawing canvas

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|---|---|
| `C` | Clear canvas and all answers |
| `Space` | Solve immediately |
| `Q` | Quit |

---

## 📋 Requirements

- Python 3.10+
- Webcam
- Groq API key (free) **or** Ollama installed locally

---

<p align="center">Built with ❤️ · Powered by Groq + MediaPipe · Real-time finger math</p>
<p align="center">If this is useful, give it a ⭐</p>
