# VidyaEdge — Offline AI Tutor

> Multilingual AI tutor for students in low-connectivity regions.
> Powered by **Gemma 4** running entirely on-device via **Ollama**.
> Built for the Gemma 4 Impact Challenge — May 2026.

---

## What it does

- Answers academic questions in **Tamil, Hindi, English, Telugu, Bengali, Malayalam, Kannada, Marathi, Gujarati** — and any other language Gemma 4 understands
- Accepts **photos** of handwritten homework (when using `gemma4:e4b`, the multimodal variant)
- Generates **step-by-step diagrams** for math and science, using Gemma 4's native JSON-structured output (not regex templates)
- Streams responses token-by-token for instant feedback
- Runs **100% offline** after one-time model download
- Installable as a **PWA / Android APK** for native phone deployment

---

## Quick start (laptop)

```bash
# 1. Install Ollama from https://ollama.com/download
ollama serve              # leave this running in one terminal

# 2. Pull the model (in another terminal)
ollama pull gemma4:e2b    # ~4 GB, fast, fits on phones
# OR
ollama pull gemma4:e4b    # ~7 GB, multimodal (vision)

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Run
python app.py

# 5. Open in any browser
# → http://localhost:7860
```

VidyaEdge will auto-detect which model you've pulled and use it.
**No fallback to non-Gemma models** — the app hard-fails if no Gemma 4 is found.

---

## Phone deployment (Termux on Android)

> **Requirements:** Android 11+, 6 GB+ RAM, ~10 GB free storage.

1. Install **Termux from F-Droid** (not Play Store): https://f-droid.org/packages/com.termux/

2. In Termux, run:
   ```bash
   pkg update -y && pkg upgrade -y
   pkg install proot-distro -y
   proot-distro install ubuntu
   proot-distro login ubuntu
   ```

3. Inside the Ubuntu environment:
   ```bash
   apt update && apt install -y curl python3 python3-pip git
   curl -fsSL https://ollama.com/install.sh | sh
   ollama serve > /tmp/ollama.log 2>&1 &
   sleep 5
   ollama pull gemma4:e2b
   ```

4. Copy `app.py` and `static/` into the Ubuntu environment (push via `adb push` from laptop, then `cp` into `~/`).

5. Install Python deps and launch:
   ```bash
   pip install -r requirements.txt --break-system-packages
   python3 app.py
   ```

6. On your phone, open Chrome → `http://localhost:7860`.

7. Optional — install as PWA: in Chrome, tap **⋮ menu → Add to Home Screen**. VidyaEdge now has its own app icon.

### Performance expectations on phone
| Model | RAM used | Speed | First token |
|-------|---------|-------|-------------|
| `gemma4:e2b` | ~3 GB | 3–5 tok/s | ~10 s |
| `gemma4:e4b` | ~5 GB | 1–2 tok/s | ~30 s |

---

## Architecture

```
┌─────────────────────────────────────────────┐
│         User (Tamil / Hindi / English)      │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  Gradio UI (PWA-enabled, mobile-first)      │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  Ollama daemon (localhost:11434)            │
│  • num_ctx=2048, INT4 quantization          │
│  • Single-instance, keep-alive enabled      │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│  Gemma 4 (e2b or e4b)                       │
│  • Multilingual via system prompt           │
│  • Multimodal via image attachment (e4b)    │
│  • Structured JSON via format="json" mode   │
└─────────────────────────────────────────────┘
```

### Key engineering choices

1. **Multilingual without `langdetect`** — Gemma 4 mirrors the input language by instruction. More reliable than client-side detection, and one fewer dependency on phone deployment.

2. **AI-generated diagrams via JSON-structured output** — instead of regex-extracting steps from response text, we make a second Gemma 4 call with `format="json"` (Ollama's constrained decoding). This guarantees valid JSON and uses Gemma 4's **native structured output capability**.

3. **No fallback to other models** — the app hard-fails if no Gemma 4 is installed. This protects submission integrity for the hackathon and prevents silent degradation in production.

4. **PWA-first** — single Python file ships a fullscreen-installable web app. No Android Studio, no native build required for the basic deployment. (TWA wrapping via Bubblewrap produces a real signed APK — see `BUBBLEWRAP.md` for that path.)

5. **Edge-tuned Ollama settings** — `num_ctx=2048` (low memory), `num_predict=512` (bounded response), `keep_alive` defaults to keep the model hot between turns.

---

## File structure

```
vidyaedge/
├── app.py                 # Main application (single file)
├── requirements.txt       # Python deps
├── README.md              # This file
└── static/
    ├── manifest.json      # PWA manifest (auto-generated)
    ├── icon-192.png       # App icon, auto-generated if missing
    └── icon-512.png       # App icon, auto-generated if missing
```

---

## Testing checklist

Before claiming the demo works, run these five tests with the phone in airplane mode:

1. ✅ English greeting: `Hello, can you help me?`
2. ✅ English math: `Solve 2x + 3 = 11 step by step`
3. ✅ Hindi science: `नमस्ते, गुरुत्वाकर्षण क्या है?`
4. ✅ Tamil greeting: `வணக்கம், சூரியன் ஏன் சூடாக இருக்கிறது?`
5. ✅ Tamil math: `2x + 3 = 11 படிப்படியாக தீர்க்கவும்`

If any reply comes back in the wrong language, restart with a stronger system prompt.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| "Cannot reach Ollama daemon" | Run `ollama serve` in another terminal |
| "No Gemma 4 model installed" | `ollama pull gemma4:e2b` |
| OOM crash on phone | Lower `num_ctx` to 1024 in `INFERENCE_OPTS` |
| Replies in wrong language | Strengthen rule 1 in `SYSTEM_PROMPT` with "CRITICAL — REPLY ONLY IN STUDENT'S LANGUAGE" |
| Slow first response (>60s) | Normal — model is loading into RAM. Subsequent responses are fast. |
| PWA icon doesn't show | Clear browser cache, reload, re-add to home screen |

---

## License & attribution

VidyaEdge is built on:
- **Gemma 4** by Google DeepMind (Gemma Terms of Use)
- **Ollama** (MIT)
- **Gradio** (Apache 2.0)

