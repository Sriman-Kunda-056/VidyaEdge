#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           VidyaEdge — Offline AI Tutor powered by Gemma 4                  ║
║           Gemma 4 Good Hackathon · May 2026                                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Track  : Future of Education (Impact · $10K)                               ║
║           Ollama Special Technology ($10K)                                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Mission : 265 million children have no reliable internet.                  ║
║            VidyaEdge brings a multilingual, multimodal AI tutor to every    ║
║            classroom — on a ₹200 USB pendrive, no cloud, no subscription.   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Setup (one-time):                                                          ║
║    ollama pull gemma4:e2b-q4_k_m        # 3.1 GB — recommended             ║
║    pip install gradio ollama pillow psutil                                  ║
║    python app.py                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import base64
import io
import re
import time
import hashlib
import sqlite3
import subprocess
import threading
from pathlib import Path
from typing import Optional, Iterator, List, Dict, Any, Tuple
from collections import Counter

import ollama
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

try:
    import numpy as np
    _NUMPY = True
except ImportError:
    _NUMPY = False


# ═══════════════════════════════════════════════════════════════════════════
# HARDWARE DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_hardware() -> Dict[str, Any]:
    """Detect GPU/CPU and RAM."""
    hw = {
        "has_gpu": False,
        "vram_mb": 0,
        "ram_gb": 4.0,
        "gpu_name": "None",
        "mode": "cpu",
    }

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            hw["gpu_name"] = parts[0].strip()
            hw["vram_mb"] = int(parts[1].strip())
            hw["has_gpu"] = hw["vram_mb"] >= 4000
            hw["mode"] = "gpu" if hw["has_gpu"] else "cpu"
    except Exception:
        pass

    if _PSUTIL:
        hw["ram_gb"] = psutil.virtual_memory().total / (1024 ** 3)
    else:
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        hw["ram_gb"] = kb / (1024 ** 2)
                        break
        except Exception:
            pass

    return hw


HARDWARE = detect_hardware()


def select_model(hw: Dict[str, Any]) -> str:
    """Pick the best Gemma 4 model based on hardware."""
    candidates = []

    if hw["has_gpu"] and hw["vram_mb"] >= 6000:
        candidates = ["gemma4:e2b-q4_k_m", "gemma4:e2b", "gemma4:e4b"]
    elif hw["has_gpu"] and hw["vram_mb"] >= 4000:
        candidates = ["gemma4:e2b-q4_k_m", "gemma4:e2b"]
    elif hw["ram_gb"] >= 8:
        candidates = ["gemma4:e2b-q4_k_m", "gemma4:e2b"]
    else:
        candidates = ["gemma4:e2b-q4_k_m", "gemma4:e2b"]

    try:
        resp = ollama.list()
        installed = set()
        for m in resp.get("models", []):
            name = m.get("model", m.get("name", ""))
            if name:
                installed.add(name)
                if ":" in name:
                    base = name.split(":")[0] + ":" + name.split(":")[1].split("-")[0]
                    installed.add(base)
    except Exception as e:
        print("=" * 64)
        print("ERROR: Cannot reach Ollama daemon.")
        print(f"  Fix: run `ollama serve` in another terminal first.")
        print("=" * 64)
        sys.exit(1)

    for candidate in candidates:
        for inst in installed:
            if inst == candidate or inst.startswith(candidate):
                return candidate

    for inst in installed:
        if "gemma4" in inst.lower():
            print(f"⚠  Using installed Gemma 4 variant: {inst}")
            return inst

    print("=" * 64)
    print("ERROR: No Gemma 4 model installed.")
    print("Install one: ollama pull gemma4:e2b-q4_k_m")
    print("=" * 64)
    sys.exit(1)


OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
os.environ["OLLAMA_HOST"] = OLLAMA_HOST
MODEL = select_model(HARDWARE)


def print_deployment_summary():
    """Print system ready message."""
    hw = HARDWARE
    mode = "GPU" if hw["has_gpu"] else "CPU"
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║              VidyaEdge — System Ready                ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Model    : {MODEL:<42}║")
    print(f"║  Mode     : {mode:<42}║")
    print(f"║  Host     : {OLLAMA_HOST:<42}║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


print_deployment_summary()


# ═══════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════════════════

COLOR_PRIMARY       = "#f97316"
COLOR_PRIMARY_DARK  = "#ea580c"
COLOR_ACCENT        = "#0d9488"
COLOR_SUCCESS       = "#16a34a"
COLOR_BG            = "#fffbf5"
COLOR_SURFACE       = "#ffffff"
COLOR_BORDER        = "#fed7aa"
COLOR_TEXT          = "#292524"
COLOR_MUTED         = "#78716c"

INFERENCE_OPTS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "num_ctx": 4096,
    "num_predict": 1024,
}

JSON_OPTS = {
    "temperature": 0.05,
    "top_p": 0.9,
    "num_ctx": 1024,
    "num_predict": 300,
}

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# SQLITE CACHE
# ═══════════════════════════════════════════════════════════════════════════

DB_PATH = Path(__file__).parent / "vidyaedge_cache.db"
_db_lock = threading.Lock()


def get_db() -> sqlite3.Connection:
    """Thread-safe SQLite connection."""
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db():
    """Initialize database with curriculum."""
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS response_cache (
                query_hash   TEXT PRIMARY KEY,
                question     TEXT NOT NULL,
                response     TEXT NOT NULL,
                language     TEXT DEFAULT 'en',
                subject      TEXT DEFAULT 'general',
                grade        INTEGER DEFAULT 0,
                hit_count    INTEGER DEFAULT 0,
                created_at   REAL DEFAULT (unixepoch()),
                updated_at   REAL DEFAULT (unixepoch())
            );

            CREATE TABLE IF NOT EXISTS student_sessions (
                session_id       TEXT PRIMARY KEY,
                started_at       REAL DEFAULT (unixepoch()),
                turn_count       INTEGER DEFAULT 0,
                topics_covered   TEXT DEFAULT '[]',
                topics_mastered  TEXT DEFAULT '[]',
                topics_struggling TEXT DEFAULT '[]',
                languages_used   TEXT DEFAULT '[]',
                quiz_scores      TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS quiz_attempts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT,
                topic        TEXT,
                difficulty   TEXT,
                question     TEXT,
                correct      INTEGER DEFAULT 0,
                created_at   REAL DEFAULT (unixepoch())
            );

            CREATE INDEX IF NOT EXISTS idx_cache_subject
                ON response_cache(subject, grade);
            CREATE INDEX IF NOT EXISTS idx_quiz_session
                ON quiz_attempts(session_id);
        """)

        CURRICULUM = [
            ("what is photosynthesis", "en", "science", 7,
             "Photosynthesis is the process by which green plants make their own food using sunlight."),
            ("pythagorean theorem", "en", "math", 8,
             "$$a^2 + b^2 = c^2$$"),
            ("area of circle", "en", "math", 6,
             "The area of a circle: $$A = \\pi r^2$$"),
        ]

        inserted = 0
        for question, lang, subject, grade, response in CURRICULUM:
            h = hashlib.md5(f"{question}:{lang}".encode()).hexdigest()
            try:
                conn.execute(
                    """INSERT OR IGNORE INTO response_cache
                       (query_hash, question, response, language, subject, grade)
                       VALUES (?,?,?,?,?,?)""",
                    (h, question, response, lang, subject, grade),
                )
                inserted += 1
            except Exception:
                pass
        conn.commit()
        print(f"✓ Cache ready: {inserted} entries → {DB_PATH.name}")


init_db()


def cache_lookup(question: str, lang: str = "en") -> Optional[str]:
    """Check cache for exact match."""
    normalized = re.sub(r"\s+", " ", question.lower().strip())
    h = hashlib.md5(f"{normalized}:{lang}".encode()).hexdigest()
    with _db_lock:
        with get_db() as conn:
            row = conn.execute(
                "SELECT response FROM response_cache WHERE query_hash = ?", (h,)
            ).fetchone()
            if row:
                conn.execute(
                    "UPDATE response_cache SET hit_count = hit_count + 1, "
                    "updated_at = unixepoch() WHERE query_hash = ?", (h,)
                )
                conn.commit()
                return row[0]
    return None


def cache_store(question: str, lang: str, response: str, subject: str = "general"):
    """Store response in cache."""
    normalized = re.sub(r"\s+", " ", question.lower().strip())
    h = hashlib.md5(f"{normalized}:{lang}".encode()).hexdigest()
    with _db_lock:
        with get_db() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO response_cache
                   (query_hash, question, response, language, subject)
                   VALUES (?,?,?,?,?)""",
                (h, normalized, response, lang, subject),
            )
            conn.commit()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM response_cache").fetchone()[0]
        hits  = conn.execute("SELECT SUM(hit_count) FROM response_cache").fetchone()[0] or 0
    return {"total_entries": total, "total_hits": hits}


# ═══════════════════════════════════════════════════════════════════════════
# SESSION TRACKING
# ═══════════════════════════════════════════════════════════════════════════

import uuid

_SESSIONS: Dict[str, Dict] = {}
_SESSION_LOCK = threading.Lock()


def new_session() -> str:
    """Create a new session."""
    sid = str(uuid.uuid4())[:8]
    with _SESSION_LOCK:
        _SESSIONS[sid] = {
            "session_id": sid,
            "started_at": time.time(),
            "turn_count": 0,
            "topics_covered": [],
            "topics_mastered": [],
            "topics_struggling": [],
            "languages_used": [],
            "quiz_scores": {},
        }
    return sid


def log_topic(sid: str, topic: str, understood: bool):
    """Log learning progress."""
    with _SESSION_LOCK:
        if sid in _SESSIONS:
            s = _SESSIONS[sid]
            s["topics_covered"].append(topic)
            if understood and topic not in s["topics_mastered"]:
                s["topics_mastered"].append(topic)
            elif not understood and topic not in s["topics_struggling"]:
                s["topics_struggling"].append(topic)


# ═══════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT & TOOLS
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are VidyaEdge, a warm offline AI tutor for students aged 10–18.

CRITICAL RULES:

1. LANGUAGE MIRRORING: Reply in the EXACT language the student wrote in.
   Tamil → Tamil only. Hindi → Hindi only. English → English only.

2. FORMATTING: Number steps 1. 2. 3. Put math on own lines between $$ and $$.

3. CONCISENESS: 5–6 sentences unless asked for more.

4. HONESTY: If unsure, say so. Never invent facts.

5. ENCOURAGEMENT: Be warm. Use simple praise.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_visual_diagram",
            "description": "Generate a visual diagram",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject_type": {"type": "string", "enum": ["math", "science"]},
                    "topic": {"type": "string"},
                    "steps": {"type": "array", "items": {"type": "string"}},
                    "answer": {"type": "string"},
                },
                "required": ["subject_type", "topic", "steps"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_quiz_question",
            "description": "Generate a quiz question",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                    "language": {"type": "string"},
                },
                "required": ["topic", "difficulty", "language"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_student_progress",
            "description": "Log learning progress",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "understood": {"type": "boolean"},
                    "subject": {"type": "string", "enum": ["math", "science", "general"]},
                },
                "required": ["topic", "understood", "subject"],
            },
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# IMAGE ENCODING
# ═══════════════════════════════════════════════════════════════════════════

def encode_image_for_ollama(image) -> Optional[str]:
    """Convert PIL Image or numpy array to base64 JPEG for Ollama."""
    if image is None:
        return None
    try:
        if _NUMPY and isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype("uint8"))
        if not isinstance(image, Image.Image):
            return None
        image = image.convert("RGB")
        max_dim = 896
        if max(image.size) > max_dim:
            ratio = max_dim / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# AGENTIC CHAT CORE
# ═══════════════════════════════════════════════════════════════════════════

def _build_messages(user_text: str, history: List[dict], image=None) -> List[dict]:
    """Build Ollama message list."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in history[-6:]:
        role = turn.get("role")
        if role not in ("user", "assistant"):
            continue
        content = turn.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ).strip()
        if content:
            messages.append({"role": role, "content": content})

    current = {"role": "user", "content": user_text or "(image only)"}
    img_b64 = encode_image_for_ollama(image)
    if img_b64:
        current["images"] = [img_b64]
    messages.append(current)
    return messages


def execute_tool(name: str, args: dict, sid: str) -> Tuple[str, Optional[Image.Image]]:
    """Execute a tool call."""
    if name == "generate_visual_diagram":
        subject = args.get("subject_type", "math")
        topic   = (args.get("topic") or "Problem")[:50]
        steps   = [str(s) for s in (args.get("steps") or []) if s][:5]
        answer  = str(args.get("answer") or "")[:50]

        if subject == "math":
            img = render_math_diagram(topic, steps, answer)
        else:
            img = render_science_diagram(topic, steps)

        return (f"[Diagram: {topic}]", img)

    elif name == "generate_quiz_question":
        topic = args.get("topic", "topic")
        return (f"[Quiz: {topic}]", None)

    elif name == "log_student_progress":
        topic = args.get("topic", "unknown")
        understood = bool(args.get("understood", True))
        log_topic(sid, topic, understood)
        return (f"[Progress: {topic}]", None)

    else:
        return (f"[Tool: {name}]", None)


def agentic_chat(
    user_text: str,
    history: List[dict],
    image=None,
    sid: str = "",
) -> Iterator[Tuple[str, Optional[Image.Image]]]:
    """Full agentic loop with function calling."""
    diagram: Optional[Image.Image] = None
    messages = _build_messages(user_text, history, image)

    try:
        r1 = ollama.chat(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            stream=False,
            options={**JSON_OPTS, "num_predict": 400},
        )
    except Exception as e:
        yield f"⚠️ Error: {e}", None
        return

    msg1 = r1.message if hasattr(r1, "message") else r1.get("message", {})

    tool_calls = (
        msg1.tool_calls
        if hasattr(msg1, "tool_calls")
        else msg1.get("tool_calls", []) if isinstance(msg1, dict)
        else []
    ) or []

    # Content the model already wrote before/during the tool call decision.
    # Must be shown to the user — the streaming call only adds the continuation.
    first_content = (
        (msg1.content if hasattr(msg1, "content") else msg1.get("content", "")) or ""
    ).strip()

    if tool_calls:
        messages.append({
            "role": "assistant",
            "content": first_content,
            "tool_calls": [
                {
                    "function": {
                        "name": tc.function.name if hasattr(tc, "function") else tc["function"]["name"],
                        "arguments": tc.function.arguments if hasattr(tc, "function") else tc["function"]["arguments"],
                    }
                }
                for tc in tool_calls
            ],
        })

        for tc in tool_calls:
            fn      = tc.function if hasattr(tc, "function") else tc["function"]
            fn_name = fn.name if hasattr(fn, "name") else fn["name"]
            fn_args = fn.arguments if hasattr(fn, "arguments") else fn["arguments"]
            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except Exception:
                    fn_args = {}

            result_text, result_img = execute_tool(fn_name, fn_args, sid)
            if result_img is not None:
                diagram = result_img

            messages.append({"role": "tool", "content": result_text})

    try:
        stream = ollama.chat(
            model=MODEL,
            messages=messages,
            stream=True,
            options=INFERENCE_OPTS,
        )
        # Seed accumulated with anything the first call already generated so the
        # user sees a seamless, complete response rather than just the tail end.
        accumulated = first_content
        if accumulated:
            yield accumulated, diagram
        for chunk in stream:
            msg_c = getattr(chunk, "message", None)
            piece = ""
            if msg_c is not None:
                piece = getattr(msg_c, "content", "") or ""
            elif isinstance(chunk, dict):
                piece = chunk.get("message", {}).get("content", "") or ""
            if piece:
                accumulated += piece
                yield accumulated, diagram
    except Exception as e:
        yield f"⚠️ Error: {e}", diagram
        return

    if accumulated and user_text:
        try:
            cache_store(user_text, "auto", accumulated)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# DIAGRAM RENDERING
# ═══════════════════════════════════════════════════════════════════════════

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Load a font with fallbacks."""
    candidates = [
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap_text(text: str, max_chars: int) -> List[str]:
    """Word-wrap text."""
    if not text:
        return [""]
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_chars:
            cur = (cur + " " + w).strip()
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [text[:max_chars]]


def render_math_diagram(topic: str, steps: List[str], answer: str) -> Image.Image:
    """Render a math step diagram."""
    W, H = 720, 500
    img  = Image.new("RGB", (W, H), COLOR_BG)
    draw = ImageDraw.Draw(img)

    f_title = _load_font(22)
    f_step  = _load_font(17)
    f_ans   = _load_font(20)

    draw.rectangle([0, 0, W, 62], fill=COLOR_PRIMARY)
    draw.text((20, 16), f"Step: {topic[:42]}", font=f_title, fill="white")

    y = 82
    for i, step in enumerate(steps[:5], 1):
        if y > H - 90:
            break
        cx, cy, r = 38, y + 16, 16
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=COLOR_PRIMARY)
        num_text = str(i)
        bbox = draw.textbbox((0, 0), num_text, font=f_step)
        tw = bbox[2] - bbox[0]
        draw.text((cx - tw // 2, cy - 10), num_text, font=f_step, fill="white")

        wrapped = _wrap_text(step, 60)
        for j, line in enumerate(wrapped[:2]):
            draw.text((68, y + j * 22), line, font=f_step, fill=COLOR_TEXT)
        line_h = max(36, 22 * len(wrapped[:2])) + 8
        draw.line([20, y + line_h, W - 20, y + line_h], fill=COLOR_BORDER, width=1)
        y += line_h + 6

    if answer:
        by = H - 68
        draw.rectangle([16, by, W - 16, by + 52], fill=COLOR_SUCCESS + "22",
                       outline=COLOR_SUCCESS, width=2)
        draw.rectangle([16, by, 96, by + 52], fill=COLOR_SUCCESS)
        draw.text((20, by + 14), "Ans", font=f_step, fill="white")
        draw.text((106, by + 14), answer[:50], font=f_ans, fill=COLOR_TEXT)

    return img


def render_science_diagram(topic: str, facts: List[str]) -> Image.Image:
    """Render a science concept card."""
    W, H = 720, 500
    img  = Image.new("RGB", (W, H), COLOR_BG)
    draw = ImageDraw.Draw(img)

    f_title  = _load_font(24)
    f_fact   = _load_font(17)
    f_number = _load_font(19)

    draw.rectangle([0, 0, W, 68], fill=COLOR_ACCENT)
    draw.text((20, 16), topic[:44], font=f_title, fill="white")

    y = 90
    for i, fact in enumerate(facts[:4], 1):
        if y > H - 50:
            break
        card_h = max(52, 26 * len(_wrap_text(fact, 58)))
        draw.rectangle([16, y, W - 16, y + card_h],
                       fill=COLOR_SURFACE, outline=COLOR_BORDER, width=1)
        draw.rectangle([16, y, 56, y + card_h], fill=COLOR_ACCENT)
        n_bbox = draw.textbbox((0, 0), str(i), font=f_number)
        nx = 16 + (40 - (n_bbox[2] - n_bbox[0])) // 2
        draw.text((nx, y + 12), str(i), font=f_number, fill="white")

        wrapped = _wrap_text(fact, 58)
        for j, line in enumerate(wrapped[:3]):
            draw.text((68, y + 8 + j * 24), line, font=f_fact, fill=COLOR_TEXT)
        y += card_h + 10

    return img


# ═══════════════════════════════════════════════════════════════════════════
# GRADIO FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def chat_fn(user_text: str, image, history: List[dict], session_state: dict):
    """Main chat function."""
    user_text = (user_text or "").strip()
    if not user_text and image is None:
        yield history, None, "", session_state
        return

    sid = session_state.get("sid", "")
    if not sid:
        sid = new_session()
        session_state = {"sid": sid}

    display_user = user_text if user_text else "📷"
    history = list(history) + [
        {"role": "user",      "content": display_user},
        {"role": "assistant", "content": ""},
    ]

    if user_text and image is None:
        cached = cache_lookup(user_text)
        if cached:
            history[-1] = {"role": "assistant", "content": cached + "\n\n*⚡ Cached*"}
            yield history, None, "", session_state
            return

    final_response = ""
    final_diagram  = None
    for partial, diagram in agentic_chat(user_text, history[:-2], image, sid):
        final_response = partial
        final_diagram  = diagram
        history[-1] = {"role": "assistant", "content": partial}
        yield history, diagram, "", session_state

    history[-1] = {"role": "assistant", "content": final_response}
    yield history, final_diagram, "", session_state


def clear_chat(session_state: dict):
    """Clear chat."""
    return [], None, "", session_state


def build_teacher_report() -> str:
    """Build teacher dashboard."""
    stats = get_cache_stats()
    return f"## Dashboard\n\nCache: {stats['total_entries']} entries, {stats['total_hits']} hits"


# ═══════════════════════════════════════════════════════════════════════════
# PWA ASSETS
# ═══════════════════════════════════════════════════════════════════════════

def ensure_pwa_assets():
    """Generate PWA manifest and icons."""
    manifest = {
        "name": "VidyaEdge",
        "short_name": "VidyaEdge",
        "description": "Offline AI Tutor",
        "start_url": "/",
        "display": "standalone",
        "background_color": COLOR_PRIMARY,
        "theme_color": COLOR_PRIMARY,
        "icons": [{"src": "/gradio_api/file=static/icon-192.png", "sizes": "192x192"}],
    }
    (STATIC_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for size in (192, 512):
        path = STATIC_DIR / f"icon-{size}.png"
        if path.exists():
            continue
        img  = Image.new("RGB", (size, size), COLOR_PRIMARY)
        draw = ImageDraw.Draw(img)
        font = _load_font(int(size * 0.4))
        bbox = draw.textbbox((0, 0), "V", font=font)
        tw = bbox[2] - bbox[0]
        draw.text(((size - tw) // 2, 50), "V", font=font, fill="white")
        img.save(path)


ensure_pwa_assets()


# ═══════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ═══════════════════════════════════════════════════════════════════════════

PWA_HEAD = f'<link rel="manifest" href="/gradio_api/file=static/manifest.json">'
CUSTOM_CSS = ""

with gr.Blocks(title="VidyaEdge") as demo:
    session_state = gr.State({"sid": ""})

    gr.HTML(
        f'<div style="background:{COLOR_SURFACE};padding:20px;border-radius:12px;margin-bottom:16px">'
        f'<div style="font-size:24px;font-weight:bold;color:{COLOR_PRIMARY}">🎓 VidyaEdge</div>'
        f'<div style="font-size:13px;color:{COLOR_PRIMARY_DARK}">Offline AI Tutor</div></div>'
    )

    with gr.Tabs():
        with gr.Tab("📚 Student"):
            chatbot = gr.Chatbot(height=430, render_markdown=True,
                                latex_delimiters=[{"left":"$$","right":"$$","display":True},
                                               {"left":"$","right":"$","display":False}])
            with gr.Row():
                img_in = gr.Image(type="pil", label="📷", height=140, scale=1)
                diagram_out = gr.Image(label="✏️", interactive=False, height=140, scale=1)

            with gr.Row():
                msg_in = gr.Textbox(placeholder="Ask...", show_label=False, scale=5)

            with gr.Row():
                send_btn = gr.Button("🚀 Ask", variant="primary", scale=4)
                clear_btn = gr.Button("🗑", scale=1)

        with gr.Tab("👩‍🏫 Teacher"):
            dashboard_out = gr.Markdown("*Dashboard*")
            refresh_btn = gr.Button("🔄 Refresh")
            refresh_btn.click(build_teacher_report, outputs=[dashboard_out])

    submit_inputs = [msg_in, img_in, chatbot, session_state]
    submit_outputs = [chatbot, diagram_out, msg_in, session_state]
    send_btn.click(chat_fn, inputs=submit_inputs, outputs=submit_outputs)
    msg_in.submit(chat_fn, inputs=submit_inputs, outputs=submit_outputs)
    clear_btn.click(clear_chat, inputs=[session_state], outputs=submit_outputs)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=10)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False,
                show_error=True, allowed_paths=[str(STATIC_DIR)],
                head=PWA_HEAD, theme=gr.themes.Soft())
