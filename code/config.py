"""
config.py — Central Configuration & 3-Tier LLM Fallback Chain
==============================================================
Loads environment variables from .env and provides a unified
`call_llm()` function with automatic fallback:
    Tier 1: Google Gemini API
    Tier 2: Groq API (fast cloud inference)
    Tier 3: Ollama Local (offline fallback)

Each LLM purpose (router, security, arbiter, vision) uses the
best-fit model for that task.
"""

import os
import json
import base64
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MEDIA_DIR = DATASET_DIR / "media"
OUTPUT_PATH = DATASET_DIR / "output.csv"
LOG_DIR = Path(os.environ.get("USERPROFILE", os.path.expanduser("~"))) / "hackerrank_orchestrate_august26"
LOG_FILE = LOG_DIR / "log.txt"

load_dotenv(PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# API Keys & Model Names (including AssemblyAI for audio transcription)
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Purpose-specific model mapping
MODELS = {
    "router": {
        "gemini": os.getenv("GEMINI_ROUTER_MODEL", "gemini-2.5-flash"),
        "groq": os.getenv("GROQ_ROUTER_MODEL", "llama-3.3-70b-versatile"),
        "ollama": os.getenv("OLLAMA_ROUTER_MODEL", "minimax-m3:cloud"),
    },
    "security": {
        "gemini": os.getenv("GEMINI_SECURITY_MODEL", "gemini-2.5-flash"),
        "groq": os.getenv("GROQ_SECURITY_MODEL", "llama-3.3-70b-versatile"),
        "ollama": os.getenv("OLLAMA_SECURITY_MODEL", "minimax-m3:cloud"),
    },
    "arbiter": {
        "gemini": os.getenv("GEMINI_ARBITER_MODEL", "gemini-2.5-flash"),
        "groq": os.getenv("GROQ_ARBITER_MODEL", "llama-3.3-70b-versatile"),
        "ollama": os.getenv("OLLAMA_ARBITER_MODEL", "minimax-m3:cloud"),
    },
    "vision": {
        "gemini": os.getenv("GEMINI_VISION_MODEL", "gemini-3.5-flash-lite"),
        "groq": os.getenv("GROQ_VISION_MODEL", "qwen3.6-27B"),
        "ollama": os.getenv("OLLAMA_VISION_MODEL", "llava:13b"),
    },
}

GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3")
ASSEMBLYAI_MODEL = os.getenv("ASSEMBLYAI_MODEL", "best")  # 'best' or 'nano'

# ---------------------------------------------------------------------------
# Allowed output values (from problem_statement.md)
# ---------------------------------------------------------------------------
VALID_ACTIONS = {"notify", "digest", "mute"}
VALID_MESSAGE_TYPES = {
    "personal", "urgent", "event", "payment",
    "business_update", "promotion", "greeting",
    "forward", "spam", "scam", "unknown",
}

# ---------------------------------------------------------------------------
# Logger Setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("NotificationRouter")


# ---------------------------------------------------------------------------
# 3-Tier LLM Fallback Chain
# ---------------------------------------------------------------------------
async def _call_gemini(prompt: str, purpose: str, image_bytes: bytes | None = None) -> str:
    """Tier 1: Google Gemini API via REST (google-genai) with 429 retry."""
    import httpx

    model = MODELS[purpose]["gemini"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"

    parts = [{"text": prompt}]
    if image_bytes and purpose == "vision":
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        parts.insert(0, {"inline_data": {"mime_type": "image/jpeg", "data": b64}})

    payload = {"contents": [{"parts": parts}]}

    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(3):
            resp = await client.post(url, json=payload)
            if resp.status_code == 429:
                logger.warning(f"[Gemini] HTTP 429 Rate Limit (attempt {attempt+1}/3). Waiting 5s...")
                await asyncio.sleep(5)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        raise RuntimeError("Gemini failed after 3 rate limit retries.")


async def _call_groq(prompt: str, purpose: str, image_bytes: bytes | None = None) -> str:
    """Tier 2: Groq API via REST with 429 retry."""
    import httpx

    model = MODELS[purpose]["groq"]
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    messages = [{"role": "user", "content": prompt}]

    if image_bytes and purpose == "vision":
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        messages = [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}]

    payload = {"model": model, "messages": messages, "temperature": 0.1, "max_tokens": 1024}

    async with httpx.AsyncClient(timeout=60.0) as client:
        for attempt in range(3):
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code == 429:
                logger.warning(f"[Groq] HTTP 429 Rate Limit (attempt {attempt+1}/3). Waiting 5s...")
                await asyncio.sleep(5)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        raise RuntimeError("Groq failed after 3 rate limit retries.")


async def _call_ollama(prompt: str, purpose: str, image_bytes: bytes | None = None) -> str:
    """Tier 3: Ollama Local API via REST."""
    import httpx

    model = MODELS[purpose]["ollama"]
    url = f"{OLLAMA_BASE_URL}/api/generate"

    payload = {"model": model, "prompt": prompt, "stream": False}
    if image_bytes and purpose == "vision":
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        payload["images"] = [b64]

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["response"]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404 and purpose == "vision":
                # Vision model like llava missing — fallback to primary router model for text prompt
                logger.warning(f"[Ollama] Vision model '{model}' returned 404. Falling back to primary router model...")
                payload["model"] = MODELS["router"]["ollama"]
                payload.pop("images", None)
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                return resp.json()["response"]
            raise


# Global rate limiter & failure tracking
_api_rate_lock = asyncio.Lock()
API_CALL_DELAY_SECONDS = 1.5

# ---------------------------------------------------------------------------
# Sticky-Tier LLM Selection
# ---------------------------------------------------------------------------
# Instead of cycling through tiers per-call, we probe once at startup,
# lock onto the first working API, and use it for ALL messages.
# If it fails mid-run, we switch to the next tier and signal a retry.
# ---------------------------------------------------------------------------

# All available tiers in priority order: Groq > Gemini > Ollama
_ALL_TIERS: list[tuple[str, callable]] = []  # populated by init_llm_tier()
_dead_tiers: set[str] = set()  # tiers that have permanently failed
_active_tier_name: str = ""  # the currently locked tier name
_active_tier_fn: callable = None  # the currently locked tier function


class AllLLMTiersFailedError(RuntimeError):
    """Raised when all LLM tiers (Groq, Gemini, Ollama) are dead."""
    pass


class LLMTierSwitchedError(RuntimeError):
    """Raised when the active tier failed mid-run and we switched to a new one.
    The caller (agent4) should retry the current message."""
    pass


def _build_tier_list() -> list[tuple[str, callable]]:
    """Build the ordered list of available tiers: Groq -> Ollama -> Gemini."""
    tiers = []
    # Priority 1: Groq (primary cloud)
    if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
        tiers.append(("Groq", _call_groq))
    # Priority 2: Ollama (minimax-m3:cloud / local)
    tiers.append(("Ollama", _call_ollama))
    # Priority 3: Gemini (cloud fallback)
    if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
        tiers.append(("Gemini", _call_gemini))
    return tiers


async def init_llm_tier() -> str:
    """
    Probe LLM tiers in priority order (Groq > Ollama > Gemini).
    Lock onto the first one that responds successfully.
    Must be called once at pipeline startup.

    Returns:
        The name of the active tier.

    Raises:
        AllLLMTiersFailedError: If no tier responds.
    """
    global _ALL_TIERS, _active_tier_name, _active_tier_fn, _dead_tiers
    _ALL_TIERS = _build_tier_list()
    _dead_tiers.clear()

    test_prompt = "Respond with exactly: OK"
    for tier_name, tier_fn in _ALL_TIERS:
        try:
            logger.info(f"[LLM Init] Probing {tier_name}...")
            await tier_fn(test_prompt, "router", None)
            _active_tier_name = tier_name
            _active_tier_fn = tier_fn
            logger.info(f"[LLM Init] === LOCKED onto {tier_name} as primary LLM ===")
            return tier_name
        except Exception as e:
            logger.warning(f"[LLM Init] {tier_name} probe failed: {e}")
            _dead_tiers.add(tier_name)

    raise AllLLMTiersFailedError("No LLM tier responded during startup probe.")


def _switch_to_next_tier() -> bool:
    """
    Mark the current active tier as dead and switch to the next available one.

    Returns:
        True if a new tier was found, False if all tiers are dead.
    """
    global _active_tier_name, _active_tier_fn
    _dead_tiers.add(_active_tier_name)
    logger.warning(f"[LLM] Tier '{_active_tier_name}' marked DEAD. Searching for next tier...")

    for tier_name, tier_fn in _ALL_TIERS:
        if tier_name not in _dead_tiers:
            _active_tier_name = tier_name
            _active_tier_fn = tier_fn
            logger.info(f"[LLM] === SWITCHED to {tier_name} as new primary LLM ===")
            return True

    logger.critical("[LLM] ALL LLM tiers are DEAD. No fallback available.")
    return False


async def call_llm(prompt: str, purpose: str = "router", image_bytes: bytes | None = None) -> str:
    """
    Call the LLM tier.
    For vision tasks (purpose == "vision"), use Gemini directly.
    For text tasks (router, security, arbiter), use sticky priority: Groq -> Ollama -> Gemini.

    Args:
        prompt: The text prompt to send.
        purpose: One of 'router', 'security', 'arbiter', 'vision'.
        image_bytes: Optional raw image bytes for vision tasks.

    Returns:
        The LLM response text.
    """
    # Direct Gemini for Vision/Images
    if purpose == "vision" or image_bytes is not None:
        if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
            try:
                async with _api_rate_lock:
                    logger.info(f"[LLM] Calling Gemini (vision)...")
                    result = await _call_gemini(prompt, "vision", image_bytes)
                    logger.info(f"[LLM] Gemini (vision) succeeded. Cooldown {API_CALL_DELAY_SECONDS}s...")
                    await asyncio.sleep(API_CALL_DELAY_SECONDS)
                return result
            except Exception as e:
                logger.warning(f"[LLM] Gemini (vision) failed: {e}. Falling back to active tier...")

    if not _active_tier_fn:
        raise AllLLMTiersFailedError("No active LLM tier. Call init_llm_tier() first.")

    try:
        async with _api_rate_lock:
            logger.info(f"[LLM] Calling {_active_tier_name} ({purpose})...")
            result = await _active_tier_fn(prompt, purpose, image_bytes)
            logger.info(f"[LLM] {_active_tier_name} ({purpose}) succeeded. Cooldown {API_CALL_DELAY_SECONDS}s...")
            await asyncio.sleep(API_CALL_DELAY_SECONDS)
        return result
    except Exception as e:
        logger.error(f"[LLM] {_active_tier_name} ({purpose}) FAILED: {e}")
        # Switch to next tier
        if _switch_to_next_tier():
            raise LLMTierSwitchedError(
                f"Tier '{_active_tier_name}' is now active after previous tier failed. Retry the message."
            ) from e
        else:
            raise AllLLMTiersFailedError(
                f"All LLM tiers exhausted. Last error: {e}"
            ) from e


def _convert_audio_with_ffmpeg(audio_path: str, target_format: str = "wav") -> str | None:
    """
    Use FFmpeg to convert audio to a target format (e.g., wav).
    Useful when APIs require specific formats or for local whisper.
    
    Args:
        audio_path: Path to the source audio file.
        target_format: Output format (default: wav).
    
    Returns:
        Path to the converted file, or None if FFmpeg not available.
    """
    import subprocess
    import tempfile

    output_path = audio_path.rsplit(".", 1)[0] + f".{target_format}"
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1", output_path],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            logger.info(f"[FFmpeg] Converted {audio_path} → {output_path}")
            return output_path
        else:
            logger.warning(f"[FFmpeg] Conversion failed: {result.stderr[:200]}")
    except FileNotFoundError:
        logger.warning("[FFmpeg] ffmpeg not found on PATH. Skipping conversion.")
    except Exception as e:
        logger.warning(f"[FFmpeg] Error: {e}")
    return None


async def call_whisper_transcription(audio_path: str) -> str:
    """
    Transcribe audio using 3-tier fallback chain:
        Tier 1: Groq Whisper API (whisper-large-v3 — highest accuracy for multilingual & Hinglish)
        Tier 2: AssemblyAI API (cloud fallback)
        Tier 3: FFmpeg conversion + local whisper library

    Args:
        audio_path: Path to the .mp3 audio file.

    Returns:
        Transcribed text string.
    """
    # Tier 1: Groq Whisper API (whisper-large-v3) — Best for multilingual / code-switched Hinglish
    if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
        try:
            import httpx
            logger.info(f"[Tier 1 - Groq Whisper] Transcribing {os.path.basename(audio_path)}...")
            url = "https://api.groq.com/openai/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
            async with httpx.AsyncClient(timeout=60.0) as client:
                with open(audio_path, "rb") as f:
                    files = {"file": (os.path.basename(audio_path), f, "audio/mpeg")}
                    data = {
                        "model": GROQ_WHISPER_MODEL,
                        "response_format": "text",
                        "prompt": "Transcribe audio accurately in original language or Hinglish/code-switched speech."
                    }
                    resp = await client.post(url, headers=headers, files=files, data=data)
                    resp.raise_for_status()
                    text = resp.text.strip()
                    if text:
                        logger.info(f"[Tier 1 - Groq Whisper] Transcribed {len(text)} chars")
                        return text
        except Exception as e:
            logger.warning(f"[Tier 1 - Groq Whisper] Transcription failed: {e}. Falling back to AssemblyAI...")

    # Tier 2: AssemblyAI API
    if ASSEMBLYAI_API_KEY and ASSEMBLYAI_API_KEY != "your_assemblyai_api_key_here":
        try:
            import httpx

            headers = {"authorization": ASSEMBLYAI_API_KEY}

            # Step 1: Upload audio file to AssemblyAI
            logger.info(f"[Tier 2 - AssemblyAI] Uploading {os.path.basename(audio_path)}...")
            async with httpx.AsyncClient(timeout=60.0) as client:
                with open(audio_path, "rb") as f:
                    upload_resp = await client.post(
                        "https://api.assemblyai.com/v2/upload",
                        headers=headers,
                        content=f.read(),
                    )
                    upload_resp.raise_for_status()
                    upload_url = upload_resp.json()["upload_url"]

                # Step 2: Submit transcription request (with automatic multi-language detection)
                transcript_resp = await client.post(
                    "https://api.assemblyai.com/v2/transcript",
                    headers=headers,
                    json={
                        "audio_url": upload_url,
                        "speech_model": ASSEMBLYAI_MODEL,
                        "language_detection": True,
                    },
                )
                transcript_resp.raise_for_status()
                transcript_id = transcript_resp.json()["id"]

                # Step 3: Poll for completion
                poll_url = f"https://api.assemblyai.com/v2/transcript/{transcript_id}"
                import asyncio
                for _ in range(60):  # max 60 polls (~60s)
                    poll_resp = await client.get(poll_url, headers=headers)
                    poll_data = poll_resp.json()
                    status = poll_data.get("status", "")

                    if status == "completed":
                        text = poll_data.get("text", "").strip()
                        if text:
                            logger.info(f"[Tier 2 - AssemblyAI] Transcribed {len(text)} chars")
                            return text
                        else:
                            break
                    elif status == "error":
                        raise RuntimeError(f"AssemblyAI error: {poll_data.get('error', 'unknown')}")

                    await asyncio.sleep(1)

        except Exception as e:
            logger.warning(f"[Tier 2 - AssemblyAI] Transcription failed: {e}")

    # Tier 3: FFmpeg conversion + local whisper library
    try:
        logger.info(f"[Tier 3 - FFmpeg + Local Whisper] Processing {os.path.basename(audio_path)}...")
        wav_path = _convert_audio_with_ffmpeg(audio_path, "wav")
        transcribe_path = wav_path if wav_path else audio_path

        import whisper
        model = whisper.load_model("base")
        result = model.transcribe(transcribe_path)
        text = result["text"].strip()
        logger.info(f"[Tier 3 - Local Whisper] Transcribed {len(text)} chars")

        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)

        return text
    except Exception as e:
        logger.warning(f"[Tier 3 - FFmpeg + Local Whisper] Transcription failed: {e}")

    logger.error(f"[Audio] All transcription tiers failed for {audio_path}")
    return "[AUDIO_TRANSCRIPTION_UNAVAILABLE]"


# ---------------------------------------------------------------------------
# Logging helper (AGENTS.md §5.2 compliance)
# ---------------------------------------------------------------------------
def append_log(title: str, summary: str, actions: str = ""):
    """Append a per-turn log entry to the mandatory log file."""
    from datetime import datetime, timezone
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    entry = f"""
## [{timestamp}] {title[:80]}

Agent Response Summary:
{summary}

Actions:
* {actions}

Context:
tool=NotificationRouter
repo_root={PROJECT_ROOT}
"""
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry)
