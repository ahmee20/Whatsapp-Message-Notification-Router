"""
agent1_perception.py — Agent 1: Perception Worker Agent
========================================================
Extracts text content from multimodal messages (images & voice notes).
Runs BEFORE the routing pipeline to normalize all messages into text.

Sub-Agents:
    1.1 — Vision OCR Sub-Agent (image → text)
    1.2 — Audio Whisper Sub-Agent (voice note → text)
"""

import os
import logging
from pathlib import Path

from config import DATASET_DIR, call_llm, call_whisper_transcription, LLMTierSwitchedError, AllLLMTiersFailedError

logger = logging.getLogger("Agent1_Perception")


##------------------------------Sub-Agent 1.1 - Vision OCR: Extract text from images------------------##

def _run_easyocr_tesseract_fallback(abs_path: Path) -> str:
    """Traditional OCR fallback using EasyOCR and Pytesseract if Gemini Vision API fails."""
    logger.info(f"[Sub-Agent 1.1] Invoking EasyOCR + Tesseract fallback for {abs_path.name}...")
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False)
        results = reader.readtext(str(abs_path))
        lines = [item[1] for item in results if isinstance(item, (list, tuple)) and len(item) >= 2]
        text = ' '.join(line.strip() for line in lines if line and line.strip())
        if text:
            logger.info(f"[Sub-Agent 1.1] EasyOCR extracted {len(text)} chars from {abs_path.name}")
            return f"[EXTRACTED_TEXT]: {text}\n[SCENE_DESCRIPTION]: Text image processed via EasyOCR fallback."
    except Exception as e:
        logger.warning(f"[Sub-Agent 1.1] EasyOCR failed: {e}")

    try:
        import pytesseract
        from PIL import Image
        text = pytesseract.image_to_string(Image.open(abs_path))
        if text and text.strip():
            logger.info(f"[Sub-Agent 1.1] Tesseract extracted {len(text.strip())} chars from {abs_path.name}")
            return f"[EXTRACTED_TEXT]: {text.strip()}\n[SCENE_DESCRIPTION]: Text image processed via Tesseract fallback."
    except Exception as e:
        logger.warning(f"[Sub-Agent 1.1] Tesseract failed: {e}")

    return "[EXTRACTED_TEXT]: NONE\n[SCENE_DESCRIPTION]: Traditional OCR fallback produced no text."


async def sub_agent_1_1_vision_ocr(image_path: str) -> str:
    """
    Sub-Agent 1.1 — Vision OCR Sub-Agent
    
    Converts image to Base64, sends to Gemini 3.5 Flash / Vision LLM to extract text & scene description.
    Retries on the CURRENT image if API fails.
    Falls back to EasyOCR + Tesseract if Gemini Vision API fails after retries.
    
    Args:
        image_path: Absolute path to the .jpg image file.
    
    Returns:
        Structured string with [EXTRACTED_TEXT] and [SCENE_DESCRIPTION].
    """
    import base64
    import asyncio
    abs_path = Path(image_path)
    if not abs_path.exists():
        logger.warning(f"[Sub-Agent 1.1] Image not found: {image_path}")
        return "[EXTRACTED_TEXT]: NONE\n[SCENE_DESCRIPTION]: Image file missing."

    logger.info(f"[Sub-Agent 1.1] Processing image with Gemini Vision: {abs_path.name}")

    image_bytes = abs_path.read_bytes()
    b64_str = base64.b64encode(image_bytes).decode("utf-8")

    prompt = (
        "You are a multimodal vision perception agent.\n"
        "Analyze this image carefully. The image might contain full text, minimal text, or no text at all.\n\n"
        "Extract and analyze the image into the following STRICT OUTPUT FORMAT:\n\n"
        "[EXTRACTED_TEXT]: <Extract ALL visible text from the image verbatim. Include headings, body text, prices, dates, phone numbers, URLs, and fine print. If no text exists in the image, write \"NONE\">\n"
        "[SCENE_DESCRIPTION]: <Provide a clear, detailed 1-2 sentence description of the overall visual scene, graphics, objects, setting, and context of the image.>\n\n"
        "Respond strictly in the above format with no additional commentary."
    )

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            result = await call_llm(prompt, purpose="vision", image_bytes=image_bytes)
            logger.info(f"[Sub-Agent 1.1] Gemini Vision output for {abs_path.name}: {len(result)} chars")
            return result.strip()
        except LLMTierSwitchedError as e:
            logger.warning(f"[Sub-Agent 1.1] LLM tier switched during Vision for {abs_path.name}. "
                           f"Retrying CURRENT image (attempt {attempt}/{max_retries}): {e}")
            continue
        except AllLLMTiersFailedError as e:
            logger.warning(f"[Sub-Agent 1.1] All LLM tiers failed for {abs_path.name}. Switching to EasyOCR + Tesseract fallback.")
            break
        except Exception as e:
            logger.error(f"[Sub-Agent 1.1] Gemini Vision failed for {abs_path.name} (attempt {attempt}/{max_retries}): {e}")
            if attempt < max_retries:
                await asyncio.sleep(2)  # Pause before retrying the exact CURRENT image
                continue

    # Fallback to EasyOCR + Tesseract on CURRENT image if Gemini Vision API retries fail
    logger.warning(f"[Sub-Agent 1.1] Gemini Vision exhausted retries for {abs_path.name}. Running EasyOCR + Tesseract fallback...")
    return _run_easyocr_tesseract_fallback(abs_path)


##------------------------------Sub-Agent 1.2 - Audio Whisper: Transcribe voice notes------------------##

async def sub_agent_1_2_audio_whisper(audio_path: str) -> str:
    """
    Sub-Agent 1.2 — Audio Whisper Sub-Agent
    
    Transcribes voice note .mp3 files into text using
    AssemblyAI (multilingual) → Groq Whisper API (99+ languages) → Local whisper library fallback.
    
    Args:
        audio_path: Absolute path to the .mp3 voice note file.
    
    Returns:
        Transcribed text from the voice note.
    """
    abs_path = Path(audio_path)
    if not abs_path.exists():
        logger.warning(f"[Sub-Agent 1.2] Audio file not found: {audio_path}")
        return "[AUDIO_NOT_FOUND]"

    logger.info(f"[Sub-Agent 1.2] Transcribing voice note: {abs_path.name}")

    try:
        result = await call_whisper_transcription(str(abs_path))
        logger.info(f"[Sub-Agent 1.2] Transcribed {len(result)} chars from {abs_path.name}")
        return result.strip()
    except Exception as e:
        logger.error(f"[Sub-Agent 1.2] Whisper transcription failed for {abs_path.name}: {e}")
        return "[AUDIO_TRANSCRIPTION_FAILED]"


##------------------------------Agent 1 Orchestrator - Process media for a message------------------##

async def process_perception(message: dict) -> str:
    """
    Agent 1 Main Orchestrator — Perception Worker Agent
    
    Determines the media type and dispatches to the appropriate
    sub-agent for text extraction. Returns the final unified text
    representation of the message.
    
    Args:
        message: A dict from messages.csv with all columns.
    
    Returns:
        The full text content of the message (original text + extracted media text).
    """
    message_text = message.get("message_text", "").strip()
    media_type = message.get("media_type", "").strip()
    media_id = message.get("media_id", "").strip()

    # If no media, return raw text as-is
    if not media_type or not media_id:
        return message_text

    extracted_text = ""

    if media_type == "image":
        # Resolve image path from indexer
        from indexer import get_image_path
        relative_path = get_image_path(media_id)
        if relative_path:
            abs_image_path = str(DATASET_DIR / relative_path)
            extracted_text = await sub_agent_1_1_vision_ocr(abs_image_path)
        else:
            logger.warning(f"[Agent 1] No image path found for media_id={media_id}")
            extracted_text = "[IMAGE_NOT_INDEXED]"

    elif media_type == "voice":
        # Resolve voice note path from indexer
        from indexer import get_voice_path
        relative_path = get_voice_path(media_id)
        if relative_path:
            abs_audio_path = str(DATASET_DIR / relative_path)
            extracted_text = await sub_agent_1_2_audio_whisper(abs_audio_path)
        else:
            logger.warning(f"[Agent 1] No voice path found for media_id={media_id}")
            extracted_text = "[VOICE_NOT_INDEXED]"

    # Combine original text (if any) with extracted media text
    parts = []
    if message_text:
        parts.append(f"[TEXT]: {message_text}")
    if extracted_text:
        if media_type == "image":
            parts.append(
                f"[IMAGE OCR EXTRACTED TEXT]:\n\"{extracted_text}\"\n"
                f"(NOTE FOR ROUTER: This portion of text was extracted from an image poster/screenshot using Vision OCR. "
                f"The pattern, layout, spacing, line breaks, or line order of the extracted text may be imperfect or unordered. "
                f"Focus on the underlying context, intent, offer, and operational/business purpose of the text rather than formatting.)"
            )
        else:
            parts.append(f"[VOICE NOTE TRANSCRIBED TEXT]: \"{extracted_text}\"")

    combined = "\n\n".join(parts) if parts else "[EMPTY_MESSAGE]"
    logger.info(f"[Agent 1] Message {message.get('message_id', '?')}: {len(combined)} chars total")
    return combined
