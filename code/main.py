"""
main.py — Entry Point: WhatsApp Message Notification Router
=============================================================
3-Tier Hierarchical Multi-Agent System
    Tier 1: Agent 4 (Supervisor & Feedback Arbiter)
    Tier 2: Agent 1 (Perception), Agent 2 (Memory RAG), Agent 3 (Router Security)
    Tier 3: 11 Specialized Sub-Agents

Pipeline:
    1. Build pre-indexes from CSV datasets (O(1) hash maps)
    2. Build TF-IDF index from message_history.csv
    3. Load incoming messages from dataset/messages.csv
    4. Process all messages through the 3-Tier pipeline
    5. Write predictions to dataset/output.csv

Usage:
    python code/main.py
"""

import csv
import sys
import time
import asyncio
import logging
from pathlib import Path

# Ensure code/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATASET_DIR, OUTPUT_PATH, LOG_DIR, LOG_FILE, append_log, logger, init_llm_tier, AllLLMTiersFailedError
from indexer import build_indexes, get_user_history, HISTORY_INDEX
from agent2_memory_rag import build_tfidf_index
from agent4_supervisor import process_all_messages, write_output_csv


def load_messages() -> list[dict]:
    """Load incoming messages from dataset/messages.csv."""
    messages_path = DATASET_DIR / "messages.csv"
    if not messages_path.exists():
        logger.error(f"Messages file not found: {messages_path}")
        sys.exit(1)

    with open(messages_path, "r", encoding="utf-8") as f:
        messages = list(csv.DictReader(f))

    logger.info(f"Loaded {len(messages)} incoming messages from {messages_path}")
    return messages


async def run_pipeline():
    """Execute the full multi-agent routing pipeline."""
    start_time = time.time()

    logger.info("=" * 70)
    logger.info("WhatsApp Message Notification Router — Starting Pipeline")
    logger.info("=" * 70)

    # Step 1: Build pre-indexes from CSV datasets
    logger.info("\n[PHASE 1] Building pre-indexes from CSV datasets...")
    phase1_start = time.time()
    build_indexes()
    phase1_time = time.time() - phase1_start
    logger.info(f"[PHASE 1] Pre-indexing complete in {phase1_time:.2f}s")

    # Step 2: Build TF-IDF index from message history
    logger.info("\n[PHASE 2] Building TF-IDF index from message_history.csv...")
    phase2_start = time.time()
    all_history = []
    for user_msgs in HISTORY_INDEX.values():
        all_history.extend(user_msgs)
    build_tfidf_index(all_history)
    phase2_time = time.time() - phase2_start
    logger.info(f"[PHASE 2] TF-IDF index built in {phase2_time:.2f}s")

    # Step 3: Probe & lock onto the best LLM API (Groq > Ollama > Gemini)
    logger.info("\n[PHASE 3] Probing LLM APIs (Groq > Ollama > Gemini)...")
    phase3_start = time.time()
    try:
        active_tier = await init_llm_tier()
    except AllLLMTiersFailedError as e:
        logger.critical(f"[PHASE 3] FATAL: {e}")
        sys.exit(1)
    phase3_time = time.time() - phase3_start
    logger.info(f"[PHASE 3] Locked onto '{active_tier}' in {phase3_time:.2f}s")

    # Step 4: Load incoming messages
    logger.info("\n[PHASE 4] Loading incoming messages...")
    messages = load_messages()

    # Step 5: Process all messages through the 3-Tier pipeline (streams rows directly to output.csv)
    logger.info(f"\n[PHASE 5] Processing {len(messages)} messages through 3-Tier Multi-Agent pipeline (Streaming to {OUTPUT_PATH})...")
    phase5_start = time.time()
    results = await process_all_messages(messages)
    phase5_time = time.time() - phase5_start
    logger.info(f"[PHASE 5] All messages processed & written to {OUTPUT_PATH} in {phase5_time:.2f}s")

    # Summary
    total_time = time.time() - start_time
    action_counts = {}
    type_counts = {}
    for r in results:
        action_counts[r["action"]] = action_counts.get(r["action"], 0) + 1
        type_counts[r["message_type"]] = type_counts.get(r["message_type"], 0) + 1

    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total messages processed: {len(results)}")
    logger.info(f"Total time: {total_time:.2f}s ({total_time/max(len(results),1):.2f}s per message)")
    logger.info(f"Action distribution: {action_counts}")
    logger.info(f"Type distribution: {type_counts}")
    logger.info(f"Output written to: {OUTPUT_PATH}")

    # AGENTS.md compliance: append log entry
    append_log(
        title="Pipeline Execution Complete",
        summary=(
            f"Processed {len(results)} messages in {total_time:.2f}s. "
            f"Actions: {action_counts}. Types: {type_counts}."
        ),
        actions=f"Wrote {len(results)} rows to {OUTPUT_PATH}",
    )

    return results


def main():
    """Main entry point."""
    # Ensure log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Run the async pipeline
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    results = asyncio.run(run_pipeline())
    return results


if __name__ == "__main__":
    main()
