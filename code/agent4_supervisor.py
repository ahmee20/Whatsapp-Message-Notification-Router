"""
agent4_supervisor.py — Agent 4: Supervisor Leader & Feedback Arbiter
=====================================================================
Orchestrates the 3-Tier multi-agent pipeline. Spawns Agent 2 (RAG)
and Agent 3 (Router) in parallel, checks consensus, arbitrates
disagreements, validates output schema, and writes output.csv.

Sub-Agents:
    4.1 — Consensus Evaluator (compare RAG hints vs Router decision)
    4.2 — Feedback Arbiter (LLM tie-breaker on disagreement)
    4.3 — Output CSV Schema Validator (enforce 6-column contract)
"""

import sys
import re
import csv
import json
import asyncio
import logging

from config import call_llm, VALID_ACTIONS, VALID_MESSAGE_TYPES, OUTPUT_PATH, AllLLMTiersFailedError, LLMTierSwitchedError
from agent1_perception import process_perception
from agent2_memory_rag import process_memory_rag
from agent3_router_security import process_router_security, _extract_json

logger = logging.getLogger("Agent4_Supervisor")
_csv_write_lock = asyncio.Lock()
_written_ids: set[str] = set()  # Dedup guard: track message_ids already written to output.csv


##------------------------------Sub-Agent 4.1 - Consensus Evaluator: Compare RAG hints vs Router decision------------------##

def _generate_rag_hint(context: dict, evidence: list) -> dict | None:
    """
    Generate a lightweight routing hint from pure context features
    (no LLM call). This is the RAG agent's "vote" to compare
    against the Router LLM's decision.
    
    Returns:
        A hint dict with action and message_type, or None if inconclusive.
    """
    conv_type = context.get("conversation_type", "")

    # --- Business Branch ---
    if conv_type == "business":
        # Domain mismatch → scam
        if context.get("domain_mismatch"):
            return {"action": "mute", "message_type": "scam"}

        # Opted-out promotions
        if context.get("allows_promotions") == 0:
            return {"action": "mute", "message_type": "promotion"}

        # Active order / delivery
        why = context.get("why_user_knows_account", "")
        if why in ("active_order", "delivery", "booking", "appointment"):
            return {"action": "notify", "message_type": "business_update"}

        # Opted-in promotions
        if context.get("allows_promotions") == 1 and context.get("business_category", "") in (
                "retail", "ecommerce", "food", "travel", "entertainment"):
            return {"action": "digest", "message_type": "promotion"}

    # --- Group Branch ---
    elif conv_type == "group":
        # High forward count → mute
        if context.get("forwarded_count", 0) >= 5:
            return {"action": "mute", "message_type": "forward"}

        # Admin sender → likely notify
        if context.get("sender_role_in_group") == "admin":
            return {"action": "notify", "message_type": "event"}

    # --- Personal Branch ---
    elif conv_type == "personal":
        # Past fast reaction → likely important sender
        avg_rt = context.get("avg_reaction_time_min")
        if avg_rt is not None and avg_rt < 5:
            return {"action": "notify", "message_type": "personal"}

    return None  # Inconclusive — let Router decide


def sub_agent_4_1_consensus_check(rag_hint: dict | None, router_decision: dict) -> dict:
    """
    Sub-Agent 4.1 — Consensus Evaluator
    
    Compares RAG-based hint against Router LLM decision.
    
    Returns:
        A dict with:
            - 'agrees': bool
            - 'rag_hint': the RAG hint (or None)
            - 'router_decision': the router decision
            - 'final_confidence_boost': confidence adjustment
    """
    if rag_hint is None:
        # RAG inconclusive → trust Router with slight confidence reduction
        return {
            "agrees": True,  # No disagreement possible
            "rag_hint": None,
            "router_decision": router_decision,
            "confidence_adjustment": -0.02,
            "reason": "RAG context inconclusive; Router decision trusted.",
        }

    rag_action = rag_hint.get("action", "")
    router_action = router_decision.get("action", "")

    if rag_action == router_action:
        # Full agreement
        return {
            "agrees": True,
            "rag_hint": rag_hint,
            "router_decision": router_decision,
            "confidence_adjustment": +0.05,
            "reason": f"RAG and Router agree on action='{rag_action}'.",
        }
    else:
        # Disagreement → needs arbitration
        return {
            "agrees": False,
            "rag_hint": rag_hint,
            "router_decision": router_decision,
            "confidence_adjustment": -0.10,
            "reason": (
                f"DISAGREEMENT: RAG suggests action='{rag_action}' "
                f"but Router decided action='{router_action}'."
            ),
        }


##------------------------------Sub-Agent 4.2 - Feedback Arbiter: LLM tie-breaker------------------##

async def sub_agent_4_2_feedback_arbiter(
    message_text: str,
    context: dict,
    rag_hint: dict,
    router_decision: dict,
    evidence: list = None,
) -> dict:
    """
    Sub-Agent 4.2 — Feedback Arbiter & 3-Pass Iterative Re-evaluation Engine
    
    When Agent 2 (RAG) and Agent 3 (Router) disagree:
    1. Arbiter writes critique feedback explaining why candidates disagree.
    2. Feedback + context is injected back into Router LLM (sub_agent_3_2_router_llm) to re-evaluate!
    3. Re-evaluates consensus up to 3 times (3 passes).
    4. If consensus reached -> accept and exit!
    5. If 3 passes expire -> Arbiter makes final decision.
    """
    from agent3_router_security import sub_agent_3_2_router_llm

    current_decision = router_decision.copy()
    evidence_list = evidence or []

    for pass_idx in range(1, 4):  # Up to 3 iterations
        logger.info(f"[Sub-Agent 4.2] Feedback Loop Pass {pass_idx}/3 for msg {context.get('message_id', '?')}")

        prompt = f"""You are a Feedback Arbiter auditing an AI WhatsApp Notification Router.

DISAGREEMENT DETECTED:
- Candidate A (RAG Context Agent): action={rag_hint.get('action', '?')}, message_type={rag_hint.get('message_type', '?')}
- Candidate B (Router LLM Proposal): action={current_decision.get('action', '?')}, message_type={current_decision.get('message_type', '?')}, reason="{current_decision.get('reason', '?')}"

MESSAGE TEXT:
\"\"\"{message_text[:1000]}\"\"\"

CONTEXT:
- Business verified: {context.get('business_verified', 'N/A')}, Domain mismatch: {context.get('domain_mismatch', 'N/A')}
- Allows promotions: {context.get('allows_promotions', 'N/A')}, Forwarded count: {context.get('forwarded_count', 0)}

INSTRUCTIONS:
Provide a clear 2-sentence critique feedback explaining why Candidate A and Candidate B disagree and how Candidate B must adjust to follow system rules.

IMPORTANT: You MUST respond with ONLY a raw JSON object. No markdown code fences, no explanation, no text before or after the JSON.

Required JSON schema:
{{
    "feedback": "<string: your 2-sentence critique feedback>",
    "suggested_action": "<string: one of notify, digest, mute>",
    "suggested_message_type": "<string: the correct message type>"
}}
/no_think"""

        try:
            arbiter_resp = await call_llm(prompt, purpose="arbiter")
            parsed = _extract_json(arbiter_resp)
            critique_feedback = ""
            if parsed:
                critique_feedback = parsed.get("feedback", "")

            # INJECT FEEDBACK BACK INTO ROUTER LLM FOR RE-EVALUATION!
            logger.info(f"[Sub-Agent 4.2] Re-injecting Feedback into Router LLM (Pass {pass_idx})...")
            new_decision = await sub_agent_3_2_router_llm(
                message_text, context, evidence_list, feedback_critique=critique_feedback
            )

            # RE-CHECK CONSENSUS
            re_consensus = sub_agent_4_1_consensus_check(rag_hint, new_decision)
            if re_consensus["agrees"]:
                logger.info(f"[Sub-Agent 4.2] Pass {pass_idx}: CONSENSUS REACHED after feedback re-evaluation!")
                new_decision["confidence"] = max(0.0, min(1.0, new_decision["confidence"] + 0.05))
                new_decision["reason"] = f"[Feedback Pass {pass_idx}] {new_decision['reason']}"
                return new_decision

            current_decision = new_decision

        except (LLMTierSwitchedError, AllLLMTiersFailedError):
            raise  # Propagate tier errors up for retry
        except Exception as e:
            logger.warning(f"[Sub-Agent 4.2] Feedback loop pass {pass_idx} error: {e}")

    # If 3 passes expire and still no consensus, return final arbitrated decision
    logger.info(f"[Sub-Agent 4.2] 3 Feedback passes completed. Finalizing calibrated decision.")
    current_decision["reason"] = f"[Arbitrated 3-Pass] {current_decision.get('reason', '')}"
    return current_decision


##------------------------------Sub-Agent 4.3 - Output CSV Schema Validator------------------##

def _calculate_dynamic_confidence(row: dict) -> float:
    """
    Bayesian Dynamic Posterior Confidence Update:
    Dynamically updates the initial decision confidence P(Decision)
    using Bayes' Theorem based on multi-agent consensus likelihood and RAG evidence score.

    No predefined fixed criteria addition constants!
    """
    # Initial evaluated decision confidence P(Decision) from LLM / Security audit
    p_init = float(row.get("confidence", 0.85) or 0.85)
    p_init = max(0.05, min(0.95, p_init))

    reason_str = str(row.get("reason", "")).lower()
    method = str(row.get("method", "")).upper()

    # 1. Multi-Agent Consensus Likelihood (M_consensus)
    if "regex" in method or "security" in method or "personal_scam_check" in method:
        m_consensus = 1.05  # Deterministic guardrail validation
    elif "[arbitrated 3-pass]" in reason_str:
        m_consensus = 0.70  # Heavy multi-pass disagreement
    elif "[feedback pass 2]" in reason_str:
        m_consensus = 0.82  # Resolved pass 2
    elif "[feedback pass 1]" in reason_str:
        m_consensus = 0.92  # Resolved pass 1
    else:
        m_consensus = 1.02  # Direct multi-agent consensus (Pass 0)

    # 2. RAG Vector Evidence Likelihood (L_rag)
    top_sim = float(row.get("top_cosine_sim", 0.0) or 0.0)
    if top_sim > 0.0:
        l_rag = 1.0 + (0.30 * top_sim)  # Continuous likelihood scaling from TF-IDF Cosine Sim
    else:
        l_rag = 0.95

    # Bayesian Posterior Update Formula:
    # P(Decision | Evidence) = (P_init * M_consensus * L_rag) / (P_init * M_consensus * L_rag + (1 - P_init))
    numerator = p_init * m_consensus * l_rag
    denominator = numerator + (1.0 - p_init)

    posterior_p = numerator / max(1e-6, denominator)
    # Cap maximum confidence at 0.95 (95%)
    return min(0.95, round(posterior_p, 2))


def sub_agent_4_3_validate_output(row: dict) -> dict:
    """
    Sub-Agent 4.3 — Output CSV Schema Validator
    
    Enforces the exact 6-column output contract before writing.
    Calculates dynamic confidence and sanitizes all values.
    
    Args:
        row: A dict with routing decision fields.
    
    Returns:
        A validated and sanitized output row dict.
    """
    # Validate action
    action = row.get("action", "digest").lower().strip()
    if action not in VALID_ACTIONS:
        logger.warning(f"[Sub-Agent 4.3] Invalid action '{action}', defaulting to 'digest'")
        action = "digest"

    # Validate message_type
    msg_type = row.get("message_type", "unknown").lower().strip()
    if msg_type not in VALID_MESSAGE_TYPES:
        logger.warning(f"[Sub-Agent 4.3] Invalid message_type '{msg_type}', defaulting to 'unknown'")
        msg_type = "unknown"

    # Calculate 3-Signal Dynamic Confidence Score
    confidence = _calculate_dynamic_confidence(row)

    # Validate reason (non-empty string)
    reason = str(row.get("reason", "Automated routing decision.")).strip()
    if not reason:
        reason = "Automated routing decision."
    # Escape commas and quotes for CSV safety
    reason = reason.replace('"', "'")

    # Validate evidence_message_ids
    evidence = str(row.get("evidence_message_ids", "none")).strip()
    if not evidence or evidence.lower() == "nan":
        evidence = "none"

    validated = {
        "message_id": row.get("message_id", ""),
        "action": action,
        "message_type": msg_type,
        "reason": reason,
        "confidence": confidence,
        "evidence_message_ids": evidence,
    }

    logger.debug(f"[Sub-Agent 4.3] Validated output for {validated['message_id']}")
    return validated


##------------------------------Agent 4 Orchestrator - Full pipeline per message------------------##

async def process_single_message(message: dict) -> dict:
    """
    Agent 4 Main Orchestrator — Process a single message through
    the full 3-Tier pipeline.
    
    Pipeline:
        1. Agent 1 (Perception) → extract text from media
        2. Agent 2 (Memory RAG) → retrieve context + evidence
        3. Agent 3 (Router Security) → route with security check
        4. Sub-Agent 4.1 → consensus check
        5. Sub-Agent 4.2 → arbitrate if disagreement
        6. Sub-Agent 4.3 → validate output schema
    
    Args:
        message: A dict from messages.csv.
    
    Returns:
        A validated output row dict ready for CSV.
    """
    msg_id = message.get("message_id", "?")
    logger.info(f"[Agent 4] ===== Processing {msg_id} =====")

    # Step 1: Agent 1 — Perception (extract text from media)
    perceived_text = await process_perception(message)

    # Step 2: Agent 2 — Memory RAG (context + evidence)
    memory_package = process_memory_rag(message, perceived_text)
    context = memory_package["context"]
    evidence = memory_package["evidence"]
    evidence_ids = memory_package["evidence_message_ids"]

    # Add message_id to context for downstream use
    context["message_id"] = msg_id

    # Step 3: Agent 3 — Router Security (security audit + LLM routing)
    router_decision = await process_router_security(perceived_text, context, evidence)

    # Step 4: Sub-Agent 4.1 — Consensus check (RAG hint vs Router)
    rag_hint = _generate_rag_hint(context, evidence)
    consensus = sub_agent_4_1_consensus_check(rag_hint, router_decision)

    # Step 5: Resolve final decision
    if consensus["agrees"]:
        # Agreement path → use Router decision with confidence boost
        final = router_decision.copy()
        adj = consensus["confidence_adjustment"]
        final["confidence"] = max(0.0, min(1.0, final["confidence"] + adj))
        logger.info(f"[Agent 4] {msg_id}: Consensus AGREED (adj={adj:+.2f})")
    else:
        # Disagreement path → Sub-Agent 4.2 Arbiter 3-Pass Feedback Loop
        logger.info(f"[Agent 4] {msg_id}: DISAGREEMENT → invoking Arbiter 3-Pass Feedback Loop")
        final = await sub_agent_4_2_feedback_arbiter(
            perceived_text, context, rag_hint, router_decision, evidence=evidence
        )

    # Add message metadata & continuous signals for dynamic confidence
    final["message_id"] = msg_id
    final["evidence_message_ids"] = evidence_ids
    final["conversation_type"] = context.get("conversation_type", "")
    final["top_cosine_sim"] = memory_package.get("top_cosine_sim", 0.0)
    final["past_opened_ratio"] = context.get("past_opened_ratio", 0.5)
    final["past_reported_count"] = context.get("past_reported_count", 0)

    # Step 6: Sub-Agent 4.3 — Schema validation & dynamic confidence calculation
    validated = sub_agent_4_3_validate_output(final)

    logger.info(f"[Agent 4] {msg_id}: FINAL → action={validated['action']}, "
                f"type={validated['message_type']}, conf={validated['confidence']:.2f}")
    return validated


async def _append_row_to_csv(row: dict, fieldnames: list[str]) -> None:
    """Thread-safe incremental CSV row writer with dedup guard."""
    mid = row.get("message_id", "")
    async with _csv_write_lock:
        if mid in _written_ids:
            logger.warning(f"[Agent 4] DEDUP: Skipping duplicate write for {mid}")
            return
        _written_ids.add(mid)
        with open(OUTPUT_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)
            f.flush()


async def process_all_messages(messages: list[dict]) -> list[dict]:
    """
    Process all incoming messages through the 3-Tier pipeline in parallel.
    Writes predictions incrementally (streaming) to dataset/output.csv as
    each message finishes.
    
    If all LLM APIs fail, execution halts immediately with a fatal error.
    
    Args:
        messages: List of message dicts from messages.csv.
    
    Returns:
        List of validated output row dicts.
    """
    logger.info(f"[Agent 4] Starting parallel batch processing of {len(messages)} messages...")

    fieldnames = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]

    # Reset dedup guard and overwrite output.csv with header at startup
    _written_ids.clear()
    try:
        with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
        logger.info(f"[Agent 4] Initialized {OUTPUT_PATH} for streaming writes ({len(messages)} messages).")
    except PermissionError:
        logger.error(f"[Agent 4] PermissionError: Cannot write header to {OUTPUT_PATH}. Close the file if open in Excel!")
        sys.exit(1)

    # Process messages with controlled concurrency (1 message at a time to strictly obey rate limits)
    semaphore = asyncio.Semaphore(1)

    async def _process_with_semaphore(msg):
        async with semaphore:
            max_retries = 2  # allow up to 2 retries on tier switch
            for attempt in range(1, max_retries + 2):
                try:
                    res = await process_single_message(msg)
                    # STREAMING WRITE: Write to output.csv immediately upon completion!
                    await _append_row_to_csv(res, fieldnames)
                    return res
                except LLMTierSwitchedError as e:
                    logger.warning(f"[Agent 4] LLM tier switched during {msg.get('message_id', '?')}. "
                                   f"Retrying message (attempt {attempt}/{max_retries + 1}): {e}")
                    continue  # retry the same message with the new tier
                except AllLLMTiersFailedError as e:
                    logger.critical(f"[Agent 4] FATAL ERROR: {e}. ALL LLM APIs FAILED! ABORTING PIPELINE IMMEDIATELY.")
                    sys.exit(1)
                except Exception as e:
                    logger.error(f"[Agent 4] Error processing {msg.get('message_id', '?')}: {e}")
                    fallback = sub_agent_4_3_validate_output({
                        "message_id": msg.get("message_id", ""),
                        "action": "digest",
                        "message_type": "unknown",
                        "reason": f"Processing error: {str(e)[:100]}. Defaulting to digest.",
                        "confidence": 0.3,
                        "evidence_message_ids": "none",
                    })
                    await _append_row_to_csv(fallback, fieldnames)
                    return fallback
            # If all retries exhausted after tier switches
            logger.error(f"[Agent 4] Exhausted retries for {msg.get('message_id', '?')} after tier switches.")
            fallback = sub_agent_4_3_validate_output({
                "message_id": msg.get("message_id", ""),
                "action": "digest",
                "message_type": "unknown",
                "reason": "LLM tier switched multiple times; defaulting to digest.",
                "confidence": 0.3,
                "evidence_message_ids": "none",
            })
            await _append_row_to_csv(fallback, fieldnames)
            return fallback

    results = await asyncio.gather(*[_process_with_semaphore(msg) for msg in messages])
    logger.info(f"[Agent 4] Batch processing complete. {len(results)} results written to {OUTPUT_PATH}.")
    return list(results)


def write_output_csv(results: list[dict]) -> None:
    """
    Write final results to dataset/output.csv.
    Handles PermissionError (e.g., file open in Excel) with retry + fallback.
    
    Args:
        results: List of validated output row dicts.
    """
    import time
    fieldnames = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]

    # Try writing with retries
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in results:
                    writer.writerow(row)
            logger.info(f"[Agent 4] Output written to {OUTPUT_PATH} ({len(results)} rows)")
            return
        except PermissionError:
            if attempt < max_retries - 1:
                logger.warning(f"[Agent 4] PermissionError writing {OUTPUT_PATH}. "
                               f"Close the file if open in Excel. Retrying in 3s... "
                               f"(attempt {attempt + 1}/{max_retries})")
                time.sleep(3)
            else:
                # Fallback: write to a different file
                fallback_path = OUTPUT_PATH.parent / "output_new.csv"
                logger.warning(f"[Agent 4] Cannot write to {OUTPUT_PATH}. "
                               f"Writing to fallback: {fallback_path}")
                with open(fallback_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for row in results:
                        writer.writerow(row)
                logger.info(f"[Agent 4] Output written to FALLBACK {fallback_path} ({len(results)} rows)")
                logger.info(f"[Agent 4] ⚠️  Close output.csv and rename output_new.csv → output.csv")
