"""
audit_messages.py — Detailed Audit Script
===========================================
Processes a batch of messages from messages.csv and prints:
- Incoming Message Text
- Context Features (DND, business verification, group admin, opt-out)
- Evidence Message IDs & Evidence Text from message_history.csv
- LLM Verdict (action, message_type, confidence)
- LLM Reason Narrative
- CRITIQUE: Evaluates whether the verdict makes total sense or is wrong!
"""

import sys
import csv
import asyncio
from pathlib import Path

# Ensure code/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from indexer import build_indexes, HISTORY_INDEX, USER_INDEX, BUSINESS_INDEX, GROUP_INDEX
from agent1_perception import process_perception
from agent2_memory_rag import build_tfidf_index, process_memory_rag
from agent3_router_security import process_router_security
from agent4_supervisor import process_single_message, _generate_rag_hint, sub_agent_4_1_consensus_check

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

async def run_audit():
    print("Building indexes...")
    build_indexes()
    
    all_history = []
    for user_msgs in HISTORY_INDEX.values():
        all_history.extend(user_msgs)
    build_tfidf_index(all_history)
    
    with open(DATASET_DIR / "messages.csv", "r", encoding="utf-8") as f:
        messages = list(csv.DictReader(f))
        
    hist = {r["message_id"]: r for r in all_history}
    
    # Audit 8 representative messages across personal, group, business
    selected_ids = ["msg_023", "msg_091", "msg_090", "msg_048", "msg_040", "msg_001", "msg_004", "msg_007"]
    audit_batch = [m for m in messages if m["message_id"] in selected_ids]
    if not audit_batch:
        audit_batch = messages[:8]
        
    print(f"\n======================================================================")
    print(f"AUDITING {len(audit_batch)} MESSAGES: TEXT vs HISTORY vs VERDICT")
    print(f"======================================================================\n")
    
    for i, msg in enumerate(audit_batch, 1):
        mid = msg["message_id"]
        conv = msg["conversation_type"]
        uid = msg["user_id"]
        sender = msg.get("sender_user_id") or msg.get("business_id") or msg.get("group_id")
        
        # Run perception + memory RAG (deterministic, no LLM required)
        perceived_text = await process_perception(msg)
        mem = process_memory_rag(msg, perceived_text)
        ctx = mem["context"]
        ev_ids = mem["evidence_message_ids"]
        ev_list = mem["evidence"]
        
        rag_hint = _generate_rag_hint(ctx, ev_list)
        
        print(f"[{i}/{len(audit_batch)}] MESSAGE ID: {mid}")
        print(f"User: {uid} (DND: {ctx.get('user_dnd_window', 'None')}) | Conv Type: {conv} | Sender: {sender}")
        
        if conv == "business":
            print(f"Business: {ctx.get('business_name')} (Verified: {ctx.get('business_verified')}, OffDomain: {ctx.get('business_official_domain')}, SentDomain: {ctx.get('business_domain_used')}, DomainMismatch: {ctx.get('domain_mismatch')}, AllowsPromos: {ctx.get('allows_promotions')})")
        elif conv == "group":
            print(f"Group: {ctx.get('group_name')} (User Muted: {ctx.get('user_group_muted')}, Sender Role: {ctx.get('sender_role_in_group')})")
            
        print(f"Message Text: {repr(msg.get('message_text'))}")
        print(f"Evidence IDs Retrieved: {ev_ids}")
        
        if ev_list:
            for e in ev_list[:2]:
                print(f"  -> Evidence [{e['message_id']}]: {repr(e['text'])}")
        else:
            print("  -> Evidence: None found in message_history.csv")
            
        print(f"RAG Data Rule Hint: {rag_hint}")
        print("-" * 60)
        
if __name__ == "__main__":
    asyncio.run(run_audit())
