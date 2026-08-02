"""
agent2_memory_rag.py — Agent 2: Context & RAG Memory Agent
===========================================================
Retrieves all contextual features for a message from pre-indexed
hash maps (O(1)) and performs vector similarity search against
message_history.csv to find relevant evidence_message_ids.

Sub-Agents:
    2.1 — Relational Hash Indexer Sub-Agent (O(1) feature retrieval)
    2.2 — Vector Similarity Matcher Sub-Agent (TF-IDF cosine search)
"""

import re
import math
import logging
from collections import Counter, defaultdict

from indexer import (
    get_user, get_group, get_membership, get_sender_membership,
    get_business, get_user_business, get_user_history,
    get_sender_history, get_event, get_user_events,
    get_daily_summaries,
)

logger = logging.getLogger("Agent2_MemoryRAG")


##------------------------------Sub-Agent 2.1 - Relational Hash Indexer: O(1) feature retrieval------------------##

def sub_agent_2_1_hash_lookup(message: dict) -> dict:
    """
    Sub-Agent 2.1 — Relational Hash Indexer Sub-Agent
    
    Performs O(1) hash map lookups to retrieve all contextual
    features for a message from pre-indexed CSV datasets.
    
    Args:
        message: A dict from messages.csv.
    
    Returns:
        A context dict with all joined features.
    """
    user_id = message.get("user_id", "")
    group_id = message.get("group_id", "")
    business_id = message.get("business_id", "")
    sender_user_id = message.get("sender_user_id", "")
    conversation_type = message.get("conversation_type", "")

    context = {
        "conversation_type": conversation_type,
        "user_id": user_id,
        "group_id": group_id,
        "business_id": business_id,
        "sender_user_id": sender_user_id,
        "forwarded_count": int(message.get("forwarded_count", 0) or 0),
        "created_at": message.get("created_at", ""),
    }

    # --- User Profile ---
    user = get_user(user_id)
    if user:
        context["user_dnd_window"] = user.get("do_not_disturb_window", "")
        context["user_opened_30d"] = int(user.get("messages_opened_30d", 0) or 0)
        context["user_replied_30d"] = int(user.get("messages_replied_30d", 0) or 0)
        context["user_dismissed_30d"] = int(user.get("notifications_dismissed_30d", 0) or 0)
        context["user_reported_30d"] = int(user.get("messages_reported_30d", 0) or 0)
    else:
        context["user_dnd_window"] = ""
        context["user_opened_30d"] = 0
        context["user_replied_30d"] = 0
        context["user_dismissed_30d"] = 0
        context["user_reported_30d"] = 0

    # --- Personal Sender History Check ---
    if conversation_type == "personal" and sender_user_id:
        from indexer import get_user_history
        past_msgs = get_user_history(user_id)
        # Filter messages involving sender_user_id
        pair_history = [
            m for m in past_msgs
            if m.get("sender_user_id") == sender_user_id or m.get("user_id") == sender_user_id
        ]
        context["history_count"] = len(pair_history)

        # Check if past history contains prior precedent of sharing codes, OTPs, or verification links
        otp_keywords = ["otp", "code", "pin", "password", "link", "verify", "login"]
        has_otp = False
        for pm in pair_history:
            pm_text = (pm.get("message_text") or "").lower()
            if any(kw in pm_text for kw in otp_keywords):
                has_otp = True
                break
        context["has_prior_otp_evidence"] = has_otp
    else:
        context["history_count"] = 0
        context["has_prior_otp_evidence"] = False

    # --- Group Context (if group message) ---
    if conversation_type == "group" and group_id:
        group = get_group(group_id)
        if group:
            context["group_name"] = group.get("group_name", "")
            context["group_type"] = group.get("group_type", "")
            context["group_member_count"] = int(group.get("member_count", 0) or 0)

        # Recipient's membership in the group
        membership = get_membership(group_id, user_id)
        if membership:
            context["user_role_in_group"] = membership.get("role", "member")
            context["user_group_muted"] = int(membership.get("group_muted_by_user", 0) or 0)
            context["user_group_dismissed_30d"] = int(membership.get("notifications_dismissed_30d", 0) or 0)
        else:
            context["user_role_in_group"] = "unknown"
            context["user_group_muted"] = 0
            context["user_group_dismissed_30d"] = 0

        # Sender's role in the group
        if sender_user_id:
            sender_membership = get_sender_membership(group_id, sender_user_id)
            if sender_membership:
                context["sender_role_in_group"] = sender_membership.get("role", "member")
            else:
                context["sender_role_in_group"] = "unknown"
        else:
            context["sender_role_in_group"] = "unknown"

    # --- Business Context (if business message) ---
    if conversation_type == "business" and business_id:
        business = get_business(business_id)
        if business:
            context["business_name"] = business.get("display_name", "")
            context["business_brand"] = business.get("brand_name", "")
            context["business_category"] = business.get("category", "")
            context["business_verified"] = int(business.get("verified", 0) or 0)
            context["business_official_domain"] = business.get("official_domain", "").strip()
            context["business_domain_used"] = business.get("domain_used_by_sender", "").strip()
            context["business_reports_30d"] = int(business.get("user_reports_30d", 0) or 0)
            context["business_domain_age_days"] = int(business.get("domain_used_by_sender_age_days", 0) or 0)

            # Domain mismatch check (critical for scam detection)
            official = context["business_official_domain"].lower()
            used = context["business_domain_used"].lower()
            context["domain_mismatch"] = (official != used and official != "" and used != "")
        else:
            context["domain_mismatch"] = False

        # User-business relationship
        user_biz = get_user_business(user_id, business_id)
        if user_biz:
            context["why_user_knows_account"] = user_biz.get("why_user_knows_account", "")
            context["allows_promotions"] = int(user_biz.get("allows_promotions", 1) or 1)
            context["biz_messages_opened_30d"] = int(user_biz.get("messages_opened_30d", 0) or 0)
            context["biz_messages_dismissed_30d"] = int(user_biz.get("messages_dismissed_30d", 0) or 0)
        else:
            context["why_user_knows_account"] = ""
            context["allows_promotions"] = 1  # default: allow
            context["biz_messages_opened_30d"] = 0
            context["biz_messages_dismissed_30d"] = 0

    # --- Daily Notification Load ---
    summaries = get_daily_summaries(user_id)
    if summaries:
        recent = summaries[-1]  # most recent day
        context["daily_notifs_sent"] = int(recent.get("notifications_sent", 0) or 0)
        context["daily_notifs_dismissed"] = int(recent.get("notifications_dismissed", 0) or 0)

    logger.info(f"[Sub-Agent 2.1] Context assembled for {message.get('message_id', '?')} "
                f"(type={conversation_type}, features={len(context)})")
    return context


##------------------------------Sub-Agent 2.2 - Vector Similarity Matcher: TF-IDF evidence search------------------##

class TFIDFMatcher:
    """
    Lightweight TF-IDF cosine similarity matcher for finding
    relevant historical messages as evidence. No external
    dependencies required — pure Python implementation.
    """

    def __init__(self):
        self._idf_cache: dict[str, float] = {}
        self._doc_count = 0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + punctuation tokenizer."""
        text = text.lower()
        tokens = re.findall(r'[a-z0-9]+', text)
        # Remove common stop words
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'shall',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'it', 'this', 'that', 'these', 'those', 'i', 'me', 'my',
            'we', 'our', 'you', 'your', 'he', 'she', 'they', 'them',
            'and', 'or', 'but', 'if', 'so', 'as', 'not', 'no', 'just',
        }
        return [t for t in tokens if t not in stop_words and len(t) > 1]

    @staticmethod
    def _tf(tokens: list[str]) -> dict[str, float]:
        """Compute term frequency."""
        counts = Counter(tokens)
        total = len(tokens) if tokens else 1
        return {term: count / total for term, count in counts.items()}

    def build_idf(self, documents: list[str]) -> None:
        """Compute IDF from a corpus of document strings."""
        self._doc_count = len(documents)
        df = defaultdict(int)
        for doc in documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                df[token] += 1
        self._idf_cache = {
            term: math.log((self._doc_count + 1) / (count + 1)) + 1
            for term, count in df.items()
        }

    def _tfidf_vector(self, text: str) -> dict[str, float]:
        """Compute TF-IDF vector for a text string."""
        tokens = self._tokenize(text)
        tf = self._tf(tokens)
        return {term: tf_val * self._idf_cache.get(term, 1.0) for term, tf_val in tf.items()}

    @staticmethod
    def _cosine_sim(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
        """Compute cosine similarity between two sparse vectors."""
        common_terms = set(vec_a.keys()) & set(vec_b.keys())
        if not common_terms:
            return 0.0
        dot = sum(vec_a[t] * vec_b[t] for t in common_terms)
        norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def find_similar(self, query: str, candidates: list[dict], top_k: int = 5) -> list[tuple[dict, float]]:
        """
        Find top-k most similar historical messages to the query text.
        
        Args:
            query: The incoming message text.
            candidates: List of message_history dicts with 'message_text' field.
            top_k: Number of top matches to return.
        
        Returns:
            List of (message_dict, similarity_score) tuples, sorted descending.
        """
        query_vec = self._tfidf_vector(query)
        scored = []
        for candidate in candidates:
            cand_text = candidate.get("message_text", "")
            if not cand_text:
                continue
            cand_vec = self._tfidf_vector(cand_text)
            sim = self._cosine_sim(query_vec, cand_vec)
            if sim > 0.05:  # threshold to filter noise
                scored.append((candidate, sim))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# Global TF-IDF matcher instance
_tfidf_matcher = TFIDFMatcher()


def build_tfidf_index(history_messages: list[dict]) -> None:
    """Build TF-IDF IDF values from the full message_history corpus."""
    corpus = [msg.get("message_text", "") for msg in history_messages if msg.get("message_text")]
    _tfidf_matcher.build_idf(corpus)
    logger.info(f"[Sub-Agent 2.2] TF-IDF index built from {len(corpus)} history messages")


def sub_agent_2_2_vector_search(
    message_text: str,
    user_id: str,
    sender_user_id: str = "",
    group_id: str = "",
    business_id: str = "",
) -> list[dict]:
    """
    Sub-Agent 2.2 — Vector Similarity Matcher Sub-Agent
    
    Searches message_history.csv for similar past messages from the
    SAME sender / business / group to the recipient user_id.
    
    Scoping Priority:
        1. Exact (user_id, sender_user_id) history
        2. Exact (user_id, business_id) history
        3. Exact (user_id, group_id) history
        4. Recipient user_id general history fallback
    
    Returns:
        List of top matching history message dicts with similarity scores.
    """
    if not message_text or message_text.startswith("["):
        return []

    # Strict multi-tiered candidate scoping
    candidates = []
    if sender_user_id:
        candidates = get_sender_history(user_id, sender_user_id)
    elif business_id:
        candidates = [m for m in get_user_history(user_id) if m.get("business_id") == business_id]

    if not candidates and group_id:
        candidates = [m for m in get_user_history(user_id) if m.get("group_id") == group_id]

    # Fallback to general recipient history if specific scope yields no candidates
    if not candidates:
        candidates = get_user_history(user_id)

    if not candidates:
        logger.info(f"[Sub-Agent 2.2] No history found for user={user_id}")
        return []

    matches = _tfidf_matcher.find_similar(message_text, candidates, top_k=5)
    logger.info(f"[Sub-Agent 2.2] Found {len(matches)} similar evidence messages for user={user_id}")
    return [{"message_id": m[0]["message_id"], "score": m[1], "text": m[0].get("message_text", "")[:100]} for m in matches]


##------------------------------Agent 2 Orchestrator - Full context + evidence retrieval------------------##

def process_memory_rag(message: dict, perceived_text: str) -> dict:
    """
    Agent 2 Main Orchestrator — Context & RAG Memory Agent
    
    Combines Sub-Agent 2.1 (hash lookup) and Sub-Agent 2.2
    (vector search) to produce the full context package.
    
    Args:
        message: A dict from messages.csv.
        perceived_text: The text after Agent 1 perception processing.
    
    Returns:
        A dict containing:
            - 'context': All joined features from Sub-Agent 2.1
            - 'evidence': List of similar historical messages from Sub-Agent 2.2
            - 'evidence_message_ids': Semicolon-separated IDs or 'none'
    """
    # Sub-Agent 2.1: Hash lookup for all context features
    context = sub_agent_2_1_hash_lookup(message)

    # Sub-Agent 2.2: Vector search for evidence scoped strictly to sender/business/group
    evidence = sub_agent_2_2_vector_search(
        perceived_text,
        user_id=message.get("user_id", ""),
        sender_user_id=message.get("sender_user_id", ""),
        group_id=message.get("group_id", ""),
        business_id=message.get("business_id", ""),
    )

    # Build evidence_message_ids string
    if evidence:
        evidence_ids = ";".join(e["message_id"] for e in evidence[:3])
    else:
        evidence_ids = "none"

    # Enrich context with user's past reaction patterns
    user_events = get_user_events(message.get("user_id", ""))
    if user_events:
        opened_count = sum(1 for e in user_events if e.get("message_opened") == "1")
        reported_count = sum(1 for e in user_events if e.get("message_reported") == "1")
        muted_count = sum(1 for e in user_events if e.get("muted_after_message") == "1")
        avg_reaction = []
        for e in user_events:
            rt = e.get("reaction_time_minutes", "")
            if rt and rt != "":
                try:
                    avg_reaction.append(float(rt))
                except ValueError:
                    pass
        context["past_opened_ratio"] = opened_count / max(len(user_events), 1)
        context["past_reported_count"] = reported_count
        context["past_muted_count"] = muted_count
        context["avg_reaction_time_min"] = sum(avg_reaction) / max(len(avg_reaction), 1) if avg_reaction else None

    top_sim = float(evidence[0]["score"]) if evidence else 0.0

    result = {
        "context": context,
        "evidence": evidence,
        "evidence_message_ids": evidence_ids,
        "top_cosine_sim": top_sim,
    }

    logger.info(f"[Agent 2] Memory package ready for {message.get('message_id', '?')} "
                f"(evidence_ids={evidence_ids}, top_sim={top_sim:.3f})")
    return result
