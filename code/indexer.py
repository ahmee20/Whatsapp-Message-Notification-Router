"""
indexer.py — Startup Pre-Indexing Engine
=========================================
Pre-indexes all relational CSV datasets into O(1) Python Hash
Dictionaries at startup. This eliminates repeated linear CSV
scans during message processing.

Hash Maps Built:
    USER_INDEX[user_id]                     → user profile dict
    GROUP_INDEX[group_id]                   → group info dict
    MEMBER_INDEX[(group_id, user_id)]       → membership dict
    BUSINESS_INDEX[business_id]             → business account dict
    USER_BIZ_INDEX[(user_id, business_id)]  → user-business relationship dict
    HISTORY_INDEX[user_id]                  → list of past message dicts
    EVENTS_INDEX[(user_id, message_id)]     → event interaction dict
    IMAGE_INDEX[image_id]                   → file_path string
    VOICE_INDEX[voice_note_id]              → file_path string
    DAILY_SUMMARY_INDEX[user_id]            → list of daily summary dicts
"""

import csv
import logging
from pathlib import Path
from collections import defaultdict
from config import DATASET_DIR

logger = logging.getLogger("Indexer")

# ---------------------------------------------------------------------------
# Global Hash Map Indexes (populated at startup)
# ---------------------------------------------------------------------------
USER_INDEX: dict[str, dict] = {}
GROUP_INDEX: dict[str, dict] = {}
MEMBER_INDEX: dict[tuple[str, str], dict] = {}
MEMBERS_BY_GROUP: dict[str, list[dict]] = defaultdict(list)
MEMBERS_BY_USER: dict[str, list[dict]] = defaultdict(list)
BUSINESS_INDEX: dict[str, dict] = {}
USER_BIZ_INDEX: dict[tuple[str, str], dict] = {}
USER_BIZ_BY_USER: dict[str, list[dict]] = defaultdict(list)
HISTORY_INDEX: dict[str, list[dict]] = defaultdict(list)
HISTORY_BY_SENDER_USER: dict[tuple[str, str], list[dict]] = defaultdict(list)
EVENTS_INDEX: dict[tuple[str, str], dict] = {}
EVENTS_BY_USER: dict[str, list[dict]] = defaultdict(list)
IMAGE_INDEX: dict[str, str] = {}
VOICE_INDEX: dict[str, str] = {}
DAILY_SUMMARY_INDEX: dict[str, list[dict]] = defaultdict(list)


def _read_csv(filename: str) -> list[dict]:
    """Read a CSV file from the dataset directory and return list of row dicts."""
    filepath = DATASET_DIR / filename
    if not filepath.exists():
        logger.warning(f"CSV file not found: {filepath}")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_indexes() -> None:
    """
    Build all O(1) hash map indexes from CSV datasets.
    Call this once at startup before processing any messages.
    """
    global USER_INDEX, GROUP_INDEX, MEMBER_INDEX, BUSINESS_INDEX
    global USER_BIZ_INDEX, IMAGE_INDEX, VOICE_INDEX

    logger.info("Building pre-indexes from CSV datasets...")

    # --- users.csv ---
    for row in _read_csv("users.csv"):
        USER_INDEX[row["user_id"]] = row
    logger.info(f"  USER_INDEX: {len(USER_INDEX)} users indexed")

    # --- groups.csv ---
    for row in _read_csv("groups.csv"):
        GROUP_INDEX[row["group_id"]] = row
    logger.info(f"  GROUP_INDEX: {len(GROUP_INDEX)} groups indexed")

    # --- group_members.csv ---
    for row in _read_csv("group_members.csv"):
        key = (row["group_id"], row["user_id"])
        MEMBER_INDEX[key] = row
        MEMBERS_BY_GROUP[row["group_id"]].append(row)
        MEMBERS_BY_USER[row["user_id"]].append(row)
    logger.info(f"  MEMBER_INDEX: {len(MEMBER_INDEX)} memberships indexed")

    # --- business_accounts.csv ---
    for row in _read_csv("business_accounts.csv"):
        BUSINESS_INDEX[row["business_id"]] = row
    logger.info(f"  BUSINESS_INDEX: {len(BUSINESS_INDEX)} businesses indexed")

    # --- user_business_history.csv ---
    for row in _read_csv("user_business_history.csv"):
        key = (row["user_id"], row["business_id"])
        USER_BIZ_INDEX[key] = row
        USER_BIZ_BY_USER[row["user_id"]].append(row)
    logger.info(f"  USER_BIZ_INDEX: {len(USER_BIZ_INDEX)} user-business relations indexed")

    # --- message_history.csv ---
    for row in _read_csv("message_history.csv"):
        HISTORY_INDEX[row["user_id"]].append(row)
        if row.get("sender_user_id"):
            HISTORY_BY_SENDER_USER[(row["user_id"], row["sender_user_id"])].append(row)
    logger.info(f"  HISTORY_INDEX: {sum(len(v) for v in HISTORY_INDEX.values())} history messages indexed")

    # --- message_events.csv ---
    for row in _read_csv("message_events.csv"):
        key = (row["user_id"], row["message_id"])
        EVENTS_INDEX[key] = row
        EVENTS_BY_USER[row["user_id"]].append(row)
    logger.info(f"  EVENTS_INDEX: {len(EVENTS_INDEX)} events indexed")

    # --- images.csv ---
    for row in _read_csv("images.csv"):
        IMAGE_INDEX[row["image_id"]] = row["file_path"]
    logger.info(f"  IMAGE_INDEX: {len(IMAGE_INDEX)} images indexed")

    # --- voice_notes.csv ---
    for row in _read_csv("voice_notes.csv"):
        VOICE_INDEX[row["voice_note_id"]] = row["file_path"]
    logger.info(f"  VOICE_INDEX: {len(VOICE_INDEX)} voice notes indexed")

    # --- daily_notification_summary.csv ---
    for row in _read_csv("daily_notification_summary.csv"):
        DAILY_SUMMARY_INDEX[row["user_id"]].append(row)
    logger.info(f"  DAILY_SUMMARY_INDEX: {sum(len(v) for v in DAILY_SUMMARY_INDEX.values())} daily summaries indexed")

    logger.info("Pre-indexing complete.")


def get_user(user_id: str) -> dict | None:
    """O(1) lookup for user profile."""
    return USER_INDEX.get(user_id)


def get_group(group_id: str) -> dict | None:
    """O(1) lookup for group info."""
    return GROUP_INDEX.get(group_id)


def get_membership(group_id: str, user_id: str) -> dict | None:
    """O(1) lookup for group membership."""
    return MEMBER_INDEX.get((group_id, user_id))


def get_sender_membership(group_id: str, sender_user_id: str) -> dict | None:
    """O(1) lookup for sender's group membership (to check admin role)."""
    return MEMBER_INDEX.get((group_id, sender_user_id))


def get_business(business_id: str) -> dict | None:
    """O(1) lookup for business account."""
    return BUSINESS_INDEX.get(business_id)


def get_user_business(user_id: str, business_id: str) -> dict | None:
    """O(1) lookup for user-business relationship."""
    return USER_BIZ_INDEX.get((user_id, business_id))


def get_user_history(user_id: str) -> list[dict]:
    """O(1) lookup for user's past messages."""
    return HISTORY_INDEX.get(user_id, [])


def get_sender_history(user_id: str, sender_user_id: str) -> list[dict]:
    """O(1) lookup for past messages from a specific sender to user."""
    return HISTORY_BY_SENDER_USER.get((user_id, sender_user_id), [])


def get_event(user_id: str, message_id: str) -> dict | None:
    """O(1) lookup for message event interaction."""
    return EVENTS_INDEX.get((user_id, message_id))


def get_user_events(user_id: str) -> list[dict]:
    """O(1) lookup for all user's message events."""
    return EVENTS_BY_USER.get(user_id, [])


def get_image_path(image_id: str) -> str | None:
    """O(1) lookup for image file path."""
    return IMAGE_INDEX.get(image_id)


def get_voice_path(voice_note_id: str) -> str | None:
    """O(1) lookup for voice note file path."""
    return VOICE_INDEX.get(voice_note_id)


def get_daily_summaries(user_id: str) -> list[dict]:
    """O(1) lookup for user's daily notification summaries."""
    return DAILY_SUMMARY_INDEX.get(user_id, [])
