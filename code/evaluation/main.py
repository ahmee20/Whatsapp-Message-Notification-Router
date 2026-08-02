"""
evaluation/main.py — Local Benchmark Evaluation Harness
========================================================
Tests prediction accuracy against dataset/sample_messages.csv
(30 solved ground-truth rows).

Metrics:
    - Action Accuracy (notify/digest/mute)
    - Message Type Accuracy
    - Combined Accuracy
    - Per-action precision breakdown
    - Confidence calibration stats

Usage:
    python code/evaluation/main.py
"""

import csv
import sys
import asyncio
from pathlib import Path
from collections import defaultdict

# Ensure code/ is on python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CODE_DIR = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

DATASET_DIR = PROJECT_ROOT / "dataset"
SAMPLE_PATH = DATASET_DIR / "sample_messages.csv"
OUTPUT_PATH = DATASET_DIR / "output.csv"


def load_sample_ground_truth() -> dict:
    """Load ground-truth labels from sample_messages.csv."""
    if not SAMPLE_PATH.exists():
        print(f"ERROR: Sample file not found: {SAMPLE_PATH}")
        sys.exit(1)

    truth = {}
    with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            # Normalize ID (e.g. sample_msg_001 -> msg_001 or keep sample_msg_001)
            raw_id = row["message_id"]
            norm_id = raw_id.replace("sample_msg_", "msg_")
            item = {
                "raw_id": raw_id,
                "action": row["action"].strip().lower(),
                "message_type": row["message_type"].strip().lower(),
                "confidence": float(row.get("confidence", 0)),
                "evidence_message_ids": row.get("evidence_message_ids", "none").strip(),
            }
            truth[raw_id] = item
            truth[norm_id] = item

    print(f"Loaded {len(truth)//2} ground-truth samples from {SAMPLE_PATH.name}")
    return truth


def load_predictions() -> dict:
    """Load predictions from output.csv."""
    if not OUTPUT_PATH.exists():
        print(f"ERROR: Output file not found: {OUTPUT_PATH}")
        print("Run 'python code/main.py' first to generate predictions.")
        sys.exit(1)

    preds = {}
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            msg_id = row["message_id"]
            preds[msg_id] = {
                "action": row["action"].strip().lower(),
                "message_type": row["message_type"].strip().lower(),
                "confidence": float(row.get("confidence", 0)),
                "reason": row.get("reason", ""),
                "evidence_message_ids": row.get("evidence_message_ids", "none").strip(),
            }

    print(f"Loaded {len(preds)} predictions from {OUTPUT_PATH.name}")
    return preds


def evaluate(truth: dict, preds: dict) -> None:
    """Run full evaluation and print results."""
    # Find overlapping message IDs
    common_ids = set(truth.keys()) & set(preds.keys())

    if not common_ids:
        print("\nNote: output.csv contains target messages (msg_001+).")
        print("Evaluating mapped ground-truth IDs where available...")

    if not common_ids:
        print("\nNo direct ID matches between output.csv and sample_messages.csv.")
        print("Run benchmark mode via python code/evaluation/main.py --test-samples")
        return

    print(f"\nEvaluating {len(common_ids)} overlapping messages...\n")

    action_correct = 0
    type_correct = 0
    both_correct = 0
    total = len(common_ids)

    action_tp = defaultdict(int)
    action_fp = defaultdict(int)
    action_fn = defaultdict(int)

    errors = []

    for msg_id in sorted(common_ids):
        t = truth[msg_id]
        p = preds[msg_id]

        a_match = (t["action"] == p["action"])
        t_match = (t["message_type"] == p["message_type"])

        if a_match:
            action_correct += 1
            action_tp[t["action"]] += 1
        else:
            action_fp[p["action"]] += 1
            action_fn[t["action"]] += 1
            errors.append({
                "id": msg_id,
                "field": "action",
                "expected": t["action"],
                "predicted": p["action"],
                "reason": p.get("reason", "")[:80],
            })

        if t_match:
            type_correct += 1
        else:
            errors.append({
                "id": msg_id,
                "field": "message_type",
                "expected": t["message_type"],
                "predicted": p["message_type"],
                "reason": p.get("reason", "")[:80],
            })

        if a_match and t_match:
            both_correct += 1

    print("=" * 65)
    print("EVALUATION RESULTS")
    print("=" * 65)
    print(f"  Action Accuracy:       {action_correct}/{total} ({100*action_correct/total:.1f}%)")
    print(f"  Message Type Accuracy: {type_correct}/{total} ({100*type_correct/total:.1f}%)")
    print(f"  Combined Accuracy:     {both_correct}/{total} ({100*both_correct/total:.1f}%)")
    print()

    print("Per-Action Breakdown:")
    print("-" * 45)
    for action in ["notify", "digest", "mute"]:
        tp = action_tp[action]
        fp = action_fp[action]
        fn = action_fn[action]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 0.001)
        print(f"  {action:8s}  TP={tp:3d}  FP={fp:3d}  FN={fn:3d}  "
              f"P={precision:.2f}  R={recall:.2f}  F1={f1:.2f}")
    print()

    conf_values = [preds[mid]["confidence"] for mid in common_ids]
    if conf_values:
        print(f"Confidence Stats:")
        print(f"  Mean:   {sum(conf_values)/len(conf_values):.3f}")
        print(f"  Min:    {min(conf_values):.3f}")
        print(f"  Max:    {max(conf_values):.3f}")
        print()

    if errors:
        print(f"ERRORS ({len(errors)} mismatches):")
        print("-" * 65)
        for err in errors[:20]:
            print(f"  {err['id']:20s} {err['field']:15s}  "
                  f"expected={err['expected']:18s}  got={err['predicted']:18s}")
    else:
        print("PERFECT SCORE! No errors.")

    print()
    print("=" * 65)


def main():
    truth = load_sample_ground_truth()
    preds = load_predictions()
    evaluate(truth, preds)


if __name__ == "__main__":
    main()
