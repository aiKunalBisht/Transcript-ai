#!/usr/bin/env python3
"""
Soft Rejection Detector — Evaluation Runner v1.1
=================================================
Fixes from v1.0:
  - Pass language to detect_soft_rejections() — Hindi/English patterns now fire
  - Normalize MINIMAL → LOW (detector internal tier not in our vocab)
  - Graceful fallback if detector doesn't accept language kwarg

Run:
    python scripts/eval_soft_rejection.py
    python scripts/eval_soft_rejection.py --verbose
    python scripts/eval_soft_rejection.py --save
    python scripts/eval_soft_rejection.py -v -s
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET_PATH = ROOT / "tests" / "eval_dataset.json"
RESULTS_DIR  = ROOT / "eval_results"

# ── Tier vocabulary (MINIMAL is detector's internal name → normalize to LOW) ──
_TIERS       = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]
_TIER_REMAP  = {"MINIMAL": "LOW"}   # detector uses MINIMAL, we use LOW


# ── Load dataset ──────────────────────────────────────────────────────────────
def load_dataset() -> list[dict]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["examples"]


# ── Prediction ────────────────────────────────────────────────────────────────
def _extract_tier(result: dict) -> str:
    tier = (
        result.get("risk_level")
        or result.get("overall_risk")
        or (result.get("risk_summary") or {}).get("level")
        or ""
    )
    if not tier:
        detected = result.get("detected") or result.get("has_soft_rejection")
        tier = "MEDIUM" if detected else "NONE"

    tier = str(tier).strip().upper()
    # Normalize detector-internal tier names to our vocab
    return _TIER_REMAP.get(tier, tier)


def predict(text: str, language: str = None) -> dict:
    """
    Run detect_soft_rejections with language context.
    Language is required for Hindi and English patterns to activate.
    Falls back gracefully if the function doesn't accept language kwarg.
    """
    from analysis.soft_rejection_detector import detect_soft_rejections
    t0 = time.perf_counter()
    try:
        # Pass language so Hindi/English/mixed patterns fire
        raw = detect_soft_rejections(text, language=language)
    except TypeError:
        # Older signature — no language param
        raw = detect_soft_rejections(text)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    return {"tier": _extract_tier(raw), "raw": raw, "latency_ms": ms}


# ── Binary helpers ────────────────────────────────────────────────────────────
def _to_binary(label_or_tier: str) -> str:
    return "no_risk" if label_or_tier in ("approval", "NONE") else "risk"


# ── Metrics ───────────────────────────────────────────────────────────────────
def _prf(tp: int, fp: int, fn: int) -> tuple:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return round(p, 3), round(r, 3), round(f, 3)


# ── Main evaluation loop ──────────────────────────────────────────────────────
def run_evaluation(verbose: bool = False) -> dict:
    examples = load_dataset()
    n = len(examples)

    print(f"\n{'═'*62}")
    print(f"  TranscriptAI — Soft Rejection Eval v1.1")
    print(f"  Dataset : {n} examples ({DATASET_PATH.name})")
    print(f"  Fix     : language passed · MINIMAL→LOW normalized")
    print(f"{'═'*62}")

    rows         = []
    total_ms     = 0.0
    tier_correct = 0
    b = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    tc = {t: {"tp": 0, "fp": 0, "fn": 0} for t in _TIERS}
    lang_counts  = defaultdict(lambda: {"correct": 0, "total": 0})

    for ex in examples:
        gt_tier    = ex["tier"].upper()
        gt_label   = ex["label"]
        gt_bin     = _to_binary(gt_label)
        lang       = ex.get("language", "en")

        pred        = predict(ex["text"], language=lang)
        pred_tier   = pred["tier"]
        pred_bin    = _to_binary(pred_tier)
        total_ms   += pred["latency_ms"]

        tier_ok    = (pred_tier == gt_tier)
        binary_ok  = (pred_bin  == gt_bin)
        if tier_ok:
            tier_correct += 1

        if pred_bin == "risk"    and gt_bin == "risk":    b["tp"] += 1
        elif pred_bin == "risk"  and gt_bin == "no_risk": b["fp"] += 1
        elif pred_bin == "no_risk" and gt_bin == "risk":  b["fn"] += 1
        else:                                              b["tn"] += 1

        for t in _TIERS:
            p_is = (pred_tier == t)
            g_is = (gt_tier   == t)
            if   p_is and g_is:      tc[t]["tp"] += 1
            elif p_is and not g_is:  tc[t]["fp"] += 1
            elif not p_is and g_is:  tc[t]["fn"] += 1

        lang_counts[lang]["total"] += 1
        if binary_ok:
            lang_counts[lang]["correct"] += 1

        rows.append({
            "id":           ex["id"],
            "language":     lang,
            "text_preview": ex["text"][:55] + ("…" if len(ex["text"]) > 55 else ""),
            "gt_label":     gt_label,
            "gt_tier":      gt_tier,
            "pred_tier":    pred_tier,
            "tier_match":   tier_ok,
            "binary_match": binary_ok,
            "latency_ms":   pred["latency_ms"],
        })

        if verbose:
            icon = "✅" if tier_ok else ("⚠ " if binary_ok else "❌")
            print(f"  {icon} [{ex['id']}] gt:{gt_tier:<10} pred:{pred_tier:<10} "
                  f"lang:{lang:<6} | {ex['text'][:42]}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    tier_acc  = round(tier_correct / n, 3)
    bin_acc   = round((b["tp"] + b["tn"]) / n, 3)
    bin_p, bin_r, bin_f1 = _prf(b["tp"], b["fp"], b["fn"])
    avg_ms    = round(total_ms / n, 1)

    tier_metrics: dict = {}
    for t in _TIERS:
        support = tc[t]["tp"] + tc[t]["fn"]
        if support > 0:
            p, r, f = _prf(tc[t]["tp"], tc[t]["fp"], tc[t]["fn"])
            tier_metrics[t] = {"precision": p, "recall": r, "f1": f, "support": support}

    lang_acc: dict = {}
    for lang, c in sorted(lang_counts.items()):
        lang_acc[lang] = {
            "accuracy": round(c["correct"] / c["total"], 3),
            "correct":  c["correct"],
            "total":    c["total"],
        }

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n── Binary Classification (risk vs no_risk) ──────────────────")
    print(f"  Accuracy  : {bin_acc:.1%}  ({b['tp']+b['tn']}/{n})")
    print(f"  Precision : {bin_p:.3f}")
    print(f"  Recall    : {bin_r:.3f}")
    print(f"  F1        : {bin_f1:.3f}  ← quote this number")
    print(f"  TP:{b['tp']:3d}  FP:{b['fp']:3d}  FN:{b['fn']:3d}  TN:{b['tn']:3d}")

    print(f"\n── Tier-Level Accuracy ──────────────────────────────────────")
    print(f"  Exact tier match: {tier_correct}/{n}  ({tier_acc:.1%})")

    print(f"\n── Per-Tier Metrics (one-vs-rest) ───────────────────────────")
    print(f"  {'Tier':<10} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    print(f"  {'─'*48}")
    for t in _TIERS:
        if t in tier_metrics:
            m = tier_metrics[t]
            print(f"  {t:<10} {m['precision']:>10.3f} {m['recall']:>8.3f} "
                  f"{m['f1']:>8.3f} {m['support']:>9}")

    print(f"\n── Per-Language Binary Accuracy ─────────────────────────────")
    for lang, s in lang_acc.items():
        bar = "█" * int(s["accuracy"] * 20)
        print(f"  {lang:<8} {s['accuracy']:.1%}  {bar}  ({s['correct']}/{s['total']})")

    print(f"\n── Performance ──────────────────────────────────────────────")
    print(f"  Avg latency : {avg_ms}ms per example")
    print(f"  Total time  : {round(total_ms)}ms for {n} examples")

    print(f"\n── Your Interview Answer ────────────────────────────────────")
    print(f"  \"Evaluated the soft rejection detector on {n} labeled examples")
    print(f"   across Japanese, Hindi, English and mixed-language inputs.")
    print(f"   Binary F1: {bin_f1:.2f} · Tier accuracy: {tier_acc:.1%} · "
          f"Avg latency: {avg_ms}ms.\"")
    print(f"{'═'*62}\n")

    return {
        "dataset":        str(DATASET_PATH),
        "n_examples":     n,
        "eval_version":   "1.1",
        "binary": {
            "accuracy":  bin_acc, "precision": bin_p,
            "recall":    bin_r,   "f1": bin_f1, **b,
        },
        "tier_accuracy":  tier_acc,
        "tier_metrics":   tier_metrics,
        "lang_accuracy":  lang_acc,
        "avg_latency_ms": avg_ms,
        "examples":       rows,
    }


def save_report(report: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "soft_rejection_eval.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved → {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--save",    "-s", action="store_true")
    args = parser.parse_args()

    report = run_evaluation(verbose=args.verbose)
    if args.save:
        save_report(report)