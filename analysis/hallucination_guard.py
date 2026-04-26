# hallucination_guard.py
# Hallucination Prevention Layer for TranscriptAI
#
# ══════════════════════════════════════════════════════════════════
# CRITICAL DESIGN DECISION — READ BEFORE MODIFYING:
#
# This guard is 100% RULE-BASED. It does NOT use any LLM.
#
# Why this matters:
#   If the LLM validated its own output, it could "hallucinate
#   that the hallucination is correct" — a circular grading trap.
#   This guard is completely independent of the model.
#
# How it works:
#   Every claim is verified against the ORIGINAL transcript using
#   deterministic token overlap scoring:
#
#     overlap_score = |claim_tokens ∩ transcript_tokens|
#                     ─────────────────────────────────
#                          |claim_tokens|
#
#   No LLM involved. No model calls. Pure Python math.
#   The transcript is the ground truth — not the model.
#
# Interview answer:
#   "The hallucination guard is entirely rule-based — Unicode-aware
#    token overlap scoring against the original transcript. The LLM
#    never validates its own output. The guard is completely
#    independent of the model."
# ══════════════════════════════════════════════════════════════════
#
# Architecture:
#   transcript ──────────────────────────────────────────┐
#       ↓                                                │ (ground truth)
#   LLM Analysis                                         │
#       ↓                                                │
#   raw_result ──→ hallucination_guard (RULE-BASED) ←───┘
#                         ↓
#               verified_result (with confidence scores)
#
# Three verification checks (all deterministic):
#   1. Keyword grounding  — token overlap between claim and transcript
#   2. Speaker grounding  — speaker name extracted from transcript labels
#   3. Confidence scoring — weighted score per field, flag if below threshold

import re
import unicodedata


# ── JAPANESE-AWARE TOKENIZER (reused from evaluator) ─────────────────────────
def _ja_tokenize(text: str) -> list:
    """Tokenizes JA+EN text. CJK chars split individually, EN words kept intact."""
    ja_pattern = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]")
    tokens = []
    current_en = []
    for char in text.lower():
        if ja_pattern.match(char):
            if current_en:
                tokens.extend("".join(current_en).split())
                current_en = []
            tokens.append(char)
        elif char in (" ", "\t", "\n"):
            if current_en:
                tokens.extend("".join(current_en).split())
                current_en = []
        else:
            current_en.append(char)
    if current_en:
        tokens.extend("".join(current_en).split())
    cjk = [t for t in tokens if ja_pattern.match(t)]
    bigrams = ["".join(cjk[i:i+2]) for i in range(len(cjk)-1)]
    return tokens + bigrams


def _overlap_score(claim: str, transcript: str) -> float:
    """
    Measures how well a claim is grounded in the transcript.
    Returns 0.0 (no grounding) to 1.0 (fully grounded).

    Fix: transcript is preprocessed to extract speaker names from
    "Name:" and "[00:00] Name:" patterns before tokenizing,
    so "Tanaka" matches "Tanaka:" correctly.
    """
    if not claim or not transcript:
        return 0.0

    # Extract speaker names from labels like "Tanaka:", "田中:", "[00:01] Sato:"
    # Pattern: optional [timestamp] then word then colon
    speaker_names = re.findall(r"([^\s\[\]:：]+)\s*[:：]", transcript)
    # Filter out timestamps like "00:01"
    speaker_names = [n for n in speaker_names if not re.match(r"^\d+$", n)]
    # Build enriched transcript: original + speaker names for better matching
    enriched_transcript = transcript + " " + " ".join(speaker_names)

    claim_tokens      = set(_ja_tokenize(claim))
    transcript_tokens = set(_ja_tokenize(enriched_transcript))

    # Remove stopwords that add noise
    stopwords = {
        "the","a","an","is","are","was","were","be","been","to","of","and",
        "or","in","on","at","for","with","by","from","that","this","it",
        "will","would","should","could","may","might","have","has","had",
        "i","we","you","he","she","they","our","your","their",
        "の","は","が","を","に","で","と","も","か","な","て","し","た",
        "です","ます","する","いる","ある","こと","ため","よう","として"
    }
    claim_tokens      = claim_tokens - stopwords
    transcript_tokens = transcript_tokens - stopwords

    if not claim_tokens:
        return 0.5  # empty claim after stopword removal — neutral

    overlap = claim_tokens & transcript_tokens
    return round(len(overlap) / len(claim_tokens), 3)


# ── GROUNDING THRESHOLDS ──────────────────────────────────────────────────────
THRESHOLDS = {
    "action_task":       0.20,  # 2.3 FIX: raised — 0.065 was being accepted as verified
    "action_owner":      0.10,  # owner names already normalized — keep low
    "action_deadline":   0.10,  # often implied
    "summary_bullet":    0.15,  # paraphrasing expected
    "sentiment_speaker": 0.05,  # pre-extracted from labels — keep very low
}

# 2.3 FIX: Semantic rescue threshold raised to match task threshold
# Items with token_overlap < 0.20 AND semantic < 0.20 = flagged
SEMANTIC_RESCUE_THRESHOLD = 0.20  # was 0.15 — too lenient


# ── ACTION ITEM VERIFICATION ──────────────────────────────────────────────────
def verify_action_items(action_items: list, transcript: str) -> dict:
    """
    Verifies each action item against the transcript.
    Fix 4: Unified confidence score combining token overlap + semantic similarity.
    Single score replaces two conflicting scores.
    """
    # Semantic cross-language grounding (separate layer)
    USE_SEMANTIC = False
    semantic_grounding_score = None
    try:
        from semantic_validator import semantic_grounding_score as _sg
        semantic_grounding_score = _sg
        USE_SEMANTIC = True
    except ImportError:
        pass

    verified   = []
    flagged    = []
    confidence_scores = []

    for item in action_items:
        task     = item.get("task", "")
        owner    = item.get("owner", "")
        deadline = item.get("deadline", "Not specified")

        task_score    = _overlap_score(task, transcript)
        owner_score   = _overlap_score(owner, transcript)
        deadline_score = (
            1.0 if deadline.lower() in ("not specified", "tbd", "—", "")
            else _overlap_score(deadline, transcript)
        )

        # Semantic score for cross-language grounding
        semantic_score = 0.0
        if USE_SEMANTIC:
            semantic_score = semantic_grounding_score(task, transcript)

        # Fix 4: Unified confidence — best of token overlap OR semantic
        # This prevents conflicting scores confusing the user
        effective_task_score = max(task_score, semantic_score)

        confidence = round(
            0.60 * effective_task_score +
            0.30 * owner_score +
            0.10 * deadline_score,
            3
        )

        item_with_score = {
            **item,
            "confidence": confidence,
            "grounding": {
                "task_token_overlap": task_score,
                "task_semantic":      round(semantic_score, 3) if USE_SEMANTIC else None,
                "task_effective":     round(effective_task_score, 3),
                "owner_score":        owner_score,
                "deadline_score":     deadline_score
            }
        }

        if effective_task_score < THRESHOLDS["action_task"]:
            item_with_score["hallucination_flag"] = True
            item_with_score["flag_reason"] = (
                f"Task '{task[:40]}' has low transcript grounding "
                f"(score: {task_score}) — may be hallucinated"
            )
            flagged.append(item_with_score)
        else:
            item_with_score["hallucination_flag"] = False
            verified.append(item_with_score)

        confidence_scores.append(confidence)

    avg_confidence = round(
        sum(confidence_scores) / len(confidence_scores), 3
    ) if confidence_scores else 0.0

    return {
        "verified":        verified,
        "flagged":         flagged,
        "total":           len(action_items),
        "verified_count":  len(verified),
        "flagged_count":   len(flagged),
        "avg_confidence":  avg_confidence,
        "hallucination_rate": round(len(flagged) / max(len(action_items), 1), 3)
    }


# ── SUMMARY VERIFICATION ──────────────────────────────────────────────────────
def verify_summary(summary: list, transcript: str) -> dict:
    """
    Verifies each summary bullet is grounded in the transcript.
    Flags bullets that appear to be invented content.
    """
    verified = []
    flagged  = []

    for bullet in summary:
        score = _overlap_score(bullet, transcript)
        bullet_result = {
            "text":               bullet,
            "grounding_score":    score,
            "hallucination_flag": score < THRESHOLDS["summary_bullet"]
        }
        if score < THRESHOLDS["summary_bullet"]:
            bullet_result["flag_reason"] = (
                f"Low transcript grounding (score: {score}) — verify this bullet"
            )
            flagged.append(bullet_result)
        else:
            verified.append(bullet_result)

    return {
        "verified":       verified,
        "flagged":        flagged,
        "verified_count": len(verified),
        "flagged_count":  len(flagged),
        "total":          len(summary)
    }


# ── SENTIMENT SPEAKER VERIFICATION ───────────────────────────────────────────
def verify_sentiment_speakers(sentiment: list, transcript: str) -> dict:
    """
    Verifies that every speaker in sentiment analysis actually
    appears in the transcript. Flags ghost speakers.
    """
    verified = []
    flagged  = []

    for entry in sentiment:
        speaker = entry.get("speaker", "")
        score   = _overlap_score(speaker, transcript)

        entry_result = {
            **entry,
            "speaker_grounding": score,
            "hallucination_flag": score < THRESHOLDS["sentiment_speaker"]
        }

        if score < THRESHOLDS["sentiment_speaker"]:
            entry_result["flag_reason"] = (
                f"Speaker '{speaker}' not found in transcript — may be hallucinated"
            )
            flagged.append(entry_result)
        else:
            verified.append(entry_result)

    return {
        "verified":       verified,
        "flagged":        flagged,
        "verified_count": len(verified),
        "flagged_count":  len(flagged)
    }


# ── MASTER GUARD ──────────────────────────────────────────────────────────────
def verify_result(result: dict, transcript: str) -> dict:
    """
    Main function. Takes raw LLM result + original transcript.
    Returns verified result with hallucination flags and confidence scores.

    Usage in analyzer.py:
        from hallucination_guard import verify_result
        result = analyze_transcript(text, language)
        result = verify_result(result, text)

    The result dict is enhanced — existing keys preserved,
    hallucination metadata added under "verification" key.
    """
    if not transcript:
        return result

    verification = {}

    # Verify action items
    if result.get("action_items"):
        ai_check = verify_action_items(result["action_items"], transcript)
        verification["action_items"] = ai_check

        # Replace action_items with verified+flagged (both kept, flagged are marked)
        result["action_items"] = (
            ai_check["verified"] + ai_check["flagged"]
        )

    # Verify summary
    if result.get("summary"):
        summary_check = verify_summary(result["summary"], transcript)
        verification["summary"] = summary_check

    # Verify sentiment speakers
    if result.get("sentiment"):
        sentiment_check = verify_sentiment_speakers(result["sentiment"], transcript)
        verification["sentiment_speakers"] = sentiment_check

    # Overall hallucination risk score
    flagged_total = (
        verification.get("action_items", {}).get("flagged_count", 0) +
        verification.get("summary", {}).get("flagged_count", 0) +
        verification.get("sentiment_speakers", {}).get("flagged_count", 0)
    )
    total_claims = (
        len(result.get("action_items", [])) +
        len(result.get("summary", [])) +
        len(result.get("sentiment", []))
    )

    overall_risk = round(flagged_total / max(total_claims, 1), 3)
    risk_label = (
        "LOW"    if overall_risk < 0.15 else
        "MEDIUM" if overall_risk < 0.35 else
        "HIGH"
    )

    result["verification"] = {
        **verification,
        "overall_hallucination_risk": overall_risk,
        "risk_label":   risk_label,
        "flagged_total": flagged_total,
        "total_claims":  total_claims,
        "note": "Items with hallucination_flag=True are low-confidence — verify manually"
    }

    return result


# ── QUICK TEST ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    transcript = """
    Tanaka: おはようございます。今日のプロジェクト更新について話しましょう。
    Sato: Good morning. Yes, the Q3 report is almost ready.
    Tanaka: そうですね。Deadline is next Friday, right?
    Sato: Correct. I will handle the financial section. Can you review by Thursday?
    Tanaka: 検討します。Also, we need to inform the client about the delay.
    Sato: Understood. I will send them an email today.
    Tanaka: ありがとうございます。Let's sync again tomorrow morning.
    """

    # Simulate LLM result with one hallucinated item
    mock_result = {
        "summary": [
            "The team discussed Q3 report progress and confirmed deadline.",
            "Sato will handle the financial section reviewed by Thursday.",
            "They will organize a company picnic next month."   # HALLUCINATED
        ],
        "action_items": [
            {"task": "Review financial section of Q3 report", "owner": "Tanaka",  "deadline": "Thursday"},
            {"task": "Send email to client about delay",       "owner": "Sato",   "deadline": "Today"},
            {"task": "Book conference room in Tokyo office",   "owner": "Yamada", "deadline": "Monday"},  # HALLUCINATED
        ],
        "sentiment": [
            {"speaker": "Tanaka", "score": "neutral", "label": "Professional"},
            {"speaker": "Sato",   "score": "neutral", "label": "Cooperative"},
            {"speaker": "Ghost",  "score": "positive", "label": "Invented speaker"},  # HALLUCINATED
        ],
        "speakers": [],
        "japan_insights": {"keigo_level": "medium", "nemawashi_signals": [], "code_switch_count": 2}
    }

    print("=== BEFORE VERIFICATION ===")
    print(f"Action items: {len(mock_result['action_items'])}")
    print(f"Summary bullets: {len(mock_result['summary'])}")

    verified = verify_result(mock_result, transcript)

    print("\n=== AFTER VERIFICATION ===")
    v = verified["verification"]
    print(f"Overall risk: {v['risk_label']} ({v['overall_hallucination_risk']})")
    print(f"Flagged: {v['flagged_total']} / {v['total_claims']} claims")

    print("\n--- Action Items ---")
    for item in verified["action_items"]:
        flag = "🚩 FLAGGED" if item.get("hallucination_flag") else "✅ VERIFIED"
        print(f"{flag} | conf:{item.get('confidence','?')} | {item['task'][:50]}")

    print("\n--- Summary ---")
    for bullet in v["summary"]["verified"]:
        print(f"✅ score:{bullet['grounding_score']} | {bullet['text'][:60]}")
    for bullet in v["summary"]["flagged"]:
        print(f"🚩 score:{bullet['grounding_score']} | {bullet['text'][:60]}")

    print("\n--- Sentiment Speakers ---")
    for s in v["sentiment_speakers"]["verified"]:
        print(f"✅ {s['speaker']} (grounding: {s['speaker_grounding']})")
    for s in v["sentiment_speakers"]["flagged"]:
        print(f"🚩 {s['speaker']} — {s.get('flag_reason','')}")

    print("\n=== FULL REPORT ===")
    print(json.dumps(verified["verification"], indent=2, ensure_ascii=False))