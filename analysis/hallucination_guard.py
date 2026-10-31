# analysis/hallucination_guard.py — v2.1
# Hallucination Prevention Layer for TranscriptAI
#
# v2.0 → v2.1 changes:
#
#   FIX-TF: TASK_FIELD mismatch — was the worst bug in this file.
#     item.get("task", "") always returns "" for LLM-produced action items
#     because the LLM schema uses task_en (TASK_FIELD), not "task".
#     _overlap_score("", transcript) = 0.0 < threshold = 0.20 → every LLM
#     action item was flagged as hallucinated on every analysis.
#     The guard was not protecting anything — it was actively wrong.
#     Fix: import TASK_FIELD / TASK_FIELD_JA from analysis_prompt and use them.
#
#   FIX-IMP: semantic_validator import path was missing the "analysis." prefix.
#     The module lives at analysis/semantic_validator.py so the correct path is
#     "from analysis.semantic_validator import ...".
#     ImportError fired on every startup → USE_SEMANTIC was always False
#     → semantic rescue never ran.
#
#   FIX-PUNCT: English tokens retained punctuation.
#     "today." ≠ "today", "report," ≠ "report". Overlap scores were
#     artificially low for any claim containing sentence-terminal words.
#     Fix: _ja_tokenize strips punctuation from English word tokens.
#
#   FIX-NONE: deadline.lower() raised AttributeError when deadline=None.
#     TASK_FIELD_JA defaults to null in the schema → task_ja=None is common.
#     If the deadline field is also None, the guard crashed silently and
#     the analyzer.py fallback set hallucination_flag=True (FIX-4 safe
#     default), which was correct but for the wrong reason.
#     Fix: str(deadline or "").lower() throughout.
#
#   FIX-JA: Japanese transcripts now verify task_ja when available.
#     For a Japanese transcript, task_en "Send email to client" scores near
#     zero against Japanese token content. task_ja "クライアントにメールを送る"
#     scores correctly against the same transcript. Guard now picks whichever
#     has content: task_en for English transcripts, task_ja for Japanese.
#
# Architecture unchanged from v2.0:
#   ══════════════════════════════════════════════════════════════════
#   CRITICAL DESIGN DECISION — READ BEFORE MODIFYING:
#
#   This guard is 100% RULE-BASED. It does NOT use any LLM.
#
#   Why: if the LLM validated its own output, it could hallucinate that
#   the hallucination is correct — a circular grading trap.
#   This guard is completely independent of the model.
#
#   How: every claim is verified against the ORIGINAL transcript using
#   deterministic token overlap scoring:
#
#     overlap_score = |claim_tokens ∩ transcript_tokens|
#                     ─────────────────────────────────
#                          |claim_tokens|
#
#   No LLM. No model calls. Pure Python math.
#   The transcript is the ground truth — not the model.
#
#   PII note: the guard receives the UNMASKED transcript (analyzer.py
#   passes the normalized text before masking). PII-restored result
#   fields (real names) therefore match the unmasked transcript tokens.
#   This is intentional and correct.
#
#   Interview answer:
#     "The hallucination guard is entirely rule-based — Unicode-aware
#      token overlap scoring against the original transcript. The LLM
#      never validates its own output."
#   ══════════════════════════════════════════════════════════════════

import re
import string

# FIX-TF: Import canonical field names. Falls back to literals if
# analysis_prompt is unavailable (e.g. import cycle during testing).
try:
    from prompts.analysis_prompt import TASK_FIELD, TASK_FIELD_JA
except ImportError:
    TASK_FIELD    = "task_en"
    TASK_FIELD_JA = "task_ja"

# FIX-PUNCT: punctuation to strip from English tokens.
# Includes ASCII punctuation and common Japanese sentence terminals.
_STRIP_PUNCT = str.maketrans(
    "", "",
    string.punctuation + "。、！？…「」『』（）【】〔〕・"
)


# ── JAPANESE-AWARE TOKENIZER ──────────────────────────────────────────────────

def _ja_tokenize(text: str) -> list[str]:
    """
    Tokenize Japanese + English mixed text.
    CJK characters are split individually (no word boundaries in Japanese).
    CJK bigrams are added for better phrase matching.
    English words are split on whitespace with punctuation stripped.

    FIX-PUNCT: English tokens now have punctuation stripped so
    "today." matches "today" and "report," matches "report".

    DSA: O(n) character scan.
    """
    ja_pattern  = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]")
    tokens: list[str] = []
    current_en: list[str] = []

    for char in text.lower():
        if ja_pattern.match(char):
            if current_en:
                raw = "".join(current_en)
                for word in raw.split():
                    cleaned = word.translate(_STRIP_PUNCT).strip()  # FIX-PUNCT
                    if cleaned:
                        tokens.append(cleaned)
                current_en = []
            tokens.append(char)
        elif char in (" ", "\t", "\n"):
            if current_en:
                raw = "".join(current_en)
                for word in raw.split():
                    cleaned = word.translate(_STRIP_PUNCT).strip()  # FIX-PUNCT
                    if cleaned:
                        tokens.append(cleaned)
                current_en = []
        else:
            current_en.append(char)

    if current_en:
        raw = "".join(current_en)
        for word in raw.split():
            cleaned = word.translate(_STRIP_PUNCT).strip()          # FIX-PUNCT
            if cleaned:
                tokens.append(cleaned)

    # CJK bigrams for phrase matching ("田中" → tokens["田","中"] + bigram "田中")
    cjk     = [t for t in tokens if ja_pattern.match(t)]
    bigrams = ["".join(cjk[i:i + 2]) for i in range(len(cjk) - 1)]
    return tokens + bigrams


def _overlap_score(claim: str, transcript: str) -> float:
    """
    Measure how well a claim is grounded in the transcript.
    Returns 0.0 (no grounding) to 1.0 (fully grounded).

    Speaker names are pre-extracted from "Name:" and "[00:00] Name:"
    patterns and added to the transcript token set so "Tanaka" matches
    "Tanaka:" speaker labels correctly.

    DSA: O(n + m) where n = claim tokens, m = transcript tokens.
    """
    if not claim or not transcript:
        return 0.0

    # Extract speaker names from labels ("Tanaka:", "田中：", "[00:01] Sato:")
    speaker_names = re.findall(r"([^\s\[\]:：]+)\s*[:：]", transcript)
    speaker_names = [n for n in speaker_names if not re.match(r"^\d+$", n)]
    enriched = transcript + " " + " ".join(speaker_names)

    claim_tokens      = set(_ja_tokenize(claim))
    transcript_tokens = set(_ja_tokenize(enriched))

    # Stopwords that inflate overlap artificially
    _STOPWORDS = {
        "the","a","an","is","are","was","were","be","been","to","of","and",
        "or","in","on","at","for","with","by","from","that","this","it",
        "will","would","should","could","may","might","have","has","had",
        "i","we","you","he","she","they","our","your","their",
        "の","は","が","を","に","で","と","も","か","な","て","し","た",
        "です","ます","する","いる","ある","こと","ため","よう","として",
    }
    claim_tokens      -= _STOPWORDS
    transcript_tokens -= _STOPWORDS

    if not claim_tokens:
        return 0.5  # empty after stopword removal — neutral, not zero

    overlap = claim_tokens & transcript_tokens
    return round(len(overlap) / len(claim_tokens), 3)


def _has_japanese_content(text: str) -> bool:
    """True when text contains CJK characters. O(n)."""
    return bool(re.search(r"[\u3040-\u9FFF\u4E00-\u9FFF]", text))


# ── GROUNDING THRESHOLDS ──────────────────────────────────────────────────────

THRESHOLDS = {
    "action_task":       0.20,  # Raised in v2.0 — 0.065 was too lenient
    "action_owner":      0.10,  # Owner names normalized — keep low
    "action_deadline":   0.10,  # Often implied in transcript
    "summary_bullet":    0.15,  # Analytical bullets have lower literal overlap
    "sentiment_speaker": 0.05,  # Pre-extracted from labels — keep very low
}

# NOTE: SEMANTIC_RESCUE_THRESHOLD is defined here but the current logic uses
# max(token_overlap, semantic_score) directly. This constant is retained for
# documentation purposes. A future version could implement the AND logic:
# flag only when BOTH token_overlap AND semantic < this threshold.
SEMANTIC_RESCUE_THRESHOLD = 0.20


# ── ACTION ITEM VERIFICATION ──────────────────────────────────────────────────

def verify_action_items(action_items: list, transcript: str) -> dict:
    """
    Verify each action item against the transcript.

    FIX-TF: reads TASK_FIELD ("task_en") first, falls back to "task"
    for rule-based extractor items that use the legacy key.

    FIX-JA: for Japanese transcripts, also scores task_ja if available,
    and uses the higher of the two scores. English task text against a
    Japanese transcript scores near zero; task_ja scores correctly.

    FIX-NONE: deadline=None no longer raises AttributeError.

    DSA: O(A · n) where A = action items, n = transcript length.
    """
    # FIX-IMP: corrected import path (was "from semantic_validator import ...")
    USE_SEMANTIC = False
    _semantic_fn = None
    try:
        from analysis.semantic_validator import semantic_grounding_score as _sg
        _semantic_fn = _sg
        USE_SEMANTIC = True
    except (ImportError, AttributeError):
        pass

    is_japanese = _has_japanese_content(transcript)

    verified:          list = []
    flagged:           list = []
    confidence_scores: list = []

    for item in action_items:
        # FIX-TF: LLM items use task_en; rule-based extractor items use task.
        # Previously item.get("task", "") always returned "" for LLM items,
        # making _overlap_score return 0.0 → every LLM item flagged.
        task_en = item.get(TASK_FIELD) or item.get("task", "")
        task_ja = item.get(TASK_FIELD_JA) or ""
        owner   = item.get("owner", "")

        # FIX-NONE: deadline may be None when TASK_FIELD_JA defaults to null.
        raw_deadline = item.get("deadline")
        deadline     = str(raw_deadline or "").strip()

        # FIX-JA: score both language variants; use the higher score.
        # For English transcripts, task_ja is empty → task_en score wins.
        # For Japanese transcripts, task_en may score near zero while task_ja
        # correctly matches Japanese transcript content.
        task_score_en = _overlap_score(task_en, transcript) if task_en else 0.0
        task_score_ja = _overlap_score(task_ja, transcript) if task_ja and is_japanese else 0.0
        task_score    = max(task_score_en, task_score_ja)

        owner_score    = _overlap_score(owner, transcript)

        # FIX-NONE: deadline.lower() raised when deadline was None.
        deadline_score = (
            1.0 if deadline.lower() in ("not specified", "tbd", "—", "", "n/a", "null")
            else _overlap_score(deadline, transcript)
        )

        semantic_score = 0.0
        if USE_SEMANTIC and _semantic_fn:
            try:
                semantic_score = _semantic_fn(task_en, transcript)
            except Exception:
                pass

        # Best of token overlap OR semantic rescue
        effective_task_score = max(task_score, semantic_score)

        confidence = round(
            0.60 * effective_task_score
            + 0.30 * owner_score
            + 0.10 * deadline_score,
            3,
        )

        item_with_score = {
            **item,
            "confidence": confidence,
            "grounding": {
                "task_token_overlap_en": task_score_en,
                "task_token_overlap_ja": task_score_ja if is_japanese else None,
                "task_token_overlap":    task_score,
                "task_semantic":         round(semantic_score, 3) if USE_SEMANTIC else None,
                "task_effective":        round(effective_task_score, 3),
                "owner_score":           owner_score,
                "deadline_score":        deadline_score,
            },
        }

        if effective_task_score < THRESHOLDS["action_task"]:
            item_with_score["hallucination_flag"] = True
            item_with_score["flag_reason"] = (
                f"Task '{(task_en or task_ja)[:40]}' has low transcript grounding "
                f"(score: {effective_task_score:.3f}) — may be hallucinated"
            )
            flagged.append(item_with_score)
        else:
            item_with_score["hallucination_flag"] = False
            verified.append(item_with_score)

        confidence_scores.append(confidence)

    avg_confidence = (
        round(sum(confidence_scores) / len(confidence_scores), 3)
        if confidence_scores else 0.0
    )

    return {
        "verified":           verified,
        "flagged":            flagged,
        "total":              len(action_items),
        "verified_count":     len(verified),
        "flagged_count":      len(flagged),
        "avg_confidence":     avg_confidence,
        "hallucination_rate": round(len(flagged) / max(len(action_items), 1), 3),
    }


# ── SUMMARY VERIFICATION ──────────────────────────────────────────────────────

def verify_summary(summary: list, transcript: str) -> dict:
    """
    Verify each summary bullet is grounded in the transcript.

    Note on threshold: summary bullets are now analytical observations
    (P4-FIX format: [Speaker] — [action] — [significance]), not extractive
    quotes. Analytical bullets have lower literal overlap with the transcript
    than extractive ones. The threshold of 0.15 requires that speaker names
    and key noun phrases from the transcript appear in the bullet — this is
    the right bar for analytical content. Purely invented content (like
    "company picnic next month" in a meeting that discussed Q3 reports)
    will still fall below 0.15.

    DSA: O(B · n) where B = bullet count, n = transcript length.
    """
    verified: list = []
    flagged:  list = []

    for bullet in summary:
        score = _overlap_score(str(bullet), transcript)
        bullet_result = {
            "text":               bullet,
            "grounding_score":    score,
            "hallucination_flag": score < THRESHOLDS["summary_bullet"],
        }
        if score < THRESHOLDS["summary_bullet"]:
            bullet_result["flag_reason"] = (
                f"Low transcript grounding (score: {score:.3f}) — "
                "verify this bullet contains content from the transcript"
            )
            flagged.append(bullet_result)
        else:
            verified.append(bullet_result)

    return {
        "verified":       verified,
        "flagged":        flagged,
        "verified_count": len(verified),
        "flagged_count":  len(flagged),
        "total":          len(summary),
    }


# ── SENTIMENT SPEAKER VERIFICATION ───────────────────────────────────────────

def verify_sentiment_speakers(sentiment: list, transcript: str) -> dict:
    """
    Verify that every speaker in sentiment analysis appears in the transcript.
    Flags ghost speakers (invented names not in any speaker label).

    DSA: O(S · n) where S = speaker count, n = transcript length.
    """
    verified: list = []
    flagged:  list = []

    for entry in sentiment:
        speaker = entry.get("speaker", "")
        score   = _overlap_score(speaker, transcript)

        entry_result = {
            **entry,
            "speaker_grounding":  score,
            "hallucination_flag": score < THRESHOLDS["sentiment_speaker"],
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
        "flagged_count":  len(flagged),
    }


# ── MASTER GUARD ──────────────────────────────────────────────────────────────

def verify_result(result: dict, transcript: str) -> dict:
    """
    Main entry point. Takes raw LLM result + original unmasked transcript.
    Returns result with hallucination flags and confidence scores added.

    PII note: pass the UNMASKED transcript (normalized text before masking).
    This is what analyzer.py already does — the `text` variable at Stage 11
    is the normalized pre-mask text, and result has had PII restored by
    Stage A4. Both sides carry real names, so overlap scoring is correct.

    The result dict is augmented — existing keys preserved,
    hallucination metadata added under the "verification" key.

    DSA: O(A · n + B · n + S · n) = O((A + B + S) · n)
    where A = action items, B = bullets, S = speakers, n = transcript length.
    """
    if not transcript:
        return result

    verification: dict = {}

    # Verify action items
    if result.get("action_items"):
        ai_check = verify_action_items(result["action_items"], transcript)
        verification["action_items"] = ai_check
        # Both verified and flagged items are retained; flagged ones carry
        # hallucination_flag=True so the UI can surface them to the user.
        result["action_items"] = ai_check["verified"] + ai_check["flagged"]

    # Verify summary bullets
    if result.get("summary"):
        summary_check = verify_summary(result["summary"], transcript)
        verification["summary"] = summary_check

    # Verify sentiment speakers
    if result.get("sentiment"):
        sentiment_check = verify_sentiment_speakers(result["sentiment"], transcript)
        verification["sentiment_speakers"] = sentiment_check

    # Overall hallucination risk
    flagged_total = (
        verification.get("action_items",        {}).get("flagged_count", 0)
        + verification.get("summary",           {}).get("flagged_count", 0)
        + verification.get("sentiment_speakers",{}).get("flagged_count", 0)
    )
    total_claims = (
        len(result.get("action_items", []))
        + len(result.get("summary",      []))
        + len(result.get("sentiment",    []))
    )

    overall_risk = round(flagged_total / max(total_claims, 1), 3)
    risk_label   = (
        "LOW"    if overall_risk < 0.15 else
        "MEDIUM" if overall_risk < 0.35 else
        "HIGH"
    )

    result["verification"] = {
        **verification,
        "overall_hallucination_risk": overall_risk,
        "risk_label":                 risk_label,
        "flagged_total":              flagged_total,
        "total_claims":               total_claims,
        "note": (
            "Items with hallucination_flag=True have low transcript grounding "
            "— verify manually before presenting to client."
        ),
    }

    return result


# ── SELF-TEST ─────────────────────────────────────────────────────────────────

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

    # Simulate LLM result using task_en (correct schema field)
    mock_result = {
        "summary": [
            "Sato — confirmed Q3 report almost ready with Thursday review deadline — first hard commitment of the meeting",
            "Tanaka — cited need to inform client about delay — risk signal; no specific owner assigned",
            "Team agreed to organize company picnic next month",   # HALLUCINATED
        ],
        "action_items": [
            # LLM items use task_en — FIX-TF ensures these score correctly
            {"task_en": "Review financial section of Q3 report", "task_ja": None,
             "owner": "Tanaka", "deadline": "Thursday"},
            {"task_en": "Send email to client about delay", "task_ja": None,
             "owner": "Sato", "deadline": "Today"},
            {"task_en": "Book conference room in Tokyo office", "task_ja": None,
             "owner": "Yamada", "deadline": "Monday"},  # HALLUCINATED
        ],
        "sentiment": [
            {"speaker": "Tanaka", "score": "neutral",  "label": "factual"},
            {"speaker": "Sato",   "score": "positive", "label": "agreeable"},
            {"speaker": "Ghost",  "score": "positive", "label": "enthusiastic"},  # HALLUCINATED
        ],
        "speakers":      [],
        "japan_insights": {
            "keigo_level": "medium", "nemawashi_signals": [], "code_switch_count": 2
        },
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
        task = item.get("task_en") or item.get("task", "?")
        conf = item.get("confidence", "?")
        eff  = item.get("grounding", {}).get("task_effective", "?")
        print(f"{flag} | conf:{conf} eff:{eff} | {task[:55]}")

    print("\n--- Summary ---")
    for b in v["summary"]["verified"]:
        print(f"✅ score:{b['grounding_score']:.3f} | {b['text'][:65]}")
    for b in v["summary"]["flagged"]:
        print(f"🚩 score:{b['grounding_score']:.3f} | {b['text'][:65]}")

    print("\n--- Sentiment Speakers ---")
    for s in v["sentiment_speakers"]["verified"]:
        print(f"✅ {s['speaker']} (grounding: {s['speaker_grounding']})")
    for s in v["sentiment_speakers"]["flagged"]:
        print(f"🚩 {s['speaker']} — {s.get('flag_reason', '')}")