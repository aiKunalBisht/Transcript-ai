# analysis/nemawashi_sequence.py
# Sequence-based nemawashi detector — meeting-level output only.
#
# ── Kayo Miura, Point 3 ───────────────────────────────────────────────────────
# "Nemawashi is not simply 'consulting a manager.'
#  It is closer to shaping the conditions before the formal answer is given —
#  aligning the relevant people, concerns, authority, timing, and possible
#  outcomes so that the formal decision is more likely to succeed.
#
#  Look for sequences of actions across multiple speakers and stages,
#  not just one sentence."
#
# ── What this detects ─────────────────────────────────────────────────────────
# The nemawashi sequence has three phases:
#
#   Phase 1 — TRIGGER
#     A speaker shows authority_deferring or escalation.
#     They cannot decide alone. They signal internal consultation.
#     Example: 「上司に相談して、2時間以内に書面でご回答します。」
#
#   Phase 2 — GAP (not visible in transcript)
#     Internal coordination happens offline.
#     The speaker consults superiors, aligns stakeholders, coordinates
#     the response. This phase is invisible but its result is detectable.
#
#   Phase 3 — RESOLUTION (evidence of nemawashi having happened)
#     The SAME speaker (or their representative) returns with a response
#     that is MORE SPECIFIC than anything they offered before the trigger.
#     Specificity markers: deadline, method, deliverable, person named.
#     The response addresses concerns raised BEFORE the trigger turn.
#
# ── What this is NOT ─────────────────────────────────────────────────────────
# - Phrase matching on 「上司に相談します」 — that is escalation alone
# - Any single-utterance detection
# - A guarantee that nemawashi occurred (it is an inference from a pattern)
#
# ── Data structure ────────────────────────────────────────────────────────────
# NemawashiResult (TypedDict):
#   nemawashi_sequence_detected : bool
#   confidence                  : float  — 0.0 to 1.0
#   trigger_speaker             : str | None
#   trigger_index               : int | None
#   trigger_utterance           : str | None
#   resolution_speaker          : str | None
#   resolution_index            : int | None
#   resolution_utterance        : str | None
#   specificity_markers         : list[str]  — what made the resolution specific
#   evidence                    : list[str]  — human-readable evidence strings
#   cultural_note               : str
#
# ── Algorithm ─────────────────────────────────────────────────────────────────
# detect_nemawashi_sequence(function_results):
#   Pass 1 — find trigger turns (authority_deferring or escalation)    O(U)
#   Pass 2 — for each trigger, scan forward for a resolution turn      O(U)
#             from the SAME speaker that is MORE SPECIFIC than
#             their pre-trigger utterances
#   Pass 3 — score the sequence: count specificity markers             O(M)
#
# Time: O(U²) worst case, O(U) typical (early match)
# Space: O(U) for the function result list
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import re
from typing import Optional, TypedDict


# ── Return type ───────────────────────────────────────────────────────────────

class NemawashiResult(TypedDict):
    nemawashi_sequence_detected: bool
    confidence:                  float
    trigger_speaker:             Optional[str]
    trigger_index:               Optional[int]
    trigger_utterance:           Optional[str]
    resolution_speaker:          Optional[str]
    resolution_index:            Optional[int]
    resolution_utterance:        Optional[str]
    specificity_markers:         list[str]
    evidence:                    list[str]
    cultural_note:               str


# ── Specificity markers ───────────────────────────────────────────────────────
# These signal that the resolution utterance is MORE SPECIFIC than a vague
# pre-trigger response. Each pattern carries a weight.
#
# High-weight (0.90): deadline with number, written commitment, named method
# Medium-weight (0.72): day name, named stakeholder, specific system/product
# Low-weight (0.55): time reference, location, meeting reference

_SPECIFICITY_PATTERNS: list[tuple[re.Pattern, float, str]] = [
    # ── Deadline with quantity ────────────────────────────────────────────────
    (re.compile(r"\d+\s*(?:時間|分|日|週間)以内", re.IGNORECASE),
     0.92, "JP time-bound deadline"),
    (re.compile(r"within\s+\d+\s*(?:hours?|minutes?|days?)", re.IGNORECASE),
     0.92, "EN time-bound deadline"),
    (re.compile(r"by\s+(?:end\s+of\s+)?(?:today|tomorrow|friday|monday|[0-9])", re.IGNORECASE),
     0.88, "named deadline"),
    (re.compile(r"今日中|本日中|明日中|明日まで", re.IGNORECASE),
     0.88, "JP today/tomorrow deadline"),

    # ── Written commitment demand or offer ───────────────────────────────────
    (re.compile(r"書面で|書面による|文書で|文書による", re.IGNORECASE),
     0.92, "JP written commitment"),
    (re.compile(r"written\s+(?:response|commitment|guarantee|report|proposal)", re.IGNORECASE),
     0.92, "EN written commitment"),

    # ── Named deliverable ────────────────────────────────────────────────────
    (re.compile(r"ご回答|ご連絡|ご報告|お送り", re.IGNORECASE),
     0.85, "JP named deliverable"),
    (re.compile(r"(?:send|provide|deliver|submit|share)\s+(?:a\s+)?(?:report|plan|proposal|document|response)", re.IGNORECASE),
     0.85, "EN named deliverable"),

    # ── Named recovery / resolution plan ─────────────────────────────────────
    (re.compile(r"24.?hour\s+(?:recovery|response|support)", re.IGNORECASE),
     0.88, "24-hour recovery structure"),
    (re.compile(r"emergency\s+(?:response|plan|team|support)", re.IGNORECASE),
     0.85, "emergency response plan"),
    (re.compile(r"緊急(?:対応|チーム|プラン)", re.IGNORECASE),
     0.85, "JP emergency response"),
    (re.compile(r"復旧(?:計画|手順|作業)", re.IGNORECASE),
     0.85, "JP recovery plan"),

    # ── Named stakeholder coordination ────────────────────────────────────────
    (re.compile(r"(?:チーム|部長|社長|役員)\s*(?:と|で)\s*(?:調整|相談|確認)", re.IGNORECASE),
     0.80, "JP stakeholder coordination"),
    (re.compile(r"(?:coordinated|aligned|discussed|confirmed)\s+with\s+(?:the\s+)?(?:team|manager|director|CEO)", re.IGNORECASE),
     0.80, "EN stakeholder coordination"),

    # ── Specific person assigned ──────────────────────────────────────────────
    (re.compile(r"担当者|担当チーム|責任者", re.IGNORECASE),
     0.75, "JP person assigned"),
    (re.compile(r"(?:assigned\s+to|handled\s+by|responsible\s+person|point\s+of\s+contact)", re.IGNORECASE),
     0.75, "EN person assigned"),

    # ── Addresses prior concern by referencing it ─────────────────────────────
    (re.compile(r"ご指摘(?:の|いただいた)", re.IGNORECASE),
     0.78, "JP references prior concern"),
    (re.compile(r"(?:as\s+you\s+(?:mentioned|raised|noted)|regarding\s+(?:your|the)\s+concern)", re.IGNORECASE),
     0.78, "EN references prior concern"),
]

# Functions that mark a TRIGGER turn (speaker cannot decide alone)
_TRIGGER_FUNCTIONS = frozenset({
    "escalation",
    "authority_deferring",
})

# Functions that mark a RESOLUTION turn (speaker is now committing)
_RESOLUTION_FUNCTIONS = frozenset({
    "commitment",
    "relationship_preservation",
    "acknowledgment_under_pressure",
})

# Minimum gap between trigger and resolution (in turns).
# If the resolution immediately follows the trigger (gap=1), it is likely
# a pre-planned response, not evidence of coordination having happened.
# Gap of 2+ turns suggests other speakers spoke between, which is realistic.
_MIN_GAP = 1   # at least 1 other turn between trigger and resolution
_MAX_GAP = 12  # don't look beyond 12 turns for the resolution


def _measure_specificity(utterance: str) -> tuple[float, list[str]]:
    """
    Score an utterance for specificity — how concrete and detailed it is.
    Returns (score, list_of_matched_marker_labels).
    DSA: O(M) where M = number of specificity patterns (~14).
    """
    total = 0.0
    markers: list[str] = []
    for pattern, weight, label in _SPECIFICITY_PATTERNS:
        if pattern.search(utterance):
            total += weight
            markers.append(label)
    return min(total, 1.0), markers


def _speaker_key(speaker: str) -> str:
    """Normalize speaker name for fuzzy matching."""
    return re.sub(r"[\s_\[\]0-9]", "", speaker).lower()


def detect_nemawashi_sequence(
    function_results: list[dict],
) -> NemawashiResult:
    """
    Detect the nemawashi sequence pattern across a meeting transcript.

    Args:
        function_results: list of FunctionResult dicts from
                          meeting_function_detector.detect_functions().
                          Each dict has: speaker, utterance_index, utterance,
                          communicative_function, secondary_functions,
                          confidence, evidence.

    Returns:
        NemawashiResult — meeting-level finding.

    Algorithm:
        Pass 1: Find trigger turns (authority_deferring / escalation)   O(U)
        Pass 2: For each trigger, scan forward for resolution from       O(U)
                the same speaker with higher specificity than their
                pre-trigger utterances
        Pass 3: Score confidence from specificity markers                O(M)

    Time: O(U²) worst case, O(U) typical with early match
    Space: O(U) for intermediate storage
    """
    if not function_results:
        return _no_sequence()

    # ── Pass 1: find trigger indices ─────────────────────────────────────────
    triggers: list[int] = [
        i for i, r in enumerate(function_results)
        if r.get("communicative_function") in _TRIGGER_FUNCTIONS
        or any(f in _TRIGGER_FUNCTIONS for f in r.get("secondary_functions", []))
    ]

    if not triggers:
        return _no_sequence()

    # ── Pass 2: for each trigger, look forward for resolution ────────────────
    best_confidence = 0.0
    best_sequence: Optional[tuple] = None  # (trigger_idx, resolution_idx, markers, evidence)

    for t_idx in triggers:
        t_result  = function_results[t_idx]
        t_speaker = _speaker_key(t_result.get("speaker", ""))

        # Pre-trigger specificity for this speaker (baseline)
        pre_utts = [
            r["utterance"] for r in function_results[:t_idx]
            if _speaker_key(r.get("speaker", "")) == t_speaker
        ]
        pre_score, _ = _measure_specificity(" ".join(pre_utts)) if pre_utts else (0.0, [])

        # Scan forward: resolution must come after a gap of ≥ _MIN_GAP turns
        search_start = t_idx + _MIN_GAP + 1
        search_end   = min(t_idx + _MAX_GAP + 1, len(function_results))

        for r_idx in range(search_start, search_end):
            r_result  = function_results[r_idx]
            r_speaker = _speaker_key(r_result.get("speaker", ""))

            # Must be the same speaker (or close variant)
            if r_speaker != t_speaker:
                continue

            # Must have a resolution-class function
            r_fn  = r_result.get("communicative_function", "")
            r_sec = r_result.get("secondary_functions", [])
            if r_fn not in _RESOLUTION_FUNCTIONS and not any(
                f in _RESOLUTION_FUNCTIONS for f in r_sec
            ):
                continue

            # Resolution must be MORE SPECIFIC than pre-trigger utterances
            r_utt          = r_result.get("utterance", "")
            post_score, markers = _measure_specificity(r_utt)

            if post_score <= pre_score and not markers:
                continue  # not more specific — not evidence of coordination

            # Build evidence strings
            gap         = r_idx - t_idx
            specificity = post_score - pre_score
            confidence  = min(1.0, post_score * 0.85 + (0.15 if specificity > 0 else 0))

            evidence = [
                f"Trigger at turn {t_idx}: {t_result.get('utterance', '')[:80]}",
                f"Resolution at turn {r_idx} (gap={gap} turns): {r_utt[:80]}",
            ]
            evidence += [f"Specificity marker: {m}" for m in markers[:4]]

            if confidence > best_confidence:
                best_confidence = confidence
                best_sequence   = (t_idx, r_idx, markers, evidence)
            break   # found a resolution for this trigger — stop inner scan

    # ── Pass 3: build result ──────────────────────────────────────────────────
    if best_sequence is None or best_confidence < 0.45:
        return _no_sequence()

    t_idx, r_idx, markers, evidence = best_sequence
    t_result = function_results[t_idx]
    r_result = function_results[r_idx]

    return NemawashiResult(
        nemawashi_sequence_detected=True,
        confidence=round(best_confidence, 3),
        trigger_speaker=t_result.get("speaker"),
        trigger_index=t_idx,
        trigger_utterance=t_result.get("utterance", "")[:200],
        resolution_speaker=r_result.get("speaker"),
        resolution_index=r_idx,
        resolution_utterance=r_result.get("utterance", "")[:200],
        specificity_markers=markers,
        evidence=evidence,
        cultural_note=(
            "Nemawashi sequence detected: a speaker who escalated internally "
            "returned with a more specific, coordinated response — suggesting "
            "prior alignment of people, authority, and conditions. "
            "The internal coordination (Phase 2) is not visible in the transcript "
            "but its result is detectable in the increased specificity of the "
            "post-escalation response. This is the distinction between nemawashi "
            "(a process) and 「上司に相談します」(a single escalation phrase)."
        ),
    )


def _no_sequence() -> NemawashiResult:
    return NemawashiResult(
        nemawashi_sequence_detected=False,
        confidence=0.0,
        trigger_speaker=None,
        trigger_index=None,
        trigger_utterance=None,
        resolution_speaker=None,
        resolution_index=None,
        resolution_utterance=None,
        specificity_markers=[],
        evidence=[],
        cultural_note=(
            "No nemawashi sequence detected. Either no escalation occurred, "
            "or the post-escalation response was not more specific than "
            "what was offered before — suggesting no internal coordination happened."
        ),
    )


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from analysis.meeting_function_detector import detect_functions

    # ── Test 1: Kayo's exact scenario — should detect nemawashi ──────────────
    transcript_nemawashi = (
        "Client: The system has been down for 6 hours. This is completely unacceptable.\n"
        "Kenji: 大変申し訳ございません。上司に相談して、ご回答いたします。\n"
        "Client: If this is not resolved by Friday we will reconsider the entire contract.\n"
        "Manager: 緊急対応チームを立ち上げて、24時間体制で復旧を進めます。\n"
        "Kenji: 2時間以内に書面でご回答します。ご指摘の問題について、担当者を配置いたします。\n"
        "Client: I need that written commitment immediately."
    )

    # ── Test 2: No nemawashi — escalation without coordinated resolution ──────
    transcript_no_nemawashi = (
        "Client: The system is down. This is unacceptable.\n"
        "Kenji: 上司に相談します。\n"
        "Client: When will you get back to me?\n"
        "Kenji: I am not sure. I need to check.\n"
        "Client: This is not acceptable."
    )

    print("=== nemawashi_sequence — Self-Tests ===\n")

    for label, transcript, expect_detected in [
        ("Kayo scenario — SHOULD detect",    transcript_nemawashi,    True),
        ("No resolution — should NOT detect", transcript_no_nemawashi, False),
    ]:
        fn_results = detect_functions(transcript)
        result     = detect_nemawashi_sequence(fn_results)
        detected   = result["nemawashi_sequence_detected"]
        ok         = detected == expect_detected
        sym        = "✓" if ok else "✗"
        print(f"  {sym}  {label}")
        print(f"       detected={detected}  confidence={result['confidence']:.2f}")
        if result["trigger_utterance"]:
            print(f"       trigger:    {result['trigger_utterance'][:70]}")
        if result["resolution_utterance"]:
            print(f"       resolution: {result['resolution_utterance'][:70]}")
        if result["specificity_markers"]:
            print(f"       markers:    {result['specificity_markers']}")
        if not ok:
            print(f"       FAIL: expected detected={expect_detected}")
        print()

    print("=== Function results for Kayo scenario ===")
    fn_results = detect_functions(transcript_nemawashi)
    for r in fn_results:
        sec = f" + {', '.join(r['secondary_functions'])}" if r["secondary_functions"] else ""
        print(f"  [{r['utterance_index']}] {r['speaker']:<10} {r['communicative_function']}{sec}")