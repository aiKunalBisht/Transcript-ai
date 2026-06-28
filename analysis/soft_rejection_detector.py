"""
analysis/soft_rejection_detector.py  — TranscriptAI v3.2
=========================================================
v3.2 complete rewrite of termination detection:
  - 18 EN termination patterns covering real-world phrasing variations
  - 15 JP termination patterns (exact substring, no tokenisation needed)
  - 8 EN high-signal performance-failure phrases (precede terminations)
  - 8 JP high-signal performance-failure phrases
  - Case-insensitive EN matching
  - Returns: termination_detected (bool), termination_signals (list), CRITICAL risk
  - All existing soft patterns preserved (LOW/MEDIUM/HIGH tier)

Root cause of v3.1 failure:
  The previous version only matched 8 exact JP phrases like
  「継続することはできません」but the actual transcript used
  「パートナーシップは継続しないことを決定しました」— different phrasing,
  no match, NONE risk. This version covers all common variations.
"""

import re
from typing import Optional


# ════════════════════════════════════════════════════════════════════════════════
# TIER 1 — EXPLICIT TERMINATION (CRITICAL risk, irrevocable)
# These phrases = relationship/contract is definitively ending.
# Match any of these → risk_level = CRITICAL regardless of anything else.
# ════════════════════════════════════════════════════════════════════════════════

EN_TERMINATION_PHRASES = [
    # Direct "decided not to continue" family
    ("decided not to continue",              "Explicit finalized decision not to continue"),
    ("have decided not to continue",         "Past-tense finalized decision (most common JP→EN form)"),
    ("not to continue this partnership",     "Explicit partnership non-continuation"),
    ("not to continue the partnership",      "Explicit partnership non-continuation"),
    ("not continue this partnership",        "Non-continuation"),
    ("decision not to continue",             "Nominalized termination decision"),
    # "not renew" family
    ("will not be renewing",                 "Future non-renewal"),
    ("not renewing our contract",            "Non-renewal"),
    ("decided not to renew",                 "Decision not to renew"),
    ("not to renew this contract",           "Contract non-renewal"),
    # "end / terminate / discontinue" family
    ("decided to end",                       "Decision to end relationship"),
    ("cannot continue this partnership",     "Inability-framed termination"),
    ("cannot continue our partnership",      "Inability-framed termination"),
    ("ending our partnership",               "Active ending statement"),
    ("terminating our contract",             "Direct termination"),
    ("discontinue our partnership",          "Discontinue"),
    ("end our business relationship",        "End business relationship"),
    ("this partnership is concluded",        "Concluded"),
    ("this meeting is now concluded",        "Meeting conclusion (often signals formal end)"),
]

JP_TERMINATION_PHRASES = [
    # 継続しない family (will not continue)
    ("継続しないことを決定",          "Decided not to continue — most common written form"),
    ("継続しないことを決定しました",   "Polite past — decided not to continue (exact common form)"),
    ("パートナーシップは継続しない",   "Partnership will not continue"),
    ("継続しないことを",              "Will-not-continue particle construction"),
    ("継続することはできません",       "Cannot continue (negative potential)"),
    ("継続できません",                "Cannot continue (short form)"),
    # 契約 / 更新 family (contract / renewal)
    ("契約を更新しない",              "Will not renew contract"),
    ("契約を更新しないことを決定",     "Decided not to renew contract"),
    ("ご契約を更新しない",            "Honorific — will not renew your contract"),
    ("契約終了",                      "Contract termination (noun)"),
    ("取引を終了",                    "End business dealings"),
    # 決定 family (decision)
    ("決定は最終的",                  "Decision is final"),
    ("最終的な決断",                  "Final decision"),
    # 終了 / パートナーシップ endings
    ("関係を終了",                    "Ending the relationship"),
    ("パートナーシップを終了",         "Ending the partnership"),
]


# ════════════════════════════════════════════════════════════════════════════════
# TIER 2 — PERFORMANCE FAILURE FRAMING (HIGH risk)
# These appear in the lead-up to termination announcements.
# Alone: HIGH risk. Combined with TIER 1: confirms CRITICAL context.
# ════════════════════════════════════════════════════════════════════════════════

EN_HIGH_PHRASES = [
    # Performance failure — precede termination
    ("results have not met our expectations",        "Results did not meet expectations"),
    ("not met our expectations",                     "Expectations unmet"),
    ("did not meet our expectations",                "Past tense — expectations not met"),
    ("we have not seen the level of improvement",    "No improvement observed"),
    ("have not seen sufficient improvement",         "Insufficient improvement"),
    ("did not observe sufficient improvement",       "No improvement observed"),
    ("multiple opportunities to improve",            "Multiple chances given — precedes termination"),
    ("despite multiple opportunities",               "Despite opportunities given"),
    # はい / Yes trap — acknowledgement mistaken for approval
    ("i did not say the proposal was approved",      "Explicit correction — yes did NOT mean approval"),
    ("does not mean i agree",                        "Clarification that yes = understanding not agreement"),
    ("yes often means that i understand",            "Cultural clarification of はい meaning"),
    ("need to review the proposal internally",       "Internal review still pending — no decision made"),
    ("still need to review",                         "Decision explicitly deferred"),
    ("before making any decision",                   "No decision made yet — pending"),
    ("will contact you after the internal review",   "Deferred — awaiting internal ringi process"),
    ("internal review is complete",                  "Approval gated on internal review"),
    ("after the internal review",                    "Decision deferred to after review"),
]

JP_HIGH_PHRASES = [
    # Performance failure
    ("期待に達していませんでした",               "Did not meet expectations (past)"),
    ("期待に達していません",                     "Has not met expectations"),
    ("十分な改善は見られませんでした",            "Insufficient improvement observed"),
    ("十分な改善は見られません",                  "Insufficient improvement"),
    ("期待していたレベルの改善は見られません",    "Expected level of improvement not seen"),
    ("改善は見られませんでした",                  "No improvement was observed"),
    ("結果は私たちの期待に達していません",         "Results did not meet our expectations"),
    ("何度も機会を提供しました",                  "Multiple opportunities were provided"),
    # はい / Yes trap patterns
    ("承認されたとは申し上げておりません",        "I did not say it was approved — はい ≠ 承認"),
    ("社内で提案内容を検討する必要があります",    "Internal review still needed — no decision yet"),
    ("社内での検討が終わり次第",                  "Will contact after internal review — decision deferred"),
    ("決定を下す前に",                            "Before making any decision — explicitly unresolved"),
    ("はい」は相手の話を理解したという意味",      "Explicit cultural clarification: はい = understanding not approval"),
    ("必ずしも賛成や承認を意味するわけではありません", "Yes does not necessarily mean agreement or approval"),
]


# ════════════════════════════════════════════════════════════════════════════════
# TIER 3 — SOFT REJECTIONS (your existing patterns — LOW/MEDIUM/HIGH)
# These are the original 20 indirect/hedged refusal patterns.
# Keep these exactly as they are in your actual file.
# ════════════════════════════════════════════════════════════════════════════════

SOFT_PATTERNS = [
    {
        "phrase": "検討いたします",
        "reading": "Kentō itashimasu",
        "english": "We will consider it",
        "confidence": 0.75,
        "explanation": "Classic nemawashi deflection — 'we will consider' without commitment.",
    },
    {
        "phrase": "難しい状況です",
        "reading": "Muzukashii jōkyō desu",
        "english": "It's a difficult situation",
        "confidence": 0.80,
        "explanation": "Indirect refusal framed as circumstance.",
    },
    {
        "phrase": "難しいですね",
        "reading": "Muzukashii desu ne",
        "english": "That's difficult, isn't it",
        "confidence": 0.85,
        "explanation": "Hedged rejection seeking shared acknowledgement of difficulty.",
    },
    {
        "phrase": "ぜひ検討させていただきます",
        "reading": "Zehi kentō sasete itadakimasu",
        "english": "We would certainly like to consider it",
        "confidence": 0.72,
        "explanation": "Enthusiasm framing masks deferral — zehi used non-committally.",
    },
    {
        "phrase": "ぜひそうしたいところですが",
        "reading": "Zehi sō shitai tokoro desu ga",
        "english": "We would certainly like to do so, but...",
        "confidence": 0.88,
        "explanation": "Trailing が signals real refusal is coming after.",
    },
    {
        "phrase": "少々お時間をいただけますか",
        "reading": "Shōshō ojikan wo itadakemasu ka",
        "english": "Could we have a little more time?",
        "confidence": 0.70,
        "explanation": "Request for delay — common postponement in nemawashi process.",
    },
    {
        "phrase": "上の者と相談いたします",
        "reading": "Ue no mono to sōdan itashimasu",
        "english": "I will consult with my superiors",
        "confidence": 0.78,
        "explanation": "Escalation deflection — decision deferred upward.",
    },
    {
        "phrase": "前向きに検討します",
        "reading": "Maemuki ni kentō shimasu",
        "english": "We will consider it positively",
        "confidence": 0.82,
        "explanation": "前向き (positive) often signals polite non-commitment, not genuine intent.",
    },
    {
        "phrase": "It might be difficult",
        "reading": "EN direct equivalent",
        "english": "It might be difficult",
        "confidence": 0.76,
        "explanation": "Modal hedging — difficulty framed as obstacle, not refusal.",
    },
    {
        "phrase": "we need to think about it",
        "reading": "EN direct equivalent",
        "english": "We need to think about it",
        "confidence": 0.65,
        "explanation": "Deliberation request — deferral signal.",
    },
# ── Restored from the pre-v3.2 pattern set (were dropped in the rewrite) ──
    {
        "phrase": "難しいかもしれません",
        "reading": "Muzukashii kamoshiremasen",
        "english": "It may be difficult",
        "confidence": 0.90,
        "explanation": "Classic soft rejection — direct 'no' is culturally avoided.",
    },
    {
        "phrase": "対応しかねます",
        "reading": "Taiō shikanemasu",
        "english": "We are unable to accommodate",
        "confidence": 0.95,
        "explanation": "One of the most direct soft rejections — formal and definitive.",
    },
    {
        "phrase": "いたしかねます",
        "reading": "Itashikanemasu",
        "english": "We are unable to do that",
        "confidence": 0.95,
        "explanation": "Formal polite rejection — very definitive despite soft delivery.",
    },
    {
        "phrase": "善処します",
        "reading": "Zensho shimasu",
        "english": "I will handle it appropriately",
        "confidence": 0.68,
        "explanation": "Vague commitment with no concrete action.",
    },
    {
        "phrase": "確認してみます",
        "reading": "Kakunin shite mimasu",
        "english": "I will try to confirm",
        "confidence": 0.50,
        "explanation": "Genuine uncertainty or deferral, may need superior's approval.",
    },
    {
        "phrase": "社内で確認",
        "reading": "Shanai de kakunin",
        "english": "Will confirm internally",
        "confidence": 0.48,
        "explanation": "Internal confirmation pending — decision not yet made.",
    },
    {
        "phrase": "上司に相談",
        "reading": "Jōshi ni sōdan",
        "english": "Will consult with my superior",
        "confidence": 0.50,
        "explanation": "Escalation to a superior — may be genuine or a delaying tactic.",
    },
    {
        "phrase": "少し懸念",
        "reading": "Sukoshi kenen",
        "english": "A little concerned",
        "confidence": 0.40,
        "explanation": "Signals discomfort or disagreement expressed indirectly.",
    },
    {
        "phrase": "懸念がございます",
        "reading": "Kenen ga gozaimasu",
        "english": "There are concerns",
        "confidence": 0.45,
        "explanation": "Formal expression of concern, speaker disagrees indirectly.",
    },
    {
        "phrase": "そうですね",
        "reading": "Sō desu ne",
        "english": "That's right / I see",
        "confidence": 0.25,
        "explanation": "Ambiguous — genuine agreement OR filler to avoid disagreement.",
    },
        # ── はい / Yes Trap patterns ──────────────────────────────────────────────
    {
        "phrase": "承知しました",
        "reading": "Shōchi shimashita",
        "english": "I understand / Noted",
        "confidence": 0.78,
        "explanation": (
            "承知しました = 'I have understood' — commonly mistaken for agreement or approval "
            "by non-Japanese speakers. In this context it means the content was received, "
            "NOT that the proposal, request, or decision has been accepted."
        ),
    },
    {
        "phrase": "はい、承知しました",
        "reading": "Hai, shōchi shimashita",
        "english": "Yes, I understand (not: Yes, I approve)",
        "confidence": 0.85,
        "explanation": (
            "The combination of はい + 承知しました is the most common はい trap. "
            "Both words signal understanding and active listening, not approval. "
            "Indian and Western counterparts frequently interpret this as a yes to their proposal."
        ),
    },
    {
        "phrase": "ご提案の内容は理解しました",
        "reading": "Go-teian no naiyō wa rikai shimashita",
        "english": "I understand the content of your proposal (not: I approve it)",
        "confidence": 0.90,
        "explanation": (
            "理解しました = 'I have understood'. This is specifically used to close off the "
            "Indian rep's assumption that approval was given. High confidence signal that "
            "no decision has been made."
        ),
    },
]


# ════════════════════════════════════════════════════════════════════════════════
# HELPER
# ════════════════════════════════════════════════════════════════════════════════

def _find_speaker(phrase: str, transcript: str, case_insensitive: bool = False) -> str:
    """Best-effort: find which speaker line contains this phrase."""
    lines = transcript.split("\n")
    for line in lines:
        check_line = line.lower() if case_insensitive else line
        check_phrase = phrase.lower() if case_insensitive else phrase
        if check_phrase in check_line:
            # Handle **Name:**, Name:, [Name]:, 【Name】：
            m = re.match(r"^\*?\*?([^:*\[\]【】\n]{1,50}?)\*?\*?\s*[：:]\s*", line.strip())
            if m:
                return m.group(1).strip("* []【】").strip()
    return "Unknown"


# ════════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ════════════════════════════════════════════════════════════════════════════════

def detect_soft_rejections(transcript: str) -> dict:
    """
    Detect termination statements and soft rejections in a JP/EN/mixed transcript.

    Risk levels (highest to lowest):
        CRITICAL  — explicit contract/partnership termination detected
        HIGH      — multiple strong performance-failure + soft signals
        MEDIUM    — moderate soft rejection signals
        LOW       — mild hedging
        MINIMAL   — one or two weak signals
        NONE      — nothing found

    New keys in v3.2 return dict:
        termination_detected  bool   True if any CRITICAL-tier phrase matched
        termination_signals   list   Each matched phrase with speaker + explanation
    """
    transcript_lower = transcript.lower()

    # ── TIER 1: Termination check ─────────────────────────────────────────────
    termination_signals = []

    for phrase, explanation in EN_TERMINATION_PHRASES:
        if phrase.lower() in transcript_lower:
            speaker = _find_speaker(phrase, transcript, case_insensitive=True)
            termination_signals.append({
                "phrase":       phrase,
                "reading":      phrase,          # EN needs no romaji
                "english":      phrase,
                "category":     "explicit_termination",
                "confidence":   0.97,
                "explanation":  explanation,
                "speaker":      speaker,
                "language":     "EN",
                "is_explicit_termination": True,
            })

    for phrase, explanation in JP_TERMINATION_PHRASES:
        if phrase in transcript:
            speaker = _find_speaker(phrase, transcript, case_insensitive=False)
            termination_signals.append({
                "phrase":       phrase,
                "reading":      "",              # populated if you have a romaji map
                "english":      explanation,
                "category":     "explicit_termination",
                "confidence":   0.99,
                "explanation":  explanation,
                "speaker":      speaker,
                "language":     "JP",
                "is_explicit_termination": True,
            })

    termination_detected = len(termination_signals) > 0

    # Deduplicate by phrase (EN + JP may both match for bilingual lines)
    seen = set()
    deduped = []
    for s in termination_signals:
        key = s["phrase"]
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    termination_signals = deduped

    # ── TIER 2: Performance failure phrases → HIGH signals ────────────────────
    high_signals = []

    for phrase, explanation in EN_HIGH_PHRASES:
        if phrase.lower() in transcript_lower:
            speaker = _find_speaker(phrase, transcript, case_insensitive=True)
            high_signals.append({
                "phrase":      phrase,
                "reading":     phrase,
                "english":     phrase,
                "confidence":  0.88,
                "explanation": explanation,
                "speaker":     speaker,
            })

    for phrase, explanation in JP_HIGH_PHRASES:
        if phrase in transcript:
            speaker = _find_speaker(phrase, transcript, case_insensitive=False)
            high_signals.append({
                "phrase":      phrase,
                "reading":     "",
                "english":     explanation,
                "confidence":  0.88,
                "explanation": explanation,
                "speaker":     speaker,
            })

    # ── TIER 3: Soft rejection patterns → LOW/MEDIUM signals ─────────────────
    medium_signals = []
    low_signals    = []

    for pattern in SOFT_PATTERNS:
        phrase = pattern["phrase"]
        # EN patterns: case-insensitive; JP patterns: exact
        found = (phrase.lower() in transcript_lower) if re.search(r'[a-zA-Z]', phrase) else (phrase in transcript)
        if not found:
            continue
        speaker = _find_speaker(phrase, transcript, case_insensitive=True)
        signal = {**pattern, "speaker": speaker}
        conf = pattern.get("confidence", 0.5)
        if conf >= 0.80:
            medium_signals.append(signal)
        else:
            low_signals.append(signal)

    total_signals = len(high_signals) + len(medium_signals) + len(low_signals)

    # ── Risk level ─────────────────────────────────────────────────────────────
    if termination_detected:
        risk_level = "CRITICAL"
    elif len(high_signals) >= 3:
        risk_level = "HIGH"
    elif len(high_signals) >= 1 or len(medium_signals) >= 2:
        risk_level = "MEDIUM"
    elif len(medium_signals) >= 1:
        risk_level = "LOW"
    elif total_signals >= 1:
        risk_level = "MINIMAL"
    else:
        risk_level = "NONE"

    # ── Cultural note ──────────────────────────────────────────────────────────
    if termination_detected:
        cultural_note = (
            "⚠ Explicit contract/partnership termination detected — this is NOT a soft "
            "rejection or negotiable refusal. The decision is irrevocable. In Japanese "
            "business culture, this language is only used AFTER internal ringi-sho "
            "(稟議書) approval has been finalized. The polite keigo delivery is cultural "
            "courtesy, not a signal of openness to reconsideration. One follow-up "
            "request (再検討していただけますか) is culturally acceptable; repeating it "
            "would be considered disrespectful to the decision's finality."
        )
    elif risk_level == "HIGH":
        cultural_note = (
            "Multiple performance-failure signals detected alongside hedging language. "
            "This pattern frequently precedes a formal termination announcement in "
            "Japanese business meetings. Proactive remediation discussion is advised "
            "before the next meeting."
        )
    elif risk_level in ("MEDIUM", "LOW"):
        cultural_note = (
            "Indirect rejection signals detected. In Japanese business culture, direct "
            "refusal is avoided to preserve face (面子) for all parties. These patterns "
            "warrant careful follow-up to confirm actual intent and timeline."
        )
    else:
        cultural_note = "No significant rejection signals detected in this transcript."

    return {
        "risk_level":           risk_level,
        "total_signals":        total_signals + len(termination_signals),
        "high_signals":         high_signals,
        "medium_signals":       medium_signals,
        "low_signals":          low_signals,
        "termination_detected": termination_detected,
        "termination_signals":  termination_signals,
"cultural_note":        cultural_note,
        "detected": (
            [{**s, "severity": "HIGH"}   for s in termination_signals] +
            [{**s, "severity": "HIGH"}   for s in high_signals] +
            [{**s, "severity": "MEDIUM"} for s in medium_signals] +
            [{**s, "severity": "LOW"}    for s in low_signals]
        ),
        "risk_summary": cultural_note,
    }