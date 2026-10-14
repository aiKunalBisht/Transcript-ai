"""
analysis/deal_outcome_detector.py
==================================
Detects EXPLICIT deal-confirmed / acceptance language — the positive
counterpart to soft_rejection_detector.py's termination/rejection detection.

v1.1 fix (P3):
  P3-FIX: compute_meeting_outcome — added HIGH_RISK_STALL verdict.
    When approval_gate_detected AND risk is HIGH or CRITICAL, the gate is
    likely a deferral tactic masking a stall, not a genuine pending review.
    Previously: approval_gate fired at priority 5, eclipsing risk level.
    Now: gate+HIGH → HIGH_RISK_STALL (priority 5), gate alone → PENDING (5b).
    DSA: O(1) — single additional elif in a linear verdict chain.
"""

import re


# ════════════════════════════════════════════════════════════════════════════════
# EXPLICIT DEAL CONFIRMATION
# ════════════════════════════════════════════════════════════════════════════════

EN_ACCEPTANCE_PHRASES = [
    ("we accept your proposal",            "Direct, explicit acceptance"),
    ("we accept the proposal",             "Direct, explicit acceptance"),
    ("we accept these terms",              "Explicit terms acceptance"),
    ("we accept your terms",               "Explicit terms acceptance"),
    ("we accept the terms",                "Explicit terms acceptance"),
    ("sign the agreement",                 "Explicit agreement commitment"),
    ("will sign the agreement",            "Explicit agreement commitment"),
    ("sign the contract",                  "Explicit contract commitment"),
    ("will sign the contract",             "Explicit contract commitment"),
    ("we agree to move forward",           "Explicit decision to proceed"),
    ("we have decided to move forward",    "Finalized decision to proceed"),
    ("decided to move forward",            "Finalized decision to proceed"),
    ("decided to proceed",                 "Finalized decision to proceed"),
    ("we will proceed with this",          "Commitment to proceed"),
    ("we will renew the contract",         "Explicit renewal commitment"),
    ("we have decided to renew",           "Finalized renewal decision"),
    ("the contract has been approved",     "Explicit contract approval"),
    ("the contract is approved",           "Explicit contract approval"),
    ("this proposal is approved",          "Explicit proposal approval"),
    ("we are pleased to confirm",          "Formal positive confirmation"),
    ("happy to confirm",                   "Positive confirmation"),
    ("we are happy to move forward",       "Explicit, enthusiastic commitment"),
    ("the deal is confirmed",              "Explicit deal confirmation"),
    ("we look forward to continuing",      "Explicit continuation commitment"),
    ("we are excited to continue",         "Explicit, enthusiastic continuation"),
    ("let's move forward with this",       "Explicit go-ahead"),
]

JP_ACCEPTANCE_PHRASES = [
    ("契約を更新することに決定しました",        "Decided to renew the contract"),
    ("契約を更新いたします",                    "Will renew the contract"),
    ("本提案を承認いたします",                  "Formally approving this proposal"),
    ("ご提案を承認します",                      "Approving your proposal"),
    ("正式に契約を締結します",                  "Formally concluding the contract"),
    ("貴社との契約を継続いたします",            "Will continue the contract with your company"),
    ("今回の提案を受け入れます",                "Accepting this proposal"),
    ("進めることで合意しました",                "Agreed to proceed"),
    ("協力を継続することに決定しました",        "Decided to continue the cooperation"),
    ("この条件で進めさせていただきます",        "Will proceed under these terms"),
    ("貴社との協力関係を継続します",            "Will continue the partnership with your company"),
    ("ぜひ一緒に進めていきましょう",            "Let's move forward together"),
]


# ════════════════════════════════════════════════════════════════════════════════
# CONDITIONAL APPROVAL
# ════════════════════════════════════════════════════════════════════════════════

EN_CONDITIONAL_PHRASES = [
    ("approved subject to",                "Approval explicitly contingent on a condition"),
    ("we accept on the condition that",    "Acceptance explicitly conditioned"),
    ("we will proceed provided that",      "Proceeding contingent on a stated condition"),
    ("conditional approval",               "Explicitly labeled as conditional"),
    ("approved with the condition",        "Approval qualified by a condition"),
    ("we agree, provided",                 "Agreement qualified by a condition"),
    ("subject to the following conditions", "Explicit conditions attached to approval"),
    ("contingent upon",                    "Outcome made contingent on something else"),
    ("approval is conditional on",         "Approval explicitly tied to a condition"),
    ("only if you can",                    "Approval gated on a specific ask being met"),
]

JP_CONDITIONAL_PHRASES = [
    ("条件付きで承認します",                  "Conditional approval — explicit qualifier"),
    ("という条件で進めます",                  "Will proceed under a stated condition"),
    ("条件が整えば進めます",                  "Will proceed once conditions are met"),
    ("一定の条件を満たせば",                  "Contingent on certain conditions being satisfied"),
    ("条件次第で承認いたします",              "Approval depends on the condition"),
]


# ════════════════════════════════════════════════════════════════════════════════
# DEFERRED — explicitly pushed to a future meeting
# ════════════════════════════════════════════════════════════════════════════════

EN_DEFERRED_PHRASES = [
    ("let's discuss this in our next meeting",  "Explicit push to a future meeting"),
    ("we will revisit this next month",         "Explicit future revisit, dated"),
    ("postponing this decision",                "Explicit postponement"),
    ("decision is postponed",                   "Explicit postponement"),
    ("let's table this for now",                "Explicit deferral"),
    ("we'll pick this up next time",            "Explicit deferral to a future session"),
    ("deferred to the next meeting",            "Explicit deferral, named"),
    ("revisit at our next meeting",             "Explicit future revisit"),
    ("let's continue this discussion next time", "Explicit deferral of the discussion itself"),
    ("confirm the meeting on",                  "Scheduling confirmation — coordination meeting"),
    ("next meeting will be on",                 "Next meeting scheduled — outcome deferred"),
    ("schedule a follow-up on",                 "Follow-up scheduled — no decision yet"),
    ("confirm the technical meeting",           "Technical meeting confirmation — informational"),
    ("we will meet again on",                   "Next session scheduled"),
    ("let us reconnect on",                     "Reconnection scheduled — decision pending"),
]

JP_DEFERRED_PHRASES = [
    ("次回の会議で改めて議論します",            "Will discuss again at the next meeting"),
    ("決定を次回に持ち越します",                "Carrying the decision over to next time"),
    ("次回まとめて検討します",                  "Will consider together at the next session"),
    ("次の機会に持ち越したいと思います",        "Would like to carry this over to the next opportunity"),
]


# ════════════════════════════════════════════════════════════════════════════════
# INFORMATIONAL — explicitly no decision needed
# ════════════════════════════════════════════════════════════════════════════════

EN_INFORMATIONAL_PHRASES = [
    ("this is just an update",                   "Explicitly framed as a status update, not a decision point"),
    ("for your information only",                "Explicit FYI framing"),
    ("no decision is needed today",              "Explicit statement that no decision is expected"),
    ("this meeting is for informational purposes", "Explicit informational framing"),
    ("purely informational",                     "Explicit informational framing"),
    ("just a status update",                     "Explicit status-update framing"),
    ("no action required at this time",          "Explicit no-action framing"),
    ("this was an informational session",        "Explicit informational framing, past tense"),
    ("this is a coordination meeting",           "Explicitly a logistics/coordination session"),
    ("this call is to align on",                 "Alignment call — no decision expected"),
    ("today we are confirming",                  "Confirmation meeting — not a decision point"),
]

JP_INFORMATIONAL_PHRASES = [
    ("本日は決定事項はありません",              "Explicit statement that there are no decisions today"),
    ("情報共有のための会議です",                "Explicitly framed as an information-sharing meeting"),
    ("本日は報告のみです",                      "Explicitly a report-only session"),
    ("決定は不要です",                          "Explicit statement that no decision is required"),
]


def _find_speaker(phrase: str, transcript: str, case_insensitive: bool = False) -> str:
    lines = transcript.split("\n")
    for line in lines:
        check_line   = line.lower() if case_insensitive else line
        check_phrase = phrase.lower() if case_insensitive else phrase
        if check_phrase in check_line:
            m = re.match(r"^\*?\*?([^:*\[\]【】\n]{1,50}?)\*?\*?\s*[：:]\s*", line.strip())
            if m:
                return m.group(1).strip("* []【】").strip()
    return "Unknown"


def detect_deal_outcome(transcript: str) -> dict:
    """
    Detect explicit deal/acceptance confirmation — plus conditional approval,
    deferral, and explicit informational framing.
    """
    transcript_lower = transcript.lower()

    def _scan(en_list, jp_list, confidence_en, confidence_jp):
        signals = []
        for phrase, explanation in en_list:
            if phrase.lower() in transcript_lower:
                signals.append({
                    "phrase": phrase, "reading": phrase, "english": phrase,
                    "confidence": confidence_en, "explanation": explanation,
                    "speaker": _find_speaker(phrase, transcript, case_insensitive=True),
                    "language": "EN",
                })
        for phrase, explanation in jp_list:
            if phrase in transcript:
                signals.append({
                    "phrase": phrase, "reading": "", "english": explanation,
                    "confidence": confidence_jp, "explanation": explanation,
                    "speaker": _find_speaker(phrase, transcript, case_insensitive=False),
                    "language": "JP",
                })
        seen = set(); deduped = []
        for s in signals:
            if s["phrase"] not in seen:
                seen.add(s["phrase"]); deduped.append(s)
        return deduped

    confirmation_signals  = _scan(EN_ACCEPTANCE_PHRASES,    JP_ACCEPTANCE_PHRASES,    0.95, 0.96)
    conditional_signals   = _scan(EN_CONDITIONAL_PHRASES,   JP_CONDITIONAL_PHRASES,   0.90, 0.92)
    deferred_signals      = _scan(EN_DEFERRED_PHRASES,      JP_DEFERRED_PHRASES,      0.88, 0.90)
    informational_signals = _scan(EN_INFORMATIONAL_PHRASES, JP_INFORMATIONAL_PHRASES, 0.85, 0.88)

    return {
        "deal_confirmed":          len(confirmation_signals) > 0,
        "confirmation_signals":    confirmation_signals,
        "conditional_detected":    len(conditional_signals) > 0,
        "conditional_signals":     conditional_signals,
        "deferred_detected":       len(deferred_signals) > 0,
        "deferred_signals":        deferred_signals,
        "informational_detected":  len(informational_signals) > 0,
        "informational_signals":   informational_signals,
        "total_signals": (
            len(confirmation_signals) + len(conditional_signals)
            + len(deferred_signals) + len(informational_signals)
        ),
    }


VERDICT_STYLES = {
    "REJECTED":         {"emoji": "🔴", "label": "Rejected",          "color": "#C84040", "meaning": "Final decision made not to proceed"},
    "APPROVED":         {"emoji": "🟢", "label": "Approved",          "color": "#2D9E6B", "meaning": "Final decision made to proceed"},
    "CONDITIONAL":      {"emoji": "🔵", "label": "Conditional",       "color": "#3B82C4", "meaning": "Approved only if certain conditions are met"},
    "DEFERRED":         {"emoji": "🟣", "label": "Deferred",          "color": "#7D4E8A", "meaning": "Decision postponed to a future meeting"},
    # P3-FIX: HIGH_RISK_STALL — approval gate present but soft rejection signals are HIGH.
    # Distinct from PENDING (genuine review) — gate is likely a deferral tactic.
    "HIGH_RISK_STALL":  {"emoji": "🟠", "label": "High Risk Stall",   "color": "#C87030", "meaning": "Approval gate present but risk signals are HIGH — internal review may be a stall tactic"},
    "PENDING":          {"emoji": "🟡", "label": "Pending",           "color": "#B8860B", "meaning": "Waiting for internal review or approval"},
    "INFORMATIONAL":    {"emoji": "⚪", "label": "Informational",     "color": "#7A7A7A", "meaning": "Discussion only, no decision expected"},
    "AT_RISK":          {"emoji": "🟠", "label": "At Risk",           "color": "#D2691E", "meaning": "Trouble signals present, no resolution stated either way"},
    "UNCLEAR":          {"emoji": "⚫", "label": "Unclear",           "color": "#3C2416", "meaning": "No decision — clear or otherwise — was stated in this transcript"},
}


def compute_meeting_outcome(soft_rejections: dict, deal_outcome: dict) -> dict:
    """
    Combines soft_rejection_detector output with deal_outcome into ONE verdict.

    Priority order (P3-FIX: HIGH_RISK_STALL inserted at 5a):
      1. REJECTED         — explicit termination
      2. CONDITIONAL      — checked before APPROVED
      3. APPROVED         — explicit, unconditional acceptance
      4. DEFERRED         — pushed to future meeting
      5a. HIGH_RISK_STALL — approval gate AND risk HIGH/CRITICAL (gate = stall tactic)
      5b. PENDING         — approval gate alone (genuine review)
      6. INFORMATIONAL    — explicitly no decision needed
      7. AT_RISK          — soft rejection risk HIGH or MEDIUM, no resolution
      8. UNCLEAR          — fallback

    P3-FIX root cause: approval_gate_detected previously fired at priority 5
    before the risk-level check at priority 7. When a meeting had 4 soft
    rejection signals (risk=HIGH) wrapped in "need internal approval" language,
    the gate eclipsed the risk and returned PENDING instead of HIGH_RISK_STALL.

    DSA: O(1) — linear if-elif verdict chain with one added branch.
    """
    soft = soft_rejections or {}
    deal = deal_outcome    or {}
    risk = (soft.get("risk_level") or "NONE").upper()

    if soft.get("termination_detected"):
        verdict, detail = "REJECTED", soft.get("cultural_note", "")

    elif deal.get("conditional_detected"):
        sig = deal.get("conditional_signals", [])
        verdict, detail = "CONDITIONAL", (sig[0]["explanation"] if sig else "Conditional approval language detected.")

    elif deal.get("deal_confirmed"):
        sig = deal.get("confirmation_signals", [])
        verdict, detail = "APPROVED", (sig[0]["explanation"] if sig else "Explicit acceptance language detected.")

    elif deal.get("deferred_detected"):
        sig = deal.get("deferred_signals", [])
        verdict, detail = "DEFERRED", (sig[0]["explanation"] if sig else "Decision explicitly postponed to a future meeting.")

    # P3-FIX: gate + HIGH/CRITICAL risk = stall, not genuine pending review
    elif soft.get("approval_gate_detected") and risk in ("HIGH", "CRITICAL"):
        verdict, detail = "HIGH_RISK_STALL", (
            soft.get("cultural_note", "")
            or "Approval gate detected alongside high soft-rejection signals — "
               "internal review may be a deferral tactic rather than a genuine review."
        )

    elif soft.get("approval_gate_detected"):
        verdict, detail = "PENDING", soft.get("cultural_note", "")

    elif deal.get("informational_detected"):
        sig = deal.get("informational_signals", [])
        verdict, detail = "INFORMATIONAL", (sig[0]["explanation"] if sig else "Explicitly framed as not requiring a decision.")

    elif risk in ("HIGH", "MEDIUM"):
        verdict, detail = "AT_RISK", soft.get("cultural_note", "")

    else:
        verdict, detail = "UNCLEAR", "Neither explicit acceptance, rejection, nor deferral language was found in this transcript."

    style = VERDICT_STYLES[verdict]
    return {
        "verdict": verdict,
        "emoji":   style["emoji"],
        "label":   style["label"],
        "color":   style["color"],
        "meaning": style["meaning"],
        "detail":  detail,
    }


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis.soft_rejection_detector import detect_soft_rejections

    tests = {
        "APPROVED":         "Client: We have reviewed everything and we accept your proposal.",
        "REJECTED":         "Director: We have decided not to renew our contract.",
        "CONDITIONAL":      "Client: We accept on the condition that you reduce the price by 10%.",
        "DEFERRED":         "Tanaka: Let's discuss this in our next meeting and decide then.",
        # HIGH_RISK_STALL: needs multiple rejection signals (→ HIGH risk) + approval gate.
        # "cannot commit" + "serious concerns" + "not convinced" = HIGH risk;
        # "approval from headquarters" = gate. Combined → HIGH_RISK_STALL.
        "HIGH_RISK_STALL": (
            "Director: We cannot commit to this at the current pricing. "
            "We have serious concerns about the delivery timeline. "
            "We are not convinced this solution meets our requirements. "
            "We will need approval from headquarters before we can consider moving forward."
        ),
        # PENDING: approval gate only, no rejection signals → risk stays LOW/NONE.
        # Gate alone → PENDING.
        "PENDING": (
            "Director: The proposal looks promising overall. "
            "We will need to get sign-off from our procurement committee before we can proceed."
        ),
        "INFORMATIONAL":    "Tanaka: This is just an update — no decision is needed today.",
        "AT_RISK":          "Director: Results have not met our expectations. We have not seen sufficient improvement.",
        "UNCLEAR":          "Tanaka: Let's continue discussing this next week.",
        "DEFERRED_sched":   "Yamamoto: We will confirm the meeting on Friday to proceed.",
        "INFO_coord":       "Tanaka: This is a coordination meeting — no decision needed.",
    }
    print("=== P3 Verdict Engine Self-Test ===")
    for expected, t in tests.items():
        deal    = detect_deal_outcome(t)
        soft    = detect_soft_rejections(t)
        outcome = compute_meeting_outcome(soft, deal)
        ok = "OK  " if outcome["verdict"].startswith(expected.split("_")[0]) else "FAIL"
        print(f"[{ok}] expected={expected:20} got={outcome['verdict']:16} {outcome['emoji']} {outcome['label']}")