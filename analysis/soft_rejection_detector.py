"""
analysis/soft_rejection_detector.py  — TranscriptAI v3.3
=========================================================
v3.3 changes vs v3.2:

C1 FIX: New TIER 1c — EN_CONTRACT_RISK_PHRASES (21 patterns)
    The single biggest gap in v3.2. The detector had explicit termination
    (already happened) and performance failure (past context), but was
    completely blind to CONDITIONAL THREATS — the most actionable signal
    in any client meeting. "Contract would be reconsidered" is the last
    warning before formal termination. v3.2 returned NONE for this.
    The demo transcript produced ZERO matches against a meeting where
    the client threatened contract reconsideration and set a Friday
    deadline. v3.3 adds conditional reconsideration, "not acceptable",
    and cannot-continue patterns as HIGH → CRITICAL signals.

C2 FIX: JP_CONTRACT_RISK_PHRASES (12 patterns)
    Japanese equivalents for conditional threats, deadline ultimatums,
    and written commitment demands in keigo register.

C3 FIX: EN_HIGH_PHRASES expanded (+18 patterns)
    Three missing signal categories added:
    — Deadline ultimatums: "if not resolved by", "must be resolved by",
      "unless this is resolved"
    — Written demand signals: "written commitment", "written response",
      "demand a written", "put it in writing"
    — SLA / downtime complaints: "system has been down", "been down for",
      "hours of downtime", "still not working", "this keeps happening"

C4 FIX: JP_HIGH_PHRASES expanded (+8 patterns)
    Japanese equivalents: written commitment demand, deadline ultimatum,
    conditional improvement threat, hours-of-downtime framing.

C5 FIX: Risk level updated
    contract_risk_detected alone → HIGH
    contract_risk_detected + any high_signal → CRITICAL
    Conditional threats + performance failure = same risk as explicit
    termination from an account management perspective.

C6 FIX: Cultural note added for contract risk tier
    Explains why conditional threats in Japanese business context are
    nearly as serious as explicit termination — ringi approval is
    required before the formal phrase can be used, so the conditional
    framing means internal discussions have already begun.

Root cause of v3.2 gap:
    v3.2 was built to fix the JP-only exact-phrase miss (「継続しないことを
    決定しました」). It covered all explicit termination variants but
    never modelled the PRE-TERMINATION conditional threat pattern, which
    is the most common form in English-language client escalations.

All v3.2 patterns preserved unchanged.
"""

import re
from typing import Optional


# ════════════════════════════════════════════════════════════════════════════════
# TIER 1 — EXPLICIT TERMINATION (CRITICAL, irrevocable)
# ════════════════════════════════════════════════════════════════════════════════

EN_TERMINATION_PHRASES = [
    ("decided not to continue",              "Explicit finalized decision not to continue"),
    ("have decided not to continue",         "Past-tense finalized decision (most common JP→EN form)"),
    ("not to continue this partnership",     "Explicit partnership non-continuation"),
    ("not to continue the partnership",      "Explicit partnership non-continuation"),
    ("not continue this partnership",        "Non-continuation"),
    ("decision not to continue",             "Nominalized termination decision"),
    ("will not be renewing",                 "Future non-renewal"),
    ("not renewing our contract",            "Non-renewal"),
    ("decided not to renew",                 "Decision not to renew"),
    ("not to renew this contract",           "Contract non-renewal"),
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
    ("継続しないことを決定",          "Decided not to continue — most common written form"),
    ("継続しないことを決定しました",   "Polite past — decided not to continue (exact common form)"),
    ("パートナーシップは継続しない",   "Partnership will not continue"),
    ("継続しないことを",              "Will-not-continue particle construction"),
    ("継続することはできません",       "Cannot continue (negative potential)"),
    ("継続できません",                "Cannot continue (short form)"),
    ("契約を更新しない",              "Will not renew contract"),
    ("契約を更新しないことを決定",     "Decided not to renew contract"),
    ("ご契約を更新しない",            "Honorific — will not renew your contract"),
    ("契約終了",                      "Contract termination (noun)"),
    ("取引を終了",                    "End business dealings"),
    ("決定は最終的",                  "Decision is final"),
    ("最終的な決断",                  "Final decision"),
    ("関係を終了",                    "Ending the relationship"),
    ("パートナーシップを終了",         "Ending the partnership"),
]


# ════════════════════════════════════════════════════════════════════════════════
# TIER 1b — APPROVAL GATE (HIGH risk, not CRITICAL)
# ════════════════════════════════════════════════════════════════════════════════

EN_APPROVAL_GATE_PHRASES = [
    ("commercial contract has not yet been approved",       "Technical approval exists but commercial contract still pending"),
    ("technical approval and contract approval are separate","Explicit split: tech approval ≠ contract approval"),
    ("technical review is complete, but",                   "Technical done BUT commercial not — gate pattern"),
    ("not be interpreted as contract approval",             "Explicit denial that meeting = approval"),
    ("today's meeting should not be interpreted",           "Meeting explicitly not an approval event"),
    ("purchasing committee must review",                    "Committee gate — final authority not present in meeting"),
    ("committee must review",                               "Review committee required before decision"),
    ("before any final decision",                           "Explicitly no final decision yet"),
    ("headquarters in tokyo must make the final decision",  "Director lacks final authority — HQ decides"),
    ("headquarters must make the final decision",           "Authority escalation to headquarters"),
    ("the board has reached a different conclusion",        "Board overrides personal support — authority conflict"),
    ("board has reached a different",                       "Organizational decision differs from personal view"),
    ("i personally support this proposal",                  "Personal support explicitly separated from org decision"),
    ("personally support",                                  "Personal opinion flagged — may not reflect org decision"),
    ("does not have the authority",                         "Authority delegation gap explicitly stated"),
    ("final decision rests with",                           "Final authority delegated elsewhere"),
    ("cannot approve this myself",                          "Speaker acknowledges own authority limit"),
    ("above my authority",                                  "Explicit authority ceiling"),
    ("need approval from",                                  "Approval chain — additional gate required"),
    ("senior management must approve",                      "Senior gate — meeting result not final"),
    ("executive committee",                                 "Executive-level gate required"),
    ("board approval required",                             "Board gate — not approvable at this level"),
    ("not within my authority",                             "Authority delegation gap"),
]

JP_APPROVAL_GATE_PHRASES = [
    ("商業契約はまだ承認されていません",        "Commercial contract not yet approved"),
    ("技術承認と契約承認は別の手続きです",      "Technical and contract approval are separate processes"),
    ("技術審査は完了していますが",              "Technical review complete BUT (commercial pending)"),
    ("契約承認を意味するものではありません",    "Does not mean contract approval — explicit denial"),
    ("購買委員会が価格と契約条件を審査",        "Purchasing committee must review pricing and terms"),
    ("最終決定の前に",                          "Before the final decision — explicitly not final yet"),
    ("委員会の決定",                            "Committee decision required"),
    ("本社が最終決定を",                        "Headquarters makes the final decision"),
    ("取締役会が異なる結論に",                  "Board reached a different conclusion"),
    ("個人的にはこの提案を支持します",          "Personally support — but org decision may differ"),
    ("承認する権限がありません",                "Does not have approval authority"),
    ("上位の承認が必要です",                    "Higher-level approval required"),
    ("稟議が必要です",                          "Ringi-sho process required — formal approval chain"),
    ("稟議を通す必要があります",               "Must pass through ringi approval process"),
    ("役員会の承認が必要",                      "Executive board approval required"),
]


# ════════════════════════════════════════════════════════════════════════════════
# TIER 1c — CONTRACT AT RISK (HIGH → CRITICAL, conditional threat)
# C1/C2 FIX: The pre-termination warning phase — most actionable signal in
# client escalations. "Contract would be reconsidered" is the last stop before
# the ringi-sho is filed. In Japanese business culture, saying this phrase
# means internal discussions about termination have already started.
# ════════════════════════════════════════════════════════════════════════════════

EN_CONTRACT_RISK_PHRASES = [
    # Direct reconsideration threats — matched the demo transcript
    ("contract would be reconsidered",         "Conditional contract reconsideration — last warning before formal termination"),
    ("contract will be reconsidered",          "Future conditional contract threat"),
    ("contract may be reconsidered",           "Possibility of reconsideration — still a critical signal"),
    ("reconsidering our contract",             "Active reconsideration of contract underway"),
    ("reconsidering the contract",             "Contract under active reconsideration"),
    ("reconsider our contract",                "Direct conditional — will reconsider"),
    ("reconsider the contract",                "Contract reconsideration warning"),
    ("reconsider our partnership",             "Partnership reconsideration"),
    ("reconsidering our partnership",          "Partnership under active reconsideration"),
    ("reconsider our business relationship",   "Business relationship reconsideration"),
    ("reviewing whether to continue",          "Active review of continuation decision"),
    # Suspension / hold
    ("put the contract on hold",               "Contract suspension signal"),
    ("contract on hold",                       "Contract held — pending resolution"),
    ("suspend the contract",                   "Active contract suspension"),
    ("contract at risk",                       "Contract explicitly flagged as at risk"),
    # Unacceptability statements — precede formal action
    ("this is unacceptable",                   "Explicit rejection of current state — strong escalation preceding formal action"),
    ("this cannot continue",                   "Cannot-continue statement — precedes reconsideration or termination"),
    ("cannot accept this situation",           "Formal rejection of current status"),
    ("cannot accept this",                     "Rejection of current state"),
    ("no longer acceptable",                   "No longer acceptable — precedes formal action"),
    ("not acceptable",                         "Explicit unacceptability — precedes formal escalation"),
]

JP_CONTRACT_RISK_PHRASES = [
    # C2 FIX: Japanese conditional threats
    ("契約を見直すことを検討",                 "Considering reviewing/reconsidering the contract"),
    ("契約の見直し",                           "Contract review/reconsideration (noun)"),
    ("契約を再検討",                           "Reconsidering the contract"),
    ("このままでは契約",                       "If this continues, the contract... (conditional threat opener)"),
    ("契約の継続が困難",                       "Continuation of contract is difficult"),
    ("この状況は受け入れられません",            "This situation is unacceptable — explicit formal rejection"),
    ("受け入れがたい状況",                     "Unacceptable situation — formal rejection framing"),
    ("このような状況が続くようであれば",        "If this situation continues — conditional warning"),
    ("金曜日までに解決されなければ",            "If not resolved by Friday — exact deadline ultimatum"),
    ("期限までに改善されなければ",              "If not improved by deadline — conditional threat"),
    ("書面でのコミットメント",                  "Written commitment demanded — formal escalation to documentation"),
    ("文書による確約",                          "Written guarantee/commitment demanded — high-stakes escalation"),
]


# ════════════════════════════════════════════════════════════════════════════════
# TIER 2 — PERFORMANCE FAILURE + DEADLINE ULTIMATUMS (HIGH risk)
# C3/C4 FIX: Three new signal categories added
# ════════════════════════════════════════════════════════════════════════════════

EN_HIGH_PHRASES = [
    # ── Retained from v3.2 ────────────────────────────────────────────────────
    ("results have not met our expectations",        "Results did not meet expectations"),
    ("not met our expectations",                     "Expectations unmet"),
    ("did not meet our expectations",                "Past tense — expectations not met"),
    ("we have not seen the level of improvement",    "No improvement observed"),
    ("have not seen sufficient improvement",         "Insufficient improvement"),
    ("did not observe sufficient improvement",       "No improvement observed"),
    ("multiple opportunities to improve",            "Multiple chances given — precedes termination"),
    ("despite multiple opportunities",               "Despite opportunities given"),
    ("i did not say the proposal was approved",      "Explicit correction — yes did NOT mean approval"),
    ("does not mean i agree",                        "Clarification that yes = understanding not agreement"),
    ("yes often means that i understand",            "Cultural clarification of meaning"),
    ("need to review the proposal internally",       "Internal review still pending — no decision made"),
    ("still need to review",                         "Decision explicitly deferred"),
    ("before making any decision",                   "No decision made yet — pending"),
    ("will contact you after the internal review",   "Deferred — awaiting internal process"),
    ("internal review is complete",                  "Approval gated on internal review"),
    ("after the internal review",                    "Decision deferred to after review"),
    # ── C3 FIX: Deadline ultimatums ───────────────────────────────────────────
    ("if not resolved by",                           "Deadline ultimatum — explicit conditional if unresolved"),
    ("if this is not resolved",                      "Resolution ultimatum — conditional escalation"),
    ("if the problem is not resolved",               "Problem-resolution deadline ultimatum"),
    ("if this is not fixed",                         "Fix-or-else conditional"),
    ("unless this is resolved",                      "Unless-clause — conditional escalation warning"),
    ("must be resolved by",                          "Hard deadline imposed for resolution"),
    ("needs to be resolved by",                      "Deadline for resolution specified"),
    ("by end of week",                               "End-of-week deadline — escalation framing"),
    ("by friday",                                    "Friday deadline — common client ultimatum pattern"),
    ("by monday",                                    "Monday deadline — common client ultimatum pattern"),
    # ── C3 FIX: Written demand signals ────────────────────────────────────────
    ("demand a written",                             "Formal written demand — complaint escalation to documentation"),
    ("written commitment",                           "Written commitment demanded — accountability escalation"),
    ("written response",                             "Written response demanded — formal escalation signal"),
    ("written guarantee",                            "Written guarantee demanded — high-stakes escalation"),
    ("put it in writing",                            "Demand for written documentation"),
    ("in writing",                                   "Written documentation demanded — formalisation of complaint"),
    # ── C3 FIX: SLA / system failure signals ──────────────────────────────────
    ("system has been down",                         "System downtime complaint — SLA failure context"),
    ("been down for",                                "Extended downtime duration — SLA breach signal"),
    ("hours of downtime",                            "Extended downtime duration complaint"),
    ("hours the system",                             "Hours-system failure framing"),
    ("still not working",                            "Persistent failure — patience exhausted signal"),
    ("still not fixed",                              "Persistent unfixed issue — patience signal"),
    ("this keeps happening",                         "Recurring failure — precedes formal escalation"),
    ("happened before",                              "Recurrence signal — second/third occurrence"),
    ("not the first time",                           "Recurrence explicitly stated"),
    # ── C3 FIX: Escalation warnings ───────────────────────────────────────────
    ("will have to escalate",                        "Formal escalation warning to higher authority"),
    ("need to escalate",                             "Escalation signal"),
    ("taking this further",                          "Escalation to higher level"),
    ("involve our legal team",                       "Legal escalation warning — extreme signal"),
    ("involve our lawyers",                          "Legal escalation — critical signal"),
]

JP_HIGH_PHRASES = [
    # ── Retained from v3.2 ────────────────────────────────────────────────────
    ("期待に達していませんでした",               "Did not meet expectations (past)"),
    ("期待に達していません",                     "Has not met expectations"),
    ("十分な改善は見られませんでした",            "Insufficient improvement observed"),
    ("十分な改善は見られません",                  "Insufficient improvement"),
    ("期待していたレベルの改善は見られません",    "Expected level of improvement not seen"),
    ("改善は見られませんでした",                  "No improvement was observed"),
    ("結果は私たちの期待に達していません",         "Results did not meet our expectations"),
    ("何度も機会を提供しました",                  "Multiple opportunities were provided"),
    ("承認されたとは申し上げておりません",        "I did not say it was approved"),
    ("社内で提案内容を検討する必要があります",    "Internal review still needed — no decision yet"),
    ("社内での検討が終わり次第",                  "Will contact after internal review — decision deferred"),
    ("決定を下す前に",                            "Before making any decision — explicitly unresolved"),
    ("はい」は相手の話を理解したという意味",      "Explicit cultural clarification: yes = understanding not approval"),
    ("必ずしも賛成や承認を意味するわけではありません", "Yes does not necessarily mean agreement or approval"),
    # ── C4 FIX: Written demand + deadline ultimatum + SLA signals ─────────────
    ("書面での回答をお願いしたい",                "Request for written response — formal escalation"),
    ("書面でのコミットメントをいただきたい",      "Request for written commitment — accountability escalation"),
    ("システムが何時間も停止しており",            "System has been down for hours — SLA failure signal"),
    ("この問題が解決されない場合",                "If this problem is not resolved — conditional threat"),
    ("金曜日までに",                              "By Friday — deadline specification in threat context"),
    ("期限を設けさせていただきます",              "Setting a deadline — formal escalation signal"),
    ("是正されない場合は",                        "If not corrected — conditional escalation"),
    ("改善が見られない場合は",                    "If no improvement is seen — conditional threat"),
]


# ════════════════════════════════════════════════════════════════════════════════
# TIER 3 — SOFT REJECTIONS (LOW/MEDIUM/HIGH)
# Unchanged from v3.2
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
        "phrase": "難しい状況ですが",
        "reading": "Muzukashii jōkyō desu ga",
        "english": "It's a difficult situation, but...",
        "confidence": 0.85,
        "explanation": "Trailing が signals refusal coming — stronger than the base form.",
    },
    {
        "phrase": "難しいですが",
        "reading": "Muzukashii desu ga",
        "english": "That's difficult, but...",
        "confidence": 0.88,
        "explanation": "が after difficulty phrase = explicit incoming refusal.",
    },
    {
        "phrase": "対応しかねますが",
        "reading": "Taiō shikanemasu ga",
        "english": "We are unable to accommodate, but...",
        "confidence": 0.93,
        "explanation": "Softened with が — still a hard refusal.",
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
            "assumption that approval was given. High confidence signal that no decision "
            "has been made."
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
        check_line   = line.lower() if case_insensitive else line
        check_phrase = phrase.lower() if case_insensitive else phrase
        if check_phrase in check_line:
            m = re.match(r"^\*?\*?([^:*\[\]【】\n]{1,50}?)\*?\*?\s*[：:]\s*", line.strip())
            if m:
                return m.group(1).strip("* []【】").strip()
    return "Unknown"


def _dedup(signals: list) -> list:
    """Remove duplicate signals by phrase key."""
    seen, out = set(), []
    for s in signals:
        if s["phrase"] not in seen:
            seen.add(s["phrase"])
            out.append(s)
    return out


# ════════════════════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ════════════════════════════════════════════════════════════════════════════════

def detect_soft_rejections(transcript: str) -> dict:
    """
    Detect termination, contract risk, and soft rejection signals in a
    JP/EN/mixed business transcript.

    Risk levels (highest → lowest):
        CRITICAL  — explicit termination OR conditional threat + performance failure
        HIGH      — conditional contract threat OR approval gate OR multiple high signals
        MEDIUM    — moderate soft rejection signals
        LOW       — mild hedging
        MINIMAL   — one or two weak signals
        NONE      — nothing found
    """
    transcript_lower = transcript.lower()

    # ── TIER 1: Explicit termination ──────────────────────────────────────────
    termination_signals = []

    for phrase, explanation in EN_TERMINATION_PHRASES:
        if phrase.lower() in transcript_lower:
            termination_signals.append({
                "phrase":      phrase,
                "reading":     phrase,
                "english":     phrase,
                "category":    "explicit_termination",
                "confidence":  0.97,
                "explanation": explanation,
                "speaker":     _find_speaker(phrase, transcript, case_insensitive=True),
                "language":    "EN",
                "is_explicit_termination": True,
            })

    for phrase, explanation in JP_TERMINATION_PHRASES:
        if phrase in transcript:
            termination_signals.append({
                "phrase":      phrase,
                "reading":     "",
                "english":     explanation,
                "category":    "explicit_termination",
                "confidence":  0.99,
                "explanation": explanation,
                "speaker":     _find_speaker(phrase, transcript, case_insensitive=False),
                "language":    "JP",
                "is_explicit_termination": True,
            })

    termination_signals  = _dedup(termination_signals)
    termination_detected = len(termination_signals) > 0

    # ── TIER 1b: Approval gate ────────────────────────────────────────────────
    approval_gate_signals = []

    for phrase, explanation in EN_APPROVAL_GATE_PHRASES:
        if phrase.lower() in transcript_lower:
            approval_gate_signals.append({
                "phrase":      phrase,
                "reading":     phrase,
                "english":     explanation,
                "category":    "approval_gate",
                "confidence":  0.93,
                "explanation": explanation,
                "speaker":     _find_speaker(phrase, transcript, case_insensitive=True),
                "language":    "EN",
            })

    for phrase, explanation in JP_APPROVAL_GATE_PHRASES:
        if phrase in transcript:
            approval_gate_signals.append({
                "phrase":      phrase,
                "reading":     "",
                "english":     explanation,
                "category":    "approval_gate",
                "confidence":  0.95,
                "explanation": explanation,
                "speaker":     _find_speaker(phrase, transcript, case_insensitive=False),
                "language":    "JP",
            })

    approval_gate_signals  = _dedup(approval_gate_signals)
    approval_gate_detected = len(approval_gate_signals) > 0

    # ── TIER 1c: Contract risk — C1/C2 FIX ───────────────────────────────────
    contract_risk_signals = []

    for phrase, explanation in EN_CONTRACT_RISK_PHRASES:
        if phrase.lower() in transcript_lower:
            contract_risk_signals.append({
                "phrase":      phrase,
                "reading":     phrase,
                "english":     explanation,
                "category":    "contract_risk",
                "confidence":  0.92,
                "explanation": explanation,
                "speaker":     _find_speaker(phrase, transcript, case_insensitive=True),
                "language":    "EN",
                "is_contract_risk": True,
            })

    for phrase, explanation in JP_CONTRACT_RISK_PHRASES:
        if phrase in transcript:
            contract_risk_signals.append({
                "phrase":      phrase,
                "reading":     "",
                "english":     explanation,
                "category":    "contract_risk",
                "confidence":  0.94,
                "explanation": explanation,
                "speaker":     _find_speaker(phrase, transcript, case_insensitive=False),
                "language":    "JP",
                "is_contract_risk": True,
            })

    contract_risk_signals  = _dedup(contract_risk_signals)
    contract_risk_detected = len(contract_risk_signals) > 0

    # ── TIER 2: Performance failure + deadline ultimatums ─────────────────────
    high_signals = []

    for phrase, explanation in EN_HIGH_PHRASES:
        if phrase.lower() in transcript_lower:
            high_signals.append({
                "phrase":      phrase,
                "reading":     phrase,
                "english":     phrase,
                "confidence":  0.88,
                "explanation": explanation,
                "speaker":     _find_speaker(phrase, transcript, case_insensitive=True),
            })

    for phrase, explanation in JP_HIGH_PHRASES:
        if phrase in transcript:
            high_signals.append({
                "phrase":      phrase,
                "reading":     "",
                "english":     explanation,
                "confidence":  0.88,
                "explanation": explanation,
                "speaker":     _find_speaker(phrase, transcript, case_insensitive=False),
            })

    # ── TIER 3: Soft rejection patterns ──────────────────────────────────────
    medium_signals = []
    low_signals    = []

    for pattern in SOFT_PATTERNS:
        phrase = pattern["phrase"]
        found  = (
            phrase.lower() in transcript_lower
            if re.search(r"[a-zA-Z]", phrase)
            else phrase in transcript
        )
        if not found:
            continue
        signal = {**pattern, "speaker": _find_speaker(phrase, transcript, case_insensitive=True)}
        if pattern.get("confidence", 0.5) >= 0.80:
            medium_signals.append(signal)
        else:
            low_signals.append(signal)

    total_signals = (
        len(termination_signals) + len(approval_gate_signals) +
        len(contract_risk_signals) + len(high_signals) +
        len(medium_signals) + len(low_signals)
    )

    # ── C5 FIX: Risk level — contract risk tier integrated ────────────────────
    if termination_detected:
        risk_level = "CRITICAL"
    elif contract_risk_detected and len(high_signals) >= 1:
        # Conditional threat + evidence of performance failure = imminent termination
        risk_level = "CRITICAL"
    elif contract_risk_detected:
        risk_level = "HIGH"
    elif approval_gate_detected:
        if len(high_signals) >= 3:
            risk_level = "CRITICAL"
        else:
            risk_level = "HIGH"
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

    # ── C6 FIX: Cultural note for contract risk ───────────────────────────────
    if termination_detected:
        cultural_note = (
            "EXPLICIT TERMINATION DETECTED — this is NOT a soft rejection or "
            "negotiable refusal. The decision is irrevocable. In Japanese business "
            "culture, this language is only used AFTER internal ringi-sho (稟議書) "
            "approval has been finalized. The polite keigo delivery is cultural courtesy, "
            "not a signal of openness to reconsideration. One follow-up request is "
            "culturally acceptable; repeating it would be considered disrespectful."
        )
    elif contract_risk_detected and len(high_signals) >= 1:
        cultural_note = (
            "CONTRACT AT CRITICAL RISK — conditional termination threat combined with "
            "performance failure signals detected. This pattern means the client has "
            "already begun internal discussions about ending the relationship. In Japanese "
            "business culture, stating 'the contract would be reconsidered' is not "
            "rhetorical — it means a senior decision-maker has endorsed this position. "
            "Immediate written response and resolution before the stated deadline is "
            "non-negotiable. Failure to respond in writing will be treated as acceptance "
            "of termination."
        )
    elif contract_risk_detected:
        cultural_note = (
            "CONTRACT AT RISK — conditional reconsideration or unacceptability signals "
            "detected. The client is formally warning that the relationship is in "
            "jeopardy. In Japanese business context, this phrasing is used only when "
            "the speaker has internal backing for the position. Treat as HIGH priority "
            "escalation — provide written commitment and resolution timeline immediately."
        )
    elif approval_gate_detected:
        cultural_note = (
            "APPROVAL GATE DETECTED — apparent agreement in this meeting does NOT "
            "constitute final approval. Commercial contracts require separate approval "
            "through procurement or an executive/purchasing committee. Do not begin "
            "work or resource allocation until written confirmation from the authorizing "
            "committee is received."
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
        "risk_level":               risk_level,
        "total_signals":            total_signals,
        "termination_detected":     termination_detected,
        "termination_signals":      termination_signals,
        "approval_gate_detected":   approval_gate_detected,
        "approval_gate_signals":    approval_gate_signals,
        "contract_risk_detected":   contract_risk_detected,   # C1 FIX: new field
        "contract_risk_signals":    contract_risk_signals,    # C1 FIX: new field
        "high_signals":             high_signals,
        "medium_signals":           medium_signals,
        "low_signals":              low_signals,
        "cultural_note":            cultural_note,
        "risk_summary":             cultural_note,
        "detected": (
            [{**s, "severity": "CRITICAL"} for s in termination_signals] +
            [{**s, "severity": "CRITICAL"} for s in contract_risk_signals] +
            [{**s, "severity": "HIGH"}     for s in approval_gate_signals] +
            [{**s, "severity": "HIGH"}     for s in high_signals] +
            [{**s, "severity": "MEDIUM"}   for s in medium_signals] +
            [{**s, "severity": "LOW"}      for s in low_signals]
        ),
    }