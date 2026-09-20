#!/usr/bin/env python3
"""
apply_sentiment_trust_patches.py — TranscriptAI v8.2
Run from your project root:  python apply_sentiment_trust_patches.py

Patches:
  S1  sentiment_engine.py    — add communicative_function + secondary_functions
                               to UtteranceResult dataclass and to_dict()
  S2  soft_rejection_detector.py — fix 上司に相談 explanation (escalation, not nemawashi)
  S3  soft_rejection_detector.py — add trust_state, trust_trajectory, trust_note
                               to detect_soft_rejections() return dict
"""

import pathlib, sys

def patch(path_str: str, old: str, new: str, label: str) -> bool:
    path = pathlib.Path(path_str)
    if not path.exists():
        print(f"  \u2717  {label}  ({path_str} not found)")
        return False
    src = path.read_text(encoding="utf-8")
    if old not in src:
        print(f"  \u2717  {label}  (pattern not found \u2014 already applied?)")
        return False
    path.write_text(src.replace(old, new, 1), encoding="utf-8")
    print(f"  \u2713  {label}")
    return True

applied = 0
print("Patching sentiment_engine.py and soft_rejection_detector.py ...\n")

# ── S1: Add communicative_function + secondary_functions to UtteranceResult ──
ok = patch(
    "analysis/sentiment_engine.py",
    old=(
        "    signals: list[SentimentSignal]      # all signals above _SCORE_THRESHOLD\n"
        "\n"
        "    def to_dict(self) -> dict:\n"
        "        return {\n"
        "            \"speaker\": self.speaker,\n"
        "            \"text\": self.text,\n"
        "            \"primary\": self.primary,\n"
        "            \"secondary\": self.secondary,\n"
        "            \"tone\": self.tone.to_dict(),\n"
        "            \"valence\": self.valence,\n"
        "            \"confidence\": self.confidence,\n"
        "            \"signals\": [\n"
        "                {\n"
        "                    \"label\": s.label,\n"
        "                    \"score\": s.score,\n"
        "                    \"valence\": s.valence,\n"
        "                    \"matched_phrases\": s.matched_phrases,\n"
        "                }\n"
        "                for s in self.signals\n"
        "            ],\n"
        "        }"
    ),
    new=(
        "    signals: list[SentimentSignal]      # all signals above _SCORE_THRESHOLD\n"
        "    # Communicative function (WHAT speaker is doing) — populated by\n"
        "    # meeting_function_detector via Stage 12c in analyzer.py.\n"
        "    # Orthogonal to emotion labels: defensive feeling \u2260 escalation function.\n"
        "    communicative_function: str = \"information_sharing\"\n"
        "    secondary_functions: list[str] = field(default_factory=list)\n"
        "\n"
        "    def to_dict(self) -> dict:\n"
        "        return {\n"
        "            \"speaker\": self.speaker,\n"
        "            \"text\": self.text,\n"
        "            \"primary\": self.primary,\n"
        "            \"secondary\": self.secondary,\n"
        "            \"communicative_function\": self.communicative_function,\n"
        "            \"secondary_functions\": self.secondary_functions,\n"
        "            \"tone\": self.tone.to_dict(),\n"
        "            \"valence\": self.valence,\n"
        "            \"confidence\": self.confidence,\n"
        "            \"signals\": [\n"
        "                {\n"
        "                    \"label\": s.label,\n"
        "                    \"score\": s.score,\n"
        "                    \"valence\": s.valence,\n"
        "                    \"matched_phrases\": s.matched_phrases,\n"
        "                }\n"
        "                for s in self.signals\n"
        "            ],\n"
        "        }"
    ),
    label="S1 \u2014 UtteranceResult: communicative_function + secondary_functions fields added",
)
if ok: applied += 1

# ── S2: Fix 上司に相談 explanation ─────────────────────────────────────────────
ok = patch(
    "analysis/soft_rejection_detector.py",
    old=(
        '        "phrase": "\u4e0a\u53f8\u306b\u76f8\u8ac7",\n'
        '        "reading": "J\u014dshi ni s\u014ddan",\n'
        '        "english": "Will consult with my superior",\n'
        '        "confidence": 0.50,\n'
        '        "explanation": "Escalation to a superior \u2014 may be genuine or a delaying tactic.",\n'
        "    },"
    ),
    new=(
        '        "phrase": "\u4e0a\u53f8\u306b\u76f8\u8ac7",\n'
        '        "reading": "J\u014dshi ni s\u014ddan",\n'
        '        "english": "Will consult with my superior",\n'
        '        "confidence": 0.45,\n'
        '        "explanation": (\n'
        '            "authority_deferring (escalation) \u2014 speaker lacks decision authority. "\n'
        '            "NOT nemawashi. Nemawashi is the internal coordination process that may "\n'
        '            "follow this phrase, not the phrase itself. Detect the sequence "\n'
        '            "(escalation \u2192 gap \u2192 specific commitment), not the single utterance."\n'
        "        ),\n"
        "    },"
    ),
    label="S2 \u2014 \u4e0a\u53f8\u306b\u76f8\u8ac7: reclassified as authority_deferring, not nemawashi",
)
if ok: applied += 1

# ── S3: Add trust_state + trust_trajectory to detect_soft_rejections() ────────
# Trust state is SEPARATE from sentiment (Kayo point 5):
# A client setting a deadline still has remaining trust.
# A client who has already decided to leave does NOT set deadlines.
ok = patch(
    "analysis/soft_rejection_detector.py",
    old='    return {\n        "risk_level":               risk_level,',
    new=(
        "    # \u2500\u2500 Kayo point 5: trust_state separate from sentiment \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        "    # An ultimatum with a deadline = remaining trust (still engaging).\n"
        "    # Explicit termination = trust already lost.\n"
        "    # These are orthogonal to sentiment.score and must be separate fields.\n"
        "    if termination_detected:\n"
        "        trust_state      = \"lost\"\n"
        "        trust_trajectory = \"lost\"\n"
        "    elif contract_risk_detected and len(high_signals) >= 1:\n"
        "        trust_state      = \"critical\"       # ultimatum + evidence = nearly gone\n"
        "        trust_trajectory = \"deteriorating\"  # but recovery window still exists\n"
        "    elif contract_risk_detected:\n"
        "        trust_state      = \"at_risk\"        # conditional threat, still engaging\n"
        "        trust_trajectory = \"deteriorating\"\n"
        "    elif approval_gate_detected or len(high_signals) >= 2:\n"
        "        trust_state      = \"fragile\"\n"
        "        trust_trajectory = \"stable\"\n"
        "    elif len(medium_signals) >= 1 or len(high_signals) >= 1:\n"
        "        trust_state      = \"fragile\"\n"
        "        trust_trajectory = \"stable\"\n"
        "    else:\n"
        "        trust_state      = \"healthy\"\n"
        "        trust_trajectory = \"stable\"\n"
        "\n"
        "    _TRUST_NOTES = {\n"
        "        \"lost\":         \"Trust is gone \u2014 termination language means internal decision is final.\",\n"
        "        \"critical\":     (\n"
        "            \"Ultimatum with deadline \u2014 trust nearly gone but a recovery window \"\n"
        "            \"still exists. A client setting a deadline is still engaging; \"\n"
        "            \"one who has decided to leave does not set deadlines. \"\n"
        "            \"Resolve before the stated deadline with a written commitment.\"\n"
        "        ),\n"
        "        \"at_risk\":      (\n"
        "            \"Conditional threat detected. Client is still engaging (they stated \"\n"
        "            \"conditions, not finality). Recovery is possible with immediate \"\n"
        "            \"written response and clear timeline.\"\n"
        "        ),\n"
        "        \"fragile\":      \"Tension present but relationship still active. Careful follow-up required.\",\n"
        "        \"healthy\":      \"No significant trust risk detected in this transcript.\",\n"
        "    }\n"
        "    trust_note = _TRUST_NOTES.get(trust_state, \"\")\n"
        "\n"
        "    return {\n"
        '        "risk_level":               risk_level,'
    ),
    label="S3 \u2014 trust_state + trust_trajectory + trust_note added to detect_soft_rejections()",
)
if ok: applied += 1

# Add the new fields to the return dict as well
ok = patch(
    "analysis/soft_rejection_detector.py",
    old=(
        '        "risk_summary":             cultural_note,\n'
        '        "detected": ('
    ),
    new=(
        '        "risk_summary":             cultural_note,\n'
        '        "trust_state":              trust_state,\n'
        '        "trust_trajectory":         trust_trajectory,\n'
        '        "trust_note":               trust_note,\n'
        '        "detected": ('
    ),
    label="S3b \u2014 trust fields added to return dict",
)
if ok: applied += 1

print(f"\n{'='*54}")
print(f"Applied {applied}/4 patches.")
if applied == 4:
    print("All done.")
else:
    print(f"{4-applied} skipped \u2014 check output above.")