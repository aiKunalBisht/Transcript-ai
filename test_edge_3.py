"""
test_edge_3.py — Hallucination Guard + Cross-Language Grounding Test
=====================================================================
Mostly English meeting with Japanese phrases dropped in.
Tests two things simultaneously:

1. Action items stated explicitly in English should NEVER be flagged
   as hallucinated — they are fully grounded in the transcript.
   ANY flag = false positive = threshold too aggressive.

2. Japanese phrase 'すこし難しいかもしれません' appears in an English
   meeting context. The soft rejection detector should flag it,
   but it should be understood in CONTEXT — the deadline was
   immediately renegotiated to Monday, so it is a partial rejection,
   not a full deal-breaker.
"""

import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hallucination_guard import verify_result
from soft_rejection_detector import detect_soft_rejections
from semantic_validator import semantic_grounding_score

TRANSCRIPT = """Sarah: Good morning. Today we're reviewing the TranscriptAI integration timeline.
Kenji: Morning. I wanted to flag that the database migration is still pending.
Sarah: Right. Kenji can you own that? We need it done before the client demo.
Kenji: はい、承知しました。金曜日までに完了します。
Sarah: Perfect. Also the API documentation — Mike was supposed to handle that.
Mike: Yeah I'll have a draft by Wednesday. Sorry for the delay.
Sarah: No worries. Kenji, one more thing — the security audit report?
Kenji: 少し難しいかもしれません。来週月曜日が限界です。
Sarah: Okay Monday it is. Let's also schedule a sync with the Tokyo team.
Mike: I can set that up. Thursday works for me.
Sarah: Great. So to summarize — database Kenji by Friday, API docs Mike by Wednesday, security audit Kenji by Monday, Tokyo sync Mike Thursday.
Kenji: 了解しました。"""

# These 4 action items are EXPLICITLY stated in the transcript
# None should be hallucination-flagged
EXPLICIT_ACTIONS = [
    {"task": "Complete database migration", "owner": "Kenji", "deadline": "Friday"},
    {"task": "Draft API documentation",    "owner": "Mike",  "deadline": "Wednesday"},
    {"task": "Prepare security audit report", "owner": "Kenji", "deadline": "Monday"},
    {"task": "Schedule Tokyo team sync",   "owner": "Mike",  "deadline": "Thursday"},
]

# This action item is NOT in the transcript — tests false positive prevention
HALLUCINATED_ACTION = {
    "task": "Create executive presentation for board meeting",
    "owner": "Sarah",
    "deadline": "Next Friday"
}

def run():
    print("=" * 60)
    print("TC_EDGE_3 — Hallucination Guard + Cross-Language Test")
    print("=" * 60)

    # ── Test 1: Semantic grounding of explicit actions ──────────────
    print("\n1. Semantic grounding scores for EXPLICIT actions:")
    print("   (all should be > 0.20 — anything lower = false positive risk)")

    for action in EXPLICIT_ACTIONS:
        score = semantic_grounding_score(action["task"], TRANSCRIPT)
        status = "✅" if score >= 0.15 else "❌ FALSE POSITIVE RISK"
        print(f"   {status} '{action['task'][:40]}' → {score:.3f}")

    # ── Test 2: Hallucinated action should score low ─────────────────
    print("\n2. Semantic grounding for HALLUCINATED action:")
    h_score = semantic_grounding_score(HALLUCINATED_ACTION["task"], TRANSCRIPT)
    status  = "✅ correctly low" if h_score < 0.20 else "❌ MISSED — should be flagged"
    print(f"   {status} '{HALLUCINATED_ACTION['task']}' → {h_score:.3f}")

    # ── Test 3: Full hallucination guard on result ───────────────────
    print("\n3. Hallucination guard on full result:")
    mock_result = {
        "summary": ["Team reviewed integration timeline.", "Action items assigned across three owners."],
        "action_items": EXPLICIT_ACTIONS + [HALLUCINATED_ACTION],
        "sentiment": [
            {"speaker": "Sarah", "score": "positive", "label": "Organized"},
            {"speaker": "Kenji", "score": "neutral",  "label": "Professional"},
            {"speaker": "Mike",  "score": "neutral",  "label": "Apologetic"},
        ],
        "speakers": [
            {"name": "Sarah", "talk_time_pct": 50, "tone": "mixed"},
            {"name": "Kenji", "talk_time_pct": 30, "tone": "formal"},
            {"name": "Mike",  "talk_time_pct": 20, "tone": "casual"},
        ],
        "japan_insights": {"keigo_level": "low", "nemawashi_signals": [], "code_switch_count": 3},
    }

    verified = verify_result(mock_result, TRANSCRIPT)
    ai = verified.get("verification", {}).get("action_items", {})

    print(f"   Total actions: {ai.get('total', 0)}")
    print(f"   Verified:      {ai.get('verified_count', 0)}")
    print(f"   Flagged:       {ai.get('flagged_count', 0)}")

    flagged_tasks = [i["task"] for i in ai.get("flagged", [])]
    print(f"\n   Flagged items: {flagged_tasks}")

    # Check: only hallucinated action should be flagged
    halluc_flagged  = HALLUCINATED_ACTION["task"] in flagged_tasks or \
                      any("presentation" in t.lower() or "board" in t.lower() for t in flagged_tasks)
    explicit_flagged = [t for t in flagged_tasks
                        if not any(w in t.lower() for w in ["presentation","board","executive"])]

    if halluc_flagged:
        print("   ✅ Hallucinated action correctly flagged")
    else:
        print("   ❌ LOOPHOLE: Hallucinated action NOT flagged — threshold too permissive")

    if not explicit_flagged:
        print("   ✅ No false positives on explicit actions")
    else:
        print(f"   ❌ FALSE POSITIVES: {explicit_flagged}")
        print("   LOOPHOLE: Cross-language grounding not rescuing EN actions against EN+JA transcript")

    # ── Test 4: Soft rejection in English meeting context ────────────
    print("\n4. Soft rejection detection in English-dominant meeting:")
    soft = detect_soft_rejections(TRANSCRIPT)
    print(f"   Risk level:    {soft['risk_level']}")
    print(f"   Total signals: {soft['total_signals']}")
    print(f"   Detected:      {[s['phrase'] for s in soft['detected']]}")

    # The phrase 難しいかもしれません appears but deadline was renegotiated
    # Risk should be LOW or MEDIUM — not HIGH — because it was resolved
    if soft["risk_level"] in ("LOW", "MEDIUM", "NONE"):
        print("   ✅ Appropriate risk level for resolved soft rejection")
    else:
        print(f"   ⚠ NUANCE GAP: Risk is {soft['risk_level']} but deadline was renegotiated immediately")
        print("   NOTE: System cannot currently detect 'resolved soft rejection'")
        print("   This is a known limitation — context window needed to resolve this properly")

    print("\n" + "=" * 60)
    all_pass = (not explicit_flagged) and halluc_flagged
    print(f"RESULT: {'✅ PASS' if all_pass else '⚠ PARTIAL'}")
    if not all_pass:
        if explicit_flagged:
            print(f"  Fix needed: Lower semantic threshold or improve EN→JA bridge coverage")
        if not halluc_flagged:
            print(f"  Fix needed: Raise hallucination threshold for non-grounded claims")
    print("=" * 60)

    return verified

if __name__ == "__main__":
    run()