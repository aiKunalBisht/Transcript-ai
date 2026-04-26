"""
test_edge_2.py — Speaker Identity Chaos Test
=============================================
Same person referred to in 5 different ways.
Tests if speaker normalization collapses them to ONE identity.

Expected: 3 unique speakers (Tanaka, Priya, Sato)
Loophole: 5+ speakers = normalization broken, duplicates in results
"""

import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transcription.speaker_normalizer import normalize_speaker_name, extract_all_speakers, unify_speakers_in_result

TRANSCRIPT = """[00:00] Tanaka (Director): Good morning everyone. Let's begin the Q4 review.
[00:45] 田中: まず、売上について報告します。目標の92%です。
[02:10] Tanaka-san: The client from last week — Suzuki-san flagged a concern.
[03:30] Director Tanaka: 承知しました。リスクについて検討いたします。
[04:15] T. Tanaka: I'll prepare the report by Friday EOD.
Priya (Backend Dev): I can help with the data extraction part.
Sato (PM): 了解しました。では、田中部長、来週月曜に確認しましょう。
Tanaka (Director): 難しいかもしれませんが、善処します。"""

# Simulate what LLM might return — messy, inconsistent names
MOCK_LLM_RESULT = {
    "summary": ["Q4 review discussed.", "Action items assigned.", "Risk concerns raised."],
    "action_items": [
        {"task": "Prepare Q4 report", "owner": "Tanaka (Director)", "deadline": "Friday EOD"},
        {"task": "Data extraction support", "owner": "Priya", "deadline": "TBD"},
        {"task": "Monday check-in", "owner": "Director Tanaka", "deadline": "Next Monday"},
    ],
    "sentiment": [
        {"speaker": "Tanaka (Director)", "score": "neutral", "label": "Professional"},
        {"speaker": "田中", "score": "neutral", "label": "Cautious"},
        {"speaker": "T. Tanaka", "score": "neutral", "label": "Formal"},
        {"speaker": "Priya", "score": "positive", "label": "Collaborative"},
        {"speaker": "Sato (PM)", "score": "neutral", "label": "Organized"},
    ],
    "speakers": [
        {"name": "Tanaka (Director)", "talk_time_pct": 25, "tone": "formal"},
        {"name": "田中", "talk_time_pct": 20, "tone": "formal"},
        {"name": "T. Tanaka", "talk_time_pct": 15, "tone": "formal"},
        {"name": "Priya (Backend Dev)", "talk_time_pct": 20, "tone": "mixed"},
        {"name": "Sato (PM)", "talk_time_pct": 20, "tone": "formal"},
    ],
    "japan_insights": {"keigo_level": "high", "nemawashi_signals": [], "code_switch_count": 6},
}

def run():
    print("=" * 60)
    print("TC_EDGE_2 — Speaker Identity Chaos Test")
    print("=" * 60)

    # Step 1: Extract speakers from transcript
    print("\n1. Speaker extraction from transcript:")
    extracted = extract_all_speakers(TRANSCRIPT)
    print(f"   Raw extracted: {extracted}")
    print(f"   Count: {len(extracted)} (should be 3-4 unique normalized names)")

    # Step 2: Normalize individual names
    print("\n2. Individual name normalization:")
    test_names = [
        "Tanaka (Director)",
        "田中",
        "Tanaka-san",
        "Director Tanaka",
        "T. Tanaka",
        "Priya (Backend Dev)",
        "Sato (PM)",
    ]
    normalized = {}
    for name in test_names:
        n = normalize_speaker_name(name)
        normalized[name] = n
        print(f"   '{name}' → '{n}'")

    # Check if Tanaka variants all normalize to same value
    tanaka_variants = ["Tanaka (Director)", "Tanaka-san", "Director Tanaka", "T. Tanaka"]
    tanaka_normalized = set(normalize_speaker_name(n) for n in tanaka_variants)
    print(f"\n   Tanaka variants normalize to: {tanaka_normalized}")
    if len(tanaka_normalized) == 1:
        print("   ✅ All Tanaka variants → single identity")
    else:
        print(f"   ⚠ Multiple identities: {tanaka_normalized}")
        print("   NOTE: T. Tanaka and Director Tanaka may not fully normalize without fuzzy matching")

    # Step 3: Run unify_speakers_in_result
    print("\n3. Running unify_speakers_in_result:")
    unified = unify_speakers_in_result(MOCK_LLM_RESULT.copy(), TRANSCRIPT)

    unique_speakers = set(s["name"] for s in unified["speakers"])
    print(f"   Speakers before: {len(MOCK_LLM_RESULT['speakers'])}")
    print(f"   Speakers after:  {len(unique_speakers)} → {unique_speakers}")

    if len(unique_speakers) <= 3:
        print("   ✅ PASS — Speakers correctly collapsed")
    else:
        print(f"   ⚠ PARTIAL — {len(unique_speakers)} unique speakers (expected ≤3)")
        print("   LOOPHOLE: T. Tanaka and Director Tanaka need fuzzy/substring matching")

    # Step 4: Check sentiment deduplication
    print("\n4. Sentiment speaker names after normalization:")
    unique_sentiment = set(s["speaker"] for s in unified["sentiment"])
    print(f"   Unique sentiment speakers: {unique_sentiment}")
    tanaka_dupes = [s for s in unified["sentiment"] if "tanaka" in s["speaker"].lower() or "田中" in s["speaker"]]
    if len(tanaka_dupes) > 1:
        print(f"   ⚠ LOOPHOLE: Tanaka appears {len(tanaka_dupes)}x in sentiment: {[s['speaker'] for s in tanaka_dupes]}")
    else:
        print(f"   ✅ Tanaka unified in sentiment")

    # Step 5: Action item owner normalization
    print("\n5. Action item owner normalization:")
    for item in unified["action_items"]:
        print(f"   '{item['task'][:30]}' → owner: '{item['owner']}'")

    print("\n" + "=" * 60)
    passed = len(unique_speakers) <= 3
    print(f"RESULT: {'✅ PASS' if passed else '⚠ PARTIAL'} — {len(unique_speakers)} unique speakers detected")
    if not passed:
        print("Next fix: Add Levenshtein distance or substring matching for 'T. Tanaka' → 'Tanaka'")
    print("=" * 60)

    return unified

if __name__ == "__main__":
    run()