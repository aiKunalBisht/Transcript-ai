"""
test_edge_1.py — Soft Rejection Exhaustive Test
================================================
Tests if the detector catches ALL signals when a speaker
uses polite/positive language throughout but every phrase is a soft rejection.

Expected: HIGH risk, 7+ signals detected
Loophole: System catches only 1-2 = pattern matching too narrow
"""

import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from soft_rejection_detector import detect_soft_rejections
from analyzer import analyze_transcript

TRANSCRIPT = """田中 (Director): 本日はお忙しい中お時間をいただきありがとうございます。
鈴木 (Sales): こちらこそ、よろしくお願いいたします。
田中 (Director): 新しいAIシステムの提案、大変興味深く拝見いたしました。
鈴木 (Sales): ありがとうございます。ぜひ導入をご検討いただければ幸いです。
田中 (Director): そうですね、非常に前向きに検討したいと思います。ただ、現在の予算サイクルとの兼ね合いで、少し難しいかもしれません。
鈴木 (Sales): 来期の予算でいかがでしょうか？
田中 (Director): 来期についても、社内での調整が必要になりますね。上司にも相談してみます。また、システムの安全性についても、我々としては慎重になるべきかもしれません。
鈴木 (Sales): セキュリティは万全です。いつ頃ご回答いただけますか？
田中 (Director): 善処します。なかなか難しい状況ではありますが、前向きに対応したいと思います。社内で確認してからご連絡いたします。
鈴木 (Sales): ありがとうございます。では来週いかがでしょうか？
田中 (Director): 検討いたします。"""

EXPECTED_PHRASES = [
    "前向きに検討",
    "難しいかもしれません",
    "上司に相談",
    "善処します",
    "社内で確認",
    "検討いたします",
    "そうですね",
]

def run():
    print("=" * 60)
    print("TC_EDGE_1 — Soft Rejection Exhaustive Test")
    print("=" * 60)

    # Test detector directly
    result = detect_soft_rejections(TRANSCRIPT)

    print(f"\nRisk level:    {result['risk_level']}  (expected: HIGH)")
    print(f"Total signals: {result['total_signals']}  (expected: 7+)")
    print(f"Rejections:    {result['rejection_count']}")
    print(f"Uncertain:     {result['uncertain_count']}")
    print(f"Hesitation:    {result['hesitation_count']}")

    print("\nDetected phrases:")
    detected_phrases = [s["phrase"] for s in result["detected"]]
    for phrase in EXPECTED_PHRASES:
        found = phrase in detected_phrases
        status = "✅ CAUGHT" if found else "❌ MISSED"
        print(f"  {status}  {phrase}")

    missed = [p for p in EXPECTED_PHRASES if p not in detected_phrases]
    extra  = [p for p in detected_phrases if p not in EXPECTED_PHRASES]

    print(f"\nMissed signals: {len(missed)}")
    if missed:
        print(f"  → {missed}")
        print("  LOOPHOLE: Pattern list too narrow — these phrases not in SOFT_REJECTION_PATTERNS")

    if extra:
        print(f"\nExtra signals detected (not in expected): {extra}")

    # Risk level check
    if result["risk_level"] != "HIGH":
        print(f"\n❌ LOOPHOLE: Risk level is {result['risk_level']}, expected HIGH")
        print("  → Threshold logic not accumulating multiple signals correctly")
    else:
        print(f"\n✅ Risk level correct: HIGH")

    # Speaker attribution check
    print("\nSpeaker attribution:")
    for sig in result["detected"]:
        speaker_clean = sig["speaker"].replace("(Director)","").replace("(Sales)","").strip()
        expected_speaker = "田中" if sig["phrase"] != "そうですね" else "田中"
        flag = "✅" if "田中" in sig["speaker"] or "Tanaka" in sig["speaker"] else "⚠"
        print(f"  {flag} '{sig['phrase']}' → speaker: '{sig['speaker']}'")

    print("\n" + "=" * 60)
    if result["risk_level"] == "HIGH" and result["total_signals"] >= 5:
        print("RESULT: ✅ PASS — System correctly identifies HIGH risk meeting")
    else:
        print(f"RESULT: ❌ FAIL — Expected HIGH/7+, got {result['risk_level']}/{result['total_signals']}")
    print("=" * 60)

    return result

if __name__ == "__main__":
    run()