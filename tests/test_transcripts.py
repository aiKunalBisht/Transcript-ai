# test_transcripts.py
# 3 stress-test transcripts designed to find loopholes
# Run: python test_transcripts.py

import sys
import json
sys.path.insert(0, ".")

# ─────────────────────────────────────────────────────────────────────────────
# TC_TRAP_1: Soft rejection DISGUISED as positive language
# Loophole tested: Does the system correctly flag 前向きに検討 even when
# surrounded by genuinely positive statements? Or does positive context
# override the signal?
# ─────────────────────────────────────────────────────────────────────────────
TC_TRAP_1 = {
    "name": "Soft rejection hidden inside positive language",
    "language": "mixed",
    "expected_soft_risk": "MEDIUM",  # 前向きに検討 = uncertain
    "expected_keigo": "high",
    "transcript": """
Yamamoto (Director): 本日は素晴らしいプレゼンテーションをありがとうございました。
Chen (Sales): Thank you so much. We are very excited about this partnership.
Yamamoto (Director): はい、私どもも大変興味深く拝聴いたしました。御社のソリューションは
    確かに革新的ですね。特にAPPI対応の部分は、弊社のコンプライアンス部門も
    高く評価するかと思います。
Chen (Sales): That's great to hear. So can we move forward with the pilot program?
Yamamoto (Director): そうですね、前向きに検討させていただきたいと思います。
    社内でも非常にポジティブな反応をいただいております。
Chen (Sales): Wonderful. When can we expect a decision?
Yamamoto (Director): 来週中には社内調整を終えて、ご連絡できると思います。
    ただ、予算の最終承認については、もう少し時間をいただければ幸いです。
Chen (Sales): Of course, we completely understand. We look forward to hearing from you.
Yamamoto (Director): こちらこそ、よろしくお願いいたします。
"""
}

# ─────────────────────────────────────────────────────────────────────────────
# TC_TRAP_2: CASCADING soft rejections — same speaker, escalating
# Loophole tested: Does risk level escalate correctly when one speaker
# uses MULTIPLE rejection signals in the same meeting?
# Expected: HIGH risk, multiple signals from Suzuki
# ─────────────────────────────────────────────────────────────────────────────
TC_TRAP_2 = {
    "name": "Cascading soft rejections — escalation test",
    "language": "ja",
    "expected_soft_risk": "HIGH",
    "expected_keigo": "high",
    "transcript": """
田中: 鈴木部長、先日ご提案した新システムの件ですが、いかがでしょうか？
鈴木: そうですね、確認してみます。
田中: 先週もお伺いしましたが、スケジュール的にはいつ頃ご判断いただけますか？
鈴木: 検討いたします。技術チームとも相談が必要でして。
田中: 予算についてはご承認いただけそうでしょうか？
鈴木: 難しい状況です。現時点では、予算の再検討も必要かもしれません。
田中: では、最低限のパイロット導入だけでも、いかがでしょうか？
鈴木: うーん、それもちょっと難しいかもしれません。上司に相談してみます。
田中: 承知しました。ご検討よろしくお願いいたします。
鈴木: 善処します。社内で確認してからご連絡いたします。
"""
}

# ─────────────────────────────────────────────────────────────────────────────
# TC_TRAP_3: Mixed language with SPEAKER IDENTITY ambiguity
# Loophole tested:
#   1. "Kenji (田中's assistant)" — role in parentheses, should normalize to "Kenji"
#   2. "田中部長" used as a reference inside speech — should NOT be treated as speaker
#   3. Priya uses Hindi soft refusal mixed into English — does hindi detection work?
#   4. Very short transcript — does summary handle it without hallucinating?
# ─────────────────────────────────────────────────────────────────────────────
TC_TRAP_3 = {
    "name": "Speaker identity ambiguity + Hindi mixed in + short transcript",
    "language": "mixed",
    "expected_soft_risk": "LOW",
    "expected_speakers_min": 3,
    "transcript": """
Kenji (Lead): Good morning everyone. Today we review Q3 with 田中部長 joining remotely.
Priya (Backend): Hi. I have the report ready. Thoda mushkil hai the data pipeline but we managed.
Kenji (Lead): 田中部長、音声が聞こえますか？
田中 (Director): はい、聞こえています。よろしくお願いします。承知いたしました。
Priya (Backend): The latency issue — dekh lete hain, should be resolved by Friday.
Kenji (Lead): Perfect. Action item: Priya to fix pipeline by Friday. 田中部長 to review report Monday.
田中 (Director): かしこまりました。確認いたします。
"""
}

# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────
def run_test(tc: dict):
    print(f"\n{'='*65}")
    print(f"TEST: {tc['name']}")
    print(f"{'='*65}")

    try:
        from analysis.analyzer import analyze_transcript
        from analysis.soft_rejection_detector import detect_soft_rejections
        from transcription.speaker_normalizer import normalize_speaker_name, extract_all_speakers

        result = analyze_transcript(tc["transcript"], tc["language"])

        # ── Soft rejection check ──────────────────────────────────────
        soft = result.get("soft_rejections", {})
        detected_risk = soft.get("risk_level", "NONE")
        expected_risk = tc.get("expected_soft_risk", "ANY")

        risk_pass = (expected_risk == "ANY" or detected_risk == expected_risk)
        print(f"\n🎭 Soft Rejection:")
        print(f"   Expected: {expected_risk}  |  Got: {detected_risk}  {'✅' if risk_pass else '❌ FAILED'}")
        print(f"   Signals : {soft.get('total_signals', 0)}")
        for sig in soft.get("detected", []):
            print(f"   → [{sig['severity']}] {sig['phrase']} (Speaker: {sig['speaker']})")

        # ── Keigo check ───────────────────────────────────────────────
        ji = result.get("japan_insights", {})
        if "expected_keigo" in tc:
            keigo_got  = ji.get("keigo_level", "unknown")
            keigo_exp  = tc["expected_keigo"]
            keigo_pass = keigo_got == keigo_exp
            print(f"\n🏯 Keigo:  Expected: {keigo_exp}  |  Got: {keigo_got}  {'✅' if keigo_pass else '⚠ PARTIAL'}")

        # ── Speaker normalization check ───────────────────────────────
        speakers_found = extract_all_speakers(tc["transcript"])
        print(f"\n🎤 Speakers extracted from transcript:")
        for norm, raw in speakers_found.items():
            print(f"   '{raw}' → '{norm}'")

        result_speakers = [s.get("name") for s in result.get("speakers", [])]
        print(f"   LLM detected speakers: {result_speakers}")

        if "expected_speakers_min" in tc:
            sp_pass = len(result_speakers) >= tc["expected_speakers_min"]
            print(f"   Min expected: {tc['expected_speakers_min']}  |  Got: {len(result_speakers)}  {'✅' if sp_pass else '❌ TOO FEW'}")

        # ── Action items ──────────────────────────────────────────────
        items = result.get("action_items", [])
        flagged = [i for i in items if i.get("hallucination_flag")]
        print(f"\n✅ Action Items: {len(items)} total, {len(flagged)} flagged")
        for item in items:
            flag = "🚩" if item.get("hallucination_flag") else "✓"
            print(f"   {flag} {item.get('task','')} | Owner: {item.get('owner','')} | Conf: {item.get('confidence','?')}")

        # ── Summary ───────────────────────────────────────────────────
        summary = result.get("summary", [])
        print(f"\n📝 Summary ({len(summary)} bullets):")
        for i, b in enumerate(summary, 1):
            print(f"   {i}. {b}")

        # ── Provider ─────────────────────────────────────────────────
        print(f"\n⚡ Provider: {result.get('_provider','?')} | {result.get('_duration_ms',0)/1000:.1f}s")

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    for tc in [TC_TRAP_1, TC_TRAP_2, TC_TRAP_3]:
        run_test(tc)
    print(f"\n{'='*65}")
    print("Done. Check results above for any ❌ or ⚠ markers.")
    print(f"{'='*65}\n")