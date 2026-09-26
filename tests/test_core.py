"""
TranscriptAI test suite — v2
Split into two tiers:
  [SMOKE]    — module loads, returns expected type, does not crash
  [BEHAVIOR] — asserts a correct VALUE, not just a return type

Target: every critical pipeline stage has at least one BEHAVIOR test.
Run: pytest tests/test_core.py -v
"""
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ═══════════════════════════════════════════════════════════════════════════════
# PII MASKER
# ═══════════════════════════════════════════════════════════════════════════════

# [BEHAVIOR] email is physically absent from masked output
def test_pii_masks_email():
    from transcription.pii_masker import mask_transcript
    masked, pii = mask_transcript("Contact kunal@example.com for info.")
    assert "kunal@example.com" not in masked

# [BEHAVIOR] phone digits are physically absent from masked output
def test_pii_masks_phone():
    from transcription.pii_masker import mask_transcript
    masked, pii = mask_transcript("Call +91-9876543210 now.")
    assert "9876543210" not in masked

# [BEHAVIOR] Japanese surname gets masked — core APPI use-case
def test_pii_masks_japanese_name():
    from transcription.pii_masker import mask_transcript
    masked, pii = mask_transcript("田中部長が契約を承認しました。")
    assert "田中" not in masked, "Japanese surname should be masked before LLM inference"

# [BEHAVIOR] bidirectional mapping — placeholder exists in mapping dict
def test_pii_mapping_contains_placeholder():
    from transcription.pii_masker import mask_transcript, PIIMask
    masked, pii = mask_transcript("Email billing@acme.co for invoice.")
    assert isinstance(pii, PIIMask)
    assert len(pii.mapping) > 0, "PIIMask.mapping must have at least one entry after masking PII"

# [BEHAVIOR] restoration reverses masking — original value re-appears
def test_pii_restoration_reverses_masking():
    from transcription.pii_masker import mask_transcript, restore_pii_in_result
    original = "Send to yamamoto@fujitsu.co.jp for approval"
    masked, pii = mask_transcript(original)
    assert "yamamoto@fujitsu.co.jp" not in masked
    fake_result = {"summary": [masked], "action_items": []}
    restored = restore_pii_in_result(fake_result, pii)
    summary_text = " ".join(restored.get("summary", []))
    assert "yamamoto@fujitsu.co.jp" in summary_text, \
        "PII restoration must return the original email in the result"

# [SMOKE] empty string does not crash
def test_pii_empty_string():
    from transcription.pii_masker import mask_transcript
    masked, pii = mask_transcript("")
    assert masked == "" or masked is not None

# [SMOKE] PII report is a dict
def test_pii_report_is_dict():
    from transcription.pii_masker import mask_transcript, get_pii_report
    _, pii = mask_transcript("Email billing@acme.co for invoice.")
    report = get_pii_report(pii)
    assert isinstance(report, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# HALLUCINATION GUARD
# ═══════════════════════════════════════════════════════════════════════════════

# [BEHAVIOR] fabricated action item with zero lexical overlap must be flagged
def test_hallucination_flags_fabricated_action_item():
    from analysis.hallucination_guard import verify_action_items
    # "Hire 500 engineers" shares no tokens with "Q3 budget"
    items = [{"task": "Hire 500 engineers by tomorrow", "owner": "unknown"}]
    transcript = "We briefly discussed the Q3 budget report."
    result = verify_action_items(items, transcript)
    assert isinstance(result, dict)
    # Accept multiple possible key shapes from the implementation
    items_out = result.get("items", result.get("action_items", []))
    if items_out:
        assert any(
            i.get("hallucination_flag") or i.get("flagged") or i.get("risk") == "high"
            for i in items_out
        ), "Fabricated action item must be flagged by the hallucination guard"
    else:
        assert (
            result.get("has_hallucinations")
            or result.get("total_flagged", 0) > 0
            or result.get("flagged")
            or result.get("hallucination_detected")
        ), "Hallucination guard must flag at least one fabricated item"

# [BEHAVIOR] grounded action item with high overlap must NOT be flagged
def test_hallucination_does_not_flag_grounded_action():
    from analysis.hallucination_guard import verify_action_items
    items = [{"task": "Send Q3 budget report", "owner": "Sarah"}]
    transcript = "Sarah will send the Q3 budget report by Friday."
    result = verify_action_items(items, transcript)
    assert isinstance(result, dict)
    items_out = result.get("items", result.get("action_items", []))
    if items_out:
        assert not all(
            i.get("hallucination_flag") or i.get("flagged")
            for i in items_out
        ), "Well-grounded action item must not be flagged as hallucination"

# [SMOKE] verify_result returns dict
def test_hallucination_verify_result_returns_dict():
    from analysis.hallucination_guard import verify_result
    fake_result = {
        "summary": ["The meeting discussed budget."],
        "action_items": [],
        "sentiment_by_speaker": []
    }
    transcript = "We discussed the Q3 budget in the meeting."
    result = verify_result(fake_result, transcript)
    assert isinstance(result, dict)

# [SMOKE] empty action items list does not crash
def test_hallucination_verify_action_items_empty():
    from analysis.hallucination_guard import verify_action_items
    result = verify_action_items([], "Some transcript text here.")
    assert isinstance(result, dict)

# [SMOKE] verify_summary returns dict
def test_hallucination_verify_summary_valid():
    from analysis.hallucination_guard import verify_summary
    summary = ["Meeting was about budget planning."]
    transcript = "Today we discussed budget planning for Q3."
    result = verify_summary(summary, transcript)
    assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# SOFT REJECTION DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

# [BEHAVIOR] classic nemawashi phrase must trigger detection
def test_soft_rejection_detects_nemawashi():
    from analysis.soft_rejection_detector import detect_soft_rejections
    # 検討いたします = "will consider it" — the prototypical soft no in JP business
    result = detect_soft_rejections("検討いたします。少し難しいかもしれません。")
    assert isinstance(result, dict)
    risk = (
        result.get("risk_level")
        or result.get("overall_risk")
        or result.get("risk_summary", {}).get("level")
    )
    detected = result.get("detected") or result.get("has_soft_rejection")
    assert (risk not in (None, "NONE", "")) or detected, \
        "Nemawashi phrase '検討いたします' must trigger soft rejection detection"

# [BEHAVIOR] explicit approval in English must be low or no risk
def test_soft_rejection_clean_approval_is_low_risk():
    from analysis.soft_rejection_detector import detect_soft_rejections
    result = detect_soft_rejections(
        "We fully approve the proposal. Let's sign the contract immediately."
    )
    assert isinstance(result, dict)
    risk = (
        result.get("risk_level", "")
        or result.get("overall_risk", "")
        or result.get("risk_summary", {}).get("level", "")
    )
    assert risk in ("NONE", "LOW", "", None), \
        f"Clear approval must return NONE or LOW risk, got: {risk!r}"

# [BEHAVIOR] explicit contract termination must be CRITICAL
def test_soft_rejection_termination_is_critical():
    from analysis.soft_rejection_detector import detect_soft_rejections
    # パートナーシップは継続しないことを決定しました = "decided not to continue the partnership"
    result = detect_soft_rejections(
        "パートナーシップは継続しないことを決定しました。契約を更新しないことを決定いたしました。"
    )
    assert isinstance(result, dict)
    risk = (
        result.get("risk_level", "")
        or result.get("overall_risk", "")
        or result.get("risk_summary", {}).get("level", "")
    )
    assert risk == "CRITICAL", \
        f"Explicit contract termination must be CRITICAL risk, got: {risk!r}"

# [BEHAVIOR] return dict has at least one standard key
def test_soft_rejection_has_expected_keys():
    from analysis.soft_rejection_detector import detect_soft_rejections
    result = detect_soft_rejections("We will look into it and get back to you.")
    assert any(k in result for k in [
        "soft_rejections", "patterns", "detected",
        "has_soft_rejection", "count", "risk_level", "overall_risk"
    ]), f"Result missing all expected keys. Got: {list(result.keys())}"

# [SMOKE] empty transcript does not crash
def test_soft_rejection_empty_transcript():
    from analysis.soft_rejection_detector import detect_soft_rejections
    result = detect_soft_rejections("")
    assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# CACHE
# ═══════════════════════════════════════════════════════════════════════════════

# [BEHAVIOR] set then get returns the same value
def test_cache_set_and_get():
    from utils.cache import set_cache, get_cached
    set_cache("Hello world test", "en", {"status": "ok"})
    val = get_cached("Hello world test", "en")
    assert val is not None
    assert val["status"] == "ok"

# [BEHAVIOR] cache miss returns None (not an error)
def test_cache_miss_returns_none():
    from utils.cache import get_cached
    val = get_cached("definitely not cached xyz 999", "en")
    assert val is None

# [BEHAVIOR] overwrite — latest value wins
def test_cache_overwrite():
    from utils.cache import set_cache, get_cached
    set_cache("overwrite test transcript", "en", {"v": 1})
    set_cache("overwrite test transcript", "en", {"v": 2})
    val = get_cached("overwrite test transcript", "en")
    assert val["v"] == 2

# [BEHAVIOR] different languages are stored independently
def test_cache_language_isolation():
    from utils.cache import set_cache, get_cached
    set_cache("cache isolation test transcript", "en", {"lang": "english"})
    set_cache("cache isolation test transcript", "ja", {"lang": "japanese"})
    en_val = get_cached("cache isolation test transcript", "en")
    ja_val = get_cached("cache isolation test transcript", "ja")
    assert en_val["lang"] == "english", "English cache entry must not be overwritten by Japanese"
    assert ja_val["lang"] == "japanese", "Japanese cache entry must not be overwritten by English"

# [SMOKE] stats is a dict
def test_cache_stats_is_dict():
    from utils.cache import get_cache_stats
    stats = get_cache_stats()
    assert isinstance(stats, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# SPEAKER NORMALIZER
# ═══════════════════════════════════════════════════════════════════════════════

# [BEHAVIOR] Japanese name returns a non-empty string
def test_normalize_speaker_japanese_name_nonempty():
    from transcription.speaker_normalizer import normalize_speaker_name
    result = normalize_speaker_name("田中")
    assert isinstance(result, str) and len(result) > 0, \
        "normalize_speaker_name must return a non-empty string for Japanese input"

# [BEHAVIOR] English name returns a non-empty string
def test_normalize_speaker_english_name_nonempty():
    from transcription.speaker_normalizer import normalize_speaker_name
    result = normalize_speaker_name("Alice")
    assert isinstance(result, str) and len(result) > 0

# [BEHAVIOR] role-only label (the v3.2 bug) does not return empty string
def test_normalize_speaker_role_only_no_empty_string():
    from transcription.speaker_normalizer import normalize_speaker_name
    # This was the v3.2 bug — "Director" was stripped to "" which then
    # matched every name via substring, merging all speakers into one
    result = normalize_speaker_name("Director")
    assert result != "", "Role-only speaker label must not normalize to empty string (v3.2 bug regression)"

# [BEHAVIOR] extract_all_speakers finds known speakers
def test_extract_all_speakers_finds_speakers():
    from transcription.speaker_normalizer import extract_all_speakers
    transcript = "Alice: Let's discuss. Bob: Agreed. Alice: Great."
    result = extract_all_speakers(transcript)
    assert isinstance(result, dict)
    speaker_names = [k.lower() for k in result.keys()]
    assert any("alice" in name for name in speaker_names), \
        "extract_all_speakers must identify 'Alice' as a speaker"

# [SMOKE] empty transcript returns dict without crashing
def test_extract_all_speakers_empty():
    from transcription.speaker_normalizer import extract_all_speakers
    result = extract_all_speakers("")
    assert isinstance(result, dict)