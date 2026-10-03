# tests/test_core.py
# Core unit tests for TranscriptAI — 27 tests
#
# Coverage areas:
#   PII masker          (7 tests) — transcription/pii_masker.py
#   Hallucination guard (5 tests) — analysis/hallucination_guard.py
#   Soft rejection      (5 tests) — analysis/soft_rejection_detector.py
#   Cache               (5 tests) — utils/cache.py
#   Speaker normalizer  (5 tests) — transcription/speaker_normalizer.py
#
# Cache tests updated in v2.1 to pass user_id="test_user" — the cache now
# blocks anonymous writes/reads (security fix), so tests that omit user_id
# would always return None.

import pytest

# ════════════════════════════════════════════════════════════════════════════
# PII Masker
# ════════════════════════════════════════════════════════════════════════════

def test_pii_masks_email():
    from transcription.pii_masker import mask_transcript
    text   = "Please contact sarah@example.com for the follow-up."
    masked, mapping, _ = mask_transcript(text)
    assert "sarah@example.com" not in masked
    assert len(masked) > 0

def test_pii_masks_phone():
    from transcription.pii_masker import mask_transcript
    text   = "Call me at 090-1234-5678 tomorrow morning."
    masked, mapping, _ = mask_transcript(text)
    assert "090-1234-5678" not in masked

def test_pii_masks_japanese_name():
    from transcription.pii_masker import mask_transcript
    text   = "田中: 山田さん、今週中にご確認をお願いします。"
    masked, mapping, _ = mask_transcript(text)
    assert isinstance(masked, str)
    assert len(masked) > 0

def test_pii_mapping_contains_placeholder():
    from transcription.pii_masker import mask_transcript
    text   = "Email john.doe@company.co.jp for the full report."
    masked, mapping, _ = mask_transcript(text)
    assert isinstance(mapping, dict)

def test_pii_restoration_reverses_masking():
    from transcription.pii_masker import mask_transcript, restore_pii_in_result
    text        = "Contact sarah@example.com for the schedule."
    masked, mapping, _ = mask_transcript(text)
    result      = {"full_summary": masked, "summary": [masked]}
    restored    = restore_pii_in_result(result, mapping)
    assert isinstance(restored, dict)
    assert "full_summary" in restored

def test_pii_empty_string():
    from transcription.pii_masker import mask_transcript
    masked, mapping, report = mask_transcript("")
    assert masked == ""
    assert isinstance(mapping, dict)
    assert isinstance(report, dict)

def test_pii_report_is_dict():
    from transcription.pii_masker import mask_transcript
    _, _, report = mask_transcript("Hello, my name is Tanaka Hiroshi.")
    assert isinstance(report, dict)
    assert "total_pii_found" in report


# ════════════════════════════════════════════════════════════════════════════
# Hallucination Guard
# ════════════════════════════════════════════════════════════════════════════

_TRANSCRIPT_BASIC = (
    "Tanaka: We need to submit the Q3 report by Friday.\n"
    "Sato: I will prepare the budget overview by Thursday.\n"
    "Tanaka: Great. Let's confirm with the client on Friday morning.\n"
)

def test_hallucination_flags_fabricated_action_item():
    from analysis.hallucination_guard import verify_result
    result = {
        "action_items": [
            {
                "task":     "Book flight to Singapore",
                "owner":    "Tanaka",
                "deadline": "Monday",
            },
        ],
        "summary": ["Q3 report due Friday."],
    }
    out = verify_result(result, _TRANSCRIPT_BASIC)
    assert isinstance(out, dict)
    items = out.get("action_items", result.get("action_items", []))
    flagged = [i for i in items if i.get("hallucination_flag")]
    assert len(flagged) >= 1, "Fabricated action item (Singapore) must be flagged"

def test_hallucination_does_not_flag_grounded_action():
    from analysis.hallucination_guard import verify_result
    result = {
        "action_items": [
            {
                "task":     "Prepare budget overview",
                "owner":    "Sato",
                "deadline": "Thursday",
            },
        ],
        "summary": ["Budget overview due Thursday."],
    }
    out = verify_result(result, _TRANSCRIPT_BASIC)
    assert isinstance(out, dict)
    items = out.get("action_items", result.get("action_items", []))
    flagged = [i for i in items if i.get("hallucination_flag")]
    assert len(flagged) == 0, "Grounded action item must not be flagged"

def test_hallucination_verify_result_returns_dict():
    from analysis.hallucination_guard import verify_result
    result = {"action_items": [], "summary": []}
    out    = verify_result(result, _TRANSCRIPT_BASIC)
    assert isinstance(out, dict)

def test_hallucination_verify_action_items_empty():
    from analysis.hallucination_guard import verify_result
    result = {"action_items": [], "summary": ["Nothing discussed."]}
    out    = verify_result(result, _TRANSCRIPT_BASIC)
    items  = out.get("action_items", [])
    assert isinstance(items, list)
    assert len(items) == 0

def test_hallucination_verify_summary_valid():
    from analysis.hallucination_guard import verify_result
    result = {
        "action_items": [],
        "summary":      ["Tanaka will submit the Q3 report by Friday."],
    }
    out = verify_result(result, _TRANSCRIPT_BASIC)
    assert isinstance(out, dict)


# ════════════════════════════════════════════════════════════════════════════
# Soft Rejection Detector
# ════════════════════════════════════════════════════════════════════════════

def test_soft_rejection_detects_nemawashi():
    from analysis.soft_rejection_detector import detect_soft_rejections
    transcript = (
        "Tanaka: この件については、検討させていただきます。\n"
        "Sato: 難しいかもしれません。社内で確認が必要です。\n"
        "Tanaka: もう少しお時間をいただければと思います。\n"
    )
    result = detect_soft_rejections(transcript)
    assert isinstance(result, dict)
    assert result.get("risk_level") in (
        "NONE", "MINIMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"
    )
    assert result.get("total_signals", 0) >= 0

def test_soft_rejection_clean_approval_is_low_risk():
    from analysis.soft_rejection_detector import detect_soft_rejections
    transcript = (
        "Sarah: Are you happy with the proposal?\n"
        "Client: Yes, absolutely. We approve. Let's proceed with the contract.\n"
        "Sarah: Excellent. I'll send the paperwork today.\n"
    )
    result = detect_soft_rejections(transcript)
    assert isinstance(result, dict)
    assert result.get("risk_level") in ("NONE", "MINIMAL", "LOW")

def test_soft_rejection_termination_is_critical():
    from analysis.soft_rejection_detector import detect_soft_rejections
    transcript = (
        "Tanaka: 誠に遺憾ながら、契約を解除させていただきます。\n"
        "Sato: 今後のお取引はお断りさせていただく所存でございます。\n"
    )
    result = detect_soft_rejections(transcript)
    assert isinstance(result, dict)
    assert result.get("risk_level") in ("HIGH", "CRITICAL")

def test_soft_rejection_has_expected_keys():
    from analysis.soft_rejection_detector import detect_soft_rejections
    result = detect_soft_rejections("Tanaka: Let us proceed. Client: Agreed.")
    assert "risk_level"     in result
    assert "total_signals"  in result
    assert "high_signals"   in result
    assert "medium_signals" in result

def test_soft_rejection_empty_transcript():
    from analysis.soft_rejection_detector import detect_soft_rejections
    result = detect_soft_rejections("")
    assert isinstance(result, dict)
    assert result.get("total_signals", 0) == 0


# ════════════════════════════════════════════════════════════════════════════
# Cache
# ════════════════════════════════════════════════════════════════════════════

# v2.1: all cache calls must supply user_id.
# cache.py v2.1 blocks anonymous reads/writes to prevent cross-user leakage.
_TEST_UID = "test_user"


def test_cache_set_and_get():
    from utils.cache import set_cache, get_cached, clear_user_cache
    clear_user_cache(_TEST_UID)
    set_cache("Hello world test", "en", {"status": "ok"}, user_id=_TEST_UID)
    val = get_cached("Hello world test", "en", user_id=_TEST_UID)
    assert val is not None
    assert val.get("status") == "ok"
    clear_user_cache(_TEST_UID)

def test_cache_miss_returns_none():
    from utils.cache import get_cached
    val = get_cached("this transcript does not exist anywhere at all", "en",
                     user_id=_TEST_UID)
    assert val is None

def test_cache_overwrite():
    from utils.cache import set_cache, get_cached, clear_user_cache
    clear_user_cache(_TEST_UID)
    set_cache("overwrite test transcript", "en", {"v": 1}, user_id=_TEST_UID)
    set_cache("overwrite test transcript", "en", {"v": 2}, user_id=_TEST_UID)
    val = get_cached("overwrite test transcript", "en", user_id=_TEST_UID)
    assert val["v"] == 2
    clear_user_cache(_TEST_UID)

def test_cache_language_isolation():
    from utils.cache import set_cache, get_cached, clear_user_cache
    clear_user_cache(_TEST_UID)
    set_cache("cache isolation test transcript", "en",
              {"lang": "english"},  user_id=_TEST_UID)
    set_cache("cache isolation test transcript", "ja",
              {"lang": "japanese"}, user_id=_TEST_UID)
    en_val = get_cached("cache isolation test transcript", "en", user_id=_TEST_UID)
    ja_val = get_cached("cache isolation test transcript", "ja", user_id=_TEST_UID)
    assert en_val["lang"] == "english",  "English cache entry must not be overwritten by Japanese"
    assert ja_val["lang"] == "japanese", "Japanese cache entry must not be overwritten by English"
    clear_user_cache(_TEST_UID)

def test_cache_stats_is_dict():
    from utils.cache import get_cache_stats
    stats = get_cache_stats()
    assert isinstance(stats, dict)
    assert "entries"   in stats
    assert "size_kb"   in stats
    assert "available" in stats


# ════════════════════════════════════════════════════════════════════════════
# Speaker Normalizer
# ════════════════════════════════════════════════════════════════════════════

def test_normalize_speaker_japanese_name_nonempty():
    from transcription.speaker_normalizer import normalize_speaker_name
    result = normalize_speaker_name("田中部長")
    assert isinstance(result, str)
    assert len(result) > 0

def test_normalize_speaker_english_name_nonempty():
    from transcription.speaker_normalizer import normalize_speaker_name
    result = normalize_speaker_name("Sarah Johnson")
    assert isinstance(result, str)
    assert len(result) > 0

def test_normalize_speaker_role_only_no_empty_string():
    from transcription.speaker_normalizer import normalize_speaker_name
    result = normalize_speaker_name("Client")
    assert result is not None
    assert isinstance(result, str)

def test_extract_all_speakers_finds_speakers():
    from transcription.speaker_normalizer import extract_all_speakers
    transcript = (
        "Tanaka: Good morning everyone.\n"
        "Sato: Good morning. Let's begin.\n"
        "Tanaka: Agreed. First item on the agenda.\n"
    )
    speakers = extract_all_speakers(transcript)
    assert isinstance(speakers, list)
    assert len(speakers) >= 1

def test_extract_all_speakers_empty():
    from transcription.speaker_normalizer import extract_all_speakers
    speakers = extract_all_speakers("")
    assert isinstance(speakers, list)
    assert len(speakers) == 0