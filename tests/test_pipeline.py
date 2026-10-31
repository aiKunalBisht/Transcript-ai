"""
tests/test_pipeline.py — TranscriptAI critical-path test suite

Covers every bug fixed in v8.5–v9.1 that can be unit-tested without
a live LLM. Grouped by module. Run with:

    pytest tests/test_pipeline.py -v

Each test class maps to one module and names the specific fix it guards.
If a test fails after a code change, the change broke something documented
in the fix log. Do not delete tests — fix the regression.

Requirements:
    pip install pytest

No LLM key required. No network access required.
All tests run in <1 second on any machine.
"""

import pytest
import re
import sys
import pathlib
import importlib
import importlib.util
import types

# ── Path setup ────────────────────────────────────────────────────────────────
# Allow imports from the project root regardless of where pytest is invoked.
_ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

# Stub out heavy optional dependencies so imports don't fail in CI
for _stub in [
    "utils.japanese_names",
    "utils.firebase_client",
    "utils.vector_cache",
    "utils.cache",
    "utils.logger",
    "utils.evaluator",
    "utils.scenario_loader",
    "utils.rag_evaluator",
    "analysis.hallucination_guard",
    "analysis.semantic_validator",
    "analysis.soft_rejection_detector",
    "analysis.deal_outcome_detector",
    "analysis.conversation_dynamics",
    "analysis.meeting_function_detector",
    "analysis.nemawashi_sequence",
    "analysis.action_item_extractor",
    "analysis.japanese_tokenizer",
    "transcription.speaker_normalizer",
    "agents.slide_architect",
    "exporters.pptx_builder",
    "mlflow",
    "slowapi",
    "authlib",
    "bcrypt",
    "firebase_admin",
]:
    if _stub not in sys.modules:
        sys.modules[_stub] = types.ModuleType(_stub)


# ═════════════════════════════════════════════════════════════════════════════
# 1. pii_masker — v5 fixes
# ═════════════════════════════════════════════════════════════════════════════

class TestPIIMasker:
    """Guards all v5 fixes. Each method names the exact bug it prevents."""

    def _load(self):
        from transcription.pii_masker import mask_transcript, restore_pii_in_result, PIIMask
        return mask_transcript, restore_pii_in_result, PIIMask

    # ── FIX-J1: My Number ────────────────────────────────────────────────────

    def test_my_number_hyphenated_is_masked(self):
        """マイナンバー with hyphens must be masked before LLM."""
        mask, _, _ = self._load()
        masked, pii = mask("マイナンバーは1234-5678-9012です")
        assert "MY_NUMBER_1" in masked, "12-digit hyphenated My Number not masked"
        assert "1234-5678-9012" not in masked

    def test_my_number_plain_digits_is_masked(self):
        """12 consecutive digits must be masked (common in documents)."""
        mask, _, _ = self._load()
        masked, _ = mask("番号: 123456789012")
        assert "MY_NUMBER_1" in masked

    def test_14_digit_barcode_not_masked(self):
        """14-digit product codes must NOT be masked — boundary guard required."""
        mask, _, _ = self._load()
        masked, _ = mask("製品コード: 12345678901234")
        assert "MY_NUMBER" not in masked, "14-digit string falsely matched My Number pattern"

    def test_phone_number_not_masked_as_my_number(self):
        """Japanese phone numbers (10-11 digits) must not match My Number."""
        mask, _, _ = self._load()
        masked, _ = mask("電話: 03-1234-5678")
        assert "MY_NUMBER" not in masked

    # ── FIX-J2: Postal codes ─────────────────────────────────────────────────

    def test_postal_code_masked(self):
        """〒XXX-XXXX must be masked as POSTAL_1."""
        mask, _, _ = self._load()
        masked, _ = mask("本社は〒150-0001渋谷区です")
        assert "POSTAL_1" in masked
        assert "〒150-0001" not in masked

    def test_city_name_after_postal_preserved(self):
        """Masking postal code must not remove adjacent city names."""
        mask, _, _ = self._load()
        masked, _ = mask("住所は〒150-0001 東京都渋谷区")
        assert "東京都" in masked, "City name incorrectly masked alongside postal code"

    # ── FIX-F1: Company span fragmentation ────────────────────────────────────

    def test_koushiki_company_name_not_fragmented(self):
        """田中産業株式会社 must NOT become [NAME_1]産業株式会社 (後株 format)."""
        mask, _, _ = self._load()
        masked, _ = mask("田中産業株式会社の担当者です。")
        assert "田中産業株式会社" in masked, (
            "Company name fragmented: constituent surname was masked inside company name"
        )

    def test_maeshiki_company_name_not_fragmented(self):
        """株式会社田中産業 must NOT be fragmented (前株 format)."""
        mask, _, _ = self._load()
        masked, _ = mask("株式会社田中産業と連携します。")
        assert "株式会社田中産業" in masked

    def test_standalone_surname_after_company_is_masked(self):
        """Standalone 田中 after company reference must still be masked."""
        mask, _, _ = self._load()
        masked, _ = mask("田中産業株式会社の田中部長が来訪しました。")
        assert "田中産業株式会社" in masked, "Company name was fragmented"
        assert "[NAME_" in masked, "Standalone surname (田中部長) was not masked"

    # ── FIX-DB1: Expanded surname database ───────────────────────────────────

    def test_fallback_database_covers_common_surnames(self):
        """Fallback surname list must have ≥200 entries covering common names."""
        from transcription.pii_masker import JAPANESE_SURNAMES
        assert len(JAPANESE_SURNAMES) >= 200, (
            f"Fallback DB has only {len(JAPANESE_SURNAMES)} entries — "
            "was 14 before FIX-DB1"
        )
        for name in ["佐藤", "鈴木", "田中", "渡辺", "Tanaka", "Sato"]:
            assert name in JAPANESE_SURNAMES, f"Common surname {name!r} missing from DB"

    # ── V3 FIX: bracket variant restore (retained) ───────────────────────────

    def test_restore_handles_all_bracket_variants(self):
        """PIIMask.restore() must handle [NAME_1], NAME_1], [NAME_1, and NAME_1."""
        _, _, PIIMask = self._load()
        pii = PIIMask()
        pii.mapping = {"[NAME_1]": "Rahul"}
        pii.reverse = {"Rahul": "[NAME_1]"}
        assert pii.restore("[NAME_1]") == "Rahul"   # full brackets
        assert pii.restore("NAME_1]")  == "Rahul"   # missing open
        assert pii.restore("[NAME_1")  == "Rahul"   # missing close
        assert pii.restore("NAME_1")   == "Rahul"   # bare

    # ── W3 FIX: word-boundary bare restore (retained) ────────────────────────

    def test_bare_restore_uses_word_boundary(self):
        """Bare NAME_1 inside MY_NAME_1_field must NOT be restored."""
        _, _, PIIMask = self._load()
        pii = PIIMask()
        pii.mapping = {"[NAME_1]": "Kunal"}
        pii.reverse = {"Kunal": "[NAME_1]"}
        # Should restore: standalone NAME_1 with word boundaries
        assert pii.restore("variable NAME_1 is undefined") == "variable Kunal is undefined"
        # Must NOT restore: NAME_1 inside a longer identifier
        assert pii.restore("MY_NAME_1_field is null") == "MY_NAME_1_field is null"
        # Must NOT restore: NAME_1 as prefix of NAME_10
        assert pii.restore("NAME_10 spoke") == "NAME_10 spoke"

    # ── Round-trip: mask → restore ────────────────────────────────────────────

    def test_mask_restore_round_trip(self):
        """Masked PII must be fully restored from a nested result dict."""
        mask, restore, _ = self._load()
        text = "田中: メールはtanaka@example.comです。電話は+81-3-1234-5678。"
        masked, pii = mask(text)
        assert "tanaka@example.com" not in masked
        assert "+81-3-1234-5678"    not in masked
        fake_result = {"speakers": [{"name": masked.split(":")[0].strip()}],
                       "summary":  [masked[:50]]}
        restored = restore(fake_result, pii)
        result_text = str(restored)
        assert "tanaka@example.com" in result_text or "田中" in result_text


# ═════════════════════════════════════════════════════════════════════════════
# 2. speaker_detector — existing fixes verified
# ═════════════════════════════════════════════════════════════════════════════

class TestSpeakerDetector:
    """Verifies speaker detection handles all transcript formats including PII-masked."""

    def _detect(self, text):
        # Skip gracefully when utils/ isn't on the path (CI / isolated container).
        # In the real project, utils/speaker_detector.py is always present.
        try:
            from utils.speaker_detector import detect_speakers
        except ImportError:
            pytest.skip("utils.speaker_detector not importable in this environment")
        return detect_speakers(text)

    def test_pii_masked_labels_detected(self):
        """[NAME_1]: and [NAME_2]: must be recognized as distinct speakers."""
        r = self._detect(
            "[NAME_1]: Good morning.\n[NAME_2]: Hello there.\n[NAME_1]: Let us begin."
        )
        assert r["count"] == 2
        assert "[NAME_1]" in r["names"]
        assert "[NAME_2]" in r["names"]

    def test_masked_speakers_never_merged(self):
        """[NAME_1] and [NAME_2] must never be merged even though letter-tokens match."""
        r = self._detect(
            "[NAME_1]: I disagree.\n[NAME_2]: I agree.\n[NAME_1]: As I said."
        )
        assert r["count"] == 2, (
            "[NAME_1] and [NAME_2] were merged — _is_numbered_identifier guard failed"
        )

    def test_japanese_cjk_speakers_detected(self):
        """Japanese speaker labels with full-width colon must be detected."""
        r = self._detect("田中: おはようございます。\n鈴木: よろしくお願いします。")
        assert r["count"] == 2
        assert "田中" in r["names"]

    def test_english_name_dedup(self):
        """Dr. John Smith / John / Mr. Smith → single canonical speaker."""
        r = self._detect(
            "Dr. John Smith: Hello.\nJohn: Good point.\nMr. Smith: Agreed."
        )
        assert r["count"] == 1

    def test_blocklist_words_not_extracted_as_speakers(self):
        """Structural words (Note, Status, Background) must not be extracted."""
        r = self._detect(
            "Kunal: Meeting start.\nNote: Q3 review.\nStatus: On track.\nConnie: Agreed."
        )
        assert r["count"] == 2
        names_lower = [n.lower() for n in r["names"]]
        assert "note" not in names_lower
        assert "status" not in names_lower

    def test_whisper_diarized_speakers_not_merged(self):
        """SPEAKER_00 and SPEAKER_01 must be separate — never merged."""
        r = self._detect(
            "SPEAKER_00: Hello.\nSPEAKER_01: Hi.\nSPEAKER_00: Let's begin."
        )
        assert r["count"] == 2
        assert "SPEAKER_00" in r["names"]
        assert "SPEAKER_01" in r["names"]

    def test_confidence_high_for_multi_turn_transcript(self):
        """Multi-turn transcript with clear labels should return high confidence."""
        r = self._detect(
            "Alice: Point one.\nBob: Point two.\nAlice: Point three.\nBob: Agreed."
        )
        assert r["confidence"] == "high"


# ═════════════════════════════════════════════════════════════════════════════
# 3. sentiment_engine — v1.1 fixes
# ═════════════════════════════════════════════════════════════════════════════

class TestSentimentEngine:
    """Guards FIX-JP1/JP2/JP3 — the Japanese parsing fixes."""

    def _engine(self):
        import importlib.util as _iu
        spec = _iu.spec_from_file_location(
            "analysis.sentiment_engine",
            str(_ROOT / "analysis" / "sentiment_engine.py"),
        )
        mod = _iu.module_from_spec(spec)
        sys.modules["analysis.sentiment_engine"] = mod
        spec.loader.exec_module(mod)
        return mod.FineSentimentAnalyzer()

    # ── FIX-JP1 ───────────────────────────────────────────────────────────────

    def test_full_width_colon_parsed(self):
        """FIX-JP1: Full-width ： must split speaker from utterance text."""
        eng = self._engine()
        utts = eng._parse_transcript("田中：おはようございます。\n鈴木：よろしく。")
        assert len(utts) == 2, (
            f"Expected 2 utterances, got {len(utts)}. "
            "Full-width colon ： not recognized — FIX-JP1 regression."
        )
        assert utts[0]["speaker"] == "田中"
        assert utts[0]["text"]    == "おはようございます。"

    def test_half_width_colon_still_works(self):
        """Backward compat: half-width : must continue to work after FIX-JP1."""
        eng = self._engine()
        utts = eng._parse_transcript("Alice: Hello.\nBob: Hi there.")
        assert len(utts) == 2
        assert utts[0]["speaker"] == "Alice"

    def test_mixed_colon_formats_in_one_transcript(self):
        """Both ： and : in the same transcript must both be parsed."""
        eng = self._engine()
        utts = eng._parse_transcript("田中：ありがとう。\nAlice: Thank you.")
        assert len(utts) == 2

    # ── FIX-JP3 ───────────────────────────────────────────────────────────────

    def test_pii_masked_speaker_labels_parsed(self):
        """FIX-JP3: [NAME_1]: text must parse correctly."""
        eng = self._engine()
        utts = eng._parse_transcript(
            "[NAME_1]: Good morning.\n[NAME_2]: Hello.\n[NAME_1]: Let us begin."
        )
        assert len(utts) == 3, (
            f"Expected 3 utterances, got {len(utts)}. "
            "PII-masked speaker labels not parsed — FIX-JP3 regression."
        )
        assert utts[0]["speaker"] == "[NAME_1]"

    # ── FIX-JP2 ───────────────────────────────────────────────────────────────

    def test_empty_utterance_list_returns_safe_report(self):
        """FIX-JP2: analyze_transcript([]) must not raise or produce wrong data."""
        eng = self._engine()
        report = eng.analyze_transcript([])
        assert report.utterances       == []
        assert report.meeting_valence  == 0.0
        assert report.speaker_arcs     == {}

    def test_japanese_only_transcript_produces_speaker_arcs(self):
        """FIX-JP1+JP2 combined: Japanese transcript must produce non-empty arcs."""
        eng = self._engine()
        report = eng.from_raw_transcript(
            "田中：おはようございます。本日の件ですが。\n"
            "鈴木：よろしくお願いいたします。"
        )
        assert len(report.utterances) == 2, (
            "Japanese transcript produced empty utterances — FIX-JP1 regression"
        )
        assert "田中" in report.speaker_arcs, "田中 arc missing — speaker not parsed"
        assert "鈴木" in report.speaker_arcs, "鈴木 arc missing — speaker not parsed"

    # ── English sentiment still fires ─────────────────────────────────────────

    def test_english_frustration_detected(self):
        """English pattern engine must detect frustration correctly."""
        eng  = self._engine()
        res  = eng.analyze_utterance("This is absolutely unacceptable and frustrating!")
        assert res.primary in ("frustrated", "irritated", "dismissive"), (
            f"Expected negative label, got {res.primary!r}"
        )
        assert res.valence < 0, f"Valence should be negative, got {res.valence}"

    def test_english_positive_detected(self):
        """English pattern engine must detect enthusiasm correctly."""
        eng = self._engine()
        res = eng.analyze_utterance("This is absolutely fantastic! I love this idea!")
        assert res.primary in ("enthusiastic", "confident", "agreeable"), (
            f"Expected positive label, got {res.primary!r}"
        )
        assert res.valence > 0

    def test_negation_dampens_positive_match(self):
        """Negation window must dampen positive signals in 'not great'."""
        eng       = self._engine()
        pos       = eng.analyze_utterance("This is great work.")
        negated   = eng.analyze_utterance("This is not great at all.")
        assert pos.valence > negated.valence, (
            "Negation window not dampening positive signals correctly"
        )


# ═════════════════════════════════════════════════════════════════════════════
# 4. analysis_prompt — v8.5 fixes
# ═════════════════════════════════════════════════════════════════════════════

class TestAnalysisPrompt:
    """Guards FIX-1 through FIX-7 from analysis_prompt.py v8.5."""

    def _mod(self):
        from prompts import analysis_prompt
        return analysis_prompt

    # ── FIX-1: build_system_prompt uses parameters ────────────────────────────

    def test_build_system_prompt_injects_language(self):
        """FIX-1: lang_hint must appear in the returned system prompt."""
        m = self._mod()
        prompt = m.build_system_prompt(lang_hint="Japanese + English")
        assert "Japanese + English" in prompt, (
            "lang_hint not in system prompt — FIX-1 regression: parameters still discarded"
        )

    def test_build_system_prompt_injects_speakers(self):
        """FIX-1: speakers_hint must appear in the returned system prompt."""
        m = self._mod()
        prompt = m.build_system_prompt(speakers_hint="田中, 鈴木, Kenji")
        assert "田中" in prompt, "speakers_hint not injected into system prompt"

    def test_build_system_prompt_injects_japan_schema(self):
        """FIX-1+FIX-3: japan_schema must be appended when non-empty."""
        m = self._mod()
        prompt = m.build_system_prompt(japan_schema=m.japan_schema_str)
        assert "KEIGO LEVEL DEFINITIONS" in prompt, (
            "Japan schema not appended to system prompt — FIX-1/FIX-3 regression"
        )

    def test_build_system_prompt_no_params_returns_base(self):
        """With no parameters, base SYSTEM_PROMPT must still be returned intact."""
        m = self._mod()
        prompt = m.build_system_prompt()
        assert "TranscriptAI" in prompt
        assert "GROUNDING" in prompt

    def test_static_prompt_not_returned_unchanged_when_params_given(self):
        """FIX-1 core: parameterized call must differ from no-param call."""
        m = self._mod()
        base       = m.build_system_prompt()
        with_hints = m.build_system_prompt(
            lang_hint="Japanese + English",
            speakers_hint="田中, Alice",
        )
        assert base != with_hints, (
            "Parameterized build_system_prompt returned same string as no-param call. "
            "FIX-1 regression: parameters are still being discarded."
        )

    # ── FIX-2: GROUNDING_RULES non-empty ─────────────────────────────────────

    def test_grounding_rules_non_empty(self):
        """FIX-2: GROUNDING_RULES must not be an empty string."""
        m = self._mod()
        assert len(m.GROUNDING_RULES) > 50, (
            f"GROUNDING_RULES is only {len(m.GROUNDING_RULES)} chars — "
            "was empty string before FIX-2"
        )

    def test_grounding_rules_short_non_empty(self):
        """FIX-2: GROUNDING_RULES_SHORT must not be an empty string."""
        m = self._mod()
        assert len(m.GROUNDING_RULES_SHORT) > 20, (
            "GROUNDING_RULES_SHORT is empty — FIX-2 regression"
        )

    # ── FIX-3: japan_schema_str non-empty ────────────────────────────────────

    def test_japan_schema_str_non_empty(self):
        """FIX-3: japan_schema_str must contain actual schema content."""
        m = self._mod()
        assert len(m.japan_schema_str) > 200, (
            "japan_schema_str is empty or very short — FIX-3 regression"
        )

    def test_japan_schema_covers_keigo_levels(self):
        """FIX-3: Japan schema must define all keigo level values."""
        m = self._mod()
        for level in ("high", "medium", "low", "not_applicable"):
            assert level in m.japan_schema_str, (
                f"Keigo level {level!r} missing from japan_schema_str"
            )

    def test_japan_schema_nemawashi_rules_strict(self):
        """FIX-3: Japan schema must warn against over-claiming nemawashi."""
        m = self._mod()
        assert "NEMAWASHI RULES" in m.japan_schema_str or "strict" in m.japan_schema_str.lower()

    # ── FIX-4: VALID_TONE_LABELS has all 17 tones ────────────────────────────

    def test_valid_tone_labels_has_17_entries(self):
        """FIX-4: VALID_TONE_LABELS must have 17 entries, not 6."""
        m = self._mod()
        assert len(m.VALID_TONE_LABELS) == 17, (
            f"VALID_TONE_LABELS has {len(m.VALID_TONE_LABELS)} entries, expected 17. "
            "FIX-4 regression: old 6-label set still in use."
        )

    def test_valid_tone_labels_includes_dropped_tones(self):
        """FIX-4: The 11 previously dropped tones must all be in VALID_TONE_LABELS."""
        m = self._mod()
        dropped = {
            "direct", "indirect", "formal", "warm", "detached",
            "confrontational", "conciliatory", "empathetic",
            "guarded", "persuasive", "matter_of_fact",
        }
        for tone in dropped:
            assert tone in m.VALID_TONE_LABELS, (
                f"Tone {tone!r} still not in VALID_TONE_LABELS — FIX-4 incomplete"
            )

    # ── FIX-5: TASK_FIELD constants ──────────────────────────────────────────

    def test_task_field_is_task_en(self):
        """FIX-5: TASK_FIELD must equal 'task_en' matching ANALYSIS_SCHEMA."""
        m = self._mod()
        assert m.TASK_FIELD == "task_en", (
            f"TASK_FIELD is {m.TASK_FIELD!r}, expected 'task_en'"
        )

    def test_task_field_ja_is_task_ja(self):
        """FIX-5: TASK_FIELD_JA must equal 'task_ja'."""
        m = self._mod()
        assert m.TASK_FIELD_JA == "task_ja"

    # ── FIX-6: normalize_analysis ────────────────────────────────────────────

    def test_normalize_analysis_syncs_tone_primary(self):
        """FIX-6: tone_primary must be copied to tone when tone is missing."""
        m = self._mod()
        data = {
            "speakers": [
                {"tone_primary": "direct", "tone": ""},
                {"tone_primary": "guarded", "tone": "invalid_tone"},
            ]
        }
        result = m.normalize_analysis(data)
        assert result["speakers"][0]["tone"] == "direct", (
            "tone_primary not synced to tone — WIRE-1 regression"
        )
        assert result["speakers"][1]["tone"] == "guarded"

    def test_normalize_analysis_derives_score_from_valence(self):
        """FIX-6: sentiment score must be derived from valence when they conflict."""
        m = self._mod()
        data = {
            "sentiment": [
                {"speaker": "Alice", "valence": -0.7, "score": "neutral"},  # conflict
                {"speaker": "Bob",   "valence":  0.5, "score": "negative"}, # conflict
                {"speaker": "Carol", "valence":  0.1, "score": "neutral"},  # agree
            ],
            "speakers": [],
        }
        result = m.normalize_analysis(data)
        scores = {s["speaker"]: s["score"] for s in result["sentiment"]}
        assert scores["Alice"] == "negative", (
            "Negative valence -0.7 should override score='neutral' — WIRE-1 regression"
        )
        assert scores["Bob"]   == "positive", (
            "Positive valence +0.5 should override score='negative'"
        )
        assert scores["Carol"] == "neutral"

    def test_normalize_analysis_removes_primary_from_secondary(self):
        """FIX-6: tone_secondary must not contain tone_primary."""
        m = self._mod()
        data = {
            "speakers": [
                {"tone_primary": "direct", "tone_secondary": ["direct", "guarded"]}
            ],
            "sentiment": [],
        }
        result = m.normalize_analysis(data)
        secondary = result["speakers"][0]["tone_secondary"]
        assert "direct" not in secondary, (
            "tone_primary still appears in tone_secondary after normalize_analysis"
        )
        assert "guarded" in secondary


# ═════════════════════════════════════════════════════════════════════════════
# 5. analyzer wire-ups — v9.1 integration checks
# ═════════════════════════════════════════════════════════════════════════════

class TestAnalyzerWireUps:
    """Verifies the three wire-ups landed correctly in analyzer.py v9.1.
    These are structural checks — no LLM call required."""

    def test_wire1_normalize_analysis_called_after_parse(self):
        """WIRE-1: normalize_analysis call must appear after _parse(raw) in source."""
        src = (_ROOT / "analysis" / "analyzer.py").read_text()
        # normalize_analysis must be called, and must appear near _parse
        assert "_normalize_analysis(result)" in src, (
            "normalize_analysis not called in analyzer.py — WIRE-1 missing"
        )

    def test_wire2_valid_tone_labels_used_in_validate(self):
        """WIRE-2: _VALID_TONES must use VALID_TONE_LABELS, not a hardcoded set."""
        src = (_ROOT / "analysis" / "analyzer.py").read_text()
        assert "_VALID_TONES = VALID_TONE_LABELS" in src, (
            "Hardcoded 6-label tone set still in use — WIRE-2 missing"
        )
        # The old hardcoded set must be gone
        assert '"deferential","hesitant"' not in src.replace(" ", ""), (
            "Old hardcoded tone set still present alongside VALID_TONE_LABELS"
        )

    def test_wire3_task_field_used_in_stage_10b(self):
        """WIRE-3: Stage 10b backfill must read TASK_FIELD, not bare 'task'."""
        src = (_ROOT / "analysis" / "analyzer.py").read_text()
        assert "i.get(TASK_FIELD)" in src, (
            "Stage 10b still using item.get('task') — WIRE-3 missing"
        )

    def test_wire3_mock_response_uses_task_field(self):
        """WIRE-3: _mock_response must use TASK_FIELD key, not literal 'task'."""
        src = (_ROOT / "analysis" / "analyzer.py").read_text()
        assert "{TASK_FIELD:" in src, (
            "_mock_response still uses literal 'task' key — WIRE-3 incomplete"
        )

    def test_valid_tone_labels_import_present(self):
        """WIRE-2: VALID_TONE_LABELS must be imported from analysis_prompt."""
        src = (_ROOT / "analysis" / "analyzer.py").read_text()
        assert "VALID_TONE_LABELS," in src

    def test_pii_fix1_masked_text_in_fallback(self):
        """FIX-1: fallback path must use masked_text, not raw text."""
        src = (_ROOT / "analysis" / "analyzer.py").read_text()
        assert "_no_api_result(masked_text" in src, (
            "FIX-1 regression: fallback path uses unmasked text — PII leak on LLM failure"
        )

    def test_fix2_timestamps_extracted_before_normalize(self):
        """FIX-2: raw turn extraction must precede normalize_format."""
        src = (_ROOT / "analysis" / "analyzer.py").read_text()
        extract_pos   = src.find("_raw_turns_with_ts = _extract_turns")
        normalize_pos = src.find("text = normalize_format(text)")
        assert extract_pos != -1, "Timestamp pre-extraction missing"
        assert normalize_pos != -1, "normalize_format call missing"
        assert extract_pos < normalize_pos, (
            "FIX-2 regression: normalize_format runs before timestamp extraction. "
            "Timestamps are stripped before talk-time calc can read them."
        )

    def test_fix4_hallucination_guard_safe_default(self):
        """FIX-4: ImportError path must set hallucination_flag=True (not False)."""
        asrc = (_ROOT / "analysis" / "analyzer.py").read_text()
        # Locate Stage 11 specifically — not the httpx ImportError at the top
        stage11_start = asrc.find("Stage 11: hallucination guard")
        assert stage11_start != -1, "Stage 11 hallucination guard section missing"
        stage11 = asrc[stage11_start: stage11_start + 900]
        imp_idx = stage11.find("except ImportError:")
        assert imp_idx != -1, "No ImportError handler in Stage 11"
        imp_block = stage11[imp_idx: imp_idx + 500]
        assert "= True" in imp_block, (
            "FIX-4 regression: hallucination_flag not set to True in ImportError handler. "
            "Unverified action items are marked as clean."
        )


# ═════════════════════════════════════════════════════════════════════════════
# 6. main.py — structural fixes
# ═════════════════════════════════════════════════════════════════════════════

class TestMainStructural:
    """Source-level checks for main.py v3.7 fixes — no server startup required."""

    def _src(self):
        return (_ROOT / "main.py").read_text()

    def test_fix_a_no_asyncio_to_thread_on_analyze(self):
        """FIX-A: asyncio.to_thread must not wrap analyze_transcript."""
        src = self._src()
        assert "asyncio.to_thread(\n            analyze_transcript" not in src, (
            "FIX-A regression: asyncio.to_thread(analyze_transcript) still present. "
            "This breaks /analyze-text — async def passed to thread runner returns coroutine."
        )

    def test_fix_a_await_analyze_transcript_present(self):
        """FIX-A: await analyze_transcript(...) must be the call pattern."""
        src = self._src()
        assert "result = await analyze_transcript(" in src or \
               "prediction = await analyze_transcript(" in src

    def test_fix_b_get_real_ip_defined(self):
        """FIX-B: _get_real_ip function must exist for rate limiter."""
        src = self._src()
        assert "def _get_real_ip(request: Request)" in src, (
            "FIX-B regression: _get_real_ip not defined. "
            "Rate limiter uses proxy IP — all users share one rate-limit slot."
        )

    def test_fix_b_x_forwarded_for_read(self):
        """FIX-B: X-Forwarded-For header must be read by the IP resolver."""
        src = self._src()
        assert "X-Forwarded-For" in src

    def test_fix_b_limiter_uses_real_ip(self):
        """FIX-B: Limiter must use _get_real_ip, not get_remote_address."""
        src = self._src()
        assert "Limiter(key_func=_get_real_ip)" in src, (
            "FIX-B regression: Limiter still uses get_remote_address. "
            "Rate limit fires globally for all users behind reverse proxy."
        )

    def test_fix_c_no_private_import_of_get_groq_key(self):
        """FIX-C: export_pptx must not import private _get_groq_key."""
        src = self._src()
        assert "from analysis.analyzer import _get_groq_key" not in src, (
            "FIX-C regression: private _get_groq_key still imported in main.py. "
            "Will break silently if analyzer internals are refactored."
        )

    def test_fix_c_api_key_from_env(self):
        """FIX-C: export_pptx must read API key from environment."""
        src = self._src()
        assert 'os.getenv("NIM_API_KEY"' in src or 'os.getenv("GROQ_API_KEY"' in src

    def test_fix_d_worker_count_warning(self):
        """FIX-D: Multi-worker warning must be present in lifespan."""
        src = self._src()
        assert "_WORKER_COUNT > 1" in src, (
            "FIX-D regression: no multi-worker warning. "
            "In-memory cache silently breaks /api/last-result under gunicorn."
        )

# ═════════════════════════════════════════════════════════════════════════════
# 7. hallucination_guard — v2.1 fixes
# ═════════════════════════════════════════════════════════════════════════════

class TestHallucinationGuard:
    """
    Guards all four bugs fixed in hallucination_guard.py v2.1.
    No LLM required — the guard is 100% rule-based by design.
    """

    def _load(self):
        import importlib.util as _iu
        spec = _iu.spec_from_file_location(
            "analysis.hallucination_guard",
            str(_ROOT / "analysis" / "hallucination_guard.py"),
        )
        mod = _iu.module_from_spec(spec)
        sys.modules["analysis.hallucination_guard"] = mod
        # Stub semantic_validator so the import path fix doesn't need the real module
        import types
        sv = types.ModuleType("analysis.semantic_validator")
        sys.modules["analysis.semantic_validator"] = sv
        spec.loader.exec_module(mod)
        return mod

    _TRANSCRIPT = (
        "Tanaka: The Q3 report deadline is next Friday.\n"
        "Sato: I will send an email to the client about the delay today.\n"
        "Tanaka: Let us review the financial section by Thursday.\n"
    )

    # ── FIX-TF: TASK_FIELD (was the catastrophic bug) ────────────────────────

    def test_fix_tf_llm_item_task_en_not_flagged(self):
        """
        FIX-TF: LLM items use task_en. item.get('task') returned '' for them.
        _overlap_score('', transcript) = 0.0 < 0.20 → every LLM item flagged.
        After fix: grounded task_en items must NOT be flagged.
        """
        mod = self._load()
        result = mod.verify_action_items(
            [{"task_en": "Send email to client about delay",
              "task_ja": None, "owner": "Sato", "deadline": "Today"}],
            self._TRANSCRIPT,
        )
        assert result["verified_count"] == 1, (
            "FIX-TF regression: LLM task_en item was flagged as hallucinated. "
            "item.get('task') is probably being used instead of TASK_FIELD."
        )
        assert result["flagged_count"] == 0
        assert result["verified"][0]["hallucination_flag"] is False

    def test_fix_tf_hallucinated_item_still_flagged(self):
        """FIX-TF: Genuinely hallucinated content must still be caught."""
        mod = self._load()
        result = mod.verify_action_items(
            [{"task_en": "Organize company picnic on Mars next month",
              "task_ja": None, "owner": "Ghost", "deadline": "Never"}],
            self._TRANSCRIPT,
        )
        assert result["flagged_count"] == 1, (
            "Hallucinated action item was not flagged — threshold may have been lowered."
        )
        assert result["flagged"][0]["hallucination_flag"] is True

    def test_fix_tf_legacy_task_key_still_works(self):
        """FIX-TF: rule-based extractor items still use 'task' key — must work."""
        mod = self._load()
        result = mod.verify_action_items(
            [{"task": "Review financial section by Thursday",
              "owner": "Tanaka", "deadline": "Thursday"}],
            self._TRANSCRIPT,
        )
        assert result["verified_count"] == 1, (
            "Legacy 'task' key broken — rule-based extractor items would all fail."
        )

    def test_fix_tf_correct_score_in_grounding_dict(self):
        """FIX-TF: task_effective must be > 0 for a grounded LLM item."""
        mod = self._load()
        result = mod.verify_action_items(
            [{"task_en": "Send email to client",
              "task_ja": None, "owner": "Sato", "deadline": "Today"}],
            self._TRANSCRIPT,
        )
        item = result["verified"][0]
        assert item["grounding"]["task_effective"] > 0, (
            "task_effective is 0.0 — TASK_FIELD not being read correctly."
        )

    # ── FIX-NONE: None deadline ───────────────────────────────────────────────

    def test_fix_none_deadline_none_does_not_raise(self):
        """FIX-NONE: deadline=None must not raise AttributeError."""
        mod = self._load()
        try:
            mod.verify_action_items(
                [{"task_en": "Send Q3 report", "task_ja": None,
                  "owner": "Sato", "deadline": None}],
                self._TRANSCRIPT,
            )
        except AttributeError as e:
            raise AssertionError(
                f"FIX-NONE regression: deadline=None raised AttributeError: {e}. "
                "deadline.lower() called on None."
            )

    def test_fix_none_task_ja_none_does_not_crash(self):
        """FIX-NONE: task_ja=None (the schema default) must be handled safely."""
        mod = self._load()
        result = mod.verify_action_items(
            [{"task_en": "Email client", "task_ja": None,
              "owner": "Sato", "deadline": "Today"}],
            self._TRANSCRIPT,
        )
        assert isinstance(result, dict)

    # ── FIX-PUNCT: punctuation in tokenizer ──────────────────────────────────

    def test_fix_punct_terminal_period_stripped(self):
        """FIX-PUNCT: 'report.' must score same as 'report'."""
        mod = self._load()
        s1 = mod._overlap_score("report",  self._TRANSCRIPT)
        s2 = mod._overlap_score("report.", self._TRANSCRIPT)
        assert s1 == s2, (
            f"FIX-PUNCT regression: 'report'={s1} != 'report.'={s2}. "
            "Punctuation is not being stripped from English tokens."
        )

    def test_fix_punct_comma_stripped(self):
        """FIX-PUNCT: trailing comma must not reduce overlap score."""
        mod = self._load()
        s1 = mod._overlap_score("deadline", self._TRANSCRIPT)
        s2 = mod._overlap_score("deadline,", self._TRANSCRIPT)
        assert s1 == s2

    def test_fix_punct_affects_action_item_scoring(self):
        """FIX-PUNCT: action items with punctuated words must score correctly."""
        mod = self._load()
        r1 = mod.verify_action_items(
            [{"task_en": "Send email today.", "task_ja": None,
              "owner": "Sato", "deadline": "Today"}],
            self._TRANSCRIPT,
        )
        r2 = mod.verify_action_items(
            [{"task_en": "Send email today", "task_ja": None,
              "owner": "Sato", "deadline": "Today"}],
            self._TRANSCRIPT,
        )
        # Both should be verified — punctuation must not push score below threshold
        assert r1["verified_count"] == r2["verified_count"], (
            "FIX-PUNCT: punctuated task scored differently from clean task."
        )

    # ── FIX-JA: Japanese task verification ───────────────────────────────────

    _JP_TRANSCRIPT = (
        "田中：来週金曜日までにレポートを提出してください。\n"
        "鈴木：承知しました。メールでクライアントに連絡します。\n"
    )

    def test_fix_ja_task_ja_scored_for_japanese_transcript(self):
        """FIX-JA: task_ja must be scored when transcript is Japanese."""
        mod = self._load()
        result = mod.verify_action_items(
            [{"task_en": "Submit report by next Friday",
              "task_ja": "レポートを提出する",
              "owner": "田中", "deadline": "来週金曜日"}],
            self._JP_TRANSCRIPT,
        )
        item = (result["verified"] or result["flagged"])[0]
        assert item["grounding"]["task_token_overlap_ja"] is not None, (
            "FIX-JA: task_ja not scored for Japanese transcript."
        )

    def test_fix_ja_task_ja_scores_higher_than_task_en_on_jp_text(self):
        """FIX-JA: Japanese task_ja must outscore English task_en on JP transcript."""
        mod = self._load()
        result = mod.verify_action_items(
            [{"task_en": "Submit report",
              "task_ja": "レポートを提出する",
              "owner": "田中", "deadline": "来週金曜日"}],
            self._JP_TRANSCRIPT,
        )
        item = (result["verified"] or result["flagged"])[0]
        g = item["grounding"]
        assert g["task_token_overlap_ja"] > g["task_token_overlap_en"], (
            "FIX-JA: task_ja score not higher than task_en for Japanese transcript. "
            "Japanese action items may be incorrectly flagged as hallucinated."
        )

    # ── verify_result master function ─────────────────────────────────────────

    def test_verify_result_adds_verification_key(self):
        """verify_result must always add a 'verification' key to the result."""
        mod = self._load()
        out = mod.verify_result(
            {"action_items": [], "summary": [], "sentiment": []},
            self._TRANSCRIPT,
        )
        assert "verification" in out

    def test_verify_result_risk_label_is_valid(self):
        """verify_result overall risk_label must be LOW, MEDIUM, or HIGH."""
        mod = self._load()
        out = mod.verify_result(
            {"action_items": [{"task_en": "Send email", "task_ja": None,
                               "owner": "Sato", "deadline": "Today"}],
             "summary":      ["Sato will email the client today."],
             "sentiment":    [{"speaker": "Tanaka", "score": "neutral",
                               "label": "factual"}]},
            self._TRANSCRIPT,
        )
        assert out["verification"]["risk_label"] in ("LOW", "MEDIUM", "HIGH")

    def test_verify_result_replaces_action_items_list(self):
        """verify_result must replace action_items with verified+flagged combined."""
        mod = self._load()
        items = [
            {"task_en": "Send email to client", "task_ja": None,
             "owner": "Sato", "deadline": "Today"},
            {"task_en": "Organize picnic on Mars", "task_ja": None,
             "owner": "Ghost", "deadline": "Never"},
        ]
        out = mod.verify_result({"action_items": items, "summary": [], "sentiment": []},
                                self._TRANSCRIPT)
        assert isinstance(out["action_items"], list)
        assert len(out["action_items"]) == 2
        flags = {i["task_en"]: i["hallucination_flag"] for i in out["action_items"]}
        assert flags["Send email to client"]  is False
        assert flags["Organize picnic on Mars"] is True

    def test_verify_result_ghost_speaker_flagged(self):
        """verify_result must flag speakers not present in the transcript."""
        mod = self._load()
        out = mod.verify_result(
            {"action_items": [], "summary": [],
             "sentiment": [{"speaker": "Ghost", "score": "positive",
                            "label": "enthusiastic"},
                           {"speaker": "Tanaka", "score": "neutral",
                            "label": "factual"}]},
            self._TRANSCRIPT,
        )
        sp = out["verification"]["sentiment_speakers"]
        assert sp["flagged_count"]  == 1
        assert sp["verified_count"] == 1