# tests/test_security.py
# Security boundary tests — these must ALL pass before any PR merge.

import pytest
from utils.cache import get_cached, set_cache, clear_user_cache

SAMPLE_TRANSCRIPT = "Tanaka: Good morning. Let's review Q3 numbers."
SAMPLE_RESULT     = {"summary": ["Q3 reviewed"], "_provider": "test"}


class TestAnonymousMD5Cache:
    def test_anonymous_get_returns_none(self):
        """Anonymous users must never get a cache hit."""
        result = get_cached(SAMPLE_TRANSCRIPT, "en", user_id=None)
        assert result is None

    def test_anonymous_set_is_noop(self, tmp_path, monkeypatch):
        """set_cache with user_id=None must not write any file."""
        monkeypatch.chdir(tmp_path)
        set_cache(SAMPLE_TRANSCRIPT, "en", SAMPLE_RESULT, user_id=None)
        cache_files = list(tmp_path.rglob("*.json"))
        assert cache_files == [], f"Anonymous cache wrote files: {cache_files}"


class TestCrossUserIsolation:
    def setup_method(self):
        set_cache(SAMPLE_TRANSCRIPT, "en", SAMPLE_RESULT, user_id="user_a")

    def teardown_method(self):
        clear_user_cache("user_a")
        clear_user_cache("user_b")

    def test_user_a_hits_own_cache(self):
        hit = get_cached(SAMPLE_TRANSCRIPT, "en", user_id="user_a")
        assert hit is not None
        assert hit.get("_from_cache") is True

    def test_user_b_cannot_read_user_a_cache(self):
        miss = get_cached(SAMPLE_TRANSCRIPT, "en", user_id="user_b")
        assert miss is None, "Cross-user cache leak detected!"


class TestLanguageCacheCollision:
    def setup_method(self):
        set_cache(SAMPLE_TRANSCRIPT, "en", {**SAMPLE_RESULT, "lang": "en"}, user_id="user_a")
        set_cache(SAMPLE_TRANSCRIPT, "ja", {**SAMPLE_RESULT, "lang": "ja"}, user_id="user_a")

    def teardown_method(self):
        clear_user_cache("user_a")

    def test_en_and_ja_are_separate_entries(self):
        en = get_cached(SAMPLE_TRANSCRIPT, "en", user_id="user_a")
        ja = get_cached(SAMPLE_TRANSCRIPT, "ja", user_id="user_a")
        assert en is not None
        assert ja is not None
        assert en.get("lang") == "en"
        assert ja.get("lang") == "ja"


class TestScenarioLoader:
    def test_loads_without_crash(self):
        from utils.scenario_loader import load_scenarios
        scenarios = load_scenarios()
        assert isinstance(scenarios, list)

    def test_no_duplicate_ids(self):
        from utils.scenario_loader import load_scenarios
        scenarios = load_scenarios()
        ids = [sc["id"] for sc in scenarios]
        assert len(ids) == len(set(ids)), f"Duplicate scenario ids: {ids}"

    def test_all_have_required_fields(self):
        from utils.scenario_loader import load_scenarios
        for sc in load_scenarios():
            assert "id"          in sc, f"{sc} missing 'id'"
            assert "transcript"  in sc, f"{sc['id']} missing 'transcript'"
            assert "expected"    in sc, f"{sc['id']} missing 'expected'"
            assert "language"    in sc, f"{sc['id']} missing 'language'"