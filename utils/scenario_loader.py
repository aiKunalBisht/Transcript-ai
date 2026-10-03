# utils/scenario_loader.py — v1.1
# Loads the 20 scenario JSON files from tests/scenarios/ and exposes them
# to /evaluate/run, replacing the old 3-case TEST_CASES from tests/test_data.py.
#
# v1.1 change:
#   Schema normalization added. Scenario files were written with a flat schema:
#     { "expected_outcome": "DEFERRED", "expected_risk": "HIGH", ... }
#   The loader now promotes these into the nested dict the evaluator expects:
#     { "expected": { "deal_outcome": "DEFERRED", "risk_level": "HIGH" } }
#   Both schemas are accepted — existing files need no edits.
#   SC011 duplicate-id skip confirmed working in CI.

import json
import sys
from pathlib import Path

SCENARIO_DIR  = Path("tests/scenarios")
_REQUIRED_KEYS = {"id", "language", "transcript"}  # "expected" added after normalization


def _normalize(data: dict) -> dict:
    """
    Promote flat expected_outcome / expected_risk fields into a nested
    `expected` dict so the evaluator and test suite see a consistent schema.

    Accepts both formats:
      FLAT (current files):
        { "expected_outcome": "DEFERRED", "expected_risk": "MEDIUM" }
      NESTED (future files / evaluator format):
        { "expected": { "deal_outcome": "DEFERRED", "risk_level": "MEDIUM" } }

    If neither is present, inserts an empty expected dict so downstream
    code doesn't have to guard every access.
    """
    if "expected" in data:
        return data  # already nested — nothing to do

    data["expected"] = {
        "deal_outcome": data.get("expected_outcome") or data.get("deal_outcome", "UNKNOWN"),
        "risk_level":   data.get("expected_risk")    or data.get("risk_level",   "UNKNOWN"),
    }
    return data


def load_scenarios() -> list[dict]:
    """
    Load, normalize, validate, and deduplicate scenario JSON files.

    Processing order:
      1. Glob *.json from SCENARIO_DIR, sorted for deterministic ordering.
      2. Skip files that fail JSON parsing (logged to stderr).
      3. Normalize schema: flat expected_outcome → nested expected dict.
      4. Skip files missing any required key (logged to stderr).
      5. Skip files whose "id" was already seen (logged to stderr).
      6. Return clean deduplicated list.
    """
    if not SCENARIO_DIR.exists():
        print(
            f"[ScenarioLoader] WARNING: {SCENARIO_DIR} does not exist.",
            file=sys.stderr, flush=True,
        )
        return []

    seen_ids:  set[str]   = set()
    scenarios: list[dict] = []

    for path in sorted(SCENARIO_DIR.glob("*.json")):
        # ── Parse ──────────────────────────────────────────────────────────────
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(
                f"[ScenarioLoader] SKIP {path.name}: JSON parse error — {exc}",
                file=sys.stderr, flush=True,
            )
            continue

        # ── Normalize schema ───────────────────────────────────────────────────
        data = _normalize(data)

        # ── Validate required keys (after normalization) ───────────────────────
        missing = (_REQUIRED_KEYS | {"expected"}) - set(data.keys())
        if missing:
            print(
                f"[ScenarioLoader] SKIP {path.name}: missing keys {sorted(missing)}",
                file=sys.stderr, flush=True,
            )
            continue

        # ── Deduplicate by id ──────────────────────────────────────────────────
        sc_id = data["id"]
        if sc_id in seen_ids:
            print(
                f"[ScenarioLoader] SKIP {path.name}: duplicate id '{sc_id}'",
                file=sys.stderr, flush=True,
            )
            continue

        seen_ids.add(sc_id)
        scenarios.append(data)

    return scenarios


def get_scenario_summary() -> dict:
    """
    Lightweight stats for the /evaluate page template.
    """
    scenarios:  list[dict] = load_scenarios()
    by_lang:    dict[str, int] = {}
    by_outcome: dict[str, int] = {}
    by_risk:    dict[str, int] = {}

    for sc in scenarios:
        lang    = sc.get("language", "unknown")
        outcome = sc["expected"].get("deal_outcome", "unknown")
        risk    = sc["expected"].get("risk_level",   "unknown")

        by_lang[lang]       = by_lang.get(lang, 0)       + 1
        by_outcome[outcome] = by_outcome.get(outcome, 0) + 1
        by_risk[risk]       = by_risk.get(risk, 0)       + 1

    return {
        "total":      len(scenarios),
        "by_lang":    by_lang,
        "by_outcome": by_outcome,
        "by_risk":    by_risk,
    }


# ── Self-test ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pprint
    scenarios = load_scenarios()
    print(f"Loaded {len(scenarios)} scenarios:")
    for sc in scenarios:
        print(
            f"  {sc['id']:12s}  lang={sc['language']:6s}  "
            f"outcome={sc['expected'].get('deal_outcome','?'):15s}  "
            f"risk={sc['expected'].get('risk_level','?')}"
        )
    print()
    pprint.pprint(get_scenario_summary())