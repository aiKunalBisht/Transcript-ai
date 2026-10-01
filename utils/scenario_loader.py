# utils/scenario_loader.py
# Loads the 20 scenario JSON files from tests/scenarios/
# and deduplicates by scenario id.
#
# Used by /evaluate/run to replace the old 3-case TEST_CASES.

import json
from pathlib import Path

SCENARIO_DIR = Path("tests/scenarios")


def load_scenarios() -> list[dict]:
    """
    Returns deduplicated scenario list sorted by filename.
    Skips files whose 'id' has already been seen (catches SC011 == SC010 bug).
    """
    seen_ids  : set[str] = set()
    scenarios : list[dict] = []

    for path in sorted(SCENARIO_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[ScenarioLoader] Skipping {path.name}: {exc}", flush=True)
            continue

        sc_id = data.get("id", path.stem)
        if sc_id in seen_ids:
            print(f"[ScenarioLoader] Duplicate id '{sc_id}' in {path.name} — skipped",
                  flush=True)
            continue

        seen_ids.add(sc_id)
        scenarios.append(data)

    return scenarios


def get_scenario_summary() -> dict:
    """Quick stats used by the evaluate page template."""
    scenarios = load_scenarios()
    langs     = {}
    outcomes  = {}
    for sc in scenarios:
        lang    = sc.get("language", "unknown")
        outcome = sc.get("expected", {}).get("deal_outcome", "unknown")
        langs[lang]       = langs.get(lang, 0) + 1
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    return {
        "total":    len(scenarios),
        "by_lang":  langs,
        "by_outcome": outcomes,
    }