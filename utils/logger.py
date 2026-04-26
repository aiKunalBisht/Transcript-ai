# logger.py
# Observability + Trend Analysis Engine for TranscriptAI
#
# Every analysis run is logged to JSONL.
# get_trends() reads those logs and returns structured intelligence:
#   - Soft rejection trend → are your deals at risk?
#   - Hallucination rate trend → is the model drifting?
#   - Performance trend → is analysis getting slower?
#   - Language / provider / keigo distribution
#
# Zero external dependencies — pure Python.
# The Trends tab in app.py reads from this module.

import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

LOG_FILE = Path("logs/transcript_ai.jsonl")


# ── Internal helpers ─────────────────────────────────────────────────────────

def _ensure_log_dir():
    LOG_FILE.parent.mkdir(exist_ok=True)


def _load_entries(last_n: int = 500) -> list:
    """Load last N successful log entries."""
    _ensure_log_dir()
    if not LOG_FILE.exists():
        return []
    entries = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    return [e for e in entries if e.get("status") == "success"][-last_n:]


def _parse_ts(ts: str) -> datetime:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return datetime.min


# ── Write ────────────────────────────────────────────────────────────────────

def log_analysis(
    transcript_length: int,
    language: str,
    provider: str,
    duration_ms: float,
    result: dict,
    error: str = None,
    session_id: str = None,
):
    """
    Log one analysis run with all key metrics.
    Called automatically by analyzer.py after every analysis.
    """
    _ensure_log_dir()

    verification        = (result or {}).get("verification", {})
    ai_check            = verification.get("action_items", {})
    soft                = (result or {}).get("soft_rejections", {})

    entry = {
        "timestamp":             datetime.now().isoformat(),
        "session_id":            session_id or "",
        "transcript_chars":      transcript_length,
        "language":              language,
        "provider":              provider,
        "duration_ms":           round(duration_ms, 1),
        "summary_bullets":       len((result or {}).get("summary", [])),
        "action_items_total":    len((result or {}).get("action_items", [])),
        "action_items_flagged":  ai_check.get("flagged_count", 0),
        "hallucination_rate":    ai_check.get("hallucination_rate", 0.0),
        "hallucination_risk":    verification.get("risk_label", "UNKNOWN"),
        "soft_rejection_risk":   soft.get("risk_level", "NONE"),
        "soft_rejection_count":  soft.get("total_signals", 0),
        "speakers_detected":     len((result or {}).get("speakers", [])),
        "keigo_level":           (result or {}).get("japan_insights", {}).get("keigo_level", "unknown"),
        "code_switches":         (result or {}).get("japan_insights", {}).get("code_switch_count", 0),
        "error":                 error,
        "status":                "error" if error else "success",
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def log_error(error_type: str, message: str, context: dict = None):
    """Log a system error with context."""
    _ensure_log_dir()
    entry = {
        "timestamp":  datetime.now().isoformat(),
        "status":     "error",
        "type":       "error",
        "error_type": error_type,
        "message":    message,
        "context":    context or {},
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Read: basic stats ────────────────────────────────────────────────────────

def get_stats(last_n: int = 100) -> dict:
    """Summary statistics across last N analyses."""
    entries = _load_entries(last_n)
    if not entries:
        return {"message": "No analyses logged yet.", "total": 0}

    total        = len(entries)
    durations    = [e.get("duration_ms", 0) for e in entries]
    halluc_rates = [e.get("hallucination_rate", 0.0) for e in entries]
    errors       = sum(1 for e in entries if e.get("error"))

    providers = defaultdict(int)
    languages = defaultdict(int)
    for e in entries:
        providers[e.get("provider", "unknown")] += 1
        languages[e.get("language", "unknown")] += 1

    high_risk = sum(1 for e in entries
                    if e.get("hallucination_risk") in ("HIGH", "MEDIUM"))

    return {
        "total_analyses":         total,
        "avg_duration_ms":        round(sum(durations) / total, 1),
        "min_duration_ms":        round(min(durations), 1),
        "max_duration_ms":        round(max(durations), 1),
        "error_rate":             round(errors / total, 3),
        "avg_hallucination_rate": round(sum(halluc_rates) / total, 3),
        "high_risk_analyses":     high_risk,
        "provider_breakdown":     dict(providers),
        "language_breakdown":     dict(languages),
    }


# ── Read: trend analysis ─────────────────────────────────────────────────────

def get_trends(last_n: int = 50) -> dict:
    """
    Trend analysis across last N analyses.

    Returns structured data for charts:
      - timestamps          → x-axis for all time-series charts
      - soft_rejection_risk → trend of rejection risk per meeting
      - hallucination_rate  → model accuracy drift over time
      - duration_ms         → performance trend
      - keigo_levels        → register distribution
      - action_items        → workload per meeting
      - code_switches       → code-switch frequency trend
      - provider_counts     → Groq vs Ollama vs Mock over time
      - language_counts     → language distribution over time
      - summary stats       → totals for metric cards
    """
    entries = _load_entries(last_n)
    if not entries:
        return {"empty": True, "message": "No analyses logged yet. Run your first analysis to see trends."}

    # Sort by timestamp
    entries.sort(key=lambda e: _parse_ts(e.get("timestamp", "")))

    # Time series lists (parallel arrays for charts)
    timestamps          = []
    soft_rejection_risk = []
    hallucination_rates = []
    durations_sec       = []
    action_item_counts  = []
    code_switch_counts  = []
    speaker_counts      = []

    # Distribution counters
    keigo_dist    = defaultdict(int)
    provider_dist = defaultdict(int)
    language_dist = defaultdict(int)
    risk_dist     = defaultdict(int)
    soft_risk_dist = defaultdict(int)

    # Risk score mapping for trend line
    risk_score_map = {"NONE": 0, "MINIMAL": 1, "LOW": 2, "MEDIUM": 3, "HIGH": 4}

    for e in entries:
        ts = _parse_ts(e.get("timestamp", ""))
        timestamps.append(ts.strftime("%m/%d %H:%M"))

        soft_risk = e.get("soft_rejection_risk", "NONE")
        soft_rejection_risk.append(risk_score_map.get(soft_risk, 0))

        hallucination_rates.append(round(e.get("hallucination_rate", 0.0) * 100, 1))
        durations_sec.append(round(e.get("duration_ms", 0) / 1000, 1))
        action_item_counts.append(e.get("action_items_total", 0))
        code_switch_counts.append(e.get("code_switches", 0))
        speaker_counts.append(e.get("speakers_detected", 0))

        keigo_dist[e.get("keigo_level", "unknown")]        += 1
        provider_dist[e.get("provider", "unknown")]        += 1
        language_dist[e.get("language", "unknown")]        += 1
        risk_dist[e.get("hallucination_risk", "UNKNOWN")]  += 1
        soft_risk_dist[soft_risk]                          += 1

    total = len(entries)

    # Trend direction (last 5 vs previous 5)
    def _trend_direction(values: list) -> str:
        if len(values) < 4:
            return "stable"
        recent = sum(values[-3:]) / 3
        older  = sum(values[-6:-3]) / 3 if len(values) >= 6 else sum(values[:3]) / 3
        if recent > older * 1.15:
            return "up"
        if recent < older * 0.85:
            return "down"
        return "stable"

    # Soft rejection trend — most important business signal
    sr_trend = _trend_direction(soft_rejection_risk)
    sr_alert = None
    if sr_trend == "up":
        sr_alert = "⚠ Soft rejection risk is increasing across recent meetings. Your deal pipeline may be at risk."
    elif soft_risk_dist.get("HIGH", 0) >= 2:
        sr_alert = "⚠ Multiple HIGH-risk soft rejection sessions detected. Follow up explicitly with these clients."

    # Hallucination rate trend — system health
    hr_trend = _trend_direction(hallucination_rates)
    hr_alert = None
    avg_hr   = sum(hallucination_rates) / total if total else 0
    if avg_hr > 30:
        hr_alert = "⚠ Hallucination rate above 30%. Consider reviewing prompt or switching providers."
    elif hr_trend == "up":
        hr_alert = "↑ Hallucination rate trending upward. Model may be drifting on your transcript style."

    # Performance trend — latency health
    avg_dur    = round(sum(durations_sec) / total, 1) if total else 0
    dur_trend  = _trend_direction(durations_sec)
    dur_alert  = None
    mock_count = provider_dist.get("mock", 0) + sum(
        v for k, v in provider_dist.items() if "mock" in k
    )
    if mock_count / total > 0.5:
        dur_alert = "⚠ More than 50% of analyses used mock data. Set GROQ_API_KEY for real analysis."

    return {
        "empty":       False,
        "total":       total,
        "first_date":  _parse_ts(entries[0].get("timestamp", "")).strftime("%b %d, %Y"),
        "last_date":   _parse_ts(entries[-1].get("timestamp", "")).strftime("%b %d, %Y"),

        # Time series for charts
        "timestamps":           timestamps,
        "soft_rejection_scores": soft_rejection_risk,
        "hallucination_rates":  hallucination_rates,
        "durations_sec":        durations_sec,
        "action_item_counts":   action_item_counts,
        "code_switch_counts":   code_switch_counts,
        "speaker_counts":       speaker_counts,

        # Distributions for pie/bar charts
        "keigo_dist":    dict(keigo_dist),
        "provider_dist": dict(provider_dist),
        "language_dist": dict(language_dist),
        "risk_dist":     dict(risk_dist),
        "soft_risk_dist":dict(soft_risk_dist),

        # Trend directions
        "soft_rejection_trend": sr_trend,
        "hallucination_trend":  hr_trend,
        "duration_trend":       dur_trend,

        # Alerts
        "soft_rejection_alert": sr_alert,
        "hallucination_alert":  hr_alert,
        "duration_alert":       dur_alert,

        # Summary stats for metric cards
        "avg_duration_sec":        avg_dur,
        "avg_hallucination_pct":   round(avg_hr, 1),
        "avg_action_items":        round(sum(action_item_counts) / total, 1) if total else 0,
        "avg_code_switches":       round(sum(code_switch_counts) / total, 1) if total else 0,
        "high_soft_rejection_pct": round(soft_risk_dist.get("HIGH", 0) / total * 100, 1) if total else 0,
        "most_used_provider":      max(provider_dist, key=provider_dist.get) if provider_dist else "none",
        "most_common_language":    max(language_dist, key=language_dist.get) if language_dist else "none",
    }


def get_recent_entries(last_n: int = 10) -> list:
    """Return last N entries for display in history sidebar."""
    return _load_entries(last_n)


def clear_logs():
    """Delete all log entries. Use with caution."""
    if LOG_FILE.exists():
        LOG_FILE.unlink()


# ── CLI test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import random

    print("Seeding test data...")
    risks = ["NONE","NONE","LOW","MEDIUM","HIGH","MEDIUM","HIGH","HIGH"]
    langs = ["ja","mixed","ja","en","mixed","ja"]
    providers = ["groq","groq","ollama","groq"]
    keigos = ["high","high","medium","low","high"]

    for i in range(12):
        log_analysis(
            transcript_length=random.randint(800, 3000),
            language=random.choice(langs),
            provider=random.choice(providers),
            duration_ms=random.uniform(2500, 90000),
            result={
                "summary": ["a","b","c"],
                "action_items": [{"task":"t","hallucination_flag": random.random() > 0.7}
                                  for _ in range(random.randint(1,5))],
                "speakers": [{"name":"A"},{"name":"B"}],
                "japan_insights": {
                    "keigo_level": random.choice(keigos),
                    "code_switch_count": random.randint(0, 20),
                },
                "verification": {
                    "action_items": {
                        "flagged_count": random.randint(0, 2),
                        "hallucination_rate": random.uniform(0, 0.4),
                    },
                    "risk_label": random.choice(["LOW","MEDIUM","LOW","NONE"]),
                },
                "soft_rejections": {
                    "risk_level": random.choice(risks),
                    "total_signals": random.randint(0, 4),
                },
            }
        )

    print("\n=== STATS ===")
    print(json.dumps(get_stats(), indent=2))

    print("\n=== TRENDS ===")
    trends = get_trends()
    # Print without the full arrays to keep it readable
    summary = {k: v for k, v in trends.items()
               if not isinstance(v, list) or len(v) <= 3}
    print(json.dumps(summary, indent=2, ensure_ascii=False))