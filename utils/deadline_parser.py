# utils/deadline_parser.py
# Converts any time-bound phrase to a structured deadline output.
#
# Called by  : analysis/action_item_extractor.py
# Replaces   : local _extract_deadline() in action_item_extractor.py
#
# ── Data structure ───────────────────────────────────────────────────────────
# DeadlineResult (dict):
#   raw_text      : str          — original utterance passed in
#   display       : str          — human-readable deadline string, or "N/A"
#   deadline_type : str          — "relative" | "absolute" | "unknown"
#   urgency_tier  : str          — "immediate" | "next_day" | "this_week"
#                                   | "standard" | "unknown"
#   iso_datetime  : str | None   — future use: ISO-8601 when meeting_ts given
#
# ── Algorithm ────────────────────────────────────────────────────────────────
# Two-pass:
#   Pass 1 — Normalize written numbers → digits ("two hours" → "2 hours")
#             so patterns like \d+ match regardless of how the speaker said it.
#   Pass 2 — Try patterns in priority order (specific duration first,
#             broad day-names last); return the first match.
#
# Time  : O(P × L)   P = pattern count (~30), L = utterance length
# Space : O(L)       one normalized copy of the utterance
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import re
from typing import Optional

# ── Type alias (plain dict — matches existing codebase style) ─────────────────
DeadlineResult = dict  # keys: raw_text, display, deadline_type, urgency_tier, iso_datetime


# ── Word-number normalization ─────────────────────────────────────────────────
_WORD_NUMBERS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "sixty": 60, "ninety": 90,
}

_WORD_NUM_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _WORD_NUMBERS) + r")\b",
    re.IGNORECASE,
)


def _normalize(text: str) -> str:
    """Replace written-out numbers with digits. O(W × L) where W = word count."""
    return _WORD_NUM_RE.sub(lambda m: str(_WORD_NUMBERS[m.group(1).lower()]), text)


# ── Urgency helpers ───────────────────────────────────────────────────────────
def _hours_urgency(hours: int) -> str:
    if hours <= 4:
        return "immediate"
    if hours <= 24:
        return "next_day"
    return "this_week"


def _days_urgency(days: int) -> str:
    if days <= 0:
        return "immediate"
    if days == 1:
        return "next_day"
    if days <= 4:
        return "this_week"
    return "standard"


# ── Pattern table ─────────────────────────────────────────────────────────────
# Each entry: (compiled_re, deadline_type, display_fn, urgency_fn)
# Order matters — most specific first, broad day-names last.
# All patterns run on the *normalized* text (digits, not words).
_PATTERNS: list[tuple] = [

    # ── Japanese: specific durations ──────────────────────────────────────────

    (re.compile(r"(\d+)時間以内"),
     "relative",
     lambda m: f"{m.group(1)}時間以内 (within {m.group(1)} hour{'s' if int(m.group(1)) > 1 else ''})",
     lambda m: _hours_urgency(int(m.group(1)))),

    (re.compile(r"(\d+)分以内"),
     "relative",
     lambda m: f"{m.group(1)}分以内 (within {m.group(1)} minute{'s' if int(m.group(1)) > 1 else ''})",
     lambda m: "immediate"),

    (re.compile(r"(\d+)日以内"),
     "relative",
     lambda m: f"{m.group(1)}日以内 (within {m.group(1)} day{'s' if int(m.group(1)) > 1 else ''})",
     lambda m: _days_urgency(int(m.group(1)))),

    (re.compile(r"(\d+)週間以内"),
     "relative",
     lambda m: f"{m.group(1)}週間以内 (within {m.group(1)} week{'s' if int(m.group(1)) > 1 else ''})",
     lambda m: "standard"),

    # ── Japanese: named day →まで ─────────────────────────────────────────────

    (re.compile(r"(月|火|水|木|金|土|日)曜日まで"),
     "absolute",
     lambda m: ({"月": "Monday", "火": "Tuesday", "水": "Wednesday",
                 "木": "Thursday", "金": "Friday", "土": "Saturday",
                 "日": "Sunday"}[m.group(1)]) + "まで",
     lambda m: "this_week"),

    (re.compile(r"金曜日"), "absolute",
     lambda m: "金曜日 (Friday)", lambda m: "this_week"),

    (re.compile(r"月曜日"), "absolute",
     lambda m: "月曜日 (Monday)", lambda m: "this_week"),

    # ── Japanese: relative today / tomorrow / this week / month ──────────────

    (re.compile(r"今日中|本日中"), "relative",
     lambda m: "今日中 (by end of today)", lambda m: "immediate"),

    (re.compile(r"明日まで|明日中"), "relative",
     lambda m: "明日 (by tomorrow)", lambda m: "next_day"),

    (re.compile(r"今週中"), "relative",
     lambda m: "今週中 (this week)", lambda m: "this_week"),

    (re.compile(r"今月中"), "relative",
     lambda m: "今月中 (this month)", lambda m: "standard"),

    # ── Japanese: specific date ───────────────────────────────────────────────

    (re.compile(r"(\d{1,2})月(\d{1,2})日まで"),
     "absolute",
     lambda m: f"{m.group(1)}/{m.group(2)} (by {m.group(1)}/{m.group(2)})",
     lambda m: "standard"),

    # ── English: ASAP / urgency keywords ─────────────────────────────────────

    (re.compile(r"\b(?:asap|immediately|right\s+away|urgently)\b", re.IGNORECASE),
     "relative",
     lambda m: "ASAP (immediate)",
     lambda m: "immediate"),

    # ── English: specific durations (normalized digits) ───────────────────────

    (re.compile(r"within\s+(\d+)\s*hours?", re.IGNORECASE),
     "relative",
     lambda m: f"within {m.group(1)} hour{'s' if int(m.group(1)) > 1 else ''}",
     lambda m: _hours_urgency(int(m.group(1)))),

    (re.compile(r"within\s+(\d+)\s*minutes?", re.IGNORECASE),
     "relative",
     lambda m: f"within {m.group(1)} minute{'s' if int(m.group(1)) > 1 else ''}",
     lambda m: "immediate"),

    (re.compile(r"within\s+(\d+)\s*days?", re.IGNORECASE),
     "relative",
     lambda m: f"within {m.group(1)} day{'s' if int(m.group(1)) > 1 else ''}",
     lambda m: _days_urgency(int(m.group(1)))),

    (re.compile(r"within\s+(\d+)\s*weeks?", re.IGNORECASE),
     "relative",
     lambda m: f"within {m.group(1)} week{'s' if int(m.group(1)) > 1 else ''}",
     lambda m: "standard"),

    # ── English: by end of day ────────────────────────────────────────────────

    (re.compile(r"by\s+end\s+of\s+(?:the\s+)?(?:day|today)|by\s+eod", re.IGNORECASE),
     "relative",
     lambda m: "by end of day",
     lambda m: "immediate"),

    # ── English: by specific time (e.g. "by 3pm") ────────────────────────────

    (re.compile(r"by\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))", re.IGNORECASE),
     "absolute",
     lambda m: f"by {m.group(1)}",
     lambda m: "immediate"),

    # ── English: by named day ─────────────────────────────────────────────────

    (re.compile(
        r"by\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        re.IGNORECASE),
     "absolute",
     lambda m: f"by {m.group(1).capitalize()}",
     lambda m: "this_week"),

    # ── English: relative day words ───────────────────────────────────────────
    # Order: tonight > today > tomorrow > this week > next week > loose day names

    (re.compile(r"\btonight\b", re.IGNORECASE),
     "relative",
     lambda m: "by tonight",
     lambda m: "immediate"),

    (re.compile(r"\btoday\b", re.IGNORECASE),
     "relative",
     lambda m: "by end of today",
     lambda m: "immediate"),

    (re.compile(r"\btomorrow\b", re.IGNORECASE),
     "relative",
     lambda m: "by tomorrow",
     lambda m: "next_day"),

    (re.compile(r"\bthis\s+week\b", re.IGNORECASE),
     "relative",
     lambda m: "by end of this week",
     lambda m: "this_week"),

    (re.compile(r"\bnext\s+week\b", re.IGNORECASE),
     "relative",
     lambda m: "by next week",
     lambda m: "standard"),

    # ── English: loose day names (lowest priority — broad match) ─────────────

    (re.compile(r"\bfriday\b", re.IGNORECASE),
     "absolute", lambda m: "by Friday", lambda m: "this_week"),

    (re.compile(r"\bmonday\b", re.IGNORECASE),
     "absolute", lambda m: "by Monday", lambda m: "this_week"),

    (re.compile(r"\bwednesday\b", re.IGNORECASE),
     "absolute", lambda m: "by Wednesday", lambda m: "this_week"),

    (re.compile(r"\bthursday\b", re.IGNORECASE),
     "absolute", lambda m: "by Thursday", lambda m: "this_week"),
]

# ── Urgency display labels ────────────────────────────────────────────────────
URGENCY_LABELS: dict[str, str] = {
    "immediate": "🔴 Immediate",
    "next_day":  "🟠 Next Day",
    "this_week": "🟡 This Week",
    "standard":  "🟢 Standard",
    "unknown":   "⚪ Unknown",
}


def parse_deadline(utterance: str, meeting_ts: Optional[str] = None) -> DeadlineResult:
    """
    Extract the most specific deadline from an utterance.

    Args:
        utterance   : raw speaker turn text (JP / EN / mixed)
        meeting_ts  : ISO-8601 meeting start timestamp — reserved for future
                      absolute datetime computation; not yet consumed.

    Returns:
        DeadlineResult dict with keys:
            display       — human-readable string, or "N/A"
            deadline_type — "relative" | "absolute" | "unknown"
            urgency_tier  — "immediate" | "next_day" | "this_week" | "standard" | "unknown"
            iso_datetime  — None (future use)

    Algorithm: two-pass
        Pass 1: normalize word-numbers → digits
        Pass 2: try patterns in priority order, return first match

    Time : O(P × L) — ~30 patterns, L = text length
    Space: O(L)     — one normalized copy
    """
    normalized = _normalize(utterance)

    for pattern, dtype, display_fn, urgency_fn in _PATTERNS:
        m = pattern.search(normalized)
        if m:
            return {
                "display":       display_fn(m),
                "deadline_type": dtype,
                "urgency_tier":  urgency_fn(m),
                "iso_datetime":  None,
            }

    return {
        "display":       "N/A",
        "deadline_type": "unknown",
        "urgency_tier":  "unknown",
        "iso_datetime":  None,
    }


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cases = [
        # (label, utterance, expected_display_contains, expected_urgency)
        ("JP 2h",           "2時間以内に回答します",                    "2時間以内",        "immediate"),
        ("EN digit hours",  "respond within 2 hours",                  "within 2 hours",   "immediate"),
        ("EN word hours",   "respond within two hours",                 "within 2 hours",   "immediate"),  # THE BUG
        ("EN word minutes", "respond within fifteen minutes",           "within 15 minute", "immediate"),
        ("EN by friday",    "I will send by Friday",                    "by Friday",        "this_week"),
        ("EN by eod",       "I'll confirm by end of day",               "by end of day",    "immediate"),
        ("EN tomorrow",     "will follow up tomorrow",                  "by tomorrow",      "next_day"),
        ("EN ASAP",         "we need this resolved ASAP",               "ASAP",             "immediate"),
        ("EN tonight",      "I'll send it tonight",                     "by tonight",       "immediate"),
        ("EN this week",    "I'll send it this week",                   "by end of this week","this_week"),
        ("EN within 3 days","I'll respond within 3 days",               "within 3 day",     "this_week"),
        ("JP today",        "今日中にご連絡します",                      "今日中",           "immediate"),
        ("JP tomorrow",     "明日中に送ります",                          "明日",             "next_day"),
        ("No deadline",     "Let me look into this",                    "N/A",              "unknown"),
    ]

    print("=== deadline_parser — Self-Tests ===\n")
    all_pass = True
    for label, utt, expected_display, expected_urgency in cases:
        result = parse_deadline(utt)
        disp_ok    = expected_display in result["display"]
        urgency_ok = result["urgency_tier"] == expected_urgency
        ok  = disp_ok and urgency_ok
        sym = "✓" if ok else "✗"
        print(f"  {sym}  {label}")
        if not ok:
            all_pass = False
            if not disp_ok:
                print(f"       display  FAIL: expected '{expected_display}' in '{result['display']}'")
            if not urgency_ok:
                print(f"       urgency  FAIL: expected '{expected_urgency}', got '{result['urgency_tier']}'")
    print(f"\nResult: {'ALL PASS ✓' if all_pass else 'FAILURES ✗'}")