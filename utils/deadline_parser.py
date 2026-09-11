# utils/deadline_parser.py
# Converts any time-bound phrase to a structured deadline output.
#
# Called by  : analysis/action_item_extractor.py
# Replaces   : local _extract_deadline() in action_item_extractor.py
#
# ── Data structure (v2) ───────────────────────────────────────────────────────
# DeadlineResult (dict):
#   display           : str       — human-readable deadline string + date hint
#   deadline_type     : str       — "relative" | "absolute" | "unknown"
#   urgency_tier      : str       — "immediate"|"next_day"|"this_week"|"standard"|"unknown"
#   days_to_deadline  : int|None  — actual calendar days from meeting_date
#   iso_datetime      : str|None  — ISO-8601 target date (YYYY-MM-DD)
#   context_note      : str|None  — e.g. "Based on Thursday, Sep 10"
#
# ── What changed in v2 ────────────────────────────────────────────────────────
# v1: "by Friday" → urgency "this_week" (heuristic, always the same)
# v2: "by Friday" on Thursday  → urgency "next_day"   (1 day),  iso 2026-09-11
#     "by Friday" on Monday    → urgency "this_week"  (4 days), iso 2026-09-13
#     "next Friday" on Thursday → urgency "standard"  (8 days), iso 2026-09-18
#
# No external libraries needed — Python's datetime handles all date math.
# When no meeting_date supplied, defaults to datetime.now() so urgency is
# always computed from the actual current date, never from a static heuristic.
#
# ── Algorithm ─────────────────────────────────────────────────────────────────
# Three-pass:
#   Pass 1 — Normalize written numbers → digits ("two hours" → "2 hours")
#   Pass 2 — Try patterns in priority order; capture match + day_fn
#   Pass 3 — If day_fn present, compute actual days from meeting_date;
#             enrich display with "(tomorrow, Sep 11)" context hint
#
# Time  : O(P × L)   P = pattern count (~36), L = utterance length
# Space : O(L)       one normalized copy of the utterance
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

# ── Type alias ────────────────────────────────────────────────────────────────
DeadlineResult = dict   # keys defined in header above

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
    """Replace written-out numbers with digits. O(W × L)."""
    return _WORD_NUM_RE.sub(lambda m: str(_WORD_NUMBERS[m.group(1).lower()]), text)


# ── Day-of-week lookup tables ─────────────────────────────────────────────────
_EN_DAYS: dict[str, int] = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

_JP_DAY_WD: dict[str, int] = {
    "月": 0, "火": 1, "水": 2, "木": 3, "金": 4, "土": 5, "日": 6,
}

_JP_DAY_EN: dict[str, str] = {
    "月": "Monday", "火": "Tuesday", "水": "Wednesday", "木": "Thursday",
    "金": "Friday",  "土": "Saturday", "日": "Sunday",
}


# ── Date math ─────────────────────────────────────────────────────────────────
def _days_until_weekday(from_date: datetime, target_weekday: int,
                        next_week: bool = False) -> int:
    """
    Calendar days from from_date to the nearest occurrence of target_weekday.

    Args:
        from_date      : reference date/time
        target_weekday : 0=Mon … 4=Fri … 6=Sun
        next_week      : True → skip to NEXT week's occurrence
                         ("next Friday" vs "Friday")

    Returns:
        int — 0 means today, 1 means tomorrow, etc.

    Examples (from_date = Thursday, weekday=3):
        Friday  (4), next_week=False → (4-3)%7 = 1   (tomorrow)
        Friday  (4), next_week=True  → 1 + 7  = 8   (next week's Friday)
        Monday  (0), next_week=False → (0-3+7)%7 = 4  (this coming Monday)
        Monday  (0), next_week=True  → 4 + 7  = 11  (next week's Monday)
        Thursday(3), next_week=False → (3-3)%7 = 0   (today)
        Thursday(3), next_week=True  → 0 → 7         (next Thursday)

    DSA: O(1) — pure integer arithmetic.
    """
    current = from_date.weekday()
    days    = (target_weekday - current) % 7
    if next_week:
        days = days + 7 if days > 0 else 7
    return days


# ── Urgency tier helpers ──────────────────────────────────────────────────────
def _hours_urgency(hours: int) -> str:
    if hours <= 4:  return "immediate"
    if hours <= 24: return "next_day"
    return "this_week"


def _days_urgency(days: int) -> str:
    if days <= 0: return "immediate"
    if days == 1: return "next_day"
    if days <= 4: return "this_week"
    return "standard"


# ── Pattern table ─────────────────────────────────────────────────────────────
# Each entry: (compiled_re, deadline_type, display_fn, urgency_fn, day_fn)
#
# day_fn : None | Callable[[re.Match, datetime], int]
#   When not None → called with (match, meeting_date) → days_to_deadline (int)
#   The returned int overrides urgency_fn and drives iso_datetime computation.
#   When None → urgency_fn(match) is the final urgency; no date math.
#
# Priority order: "next [day]" > "this [day]" > "by [day]" > bare day names.
# "next X" MUST come before bare "X" patterns so it is matched first.

_DFN = Callable  # type alias for day_fn

_PATTERNS: list[tuple] = [

    # ── Japanese: specific durations ──────────────────────────────────────────

    (re.compile(r"(\d+)時間以内"),
     "relative",
     lambda m: f"{m.group(1)}時間以内 (within {m.group(1)} hour{'s' if int(m.group(1))>1 else ''})",
     lambda m: _hours_urgency(int(m.group(1))),
     None),

    (re.compile(r"(\d+)分以内"),
     "relative",
     lambda m: f"{m.group(1)}分以内 (within {m.group(1)} minute{'s' if int(m.group(1))>1 else ''})",
     lambda m: "immediate",
     None),

    (re.compile(r"(\d+)日以内"),
     "relative",
     lambda m: f"{m.group(1)}日以内 (within {m.group(1)} day{'s' if int(m.group(1))>1 else ''})",
     lambda m: _days_urgency(int(m.group(1))),
     None),

    (re.compile(r"(\d+)週間以内"),
     "relative",
     lambda m: f"{m.group(1)}週間以内 (within {m.group(1)} week{'s' if int(m.group(1))>1 else ''})",
     lambda m: "standard",
     None),

    # ── Japanese: named day →まで ─────────────────────────────────────────────

    (re.compile(r"(月|火|水|木|金|土|日)曜日まで"),
     "absolute",
     lambda m: _JP_DAY_EN[m.group(1)] + "まで",
     lambda m: "this_week",
     lambda m, d: _days_until_weekday(d, _JP_DAY_WD[m.group(1)], False)),

    (re.compile(r"金曜日"),
     "absolute",
     lambda m: "金曜日 (Friday)",
     lambda m: "this_week",
     lambda m, d: _days_until_weekday(d, 4, False)),

    (re.compile(r"月曜日"),
     "absolute",
     lambda m: "月曜日 (Monday)",
     lambda m: "this_week",
     lambda m, d: _days_until_weekday(d, 0, False)),

    # ── Japanese: relative today / tomorrow / this week / month ──────────────

    (re.compile(r"今日中|本日中"),
     "relative",
     lambda m: "今日中 (by end of today)",
     lambda m: "immediate",
     lambda m, d: 0),

    (re.compile(r"明日まで|明日中"),
     "relative",
     lambda m: "明日 (by tomorrow)",
     lambda m: "next_day",
     lambda m, d: 1),

    (re.compile(r"今週中"),
     "relative",
     lambda m: "今週中 (this week)",
     lambda m: "this_week",
     lambda m, d: _days_until_weekday(d, 4, False) or 7),

    (re.compile(r"今月中"),
     "relative",
     lambda m: "今月中 (this month)",
     lambda m: "standard",
     None),

    # ── Japanese: specific date ───────────────────────────────────────────────

    (re.compile(r"(\d{1,2})月(\d{1,2})日まで"),
     "absolute",
     lambda m: f"{m.group(1)}/{m.group(2)} (by {m.group(1)}/{m.group(2)})",
     lambda m: "standard",
     None),

    # ── English: ASAP / urgency keywords ─────────────────────────────────────

    (re.compile(r"\b(?:asap|immediately|right\s+away|urgently)\b", re.IGNORECASE),
     "relative",
     lambda m: "ASAP (immediate)",
     lambda m: "immediate",
     lambda m, d: 0),

    # ── English: specific durations (normalized digits) ───────────────────────

    (re.compile(r"within\s+(\d+)\s*hours?", re.IGNORECASE),
     "relative",
     lambda m: f"within {m.group(1)} hour{'s' if int(m.group(1))>1 else ''}",
     lambda m: _hours_urgency(int(m.group(1))),
     None),

    (re.compile(r"within\s+(\d+)\s*minutes?", re.IGNORECASE),
     "relative",
     lambda m: f"within {m.group(1)} minute{'s' if int(m.group(1))>1 else ''}",
     lambda m: "immediate",
     None),

    (re.compile(r"within\s+(\d+)\s*days?", re.IGNORECASE),
     "relative",
     lambda m: f"within {m.group(1)} day{'s' if int(m.group(1))>1 else ''}",
     lambda m: _days_urgency(int(m.group(1))),
     None),

    (re.compile(r"within\s+(\d+)\s*weeks?", re.IGNORECASE),
     "relative",
     lambda m: f"within {m.group(1)} week{'s' if int(m.group(1))>1 else ''}",
     lambda m: "standard",
     None),

    # ── English: by end of day ────────────────────────────────────────────────

    (re.compile(r"by\s+end\s+of\s+(?:the\s+)?(?:day|today)|by\s+eod", re.IGNORECASE),
     "relative",
     lambda m: "by end of day",
     lambda m: "immediate",
     lambda m, d: 0),

    # ── English: by specific time (e.g. "by 3pm") ────────────────────────────

    (re.compile(r"by\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))", re.IGNORECASE),
     "absolute",
     lambda m: f"by {m.group(1)}",
     lambda m: "immediate",
     lambda m, d: 0),

    # ── English: "next [day]" — MUST be before bare day patterns ─────────────
    # "next Friday" = the Friday of NEXT week, not the upcoming one.

    (re.compile(
        r"next\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        re.IGNORECASE),
     "absolute",
     lambda m: f"next {m.group(1).capitalize()}",
     lambda m: "standard",
     lambda m, d: _days_until_weekday(d, _EN_DAYS[m.group(1).lower()], True)),

    # ── English: "this [day]" = the upcoming occurrence this week ─────────────

    (re.compile(
        r"this\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        re.IGNORECASE),
     "absolute",
     lambda m: f"this {m.group(1).capitalize()}",
     lambda m: "this_week",
     lambda m, d: _days_until_weekday(d, _EN_DAYS[m.group(1).lower()], False)),

    # ── English: "by [day]" — upcoming occurrence ─────────────────────────────

    (re.compile(
        r"by\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        re.IGNORECASE),
     "absolute",
     lambda m: f"by {m.group(1).capitalize()}",
     lambda m: "this_week",
     lambda m, d: _days_until_weekday(d, _EN_DAYS[m.group(1).lower()], False)),

    # ── English: relative day words ───────────────────────────────────────────
    # Order: tonight > today > tomorrow > this week > next week

    (re.compile(r"\btonight\b", re.IGNORECASE),
     "relative",
     lambda m: "by tonight",
     lambda m: "immediate",
     lambda m, d: 0),

    (re.compile(r"\btoday\b", re.IGNORECASE),
     "relative",
     lambda m: "by end of today",
     lambda m: "immediate",
     lambda m, d: 0),

    (re.compile(r"\btomorrow\b", re.IGNORECASE),
     "relative",
     lambda m: "by tomorrow",
     lambda m: "next_day",
     lambda m, d: 1),

    (re.compile(r"\bthis\s+week\b", re.IGNORECASE),
     "relative",
     lambda m: "by end of this week",
     lambda m: "this_week",
     lambda m, d: _days_until_weekday(d, 4, False) or 7),

    (re.compile(r"\bnext\s+week\b", re.IGNORECASE),
     "relative",
     lambda m: "by next week",
     lambda m: "standard",
     lambda m, d: _days_until_weekday(d, 4, True)),

    # ── English: bare day names (lowest priority — caught last) ──────────────
    # "Friday" alone = the upcoming Friday, same as "by Friday".
    # "next Friday" is already matched above.

    (re.compile(r"\bfriday\b",    re.IGNORECASE), "absolute",
     lambda m: "by Friday",    lambda m: "this_week",
     lambda m, d: _days_until_weekday(d, 4, False)),

    (re.compile(r"\bmonday\b",    re.IGNORECASE), "absolute",
     lambda m: "by Monday",    lambda m: "this_week",
     lambda m, d: _days_until_weekday(d, 0, False)),

    (re.compile(r"\bwednesday\b", re.IGNORECASE), "absolute",
     lambda m: "by Wednesday", lambda m: "this_week",
     lambda m, d: _days_until_weekday(d, 2, False)),

    (re.compile(r"\bthursday\b",  re.IGNORECASE), "absolute",
     lambda m: "by Thursday",  lambda m: "this_week",
     lambda m, d: _days_until_weekday(d, 3, False)),

    (re.compile(r"\btuesday\b",   re.IGNORECASE), "absolute",
     lambda m: "by Tuesday",   lambda m: "this_week",
     lambda m, d: _days_until_weekday(d, 1, False)),

    (re.compile(r"\bsaturday\b",  re.IGNORECASE), "absolute",
     lambda m: "by Saturday",  lambda m: "this_week",
     lambda m, d: _days_until_weekday(d, 5, False)),

    (re.compile(r"\bsunday\b",    re.IGNORECASE), "absolute",
     lambda m: "by Sunday",    lambda m: "this_week",
     lambda m, d: _days_until_weekday(d, 6, False)),
]


# ── Urgency display labels ────────────────────────────────────────────────────
URGENCY_LABELS: dict[str, str] = {
    "immediate": "🔴 Immediate",
    "next_day":  "🟠 Next Day",
    "this_week": "🟡 This Week",
    "standard":  "🟢 Standard",
    "unknown":   "⚪ Unknown",
}


# ── Public API ────────────────────────────────────────────────────────────────

def parse_deadline(
    utterance:    str,
    meeting_date: Optional[datetime] = None,
    # Legacy alias kept for backward compatibility
    meeting_ts:   Optional[str]      = None,
) -> DeadlineResult:
    """
    Extract the most specific deadline from an utterance.

    Args:
        utterance    : raw speaker turn text (JP / EN / mixed)
        meeting_date : when the meeting is happening — used to compute actual
                       days_to_deadline for day-name phrases like "by Friday".
                       Defaults to datetime.now() when not supplied, so urgency
                       is always computed from the real current date.
        meeting_ts   : legacy ISO-8601 string alias; ignored when meeting_date
                       is provided.

    Returns:
        DeadlineResult dict:
            display          — human-readable deadline, enriched with date hint
                               when meeting_date is known, e.g.
                               "by Friday  [tomorrow, Sep 11]"
            deadline_type    — "relative" | "absolute" | "unknown"
            urgency_tier     — computed from actual days_to_deadline when
                               meeting_date is known, otherwise from heuristic
            days_to_deadline — int (0=today) or None for duration phrases
            iso_datetime     — "YYYY-MM-DD" or None
            context_note     — e.g. "Based on Thursday, Sep 10"

    Algorithm: three-pass
        Pass 1: word-number normalization ("two" → "2")
        Pass 2: pattern match in priority order
        Pass 3: if day_fn present, compute actual days from meeting_date
                and enrich display with "(tomorrow, Sep 11)" hint

    Time : O(P × L) — ~36 patterns, L = utterance length
    Space: O(L)     — one normalized copy
    """
    # Default to today if no date provided
    if meeting_date is None:
        if meeting_ts:
            try:
                meeting_date = datetime.fromisoformat(meeting_ts)
            except ValueError:
                meeting_date = datetime.now()
        else:
            meeting_date = datetime.now()

    normalized = _normalize(utterance)

    for pattern, dtype, display_fn, urgency_fn, day_fn in _PATTERNS:
        m = pattern.search(normalized)
        if m:
            base_display     = display_fn(m)
            urgency          = urgency_fn(m)
            days_to_deadline = None
            iso_datetime     = None
            display          = base_display

            if day_fn is not None:
                # Date-aware computation
                days_to_deadline = day_fn(m, meeting_date)
                urgency          = _days_urgency(days_to_deadline)
                target           = (meeting_date + timedelta(days=days_to_deadline)).date()
                iso_datetime     = target.isoformat()

                # Enrich display with human-readable date context
                if days_to_deadline == 0:
                    day_label = "today"
                elif days_to_deadline == 1:
                    day_label = "tomorrow"
                else:
                    day_label = f"in {days_to_deadline} days"

                display = f"{base_display}  [{day_label}, {target.strftime('%b %d')}]"

            context_note = (
                f"Based on {meeting_date.strftime('%A, %b %d')}"
                if day_fn is not None else None
            )

            return {
                "display":          display,
                "deadline_type":    dtype,
                "urgency_tier":     urgency,
                "days_to_deadline": days_to_deadline,
                "iso_datetime":     iso_datetime,
                "context_note":     context_note,
            }

    return {
        "display":          "N/A",
        "deadline_type":    "unknown",
        "urgency_tier":     "unknown",
        "days_to_deadline": None,
        "iso_datetime":     None,
        "context_note":     None,
    }


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from datetime import date

    # Fixed reference: Thursday, September 10, 2026 (weekday=3)
    REF = datetime(2026, 9, 10, 14, 0, 0)   # Thursday 2pm

    cases: list[tuple] = [
        # (label, utterance, meeting_date, exp_display_contains, exp_urgency, exp_days)

        # ── Existing v1 tests — still pass ───────────────────────────────────
        ("JP 2h",           "2時間以内に回答します",        None,  "2時間以内",        "immediate",  None),
        ("EN digit hours",  "respond within 2 hours",        None,  "within 2 hours",   "immediate",  None),
        ("EN word hours",   "respond within two hours",       None,  "within 2 hours",   "immediate",  None),
        ("EN word minutes", "respond within fifteen minutes", None,  "within 15 minute", "immediate",  None),
        ("EN by eod",       "I'll confirm by end of day",     None,  "by end of day",    "immediate",  None),
        ("EN ASAP",         "resolved ASAP",                  None,  "ASAP",             "immediate",  None),
        ("EN this week",    "I'll send it this week",         datetime(2026, 9, 7),  "end of this week", "this_week",  4),   # Monday Sep 7 → Friday is 4 days
        ("EN within 3 days","respond within 3 days",          None,  "within 3 day",     "this_week",  None),
        ("JP today",        "今日中にご連絡します",           None,  "今日中",           "immediate",  None),
        ("JP tomorrow",     "明日中に送ります",               None,  "明日",             "next_day",   None),
        ("No deadline",     "Let me look into this",          None,  "N/A",              "unknown",    None),

        # ── Date-aware: Thursday Sep 10, 2026 ────────────────────────────────
        ("by Friday — Thu",     "finish by Friday",   REF, "by Friday",    "next_day",  1),   # Friday is tomorrow
        ("next Friday — Thu",   "next Friday",        REF, "next Friday",  "standard",  8),   # 8 days
        ("by Monday — Thu",     "done by Monday",     REF, "by Monday",    "this_week", 4),   # 4 days
        ("next Monday — Thu",   "next Monday",        REF, "next Monday",  "standard",  11),  # 11 days
        ("this Friday — Thu",   "this Friday",        REF, "this Friday",  "next_day",  1),   # tomorrow
        ("bare Friday — Thu",   "have it Friday",     REF, "by Friday",    "next_day",  1),   # tomorrow
        ("tomorrow — Thu",      "call tomorrow",      REF, "by tomorrow",  "next_day",  1),
        ("today — Thu",         "send today",         REF, "by end of today","immediate", 0),
        ("tonight — Thu",       "finish tonight",     REF, "by tonight",   "immediate", 0),
        ("next week — Thu",     "deliver next week",  REF, "by next week", "standard",  8),   # next Friday
        ("ISO date",            "send it by tomorrow",REF, "by tomorrow",  "next_day",  1),

        # ── Date context in display ───────────────────────────────────────────
        ("display hint Fri",  "resolve by Friday",  REF, "tomorrow",    "next_day",  1),
        ("display hint Mon",  "resolve by Monday",  REF, "in 4 days",   "this_week", 4),
    ]

    print("=== deadline_parser v2 — Self-Tests ===\n")
    all_pass = True

    for label, utt, mtg_date, exp_disp, exp_urgency, exp_days in cases:
        result = parse_deadline(utt, meeting_date=mtg_date)

        disp_ok    = exp_disp in result["display"]
        urgency_ok = result["urgency_tier"] == exp_urgency
        days_ok    = (result["days_to_deadline"] == exp_days) if exp_days is not None else True

        ok  = disp_ok and urgency_ok and days_ok
        sym = "✓" if ok else "✗"
        print(f"  {sym}  {label}")

        if not ok:
            all_pass = False
            if not disp_ok:
                print(f"       display  FAIL: expected '{exp_disp}' in '{result['display']}'")
            if not urgency_ok:
                print(f"       urgency  FAIL: expected '{exp_urgency}', got '{result['urgency_tier']}'")
            if not days_ok:
                print(f"       days     FAIL: expected {exp_days}, got {result['days_to_deadline']}")
            print(f"       full result: {result}")

    print(f"\nResult: {'ALL PASS ✓' if all_pass else 'FAILURES ✗'}")