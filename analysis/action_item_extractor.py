# analysis/action_item_extractor.py
# Rule-based action item extraction — zero LLM dependency.
#
# Catches explicit commitment phrases with deadlines in:
#   Japanese  — します/いたします + 〜以内/〜まで/今日中
#   English   — "I will/I'll [verb]" + "by/within [deadline]"
#   Mixed     — Hinglish commitment + EN deadline, or EN phrase + JP deadline
#
# Used in _no_api_result() when Groq quota is exhausted.
# Also runs as a post-LLM pass to catch commitments the LLM missed.
#
# Specific case that triggered this:
#   "上司に相談して、2時間以内に書面でご回答します"
#   → owner=Kenji, deadline="2時間以内 (within 2 hours)"
#   Previously returned [] because action_items was marked LLM-only.

import re
from typing import Optional


# ── JP commitment verb endings ────────────────────────────────────────────────
_JP_COMMIT_PATTERN = re.compile(
    r"[^。\n]{3,80}"                   # lead-up text (min 3 chars, max 80)
    r"(?:"
    r"ご回答します|ご連絡します|お送りします|お伝えします|お知らせします"
    r"|対応いたします|確認いたします|提出いたします|報告いたします"
    r"|させていただきます"
    r"|いたします"                      # broad catch — after specific ones
    r"|します"                          # broad catch — after specific ones
    r")"
    r"[^。\n]{0,30}",                  # optional trailing context
    re.MULTILINE,
)

# ── EN commitment patterns ────────────────────────────────────────────────────
_EN_COMMIT_PATTERN = re.compile(
    r"(?:I(?:'ll| will| shall)|We(?:'ll| will| shall)|Will|Going to|Plan to)"
    r"\s+"
    r"(?:send|deliver|provide|share|submit|respond|reply|follow[\s\-]?up"
    r"|update|confirm|check|review|prepare|complete|finish|handle"
    r"|take care of|look into|get back|escalate|arrange|coordinate"
    r"|schedule|book|set up|draft|write|compile|investigate)[^.!?\n]{0,80}",
    re.IGNORECASE,
)

# ── Deadline extraction ───────────────────────────────────────────────────────
_DEADLINE_RULES = [
    # JP — specific durations
    (re.compile(r"(\d+)時間以内"),
     lambda m: f"{m.group(1)}時間以内 (within {m.group(1)} hour{'s' if int(m.group(1))>1 else ''})"),

    (re.compile(r"(\d+)分以内"),
     lambda m: f"{m.group(1)}分以内 (within {m.group(1)} minute{'s' if int(m.group(1))>1 else ''})"),

    (re.compile(r"(\d+)日以内"),
     lambda m: f"{m.group(1)}日以内 (within {m.group(1)} day{'s' if int(m.group(1))>1 else ''})"),

    (re.compile(r"(\d+)週間以内"),
     lambda m: f"{m.group(1)}週間以内 (within {m.group(1)} week{'s' if int(m.group(1))>1 else ''})"),

    # JP — named day deadlines
    (re.compile(r"(月|火|水|木|金|土|日)曜日まで"),
     lambda m: {"月":"Monday","火":"Tuesday","水":"Wednesday","木":"Thursday",
                "金":"Friday","土":"Saturday","日":"Sunday"}[m.group(1)] + "まで"),

    (re.compile(r"金曜日"), lambda m: "金曜日 (Friday)"),
    (re.compile(r"月曜日"), lambda m: "月曜日 (Monday)"),

    # JP — relative today/this week
    (re.compile(r"今日中|本日中"),  lambda m: "今日中 (by end of today)"),
    (re.compile(r"今週中"),         lambda m: "今週中 (this week)"),
    (re.compile(r"今月中"),         lambda m: "今月中 (this month)"),
    (re.compile(r"明日まで|明日中"), lambda m: "明日 (by tomorrow)"),

    # JP — specific date
    (re.compile(r"(\d{1,2})月(\d{1,2})日まで"),
     lambda m: f"{m.group(1)}/{m.group(2)} (by {m.group(1)}/{m.group(2)})"),

    # EN — duration
    (re.compile(r"within (\d+)\s*hours?", re.IGNORECASE),
     lambda m: f"within {m.group(1)} hour{'s' if int(m.group(1))>1 else ''}"),

    (re.compile(r"within (\d+)\s*minutes?", re.IGNORECASE),
     lambda m: f"within {m.group(1)} minute{'s' if int(m.group(1))>1 else ''}"),

    (re.compile(r"within (\d+)\s*days?", re.IGNORECASE),
     lambda m: f"within {m.group(1)} day{'s' if int(m.group(1))>1 else ''}"),

    # EN — named day
    (re.compile(r"by\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", re.IGNORECASE),
     lambda m: f"by {m.group(1).capitalize()}"),

    (re.compile(r"by\s+end\s+of\s+(?:the\s+)?(?:day|today)|by\s+eod", re.IGNORECASE),
     lambda m: "by end of day"),

    (re.compile(r"by\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))", re.IGNORECASE),
     lambda m: f"by {m.group(1)}"),

    # EN — relative
    (re.compile(r"\btoday\b|\btonight\b", re.IGNORECASE), lambda m: "by end of today"),
    (re.compile(r"\btomorrow\b", re.IGNORECASE),           lambda m: "by tomorrow"),
    (re.compile(r"\bthis week\b", re.IGNORECASE),          lambda m: "by end of this week"),
    (re.compile(r"\bfriday\b", re.IGNORECASE),             lambda m: "by Friday"),
    (re.compile(r"\bmonday\b", re.IGNORECASE),             lambda m: "by Monday"),
]


def _extract_deadline(utterance: str) -> str:
    """
    Scan utterance for any deadline pattern.
    Returns formatted deadline string or "N/A".
    Tries most-specific patterns first (durations before day names).
    """
    for pattern, formatter in _DEADLINE_RULES:
        m = pattern.search(utterance)
        if m:
            return formatter(m)
    return "N/A"


def _split_turns(text: str) -> list[tuple[str, str]]:
    """
    Split transcript into (speaker, utterance) pairs.
    Handles standard "Name: text" format and multi-line turns.
    Returns list of (speaker_name, full_utterance) tuples.
    DSA: O(n) — single pass through lines.
    """
    speaker_pat = re.compile(
        r"^\s*([A-Za-z\u3040-\u9FFF][^\n:：\[\]]{0,40}?)\s*[:：]\s*(.*)$",
        re.MULTILINE,
    )
    turns: list[tuple[str, str]] = []
    current_speaker: Optional[str] = None
    current_lines: list[str] = []

    for line in text.split("\n"):
        m = speaker_pat.match(line)
        if m:
            if current_speaker and current_lines:
                turns.append((current_speaker, " ".join(current_lines).strip()))
            current_speaker = m.group(1).strip()
            first_line = m.group(2).strip()
            current_lines = [first_line] if first_line else []
        elif current_speaker:
            stripped = line.strip()
            if stripped:
                current_lines.append(stripped)

    if current_speaker and current_lines:
        turns.append((current_speaker, " ".join(current_lines).strip()))

    return turns


def extract_action_items(text: str) -> list[dict]:
    """
    Extract explicit action items from transcript without any LLM call.

    Scans each speaker turn for:
      JP: commitment verb endings (します/いたします/etc.) + deadline (〜以内/まで/今日中)
      EN: "I will/I'll [verb]" constructions + "by/within [deadline]"

    Returns list of action item dicts compatible with LLM output schema:
        {"task": str, "owner": str, "deadline": str,
         "source": "rule_based", "confidence": float}

    Deduplication: tasks whose first 40 normalised chars match an existing
    entry are skipped — prevents the same sentence matching both a broad
    and a specific commitment pattern.

    DSA: O(n · p) — n = chars in transcript, p = number of patterns (2 JP + 1 EN).
    """
    turns = _split_turns(text)
    results: list[dict] = []
    seen_tasks: set[str] = set()       # dedup by normalised task prefix

    def _add(task: str, owner: str, deadline: str, confidence: float) -> None:
        task = task.strip()
        # Strip leading speaker label if captured
        task = re.sub(r"^[A-Za-z\u3040-\u9FFF][^\n:：]{0,40}[:：]\s*", "", task).strip()
        if len(task) < 8:
            return
        key = re.sub(r"\s+", "", task)[:40].lower()
        if key in seen_tasks:
            return
        seen_tasks.add(key)
        results.append({
            "task":       task,
            "owner":      owner,
            "deadline":   deadline,
            "source":     "rule_based",
            "confidence": confidence,
        })

    for speaker, utterance in turns:
        if not utterance:
            continue

        deadline = _extract_deadline(utterance)

        # ── Japanese commitment phrases ───────────────────────────────────────
        for m in _JP_COMMIT_PATTERN.finditer(utterance):
            task_text = m.group().strip()
            # Must contain a real commitment verb (not just a suffix match on noise)
            if re.search(r"(?:します|いたします|させていただきます)", task_text):
                _add(task_text, speaker, deadline, confidence=0.90)

        # ── English commitment phrases ────────────────────────────────────────
        for m in _EN_COMMIT_PATTERN.finditer(utterance):
            task_text = m.group().strip()
            _add(task_text, speaker, deadline, confidence=0.84)

    return results


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    tests = [
        (
            "Kenji exact phrase",
            "Kenji: 上司に相談して、2時間以内に書面でご回答します。",
            [{"owner": "Kenji", "deadline_contains": "2時間以内"}],
        ),
        (
            "English commitment with Friday deadline",
            "Client: The system has been down for 6 hours.\n"
            "Kenji: I will provide a written response by Friday.",
            [{"owner": "Kenji", "deadline_contains": "Friday"}],
        ),
        (
            "Multiple commitments different speakers",
            "Priya: I'll send the report by end of day.\n"
            "Kunal: 確認して明日中に共有します。",
            [
                {"owner": "Priya", "deadline_contains": "end of day"},
                {"owner": "Kunal", "deadline_contains": "明日"},
            ],
        ),
        (
            "No commitment — should return empty",
            "Client: The system has been down for 6 hours. This is unacceptable.",
            [],
        ),
        (
            "Within-hours English",
            "Manager: We will escalate this and respond within 2 hours.",
            [{"owner": "Manager", "deadline_contains": "2 hour"}],
        ),
        (
            "Mixed JP text with EN deadline",
            "Tanaka: ご確認して、by Friday にご連絡いたします。",
            [{"owner": "Tanaka", "deadline_contains": "Friday"}],
        ),
    ]

    print("=== Action Item Extractor — Self-Tests ===\n")
    all_pass = True
    for label, transcript, expected in tests:
        items = extract_action_items(transcript)
        if not expected:
            ok = len(items) == 0
            sym = "✓" if ok else "✗"
            print(f"  {sym}  {label}")
            if not ok:
                print(f"       Got unexpected items: {items}")
                all_pass = False
        else:
            for exp in expected:
                owner_ok    = any(i["owner"] == exp["owner"] for i in items)
                deadline_ok = any(exp["deadline_contains"] in i["deadline"] for i in items)
                ok = owner_ok and deadline_ok
                sym = "✓" if ok else "✗"
                if not ok:
                    all_pass = False
                print(f"  {sym}  {label}")
                for item in items:
                    print(f"       owner={item['owner']}  deadline={item['deadline']}")
                    print(f"       task={item['task'][:70]}")
                if not owner_ok:
                    print(f"       FAIL: expected owner='{exp['owner']}', got {[i['owner'] for i in items]}")
                if not deadline_ok:
                    print(f"       FAIL: expected deadline containing '{exp['deadline_contains']}'")
        print()

    print(f"Result: {'ALL PASS ✓' if all_pass else 'FAILURES ✗'}")