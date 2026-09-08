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
#   → owner=Kenji, deadline="within 2 hours", urgency_tier="immediate"
#
# Changelog:
#   v2 — Replaced local _DEADLINE_RULES / _extract_deadline() with
#         utils.deadline_parser.parse_deadline(). This fixes word-number
#         deadlines ("within two hours" → deadline: None was the bug).
#         Added urgency_tier and description fields to output schema.

import re
from typing import Optional

# Import the shared deadline parser — single source of truth.
# All deadline pattern logic lives there; not duplicated here.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.deadline_parser import parse_deadline


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

# ── EN commitment verb prefix — stripped from description ─────────────────────
# Converts "I will provide a report" → "Provide a report"
_EN_PREFIX_RE = re.compile(
    r"^(?:I(?:'ll| will| shall)|We(?:'ll| will| shall)|Will|Going to|Plan to)\s+",
    re.IGNORECASE,
)

# ── EN deadline suffix — stripped from description (it lives in deadline field)
_EN_DEADLINE_SUFFIX_RE = re.compile(
    r"\s+(?:by|within|before|until)\s+.{0,40}$",
    re.IGNORECASE,
)


def _clean_description(raw_task: str, is_japanese: bool) -> str:
    """
    Produce a clean, readable action item description.

    English:
      "I will provide a written response by Friday"
      → "Provide a written response"      (strip prefix + deadline suffix)

    Japanese:
      "上司に相談して、2時間以内に書面でご回答します"
      → kept as-is (no LLM translation available in rule-based path)

    DSA: O(L) — two regex scans over task length L.
    """
    task = raw_task.strip()

    if is_japanese:
        # Cannot translate without LLM — return as-is, caller adds deadline field
        return task

    # Strip "I will / I'll / We will / ..." prefix
    task = _EN_PREFIX_RE.sub("", task).strip()

    # Strip trailing deadline clause — it's already captured in the deadline field
    task = _EN_DEADLINE_SUFFIX_RE.sub("", task).strip()

    # Capitalize first word
    if task:
        task = task[0].upper() + task[1:]

    return task if len(task) >= 5 else raw_task.strip()


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


def extract_action_items(text: str, meeting_ts: Optional[str] = None) -> list[dict]:
    """
    Extract explicit action items from transcript without any LLM call.

    Scans each speaker turn for:
      JP: commitment verb endings (します/いたします/etc.) + deadline (〜以内/まで/今日中)
      EN: "I will/I'll [verb]" constructions + "by/within [deadline]"

    Output schema (compatible with LLM output, extended with new fields):
        {
          "task":         str,    # raw matched commitment phrase
          "description":  str,    # clean, human-readable version of task
          "owner":        str,    # speaker who made the commitment
          "deadline":     str,    # human-readable deadline string, or "N/A"
          "urgency_tier": str,    # "immediate"|"next_day"|"this_week"|"standard"|"unknown"
          "source":       str,    # "rule_based"
          "confidence":   float,
        }

    Deduplication: tasks whose first 40 normalised chars match an existing
    entry are skipped — prevents the same sentence matching both a broad
    and a specific commitment pattern.

    Args:
        text       : full transcript text
        meeting_ts : ISO-8601 meeting timestamp (passed to deadline_parser
                     for future absolute deadline computation)

    DSA: O(n · p) — n = chars in transcript, p = number of patterns (2 JP + 1 EN).
    """
    turns = _split_turns(text)
    results: list[dict] = []
    seen_tasks: set[str] = set()       # dedup by normalised task prefix

    def _add(
        task: str, owner: str, deadline_result: dict,
        confidence: float, is_japanese: bool,
    ) -> None:
        task = task.strip()
        # Strip leading speaker label if accidentally captured
        task = re.sub(r"^[A-Za-z\u3040-\u9FFF][^\n:：]{0,40}[:：]\s*", "", task).strip()
        if len(task) < 8:
            return
        key = re.sub(r"\s+", "", task)[:40].lower()
        if key in seen_tasks:
            return
        seen_tasks.add(key)
        results.append({
            "task":         task,
            "description":  _clean_description(task, is_japanese),
            "owner":        owner,
            "deadline":     deadline_result["display"],
            "urgency_tier": deadline_result["urgency_tier"],
            "source":       "rule_based",
            "confidence":   confidence,
        })

    for speaker, utterance in turns:
        if not utterance:
            continue

        # parse_deadline now handles word-numbers + all patterns — single call per turn
        deadline_result = parse_deadline(utterance, meeting_ts=meeting_ts)

        # ── Japanese commitment phrases ───────────────────────────────────────
        for m in _JP_COMMIT_PATTERN.finditer(utterance):
            task_text = m.group().strip()
            if re.search(r"(?:します|いたします|させていただきます)", task_text):
                _add(task_text, speaker, deadline_result, 0.90, is_japanese=True)

        # ── English commitment phrases ────────────────────────────────────────
        for m in _EN_COMMIT_PATTERN.finditer(utterance):
            task_text = m.group().strip()
            _add(task_text, speaker, deadline_result, 0.84, is_japanese=False)

    return results


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        (
            "Kenji exact phrase (JP + digit hours)",
            "Kenji: 上司に相談して、2時間以内に書面でご回答します。",
            [{"owner": "Kenji", "deadline_contains": "2時間以内", "urgency": "immediate"}],
        ),
        (
            "THE BUG — EN word-number hours (was deadline: None)",
            "Manager: We will escalate and respond within two hours.",
            [{"owner": "Manager", "deadline_contains": "within 2 hour", "urgency": "immediate"}],
        ),
        (
            "English commitment with Friday deadline",
            "Client: The system has been down for 6 hours.\n"
            "Kenji: I will provide a written response by Friday.",
            [{"owner": "Kenji", "deadline_contains": "Friday", "urgency": "this_week"}],
        ),
        (
            "Multiple commitments different speakers",
            "Priya: I'll send the report by end of day.\n"
            "Kunal: 確認して明日中に共有します。",
            [
                {"owner": "Priya",  "deadline_contains": "end of day", "urgency": "immediate"},
                {"owner": "Kunal",  "deadline_contains": "明日",       "urgency": "next_day"},
            ],
        ),
        (
            "No commitment — should return empty",
            "Client: The system has been down for 6 hours. This is unacceptable.",
            [],
        ),
        (
            "Within-hours EN (digit)",
            "Manager: We will escalate this and respond within 2 hours.",
            [{"owner": "Manager", "deadline_contains": "2 hour", "urgency": "immediate"}],
        ),
        (
            "Mixed JP text with EN deadline",
            "Tanaka: ご確認して、by Friday にご連絡いたします。",
            [{"owner": "Tanaka", "deadline_contains": "Friday", "urgency": "this_week"}],
        ),
        (
            "Description cleaning — EN should strip 'I will'",
            "Alice: I will send the updated document by tomorrow.",
            [{"owner": "Alice", "deadline_contains": "tomorrow", "urgency": "next_day",
              "description_not_starts_with": "I will"}],
        ),
    ]

    print("=== Action Item Extractor v2 — Self-Tests ===\n")
    all_pass = True
    for label, transcript, expected in tests:
        items = extract_action_items(transcript)

        if not expected:
            ok = len(items) == 0
            sym = "✓" if ok else "✗"
            print(f"  {sym}  {label}")
            if not ok:
                print(f"       Got unexpected items: {[i['task'][:50] for i in items]}")
                all_pass = False
        else:
            for exp in expected:
                matched = [i for i in items if i["owner"] == exp["owner"]]
                owner_ok    = bool(matched)
                deadline_ok = any(exp["deadline_contains"] in i["deadline"] for i in matched)
                urgency_ok  = any(i["urgency_tier"] == exp.get("urgency","unknown") for i in matched)
                desc_ok     = True
                if "description_not_starts_with" in exp:
                    desc_ok = all(
                        not i["description"].startswith(exp["description_not_starts_with"])
                        for i in matched
                    )
                ok = owner_ok and deadline_ok and urgency_ok and desc_ok
                sym = "✓" if ok else "✗"
                if not ok:
                    all_pass = False
                print(f"  {sym}  {label}")
                for item in matched:
                    print(f"       owner={item['owner']}")
                    print(f"       deadline={item['deadline']}  urgency={item['urgency_tier']}")
                    print(f"       description={item['description'][:70]}")
                    print(f"       task={item['task'][:60]}")
                if not owner_ok:
                    print(f"       FAIL: expected owner='{exp['owner']}', got {[i['owner'] for i in items]}")
                if owner_ok and not deadline_ok:
                    print(f"       FAIL: expected deadline containing '{exp['deadline_contains']}'")
                if owner_ok and not urgency_ok:
                    print(f"       FAIL: expected urgency='{exp.get('urgency')}', "
                          f"got {[i['urgency_tier'] for i in matched]}")
                if owner_ok and not desc_ok:
                    print(f"       FAIL: description should not start with "
                          f"'{exp['description_not_starts_with']}'")
        print()

    print(f"Result: {'ALL PASS ✓' if all_pass else 'FAILURES ✗'}")