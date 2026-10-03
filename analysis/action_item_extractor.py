# analysis/action_item_extractor.py
# Rule-based action item extraction — zero LLM dependency.
#
# v3 — Step 2 fixes:
#   S1: Vagueness filter — skips commitments with no specific deliverable
#       "全力で対応いたします", "We will not let that happen" → filtered out
#   S2: Bilingual output — task_en always present, task_ja for JP transcripts
#       Matches LLM output schema exactly so html_renderer uses one field for both
#   S3: _jp_to_en() — rule-based JP→EN translation via _JP_VERB_EN map
#       No LLM required. "書面でご回答します" → "Provide a written response"
#
# v2 retained:
#   _split_turns(), _clean_description(), parse_deadline integration

import re
from typing import Optional

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
    r"|いたします"
    r"|します"
    r")"
    r"[^。\n]{0,30}",
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
_EN_PREFIX_RE = re.compile(
    r"^(?:I(?:'ll| will| shall)|We(?:'ll| will| shall)|Will|Going to|Plan to)\s+",
    re.IGNORECASE,
)

# ── EN deadline suffix — stripped from description ────────────────────────────
_EN_DEADLINE_SUFFIX_RE = re.compile(
    r"\s+(?:by|within|before|until)\s+.{0,40}$",
    re.IGNORECASE,
)

# ── S1 NEW: Vagueness filter ──────────────────────────────────────────────────
# Catches commitments with no specific deliverable — these are not action items.
# "全力で対応いたします" (We will do our best), "We will not let that happen"
_VAGUE_JP_RE = re.compile(
    r"^(?:全力で|全力を尽くして|最善を尽くして|頑張|がんばり)?"
    r"(?:対応いたします|やります|努力いたします|取り組みます|させていただきます)$"
)
_VAGUE_EN_RE = re.compile(
    r"^(?:I(?:'ll| will)|We(?:'ll| will)|Will)\s+"
    r"(?:make sure|ensure|do (?:my|our) best|handle it|take care of it|"
    r"not let that happen|do everything|make it work|get it done)\b",
    re.IGNORECASE,
)

# ── S3 NEW: JP verb endings → rough EN translation ────────────────────────────
# O(1) lookup via ordered list — first match wins (specific → generic).
_JP_VERB_EN: list[tuple[str, str]] = [
    ("書面でご回答します",      "Provide a written response"),
    ("書面でご回答いたします",  "Provide a written response"),
    ("ご回答します",            "Respond"),
    ("ご回答いたします",        "Respond"),
    ("ご連絡します",            "Follow up"),
    ("ご連絡いたします",        "Follow up"),
    ("お送りします",            "Send"),
    ("お送りいたします",        "Send"),
    ("確認します",              "Confirm"),
    ("確認いたします",          "Confirm"),
    ("報告します",              "Report back"),
    ("報告いたします",          "Report back"),
    ("提出します",              "Submit"),
    ("提出いたします",          "Submit"),
    ("共有します",              "Share"),
    ("共有いたします",          "Share"),
    ("準備します",              "Prepare"),
    ("準備いたします",          "Prepare"),
    ("相談します",              "Consult internally"),
    ("相談いたします",          "Consult internally"),
    ("対応します",              "Address this"),
    ("対応いたします",          "Address this"),
    ("させていただきます",      "Will proceed as discussed"),
    ("いたします",              "Will handle"),
    ("します",                  "Will action"),
]


def _clean_description(raw_task: str, is_japanese: bool) -> str:
    """
    Produce a clean, readable action item description.

    English:
      "I will provide a written response by Friday"
      → "Provide a written response"      (strip prefix + deadline suffix)

    Japanese:
      "上司に相談して、2時間以内に書面でご回答します"
      → kept as-is (translation handled by _jp_to_en)

    DSA: O(L) — two regex scans over task length L.
    """
    task = raw_task.strip()
    if is_japanese:
        return task
    task = _EN_PREFIX_RE.sub("", task).strip()
    task = _EN_DEADLINE_SUFFIX_RE.sub("", task).strip()
    if task:
        task = task[0].upper() + task[1:]
    return task if len(task) >= 5 else raw_task.strip()


def _is_vague(task: str, is_japanese: bool) -> bool:
    """
    S1: Return True if the commitment has no specific deliverable.

    Vague (skip):    "全力で対応いたします", "We will not let that happen"
    Not vague (keep): "2時間以内に書面でご回答します", "Send the report by Friday"

    DSA: O(1) — regex match + word count check.
    """
    task = task.strip()
    if is_japanese:
        # Strip deadline context — check only the verb phrase
        verb_only = re.sub(r".+[、,]\s*", "", task)
        if _VAGUE_JP_RE.match(verb_only):
            return True
        # Task with < 4 chars before the first verb = no object = vague
        verb_start = next(
            (task.rfind(v) for v, _ in _JP_VERB_EN if v in task and task.rfind(v) > 0),
            -1,
        )
        if verb_start != -1 and verb_start < 4:
            return True
    else:
        if _VAGUE_EN_RE.match(task):
            return True
        stripped = _EN_PREFIX_RE.sub("", task).strip()
        if len(stripped.split()) < 3:
            return True
    return False


def _jp_to_en(task_ja: str, deadline_display: str) -> str:
    """
    S3: Rough JP → EN translation of a commitment phrase using _JP_VERB_EN map.
    No LLM required.

    DSA: O(V) where V = len(_JP_VERB_EN) ≈ 25 — first match wins.
    """
    clean = re.sub(r"^[^：:]{1,40}[：:]\s*", "", task_ja).strip()

    for jp_verb, en_verb in _JP_VERB_EN:
        if jp_verb in clean:
            prefix = clean[: clean.index(jp_verb)].strip()
            # Remove deadline clause from prefix
            prefix = re.sub(r"\d+時間以内|今日中|明日中|今週中|来週.*", "", prefix).strip("、,・ ")
            if prefix:
                return f"{en_verb}: {prefix}"
            return en_verb

    return f"[JP commitment — see original]: {clean[:80]}"


def _split_turns(text: str) -> list[tuple[str, str]]:
    """
    Split transcript into (speaker, utterance) pairs.
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

    Output schema (v3 — compatible with LLM output):
        {
          "task":         str,    # raw matched commitment phrase
          "task_en":      str,    # NEW — English version, always present
          "task_ja":      str|None, # NEW — Japanese original, None for EN transcripts
          "description":  str,    # same as task_en (backward compat)
          "owner":        str,
          "deadline":     str,
          "urgency_tier": str,
          "source":       str,    # "rule_based"
          "confidence":   float,
        }

    DSA: O(n · p) — n = chars in transcript, p = patterns.
    """
    turns = _split_turns(text)
    results: list[dict] = []
    seen_tasks: set[str] = set()

    def _add(
        task: str, owner: str, deadline_result: dict,
        confidence: float, is_japanese: bool,
    ) -> None:
        task = task.strip()
        task = re.sub(r"^[A-Za-z\u3040-\u9FFF][^\n:：]{0,40}[:：]\s*", "", task).strip()
        if len(task) < 8:
            return
        # S1: vagueness filter — skip commitments with no specific deliverable
        if _is_vague(task, is_japanese):
            return
        key = re.sub(r"\s+", "", task)[:40].lower()
        if key in seen_tasks:
            return
        seen_tasks.add(key)

        # S2: bilingual output
        if is_japanese:
            task_ja = task
            task_en = _jp_to_en(task, deadline_result["display"])
        else:
            task_en = _clean_description(task, is_japanese=False)
            task_ja = None   # only populated for JP transcripts

        results.append({
            "task":         task,
            "task_en":      task_en,         # always present
            "task_ja":      task_ja,         # None for EN transcripts
            "description":  task_en,         # backward compat
            "owner":        owner,
            "deadline":     deadline_result["display"],
            "urgency_tier": deadline_result["urgency_tier"],
            "source":       "rule_based",
            "confidence":   confidence,
        })

    for speaker, utterance in turns:
        if not utterance:
            continue

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
            [{"owner": "Kenji", "deadline_contains": "2時間以内", "urgency": "immediate",
              "task_en_contains": "written response"}],
        ),
        (
            "S1 VAGUENESS FILTER — 全力で対応いたします should be skipped",
            "Kenji: 全力で対応いたします。",
            [],   # must return empty
        ),
        (
            "S1 VAGUENESS FILTER — 'We will not let that happen' should be skipped",
            "Kenji: We will not let that happen.",
            [],
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
            [{"owner": "Kenji", "deadline_contains": "Friday"}],
        ),
        (
            "S2 task_ja populated for JP, None for EN",
            "Kenji: I will send the report by tomorrow.",
            [{"owner": "Kenji", "task_ja_is_none": True}],
        ),
        (
            "S2 task_en present for JP commitment",
            "Kenji: 上司に相談して、2時間以内に書面でご回答します。",
            [{"owner": "Kenji", "task_en_contains": "written response",
              "task_ja_is_none": False}],
        ),
        (
            "Multiple commitments different speakers",
            "Priya: I'll send the report by end of day.\n"
            "Kunal: 確認して明日中に共有します。",
            [
                {"owner": "Priya", "deadline_contains": "end of day", "urgency": "immediate"},
                {"owner": "Kunal", "deadline_contains": "明日", "urgency": "next_day"},
            ],
        ),
        (
            "No commitment — should return empty",
            "Client: The system has been down for 6 hours. This is unacceptable.",
            [],
        ),
        (
            "Description cleaning — EN should strip 'I will'",
            "Alice: I will send the updated document by tomorrow.",
            [{"owner": "Alice", "deadline_contains": "tomorrow", "urgency": "next_day",
              "description_not_starts_with": "I will"}],
        ),
    ]

    print("=== Action Item Extractor v3 — Self-Tests ===\n")
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
                matched = [i for i in items if i["owner"] == exp.get("owner", "")]
                owner_ok    = bool(matched)

                deadline_ok = True
                if "deadline_contains" in exp:
                    deadline_ok = any(exp["deadline_contains"] in i["deadline"] for i in matched)

                urgency_ok = True
                if "urgency" in exp:
                    urgency_ok = any(i["urgency_tier"] == exp["urgency"] for i in matched)

                desc_ok = True
                if "description_not_starts_with" in exp:
                    desc_ok = all(
                        not i["description"].startswith(exp["description_not_starts_with"])
                        for i in matched
                    )

                task_en_ok = True
                if "task_en_contains" in exp:
                    task_en_ok = any(exp["task_en_contains"].lower() in i.get("task_en", "").lower()
                                     for i in matched)

                task_ja_ok = True
                if "task_ja_is_none" in exp:
                    if exp["task_ja_is_none"]:
                        task_ja_ok = all(i.get("task_ja") is None for i in matched)
                    else:
                        task_ja_ok = all(i.get("task_ja") is not None for i in matched)

                ok = owner_ok and deadline_ok and urgency_ok and desc_ok and task_en_ok and task_ja_ok
                sym = "✓" if ok else "✗"
                if not ok:
                    all_pass = False
                print(f"  {sym}  {label}")
                for item in matched:
                    print(f"       owner={item['owner']}")
                    print(f"       task_en={item.get('task_en','—')[:60]}")
                    print(f"       task_ja={item.get('task_ja','—')[:50] if item.get('task_ja') else None}")
                    print(f"       deadline={item['deadline']}  urgency={item['urgency_tier']}")
                if not owner_ok:
                    print(f"       FAIL: expected owner='{exp.get('owner')}', got {[i['owner'] for i in items]}")
                if owner_ok and not deadline_ok:
                    print(f"       FAIL: expected deadline containing '{exp.get('deadline_contains')}'")
                if owner_ok and not task_en_ok:
                    print(f"       FAIL: task_en should contain '{exp.get('task_en_contains')}'")
                    print(f"             got: {[i.get('task_en','') for i in matched]}")
        print()

    print(f"Result: {'ALL PASS ✓' if all_pass else 'FAILURES ✗'}")