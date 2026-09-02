# utils/speaker_detector.py
# ─────────────────────────────────────────────────────────────────────────────
# Robust speaker detection for 1–40+ speaker transcripts. Zero LLM required.
#
# Handles all real-world transcript formats:
#   Standard        "Name: text"
#   LinkedIn/Chat   "Name\ntext"    (FIX-22 normalizes first)
#   Timestamped     "[00:01:23] Name: text"
#   Zoom/Teams      "Name  12:34 PM" as standalone line
#   Whisper         "SPEAKER_00: text"
#   Markdown bold   "**Name:** text"
#   Japanese CJK    "田中： text"
#   PII masked      "[NAME_1]: text"
#   Chat header     "Name  12:34 PM"
#
# Deduplication rules:
#   Numbered IDs  — [NAME_1] vs [NAME_2], SPEAKER_00 vs SPEAKER_01 → NEVER merged
#   English names — token intersection: "Dr. John Smith" ↔ "John" → same person
#   Japanese names — substring match: "田中" ↔ "田中部長" → same person
#   Titles stripped before matching: Dr., Mr., Ms., -san, -bucho, etc.
#
# Critical: ALWAYS call with full (non-truncated) text.
#   Truncation is only for LLM token budgets.
#   Speakers appearing only in the middle section would be lost otherwise.

import re
from collections import Counter, defaultdict


# ── Blocklist — structural words that appear before colons but are not names ──
_NOT_SPEAKER = {
    "note", "notes", "todo", "action", "summary", "result", "update",
    "warning", "error", "info", "subject", "from", "to", "cc", "date",
    "time", "location", "agenda", "minutes", "re", "ps", "fyi",
    "background", "context", "status", "priority", "description",
    "project", "task", "issue", "question", "answer", "decision",
    "objective", "goal", "outcome", "risk", "blocker", "deadline",
    "http", "https", "www", "sent", "edited", "deleted",
    "owner", "assignee", "reviewer", "attendee", "host", "moderator",
    "conclusion", "discussion", "resolution", "follow", "reference",
}

# Honorific noise to strip before name token comparison
_HONORIFICS = {
    "mr", "mrs", "ms", "miss", "dr", "prof", "professor",
    "sir", "madam", "rev", "hon",
    "san", "kun", "chan", "sama", "sensei", "bucho", "kacho",
    "director", "manager", "lead", "chief", "head", "vp", "ceo", "cto",
}

# ── Multi-format extraction patterns ─────────────────────────────────────────
_PATTERNS = [
    # 1. Standard inline "Name: text"
    #    (?!\d{2}) rejects "12:" in timestamps — prevents "Name  12" capture in Zoom format
    (re.compile(
        r"(?:^|\n)\s*"
        r"([A-Za-z\u3040-\u9FFF][^\n::\[\]]{0,40}?)"
        r"\s*[:：](?!\d{2})\s*\S",
        re.MULTILINE,
    ), "standard"),

    # 2. Timestamped "[00:01:23] Name: text"
    (re.compile(
        r"\[\d{1,2}:\d{2}(?::\d{2})?\]\s+"
        r"([A-Za-z\u3040-\u9FFF][^\n:]{0,40}?)"
        r"\s*[:]",
        re.MULTILINE,
    ), "timestamped"),

    # 3. Zoom/Teams/Chat "Name  12:34 PM" on its own line
    #    FIX: use literal space in char class — NOT \s — to prevent crossing newlines
    (re.compile(
        r"^([A-Za-z][A-Za-z \.\-\']{1,30})"
        r"\s+(?:\([^)]+\)\s+)?"
        r"\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\s*$",
        re.MULTILINE | re.IGNORECASE,
    ), "zoom"),

    # 4. Markdown bold "**Name:**"
    (re.compile(
        r"\*\*([A-Za-z\u3040-\u9FFF][^\*\n]{1,40}?)\*\*\s*[:]",
    ), "markdown"),

    # 5. Whisper diarized "SPEAKER_00:" or "SPEAKER 1:"
    (re.compile(
        r"(?:^|\n)\s*(SPEAKER[_\s]\d+)\s*[:]",
        re.MULTILINE,
    ), "whisper"),

    # 6. PII masked "[NAME_1]:" or "[EMAIL_1]:"
    (re.compile(
        r"(?:^|\n)\s*(\[[A-Z]+_\d+\])\s*[:]",
        re.MULTILINE,
    ), "masked"),

    # 7. Japanese CJK "田中：" or "山本部長："
    (re.compile(
        r"(?:^|\n)\s*([\u3040-\u9FFF]{1,8}(?:[　 ][\u3040-\u9FFF]{1,8})?)"
        r"\s*[：]",
        re.MULTILINE,
    ), "japanese"),
]


def _has_cjk(s: str) -> bool:
    return bool(re.search(r"[\u3040-\u9fff\u4e00-\u9fff]", s))


def _is_numbered_identifier(name: str) -> bool:
    """
    Returns True for names that are distinguished purely by number:
      [NAME_1], [NAME_2]   — PII placeholders
      SPEAKER_00, SPEAKER_01 — Whisper diarization
      Speaker01, Speaker 1   — numbered fallback labels

    These must NEVER be merged by _same_person even though their
    letter-only tokens are identical ("speaker", "name").
    """
    s = name.strip()
    return bool(
        re.match(r"^\[?[A-Z]+[_\s]?\d+\]?$", s)    # [NAME_1], SPEAKER_00, NAME_1
        or re.match(r"^[Ss]peaker[\s_]?\d+$", s)    # Speaker01, Speaker_01
        or re.match(r"^[Ss]peaker\s+\d+$", s)       # Speaker 1, Speaker 42
    )


def _name_tokens(name: str) -> frozenset:
    """
    Extract meaningful name tokens, removing honorifics and noise.
      "Dr. John Smith" → frozenset({'john', 'smith'})
      "田中部長"        → frozenset({'田中部長'})   (CJK kept whole)
    """
    if _has_cjk(name):
        return frozenset({name.strip()})
    tokens = set(re.findall(r"[a-zA-Z']+", name.lower()))
    tokens -= _HONORIFICS
    return frozenset(tokens) if tokens else frozenset({name.lower().strip()})


def _same_person(a: str, b: str) -> bool:
    """
    True if two name strings likely refer to the same speaker.

    Rules (in order):
    1. Exact match           → always same
    2. Numbered identifier   → never same (SPEAKER_00 ≠ SPEAKER_01)
    3. CJK names             → substring match in either direction
    4. English/Latin names   → shorter token set ⊆ longer token set,
                               with minimum 3-char token guard
    """
    if a.lower().strip() == b.lower().strip():
        return True

    # Numbered IDs are ONLY the same if strings match exactly (handled above)
    if _is_numbered_identifier(a) or _is_numbered_identifier(b):
        return False

    # Japanese/CJK: substring in either direction
    if _has_cjk(a) or _has_cjk(b):
        return a.lower() in b.lower() or b.lower() in a.lower()

    # Latin/English: token subset matching
    ta, tb = _name_tokens(a), _name_tokens(b)
    if not ta or not tb:
        return False

    shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)

    # Single-token match: token must be ≥3 chars (prevents "Jo" matching "Johnson")
    if len(shorter) == 1:
        tok = next(iter(shorter))
        return tok in longer and len(tok) >= 3

    # Multi-token: all tokens of shorter name must appear in longer
    return shorter.issubset(longer)


def _is_valid_name(name: str) -> bool:
    """Filter structural words, numbers, URLs, and single-char noise."""
    name = name.strip()
    if not name or len(name) < 2:
        return False
    if re.match(r"^\d+$", name):
        return False
    if re.match(r"^https?://", name, re.IGNORECASE):
        return False
    first_word = name.lower().split()[0] if name.split() else ""
    if first_word in _NOT_SPEAKER:
        return False
    if not re.match(r"^[A-Za-z\u3040-\u9FFF\[]", name):
        return False
    return True


def detect_speakers(text: str) -> dict:
    """
    Detect all unique speakers in a transcript. No LLM required.

    Always call with the FULL non-truncated transcript. Truncation is only
    for LLM token budgets — speakers in the middle section would be lost.

    Returns:
        count              int        — number of unique speakers
        names              list[str]  — deduplicated canonical names,
                                        sorted by turn count descending
        turns_per_speaker  dict       — canonical name → turn count
        total_turns        int        — total speaker turns found
        formats_detected   list[str]  — which patterns fired
        confidence         str        — "high" | "medium" | "low"

    DSA:
        Extraction : O(n · p)  n = text chars, p = number of patterns (7)
        Dedup      : O(k²)     k = unique raw names, typically <100
        Overall    : O(n) dominant
    """
    name_turn_count: Counter = Counter()
    formats_seen: set        = set()

    # ── Step 1: Extract raw names across all patterns ─────────────────────────
    for pattern, fmt in _PATTERNS:
        for m in pattern.finditer(text):
            raw = m.group(1).strip()
            raw = re.sub(r"\s+", " ", raw)           # collapse internal whitespace
            raw = re.sub(r"\s*\([^)]*\)\s*$", "", raw)  # strip "(role/title)" suffix
            if _is_valid_name(raw):
                name_turn_count[raw] += 1
                formats_seen.add(fmt)

    if not name_turn_count:
        return {
            "count": 0, "names": [], "turns_per_speaker": {},
            "total_turns": 0, "formats_detected": [], "confidence": "low",
        }

    # ── Step 2: Deduplicate — merge names that refer to the same person ───────
    # Sort longest → shortest: canonical form is the most complete name
    sorted_names = sorted(name_turn_count.keys(), key=len, reverse=True)
    canonical_of: dict[str, str] = {}   # raw → canonical

    for name in sorted_names:
        matched = None
        for existing_canon in set(canonical_of.values()):
            if _same_person(name, existing_canon):
                # Prefer longer / more complete form as canonical
                if len(name) > len(existing_canon):
                    for k in canonical_of:
                        if canonical_of[k] == existing_canon:
                            canonical_of[k] = name
                    canonical_of[name] = name
                else:
                    canonical_of[name] = existing_canon
                matched = existing_canon
                break
        if matched is None:
            canonical_of[name] = name

    # ── Step 3: Aggregate turn counts to canonical names ─────────────────────
    turns: dict[str, int] = defaultdict(int)
    for raw, count in name_turn_count.items():
        turns[canonical_of.get(raw, raw)] += count

    # Sort by turn count descending (most active speaker first)
    canonical_names = sorted(turns.keys(), key=lambda n: turns[n], reverse=True)
    total_turns     = sum(turns.values())

    # ── Step 4: Confidence scoring ────────────────────────────────────────────
    min_turns = min(turns.values()) if turns else 0
    count     = len(canonical_names)
    confidence = (
        "high"   if count >= 2 and min_turns >= 2 else
        "medium" if count >= 1 and min_turns >= 1 else
        "low"
    )

    return {
        "count":             count,
        "names":             canonical_names,
        "turns_per_speaker": dict(turns),
        "total_turns":       total_turns,
        "formats_detected":  sorted(formats_seen),
        "confidence":        confidence,
    }


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        (
            "Standard 2-speaker",
            "Kunal: Good morning.\nConnie: Hello!\nKunal: Let's begin.\nConnie: Agreed.",
            2, {"Kunal", "Connie"},
        ),
        (
            "Dedup: John Smith / John / Mr. Smith → 1 person",
            "John Smith: Hello.\nJohn: Good point.\nMr. Smith: Agreed.\nJohn Smith: Thanks.",
            1, {"John Smith"},
        ),
        (
            "Japanese: 田中 / 田中部長 → 1 person, 鈴木 → separate",
            "田中: よろしく。\n田中部長: ありがとう。\n鈴木: はい。\n田中: 検討します。",
            2, {"田中部長", "鈴木"},
        ),
        (
            "Zoom format — no cross-line capture",
            "Kunal Bisht  12:34 PM\nHello everyone\nConnie L.  12:35 PM\nHi Kunal!",
            2, {"Kunal Bisht", "Connie L."},
        ),
        (
            "Whisper diarized — SPEAKER_00 ≠ SPEAKER_01",
            "SPEAKER_00: Good morning.\nSPEAKER_01: Hello.\nSPEAKER_00: Let's begin.\nSPEAKER_01: Sure.",
            2, {"SPEAKER_00", "SPEAKER_01"},
        ),
        (
            "PII masked — [NAME_1] ≠ [NAME_2]",
            "[NAME_1]: Good morning.\n[NAME_2]: Hello.\n[NAME_1]: Let's start.\n[NAME_2]: Ready.",
            2, {"[NAME_1]", "[NAME_2]"},
        ),
        (
            "Blocklist: Note/Status/Background are not speakers",
            "Kunal: Meeting start.\nNote: Q3 review.\nStatus: On track.\nConnie: Agreed.\nKunal: Thanks.",
            2, {"Kunal", "Connie"},
        ),
        (
            "40 unique speakers",
            "\n".join(
                f"Speaker{i:02d}: Hello, I am speaker number {i}. Here is my contribution."
                for i in range(40)
            ),
            40, set(),
        ),
        (
            "Mixed JP + EN speakers",
            "田中: 検討いたします。\nKenji: Thank you.\n田中: 難しいですね。\nConnie: I see.\nKenji: Noted.",
            3, {"田中", "Kenji", "Connie"},
        ),
        (
            "Timestamped format",
            "[00:01:23] Kunal: Let's discuss Q3.\n[00:02:15] Priya: Agreed.\n[00:03:00] Kunal: Moving on.",
            2, {"Kunal", "Priya"},
        ),
        (
            "Titles stripped: Dr. Smith / Smith → same",
            "Dr. Smith: Good morning.\nSmith: As I was saying.\nJones: Thank you Dr. Smith.",
            2, {"Dr. Smith", "Jones"},
        ),
    ]

    print("=== Speaker Detector — Full Test Suite ===\n")
    all_pass = True
    for label, transcript, expected_count, expected_subset in tests:
        r      = detect_speakers(transcript)
        c_ok   = r["count"] == expected_count
        n_ok   = expected_subset.issubset(set(r["names"])) if expected_subset else True
        passed = c_ok and n_ok
        if not passed:
            all_pass = False
        sym = "✓" if passed else "✗"
        print(f"  {sym}  {label}")
        if not c_ok:
            print(f"       count: got {r['count']}, expected {expected_count}")
            print(f"       names: {r['names']}")
        if not n_ok:
            print(f"       missing: {expected_subset - set(r['names'])}")
        else:
            top = sorted(r["turns_per_speaker"].items(), key=lambda x: -x[1])[:3]
            print(f"       count={r['count']} conf={r['confidence']} "
                  f"formats={r['formats_detected']} top_speakers={top}")
        print()

    print(f"Result: {'ALL PASS ✓' if all_pass else 'FAILURES FOUND ✗'}")