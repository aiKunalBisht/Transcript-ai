# speaker_normalizer.py — v3.1
# Speaker Normalization + Role Extraction
#
# v3.0 → v3.1 changes:
#
# N1 FIX: extract_all_speakers() now delegates to speaker_detector.detect_speakers().
#         The previous implementation used its own single-pattern regex which had the
#         same timestamp-colon bug fixed in speaker_detector: "Kunal Bisht  12:34 PM"
#         produced "Kunal Bisht  12" as the speaker name because the colon in 12:34
#         was matched before the speaker colon. This made unify_speakers_in_result()
#         build known_speakers from wrong names → _best_match() could never find them.
#
# N2 FIX: _best_match() substring check replaced with _same_person() from
#         speaker_detector. The old "if n in known or known in n" was too broad:
#         "Priya" matched "Priyanka" (substring), "Ali" matched "Alicia" (substring).
#         _same_person() uses token intersection with a 3-char minimum guard so
#         short strings don't false-positive against longer unrelated names.
#
# N3 FIX: Speaker deduplication at the bottom of unify_speakers_in_result() now
#         uses _same_person() instead of exact dict-key matching. Previously
#         "田中" and "Tanaka" (same person, different scripts) were kept as two
#         separate speakers with split talk_time_pct. Now they are merged.
#
# Retained from v3.0:
#   normalize_speaker_name() — strips role suffixes from raw labels
#   extract_role_hint()      — seniority info (separate from name normalization)
#   extract_role_hints()     — role hints for all speakers in transcript
#   KNOWN_NAME_PAIRS         — Kanji↔Romaji cross-script mapping
#   ROLE_ONLY_LABELS         — pure role labels skipped as speaker names

import re
from utils.speaker_detector import detect_speakers

# Common role suffixes to strip
ROLE_PATTERNS = [
    r"\s*\([^)]*\)",
    r"\s*【[^】]*】",
    r"[\s\-]*(さん|様|くん|ちゃん|先生|部長|課長|社長|専務|常務|係長|主任|San|san)\s*",  # \- added for Sato-san
]

# Kanji↔Romaji name mapping
try:
    from japanese_names import ROMAJI_TO_KANJI as _ROMAJI_TO_KANJI_FULL, KANJI_TO_ROMAJI
    KNOWN_NAME_PAIRS = {k: [v, v.capitalize()] for k, v in KANJI_TO_ROMAJI.items()}
except ImportError:
    KNOWN_NAME_PAIRS = {
        "田中": ["tanaka", "Tanaka"],
        "鈴木": ["suzuki", "Suzuki"],
        "山本": ["yamamoto", "Yamamoto"],
        "佐藤": ["sato", "Sato"],
    }
    _ROMAJI_TO_KANJI_FULL = {}

_ROMAJI_TO_KANJI: dict[str, str] = {}
for _kanji, _romaji_list in KNOWN_NAME_PAIRS.items():
    for _r in _romaji_list:
        _ROMAJI_TO_KANJI[_r.lower()] = _kanji

# Pure role-only labels to skip as speaker names
ROLE_ONLY_LABELS = {
    "director", "pm", "manager", "lead", "developer",  # "dev" removed — common South Asian name
    "engineer", "sales", "hr", "cto", "ceo", "coo", "vp",
    "部長", "課長", "係長", "主任", "社長", "専務", "常務",
    "backend", "frontend", "backend dev", "frontend dev",
}


def normalize_speaker_name(raw: str) -> str:
    """
    Strips role suffixes and normalizes a speaker label to the bare name.

    "Tanaka (Director)" → "Tanaka"
    "田中部長"           → "田中"
    "Sato-san"           → "Sato"
    "(PM)"               → ""  (role-only → empty)
    "Dev)"               → "Dev"  (orphaned bracket stripped)
    """
    name = raw.strip()

    # Strip orphaned trailing ) with no opening (
    if name.endswith(")") and "(" not in name:
        name = name[:-1].strip()

    # Strip opening ( with no closing
    if name.startswith("(") and ")" not in name:
        name = name[1:].strip()

    for pattern in ROLE_PATTERNS:
        name = re.sub(pattern, "", name, flags=re.IGNORECASE)
    name = name.strip()

    if name.lower() in ROLE_ONLY_LABELS:
        return ""

    return name


def extract_all_speakers(transcript: str) -> dict:
    """
    Returns {normalized_name: raw_label} for every speaker found in the transcript.

    N1 FIX: Now delegates to speaker_detector.detect_speakers() for robust
    multi-format extraction (Standard, Zoom, Whisper, Timestamped, Masked, CJK).
    The previous regex had the timestamp-colon bug: "Kunal Bisht  12:34 PM"
    produced "Kunal Bisht  12" because "12:34" was matched before the speaker colon.

    Falls back to a minimal local regex if speaker_detector is not yet installed.
    """
    # N1 FIX: delegate to speaker_detector
    try:
        from utils.speaker_detector import detect_speakers
        result = detect_speakers(transcript)
        # detect_speakers returns canonical deduplicated names
        # Map each to itself as raw (role stripping happens in normalize_speaker_name)
        return {name: name for name in result["names"]}
    except ImportError:
        pass

    # Fallback: minimal safe regex (timestamp-colon bug fixed with (?!\d{2}))
    fallback_pattern = re.compile(
        r"(?:\[\d{2}:\d{2}(?::\d{2})?\]\s*)?"
        r"([A-Za-z\u3040-\u9FFF][^\n:：\[\]]{0,40}?)"
        r"\s*[:：](?!\d{2})",          # N1 FIX: reject "12:" in timestamps
        re.MULTILINE,
    )
    speakers: dict[str, str] = {}
    for m in fallback_pattern.finditer(transcript):
        raw = m.group(1).strip()
        if re.match(r"^[0-9]+$", raw):
            continue
        normalized = normalize_speaker_name(raw)
        if normalized and len(normalized) >= 2:
            speakers[normalized] = raw
    return speakers


def unify_speakers_in_result(result: dict, transcript: str) -> dict:
    """
    Unifies speaker names across all result fields after LLM analysis.

    Problem this solves:
    - LLM returns "Tanaka" in sentiment but "田中" in speakers array
    - Hallucination guard flags "Tanaka" as a ghost speaker
    - This function normalizes all names to match the transcript's actual labels

    After this runs:
    - sentiment[].speaker  → normalized and matched to transcript
    - speakers[].name      → normalized and matched
    - action_items[].owner → normalized and matched

    N1/N2/N3 FIX: uses corrected extraction and matching logic throughout.
    """
    known_speakers = extract_all_speakers(transcript)   # N1 FIX
    normalized_names = set(known_speakers.keys())

    # N2 FIX: load _same_person from speaker_detector for safe matching
    try:
        from utils.speaker_detector import _same_person as _sp_func
        _same_person = _sp_func
    except ImportError:
        # Fallback: token-intersection with 3-char guard (mirrors speaker_detector logic)
        def _same_person(a: str, b: str) -> bool:
            if a.lower().strip() == b.lower().strip():
                return True
            ta = set(t for t in re.findall(r"[a-zA-Z']+", a.lower())
                     if t not in {"mr","ms","dr","san","kun"})
            tb = set(t for t in re.findall(r"[a-zA-Z']+", b.lower())
                     if t not in {"mr","ms","dr","san","kun"})
            if not ta or not tb:
                return False
            shorter, longer = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
            if len(shorter) == 1:
                tok = next(iter(shorter))
                return tok in longer and len(tok) >= 3
            return shorter.issubset(longer)

    def _best_match(name: str) -> str:
        """
        Find the best normalized transcript speaker name for an LLM-returned name.

        Priority:
        1. Direct match (exact string)
        2. Case-insensitive match
        3. Kanji↔Romaji cross-script (田中 ↔ Tanaka)
        4. _same_person() token match  ← N2 FIX (replaced loose substring check)
        5. Return as-is (no match found)
        """
        n = normalize_speaker_name(name)

        # 1. Direct
        if n in normalized_names:
            return n

        # 2. Case-insensitive
        for known in normalized_names:
            if known.lower() == n.lower():
                return known

        # 3. Kanji↔Romaji cross-script
        if n.lower() in _ROMAJI_TO_KANJI:
            kanji = _ROMAJI_TO_KANJI[n.lower()]
            if kanji in normalized_names:
                return kanji

        for known in normalized_names:
            romaji_versions = KNOWN_NAME_PAIRS.get(known, [])
            if n in romaji_versions or n.lower() in [r.lower() for r in romaji_versions]:
                return known

        # 4. N2 FIX: _same_person() instead of "n in known or known in n"
        #    Old code: "Priya" matched "Priyanka" (substring false positive)
        #    New code: token intersection with 3-char guard prevents this
        for known in normalized_names:
            if _same_person(n, known):
                return known

        return n  # no match — return as-is

    # Normalize all speaker fields
    for entry in result.get("sentiment", []):
        entry["speaker"] = _best_match(entry.get("speaker", ""))

    for entry in result.get("speakers", []):
        entry["name"] = _best_match(entry.get("name", ""))

    for item in result.get("action_items", []):
        raw_owner = item.get("owner", "")
        if raw_owner and raw_owner.lower() not in ("tbd", "both", "all", "team"):
            item["owner"] = _best_match(raw_owner)

    # ── N3 FIX: deduplicate speakers using _same_person() + cross-script map ──
    # Old code: exact dict-key match only — missed cross-script duplicates
    # Extended: also checks Romaji↔Kanji map so "Tanaka" ↔ "田中" are merged.
    # _same_person() alone cannot do this — it has no knowledge of the name map.
    def _same_person_ext(a: str, b: str) -> bool:
        if _same_person(a, b):
            return True
        # Cross-script: Romaji → Kanji
        if a.lower() in _ROMAJI_TO_KANJI and _ROMAJI_TO_KANJI[a.lower()] == b:
            return True
        if b.lower() in _ROMAJI_TO_KANJI and _ROMAJI_TO_KANJI[b.lower()] == a:
            return True
        return False

    speakers = result.get("speakers", [])
    deduped: list[dict] = []

    for spk in speakers:
        matched = next(
            (d for d in deduped if _same_person_ext(d["name"], spk["name"])),
            None,
        )
        if matched:
            matched["talk_time_pct"] = (
                matched.get("talk_time_pct", 0) + spk.get("talk_time_pct", 0)
            )
        else:
            deduped.append(spk)

    # Clamp talk_time_pct to 100 after merges
    total_pct = sum(s.get("talk_time_pct", 0) for s in deduped)
    if total_pct > 0 and total_pct != 100:
        for s in deduped:
            s["talk_time_pct"] = round(s.get("talk_time_pct", 0) * 100 / total_pct)
    if deduped:
        diff = 100 - sum(s["talk_time_pct"] for s in deduped)
        if diff:
            deduped[0]["talk_time_pct"] += diff

    result["speakers"] = deduped
    return result


# ── ROLE / SENIORITY HINTS — read-only, never modifies name fields ────────────
_SENIORITY_RANK_JA = {
    "社長": 8, "代表": 8, "専務": 7, "常務": 6,
    "部長": 5, "課長": 4, "係長": 3, "主任": 2,
}

_SENIORITY_RANK_EN = {
    "ceo": 8, "president": 8, "coo": 7, "cto": 7,
    "vp": 6, "director": 5, "head": 5,
    "manager": 4, "senior": 3, "lead": 2, "pm": 2,
}


def extract_role_hint(raw_label: str) -> dict:
    """
    Pulls seniority-relevant role from ONE raw speaker label without modifying name.
    Returns {"role": str, "rank": int}. rank=0 means unknown, not junior.

    "Tanaka (Director)" → {"role": "Director", "rank": 5}
    "田中部長"           → {"role": "部長",     "rank": 5}
    "Sato"               → {"role": "",          "rank": 0}
    """
    raw  = raw_label.strip()
    role = ""
    rank = 0

    paren      = re.search(r"[\(（【]([^\)）】]*)[\)）】]", raw)
    paren_text = paren.group(1).strip() if paren else ""

    for text, is_paren in ((paren_text, True), (raw, False)):
        if not text:
            continue
        low = text.lower()
        for word, r in _SENIORITY_RANK_JA.items():
            if word in text and r > rank:
                role, rank = (text if is_paren else word), r
        for word, r in _SENIORITY_RANK_EN.items():
            if re.search(rf"\b{re.escape(word)}\b", low) and r > rank:
                role, rank = (text if is_paren else word), r

    return {"role": role, "rank": rank}


def extract_role_hints(transcript: str) -> dict:
    """
    Returns {normalized_name: {"role": str, "rank": int}} for all speakers.
    Names align exactly with the rest of the pipeline via extract_all_speakers().
    """
    return {name: extract_role_hint(raw)
            for name, raw in extract_all_speakers(transcript).items()}


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    transcript = (
        "Kunal (Lead Engineer): Good morning everyone.\n"
        "Tanaka (Director): ありがとうございます。\n"
        "Sato (PM): We have reviewed the proposal.\n"
        "田中: セキュリティについて確認させてください。\n"
        "Tanaka: I will follow up on that.\n"
    )

    print("=== extract_all_speakers ===")
    speakers = extract_all_speakers(transcript)
    print(json.dumps(speakers, ensure_ascii=False, indent=2))

    print("\n=== extract_role_hints ===")
    hints = extract_role_hints(transcript)
    print(json.dumps(hints, ensure_ascii=False, indent=2))

    # N3 FIX test: 田中 and Tanaka should merge
    result = {
        "sentiment": [
            {"speaker": "Tanaka (Director)", "score": "neutral"},
            {"speaker": "田中",               "score": "neutral"},
            {"speaker": "Sato",               "score": "positive"},
        ],
        "speakers": [
            {"name": "Tanaka",  "talk_time_pct": 30},
            {"name": "田中",    "talk_time_pct": 20},   # same person — should merge
            {"name": "Kunal",   "talk_time_pct": 30},
            {"name": "Sato",    "talk_time_pct": 20},
        ],
        "action_items": [
            {"task": "Review proposal", "owner": "Tanaka (Director)", "deadline": "Friday"},
        ],
    }

    fixed = unify_speakers_in_result(result, transcript)

    print("\n=== unify_speakers_in_result ===")
    print("Speakers after merge:")
    for s in fixed["speakers"]:
        print(f"  {s['name']:<20} talk_time={s['talk_time_pct']}%")

    print("\nSentiment after normalization:")
    for s in fixed["sentiment"]:
        print(f"  {s['speaker']:<20} score={s['score']}")

    print("\nAction item owner:")
    for a in fixed["action_items"]:
        print(f"  owner={a['owner']}")

    # Verify
    speaker_names = [s["name"] for s in fixed["speakers"]]
    tanaka_count  = sum(1 for n in speaker_names if "tanaka" in n.lower() or "田中" in n)
    print(f"\nN3 FIX: 田中 + Tanaka merged into 1 entry: {'✓' if tanaka_count == 1 else '✗'}")
    total_pct = sum(s["talk_time_pct"] for s in fixed["speakers"])
    print(f"talk_time_pct sums to 100: {'✓' if total_pct == 100 else f'✗ ({total_pct})'}")