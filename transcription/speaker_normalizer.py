# speaker_normalizer.py
# Fix 1: Speaker Normalization
#
# Problem: Transcript has "Tanaka (Director):" but LLM returns "Tanaka" or "田中"
# This causes hallucination guard to flag real speakers as ghosts.
#
# Solution: Extract clean name from role+name patterns before any processing.

import re

# Common role suffixes to strip
ROLE_PATTERNS = [
    r"\s*\([^)]*\)",
    r"\s*【[^】]*】",
    r"\s*(さん|様|くん|ちゃん|先生|部長|課長|社長|専務|常務|係長|主任|San|san)\s*",
]

# Fix 3: Use full JMnedict-derived database
try:
    from japanese_names import ROMAJI_TO_KANJI as _ROMAJI_TO_KANJI_FULL, KANJI_TO_ROMAJI
    KNOWN_NAME_PAIRS = {k: [v, v.capitalize()] for k, v in KANJI_TO_ROMAJI.items()}
except ImportError:
    KNOWN_NAME_PAIRS = {"田中":["tanaka","Tanaka"],"鈴木":["suzuki","Suzuki"]}
    _ROMAJI_TO_KANJI_FULL = {}

# Build reverse map
_ROMAJI_TO_KANJI = {}
for kanji, romaji_list in KNOWN_NAME_PAIRS.items():
    for r in romaji_list:
        _ROMAJI_TO_KANJI[r.lower()] = kanji


# Pure role-only labels to skip entirely
ROLE_ONLY_LABELS = {
    "director", "pm", "manager", "lead", "dev", "developer",
    "engineer", "sales", "hr", "cto", "ceo", "coo", "vp",
    "部長", "課長", "係長", "主任", "社長", "専務", "常務",
    "backend", "frontend", "backend dev", "frontend dev",
}


def normalize_speaker_name(raw: str) -> str:
    """
    Strips role suffixes and normalizes speaker name.
    "Tanaka (Director)" → "Tanaka"
    "田中部長"          → "田中"
    "Sato-san"          → "Sato"
    "(PM)"              → "" (role-only)
    "Dev)"              → "Dev" (2.4 FIX: trailing ) stripped)
    """
    name = raw.strip()

    # 2.4 FIX: Strip orphaned trailing ) not matched by role patterns
    # "Dev)" has no opening ( so ROLE_PATTERNS miss it
    if name.endswith(")") and "(" not in name:
        name = name[:-1].strip()

    # Strip opening ( with no closing
    if name.startswith("(") and ")" not in name:
        name = name[1:].strip()

    # Apply role pattern stripping
    for pattern in ROLE_PATTERNS:
        name = re.sub(pattern, "", name, flags=re.IGNORECASE)
    name = name.strip()

    # If what remains is a pure role label, return empty string
    if name.lower() in ROLE_ONLY_LABELS:
        return ""

    return name


def extract_all_speakers(transcript: str) -> dict:
    """
    Extracts all speakers from transcript with their raw and normalized names.
    Returns: {normalized_name: raw_label}

    Handles:
    - "Tanaka (Director):" → "Tanaka"
    - "田中 (部長):" → "田中"
    - "[00:01] Sato:" → "Sato"
    - "Priya:" → "Priya"
    """
    pattern = re.compile(
        r"(?:\[\d{2}:\d{2}(?::\d{2})?\]\s*)?"  # optional timestamp
        r"([^\n:：\[\]]+)"                        # speaker name/role
        r"\s*[:：]"                               # colon
    )
    speakers = {}
    for match in pattern.finditer(transcript):
        raw = match.group(1).strip()
        # Skip timestamps like "00:01"
        if re.match(r"^[0-9]+$", raw):
            continue
        normalized = normalize_speaker_name(raw)
        # Skip empty (role-only) and very short labels
        if normalized and len(normalized) >= 2:
            speakers[normalized] = raw
    return speakers


def unify_speakers_in_result(result: dict, transcript: str) -> dict:
    """
    Main fix: unifies speaker names across all result fields.

    Problem this solves:
    - LLM returns "Tanaka" in sentiment but "田中" in speakers
    - Hallucination guard flags "Tanaka" as ghost because it looks for "田中"
    - This function normalizes ALL names to match transcript labels

    After this runs:
    - sentiment[].speaker → normalized
    - speakers[].name → normalized
    - action_items[].owner → normalized
    """
    known_speakers = extract_all_speakers(transcript)
    normalized_names = set(known_speakers.keys())

    def _best_match(name: str) -> str:
        """Find best normalized match for a name."""
        n = normalize_speaker_name(name)

        # Direct match
        if n in normalized_names:
            return n

        # Case-insensitive match
        for known in normalized_names:
            if known.lower() == n.lower():
                return known

        # Kanji↔Romaji cross-script
        if n.lower() in _ROMAJI_TO_KANJI:
            kanji = _ROMAJI_TO_KANJI[n.lower()]
            if kanji in normalized_names:
                return kanji

        for known in normalized_names:
            if known in _ROMAJI_TO_KANJI:
                romaji_versions = KNOWN_NAME_PAIRS.get(known, [])
                if n in romaji_versions or n.lower() in [r.lower() for r in romaji_versions]:
                    return known

        # Substring match (Tanaka matches Tanaka Director)
        for known in normalized_names:
            if n in known or known in n:
                return known

        return n  # return as-is if no match found

    # Fix sentiment speakers
    for entry in result.get("sentiment", []):
        entry["speaker"] = _best_match(entry.get("speaker", ""))

    # Fix speaker list
    for entry in result.get("speakers", []):
        entry["name"] = _best_match(entry.get("name", ""))

    # Fix action item owners
    for item in result.get("action_items", []):
        raw_owner = item.get("owner", "")
        if raw_owner and raw_owner.lower() not in ("tbd", "both", "all", "team"):
            item["owner"] = _best_match(raw_owner)

    # Deduplicate speakers (田中 and Tanaka are same person)
    speakers = result.get("speakers", [])
    seen = {}
    deduped = []
    for spk in speakers:
        name = spk["name"]
        if name not in seen:
            seen[name] = spk
            deduped.append(spk)
        else:
            # Merge talk time
            seen[name]["talk_time_pct"] = seen[name].get("talk_time_pct", 0) + spk.get("talk_time_pct", 0)
    result["speakers"] = deduped

    return result


# ── ROLE / SENIORITY HINTS — side channel only ────────────────────────────────
# v3 ADD: the keigo-register-shifting and senior-silence patterns in the
# culture doc need seniority info, but normalize_speaker_name() above
# deliberately throws that info away (it's the fix for a hallucination-guard
# bug — see module docstring). So this is a SEPARATE, read-only pass over the
# raw labels. It never writes back into name/owner fields. Anything that
# wants seniority (conversation_dynamics.py) must call this explicitly.

_SENIORITY_RANK_JA = {
    "社長": 8, "代表": 8,
    "専務": 7,
    "常務": 6,
    "部長": 5,
    "課長": 4,
    "係長": 3,
    "主任": 2,
}

_SENIORITY_RANK_EN = {
    "ceo": 8, "president": 8,
    "coo": 7, "cto": 7,
    "vp": 6,
    "director": 5, "head": 5,
    "manager": 4,
    "senior": 3,
    "lead": 2, "pm": 2,
}


def extract_role_hint(raw_label: str) -> dict:
    """
    Pulls seniority-relevant role info out of ONE raw speaker label without
    touching the name. Returns {"role": str, "rank": int}.

    rank 0 means "no title found", not "junior" — absence of a title in a
    transcript label is not evidence the speaker is low-ranked, so callers
    should treat rank 0 as unknown, not as the bottom of the scale.

    "Tanaka (Director)" -> {"role": "Director", "rank": 5}
    "田中部長"           -> {"role": "部長", "rank": 5}
    "Sato"               -> {"role": "", "rank": 0}

    EN titles use \\b word boundaries — without that, short abbreviations
    like "cto"/"coo" false-match as substrings inside ordinary words
    (e.g. "director" contains the letters "cto" in sequence).
    """
    raw = raw_label.strip()
    role = ""
    rank = 0

    paren = re.search(r"[\(（【]([^\)）】]*)[\)）】]", raw)
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
    Returns {normalized_name: {"role": str, "rank": int}} for every speaker
    found in the transcript. Built on top of extract_all_speakers() so the
    names line up exactly with the rest of the pipeline.
    """
    raw_speakers = extract_all_speakers(transcript)  # {normalized: raw}
    return {name: extract_role_hint(raw) for name, raw in raw_speakers.items()}


if __name__ == "__main__":
    transcript = """
    Kunal (Lead Engineer): Good morning everyone.
    Tanaka (Director): ありがとうございます。
    Sato (PM): We have reviewed the proposal.
    田中: セキュリティについて確認させてください。
    """
    speakers = extract_all_speakers(transcript)
    print("Extracted speakers:", speakers)
    print("Role hints:", extract_role_hints(transcript))

    result = {
        "sentiment": [
            {"speaker": "Tanaka (Director)", "score": "neutral"},
            {"speaker": "田中", "score": "neutral"},
            {"speaker": "Sato", "score": "positive"},
        ],
        "speakers": [
            {"name": "Tanaka", "talk_time_pct": 30},
            {"name": "田中", "talk_time_pct": 20},
        ],
        "action_items": [
            {"task": "Review proposal", "owner": "Tanaka (Director)", "deadline": "Friday"}
        ]
    }

    fixed = unify_speakers_in_result(result, transcript)
    import json
    print(json.dumps(fixed, indent=2, ensure_ascii=False))