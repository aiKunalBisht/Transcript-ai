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


if __name__ == "__main__":
    transcript = """
    Kunal (Lead Engineer): Good morning everyone.
    Tanaka (Director): ありがとうございます。
    Sato (PM): We have reviewed the proposal.
    田中: セキュリティについて確認させてください。
    """
    speakers = extract_all_speakers(transcript)
    print("Extracted speakers:", speakers)

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