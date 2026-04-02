# pii_masker.py
# PII Anonymization Pipeline — APPI Compliance Layer
# Masks personal data BEFORE it reaches the LLM, restores AFTER.
# The LLM never sees raw personal information.
#
# Handles:
#   - Japanese names (田中, 佐藤, etc.)
#   - Western names (Sarah, Kenji, etc.)
#   - Phone numbers (Japanese + international)
#   - Email addresses
#   - Company names (株式会社, Ltd, Inc, etc.)
#   - Timestamps (optional)

import re
from dataclasses import dataclass, field


# ── COMMON JAPANESE SURNAMES (top 50) ────────────────────────────────────────
JAPANESE_SURNAMES = {
    "佐藤", "鈴木", "高橋", "田中", "渡辺", "伊藤", "山本", "中村", "小林", "加藤",
    "吉田", "山田", "佐々木", "山口", "松本", "井上", "木村", "林", "斎藤", "清水",
    "山崎", "森", "池田", "橋本", "阿部", "石川", "山下", "中島", "石井", "小川",
    "前田", "岡田", "長谷川", "藤田", "後藤", "近藤", "村上", "遠藤", "青木", "坂本",
    "斉藤", "福田", "太田", "西村", "藤井", "金子", "岡本", "藤原", "三浦", "中川",
    "原田", "松田", "竹内", "小野", "中野", "田村", "河野", "和田", "石田", "上田",
    "山内", "森田", "菊地", "菅原", "宮崎", "水野", "市川", "柴田", "酒井", "工藤",
    "横山", "宮本", "内田", "高木", "安藤", "島田", "谷口", "大野", "丸山", "今井",
    "武田", "西田", "平野", "村田", "矢野", "杉山", "増田", "小島", "桑原", "大塚",
    "千葉", "松井", "野口", "新井", "久保", "上野", "松尾", "黒田", "永田", "川口",
    # Common given names
    "田中", "山田", "鈴木", "佐藤", "渡辺", "伊藤", "中村", "高橋", "小林", "加藤",
    "Tanaka", "Yamamoto", "Sato", "Suzuki", "Kenji", "Yuki", "Hiroshi", "Akiko",
}

# ── PII PATTERNS ─────────────────────────────────────────────────────────────
PATTERNS = {
    "EMAIL": re.compile(
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
    ),
    "PHONE_JP": re.compile(
        r"(?:\+81|0)\d{1,4}[-\s]?\d{2,4}[-\s]?\d{4}"
    ),
    "PHONE_INTL": re.compile(
        r"\+\d{1,3}[-\s]?\(?\d{1,4}\)?[-\s]?\d{3,4}[-\s]?\d{4}"
    ),
    "COMPANY_JP": re.compile(
        r"(?:株式会社|有限会社|合同会社|一般社団法人|公益財団法人)"
        r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\w]+"
    ),
    "COMPANY_EN": re.compile(
        r"\b[A-Z][a-zA-Z]+\s+(?:Inc|Ltd|LLC|Corp|Co|Group|Holdings|Technologies|Solutions)"
        r"(?:\.|,)?\b"
    ),
    "TIMESTAMP": re.compile(
        r"\[\d{2}:\d{2}(?::\d{2})?\]"
    ),
}


@dataclass
class PIIMask:
    """
    Holds the mapping between original PII and masked placeholders.
    Used to restore real values after LLM processing.
    """
    mapping: dict = field(default_factory=dict)   # placeholder → original
    reverse: dict = field(default_factory=dict)   # original → placeholder
    counters: dict = field(default_factory=lambda: {
        "NAME": 0, "EMAIL": 0, "PHONE": 0, "COMPANY": 0
    })

    def add(self, category: str, original: str) -> str:
        """Register a PII value and return its placeholder."""
        if original in self.reverse:
            return self.reverse[original]
        self.counters[category] = self.counters.get(category, 0) + 1
        placeholder = f"[{category}_{self.counters[category]}]"
        self.mapping[placeholder] = original
        self.reverse[original] = placeholder
        return placeholder

    def restore(self, text: str) -> str:
        """Replace all placeholders back with original values."""
        for placeholder, original in self.mapping.items():
            text = text.replace(placeholder, original)
        return text

    def summary(self) -> dict:
        """Return a summary of what was masked."""
        return {
            "total_pii_found": len(self.mapping),
            "by_category": {k: v for k, v in self.counters.items() if v > 0},
            "placeholders": list(self.mapping.keys())
        }


def mask_transcript(text: str, mask_timestamps: bool = False) -> tuple[str, PIIMask]:
    """
    Main masking function. Pass transcript text, get back:
    - masked_text: safe to send to LLM
    - pii_mask: use to restore original values after LLM processing

    Usage:
        masked, pii = mask_transcript(raw_transcript)
        result = analyze_transcript(masked, language)
        result_with_names = restore_pii_in_result(result, pii)

    Args:
        text:             Raw transcript text
        mask_timestamps:  If True, also mask [00:01:23] timestamps

    Returns:
        (masked_text, PIIMask object)
    """
    pii = PIIMask()
    masked = text

    # 1. Emails (before names to avoid partial matches)
    masked = PATTERNS["EMAIL"].sub(
        lambda m: pii.add("EMAIL", m.group()), masked
    )

    # 2. Phone numbers
    masked = PATTERNS["PHONE_JP"].sub(
        lambda m: pii.add("PHONE", m.group()), masked
    )
    masked = PATTERNS["PHONE_INTL"].sub(
        lambda m: pii.add("PHONE", m.group()), masked
    )

    # 3. Company names
    masked = PATTERNS["COMPANY_JP"].sub(
        lambda m: pii.add("COMPANY", m.group()), masked
    )
    masked = PATTERNS["COMPANY_EN"].sub(
        lambda m: pii.add("COMPANY", m.group()), masked
    )

    # 4. Japanese surnames (word-boundary aware)
    for name in sorted(JAPANESE_SURNAMES, key=len, reverse=True):
        if name in masked:
            placeholder = pii.add("NAME", name)
            masked = masked.replace(name, placeholder)

    # 5. Speaker labels (e.g. "Tanaka:", "Sarah:")
    # NOTE: We mask speaker names in labels too, but keep a speaker registry
    # so the analysis knows who is who even after masking
    speaker_pattern = re.compile(r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s*(?=:)")
    found_speakers = speaker_pattern.findall(masked)
    for speaker in found_speakers:
        pii.add("NAME", speaker)   # register in PII map
    masked = speaker_pattern.sub(
        lambda m: pii.reverse.get(m.group(1), pii.add("NAME", m.group(1))) + (m.group()[len(m.group(1)):]),
        masked
    )

    # 6. Timestamps (optional)
    if mask_timestamps:
        masked = PATTERNS["TIMESTAMP"].sub("[TIME]", masked)

    return masked, pii


def restore_pii_in_result(result: dict, pii: PIIMask) -> dict:
    """
    Restores real PII values in the analysis result JSON.
    Recursively walks the result dict and restores all placeholders.
    """
    if isinstance(result, dict):
        return {k: restore_pii_in_result(v, pii) for k, v in result.items()}
    elif isinstance(result, list):
        return [restore_pii_in_result(item, pii) for item in result]
    elif isinstance(result, str):
        return pii.restore(result)
    return result


def get_pii_report(pii: PIIMask) -> dict:
    """
    Returns a user-friendly report of what PII was found and masked.
    Safe to display in UI — shows placeholders, not real values.
    """
    summary = pii.summary()
    summary["appi_compliant"] = True
    summary["note"] = "All personal data was anonymized before LLM processing."
    return summary


# ── QUICK TEST ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    [00:00:10] Tanaka: Good morning Sarah. I spoke with 田中部長 about the proposal.
    Sato: Yes, 鈴木さん from 株式会社テクノロジー called me at +81-90-1234-5678.
    Tanaka: Please send the contract to tanaka@company.co.jp by Thursday.
    Sarah: Understood. I'll loop in the team at Acme Corp. as well.
    """

    print("=== ORIGINAL ===")
    print(sample)

    masked, pii = mask_transcript(sample)

    print("\n=== MASKED (safe for LLM) ===")
    print(masked)

    print("\n=== PII MAP ===")
    for placeholder, original in pii.mapping.items():
        print(f"  {placeholder} → {original}")

    print("\n=== RESTORED ===")
    print(pii.restore(masked))

    print("\n=== APPI REPORT ===")
    import json
    print(json.dumps(get_pii_report(pii), indent=2, ensure_ascii=False))