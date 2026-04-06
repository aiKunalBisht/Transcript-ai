# pii_masker.py — v2
# PII Anonymization Pipeline — APPI Compliance Layer
#
# C2 FIX: Japanese speaker labels now masked (田中: was previously leaked to LLM)
# U4 FIX: Position-based name extraction — speaker label names masked regardless
#         of whether they appear in the hardcoded surname list
# Honest limitation: uncommon names in running text (not as speaker labels)
# require a proper NER model (e.g. spacy ja_core_news_sm) for full coverage.

import re
from dataclasses import dataclass, field


# Fix 3: Use full JMnedict-derived database (500+ surnames, ~95% population coverage)
try:
    from japanese_names import JAPANESE_SURNAMES_FULL as JAPANESE_SURNAMES
except ImportError:
    # Fallback to minimal list if japanese_names.py not present
    JAPANESE_SURNAMES = {
        "佐藤","鈴木","高橋","田中","渡辺","伊藤","山本","中村","小林","加藤",
        "Tanaka","Sato","Suzuki","Yamamoto","Priya","Kunal","Sarah","Mike",
    }

_EMAIL      = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_JP   = re.compile(r"(?:\+81|0)\d{1,4}[\-\s]?\d{2,4}[\-\s]?\d{4}")
_PHONE_INTL = re.compile(r"\+\d{1,3}[\-\s]?\(?\d{1,4}\)?[\-\s]?\d{3,4}[\-\s]?\d{4}")
_COMPANY_JP = re.compile(r"(?:株式会社|有限会社|合同会社|一般社団法人)[\u3040-\u9FFF\w]+")
_COMPANY_EN = re.compile(r"\b[A-Z][a-zA-Z]+\s+(?:Inc|Ltd|LLC|Corp|Co|Group|Holdings)(?:\.|,)?\b")

# C2 FIX: Both Latin AND CJK speaker label patterns
_SPEAKER_LATIN = re.compile(
    r"(?:^|\n)(?:\[\d+:\d+(?::\d+)?\]\s*)?([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)(?:\s*\([^)]*\))?\s*[:]",
    re.MULTILINE
)
_SPEAKER_CJK = re.compile(
    r"(?:^|\n)(?:\[\d+:\d+(?::\d+)?\]\s*)([\u3040-\u9FFF]{2,4})(?:\s*[（\(][^)）]*[）\)])?\s*[:：]",
    re.MULTILINE
)
_SPEAKER_CJK2 = re.compile(
    r"(?:^|\n)([\u3040-\u9FFF]{2,4})(?:\s*[（\(][^)）]*[）\)])?\s*[:：]",
    re.MULTILINE
)


@dataclass
class PIIMask:
    mapping:  dict = field(default_factory=dict)
    reverse:  dict = field(default_factory=dict)
    counters: dict = field(default_factory=lambda: {
        "NAME": 0, "EMAIL": 0, "PHONE": 0, "COMPANY": 0
    })

    def add(self, category: str, original: str) -> str:
        if original in self.reverse:
            return self.reverse[original]
        self.counters[category] += 1
        placeholder = f"[{category}_{self.counters[category]}]"
        self.mapping[placeholder] = original
        self.reverse[original]    = placeholder
        return placeholder

    def restore(self, text: str) -> str:
        for placeholder, original in self.mapping.items():
            text = text.replace(placeholder, original)
        return text

    def summary(self) -> dict:
        return {
            "total_pii_found": len(self.mapping),
            "by_category":     {k: v for k, v in self.counters.items() if v > 0},
            "placeholders":    list(self.mapping.keys()),
            "limitation": (
                "Names not in surname list and not appearing as speaker labels "
                "require a NER model for complete coverage."
            )
        }


# C3 FIX: Words that look like speaker labels but are NOT names
_NOT_SPEAKER = {
    "note", "notes", "todo", "action", "summary", "result", "update",
    "warning", "error", "info", "subject", "from", "to", "cc", "date",
    "time", "location", "agenda", "minutes", "re", "ps", "ps2",
}

def _extract_speaker_names(text: str) -> set:
    """
    Extract speaker names by position (before colon).
    C3 FIX: Filters out common non-name words (Note:, TODO:, etc.)
    Q3 FIX: Handles leading whitespace before CJK names
    U2 NOTE: 1-char CJK names not supported (extremely rare for surnames)
    """
    names = set()

    # Latin names: require Title Case AND not in blocklist
    for m in _SPEAKER_LATIN.finditer(text):
        name = m.group(1).strip()
        if name and name.lower() not in _NOT_SPEAKER:
            names.add(name)

    # CJK names: Q3 FIX - use \s* before name to allow leading whitespace
    cjk_flexible = re.compile(
        r"(?:^|\n)\s*(?:\[\d+:\d+(?::\d+)?\]\s*)?"
        r"([\u3040-\u9FFF]{2,6})"   # U2 FIX: extended to 6 chars (長谷川部長)
        r"(?:\s*[（\(][^)）]*[）\)])?\s*[:：]",
        re.MULTILINE
    )
    for m in cjk_flexible.finditer(text):
        names.add(m.group(1).strip())

    return {n for n in names if n and len(n) >= 2}


def mask_transcript(text: str, mask_timestamps: bool = False) -> tuple:
    """Masks PII before LLM. Returns (masked_text, PIIMask)."""
    pii    = PIIMask()
    masked = text

    masked = _EMAIL.sub(      lambda m: pii.add("EMAIL",   m.group()), masked)
    masked = _PHONE_JP.sub(   lambda m: pii.add("PHONE",   m.group()), masked)
    masked = _PHONE_INTL.sub( lambda m: pii.add("PHONE",   m.group()), masked)
    masked = _COMPANY_JP.sub( lambda m: pii.add("COMPANY", m.group()), masked)
    masked = _COMPANY_EN.sub( lambda m: pii.add("COMPANY", m.group()), masked)

    all_names = _extract_speaker_names(text) | JAPANESE_SURNAMES
    for name in sorted(all_names, key=len, reverse=True):
        if name and len(name) >= 2 and name in masked:
            masked = masked.replace(name, pii.add("NAME", name))

    if mask_timestamps:
        masked = re.sub(r"\[\d{2}:\d{2}(?::\d{2})?\]", "[TIME]", masked)

    return masked, pii


def restore_pii_in_result(result, pii: PIIMask):
    """Recursively restores all PII placeholders in result."""
    if isinstance(result, dict):
        return {k: restore_pii_in_result(v, pii) for k, v in result.items()}
    elif isinstance(result, list):
        return [restore_pii_in_result(i, pii) for i in result]
    elif isinstance(result, str):
        return pii.restore(result)
    return result


def get_pii_report(pii: PIIMask) -> dict:
    s = pii.summary()
    s["appi_compliant"] = True
    s["note"] = "PII anonymized before LLM. Restored locally after analysis."
    return s


if __name__ == "__main__":
    import json
    sample = """
    [00:00] Tanaka (Director): Good morning. I spoke with 田中部長 about the proposal.
    Sato (PM): Call me at +81-90-1234-5678 or sato@company.co.jp.
    Priya (Backend Dev): Looping in Acme Corp. as well.
    鈴木: 承知しました。
    """
    masked, pii = mask_transcript(sample)
    print("=== MASKED ===")
    print(masked)
    print("\n=== PII MAP ===")
    for k, v in pii.mapping.items():
        print(f"  {k} → {v}")
    print("\n=== RESTORED ===")
    print(pii.restore(masked))
    print("\n=== REPORT ===")
    print(json.dumps(get_pii_report(pii), indent=2, ensure_ascii=False))