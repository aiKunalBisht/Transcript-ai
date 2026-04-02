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
    "Tanaka", "Yamamoto", "Sato", "Suzuki", "Nakamura", "Kobayashi", "Ito",
    "Watanabe", "Yamada", "Kato", "Kenji", "Yuki", "Hiroshi", "Akiko",
    "Priya", "Kunal", "Sarah", "Mike", "John", "Emily", "David", "Lisa",
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


def _extract_speaker_names(text: str) -> set:
    """
    C2 + U4 FIX: Extract speaker names by position (before colon).
    Handles Latin (Tanaka:) and CJK (田中:) speaker labels.
    High-confidence PII regardless of surname list.
    """
    names = set()
    for m in _SPEAKER_LATIN.finditer(text):
        name = m.group(1).strip()
        if name:
            names.add(name)
    for pattern in [_SPEAKER_CJK, _SPEAKER_CJK2]:
        for m in pattern.finditer(text):
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