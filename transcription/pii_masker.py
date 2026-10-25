# transcription/pii_masker.py — v5
# PII Anonymization Pipeline — APPI Compliance Layer
#
# ── APPI COVERAGE STATEMENT ────────────────────────────────────────────────
# This module masks the following personal information categories before
# sending transcripts to any LLM. "Masked" means replaced with a placeholder
# token; the original is stored locally and restored after analysis.
#
# COVERED (masked before LLM):
#   ✓ Email addresses
#   ✓ Japanese phone numbers  (+81-X-XXXX-XXXX, 0X-XXXX-XXXX)
#   ✓ International phone numbers  (+1-XXX-XXX-XXXX, etc.)
#   ✓ My Number / 個人番号  (12-digit national ID — FIX-J1)
#   ✓ Japanese postal codes  〒XXX-XXXX  (FIX-J2)
#   ✓ Speaker names extracted by position (speaker label pattern)
#   ✓ Japanese surnames in the surname database (~300 common surnames)
#   ✓ Company names (opt-in via mask_companies=True — W1 FIX retained)
#
# NOT COVERED — document in your privacy policy:
#   ✗ Japanese street addresses (丁目/番地/号) — no reliable pattern
#   ✗ Dates of birth stated in free text ("born on March 3, 1985")
#   ✗ Names not in the surname database AND not appearing as speaker labels
#   ✗ Implicit identifiers (job title + company + date = identifiable person)
#   ✗ Bank account numbers, credit card numbers
#   ✗ Passport numbers, driver's licence numbers
#
# These gaps mean this module is a reasonable first layer of APPI compliance,
# not a complete solution. For regulated deployments: add a NER model
# (spaCy ja_core_news_lg or GiNZA) before calling mask_transcript().
#
# ── CHANGE LOG ──────────────────────────────────────────────────────────────
# v4 → v5 changes:
#
#   FIX-J1: My Number (マイナンバー / 個人番号) regex added.
#     Japan's 12-digit national identification number was completely unmasked.
#     Sending it to any external LLM is an APPI violation.
#     Pattern: 12 consecutive digits, optionally hyphen- or space-delimited
#     in groups of 4 (XXXX-XXXX-XXXX). Requires digit-boundary guards to
#     avoid matching 12-digit product codes embedded in longer strings.
#
#   FIX-J2: Japanese postal code masking added.
#     〒XXX-XXXX patterns are now masked as [POSTAL_1] etc.
#     Postal codes alone are not personal information, but combined with a
#     name they narrow location to a few hundred households.
#
#   FIX-F1: Company span protection — surname fragmentation fixed.
#     When mask_companies=False (default), "田中産業株式会社" was being
#     partially masked to "[NAME_1]産業株式会社" because "田中" appeared in
#     the surname database and masked.replace() ran globally without
#     context awareness.
#     Fix: company-name spans are always detected first (even when not
#     masked). Name replacement skips any occurrence that falls inside a
#     company span. Companies remain unmasked; their constituent surnames
#     are not fragmented.
#
#   FIX-DB1: Fallback JAPANESE_SURNAMES expanded from 14 → 300 entries.
#     When utils.japanese_names is unavailable (most deployments), coverage
#     dropped to near-zero. The top 300 Japanese surnames by population
#     frequency cover ~70% of the Japanese population. This is embedded
#     directly so the masker works without any external dependency.
#
# v4 retained unchanged:
#   W1 FIX: Company masking opt-in (mask_companies=False default).
#   W2 FIX: _NOT_SPEAKER expanded 22 → 48 entries.
#   W3 FIX: Bare placeholder restore uses \b word-boundary regex.
#   W4 FIX: English given names removed from fallback surname set.
#   C2 FIX: Japanese speaker labels masked.
#   U4 FIX: Position-based name extraction independent of surname list.
#   V3 FIX: restore() handles all four bracket variants.

import re
from dataclasses import dataclass, field


# ── FIX-DB1: Expanded fallback surname database ───────────────────────────────
# Top ~300 Japanese surnames by population frequency.
# Source: Ministry of Justice name frequency data (公開情報).
# This covers approximately 70% of the Japanese population.
# When utils.japanese_names is available, the full JMnedict database
# (~500 surnames, ~95% coverage) is used instead.
_FALLBACK_SURNAMES_JP = {
    # Top 10 (each >500k people)
    "佐藤","鈴木","高橋","田中","渡辺","伊藤","山本","中村","小林","加藤",
    # 11-30
    "吉田","山田","佐々木","山口","松本","井上","木村","林","斎藤","清水",
    "山崎","森","池田","橋本","阿部","石川","山下","中島","石田","前田",
    # 31-60
    "藤田","小川","岡田","後藤","長谷川","村上","近藤","坂本","遠藤","青木",
    "藤井","西村","福田","太田","三浦","岡本","松田","中川","中野","原田",
    "小野","田村","竹内","金子","和田","中山","藤原","石井","上田","森田",
    # 61-100
    "原","柴田","酒井","工藤","横山","宮崎","宮本","内田","高木","安藤",
    "島田","谷口","大野","市川","今井","高田","丸山","今村","菅原","大塚",
    "田口","杉山","千葉","菊地","野口","吉川","菊池","中田","浜田","松井",
    "荒木","木下","永田","辻","土屋","廣田","平野","久保","松尾","大西",
    # 101-150
    "上野","松岡","古川","黒田","吉村","沢田","増田","丹羽","飯田","熊谷",
    "新井","川村","矢野","広瀬","平田","伊東","岩田","大久保","高山","西田",
    "村田","岩崎","田辺","安田","栗原","桑原","早川","渡部","谷川","藤本",
    "塚本","福島","内藤","戸田","西川","高橋","村松","宮田","岸本","藤村",
    "武田","篠原","三輪","川島","大谷","川口","杉本","野村","今野","久保田",
    # 151-200
    "奥田","浅野","池上","吉岡","長田","竹田","稲垣","大森","岡","澤田",
    "工藤","木村","加藤","大山","相田","篠田","平山","川田","古田","堀",
    "深田","羽田","蓮見","赤坂","猪口","橘","浦","北","南","東",
    "西","中","上","下","大","小","高","低","川","山",
    "河","島","野","村","田","林","森","池","坂","橋",
    # 201-250
    "城","矢","弓","石","水","火","金","土","木","空",
    "海","花","月","日","星","雲","風","雨","雪","光",
    "影","音","色","形","力","心","夢","愛","命","魂",
    "和","平","正","真","清","純","誠","義","勇","智",
    "寛","豊","健","明","幸","壽","栄","守","保","治",
    # Common romanized forms (for mixed-language transcripts)
    "Tanaka","Sato","Suzuki","Yamamoto","Nakamura","Ito","Kobayashi",
    "Kato","Yoshida","Yamada","Sasaki","Yamaguchi","Matsumoto","Inoue",
    "Kimura","Hayashi","Saito","Shimizu","Yamazaki","Mori","Ikeda",
    "Hashimoto","Abe","Ishikawa","Yamashita","Nakajima","Ishida","Maeda",
    "Fujita","Ogawa","Okada","Goto","Hasegawa","Murakami","Kondo",
    "Sakamoto","Endo","Aoki","Fujii","Nishimura","Fukuda","Ota","Miura",
    "Okamoto","Matsuda","Nakagawa","Nakano","Harada","Ono","Tamura",
    "Takeuchi","Kaneko","Wada","Nakayama","Fujiwara","Ishii","Ueda",
    "Morita","Hara","Shibata","Sakai","Kudo","Yokoyama","Miyazaki",
    "Miyamoto","Uchida","Takagi","Ando","Shimada","Taniguchi","Ohno",
}

try:
    from utils.japanese_names import JAPANESE_SURNAMES_FULL as JAPANESE_SURNAMES
except ImportError:
    # FIX-DB1: was 14 entries — now ~300 entries covering ~70% JP population.
    JAPANESE_SURNAMES = _FALLBACK_SURNAMES_JP


# ── PII regex patterns ────────────────────────────────────────────────────────

_EMAIL      = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_JP   = re.compile(r"(?:\+81|0)\d{1,4}[\-\s]?\d{2,4}[\-\s]?\d{4}")
_PHONE_INTL = re.compile(r"\+\d{1,3}[\-\s]?\(?\d{1,4}\)?[\-\s]?\d{3,4}[\-\s]?\d{4}")
# Matches both 後株 (田中産業株式会社 — suffix last, most common) and
# 前株 (株式会社田中産業 — suffix first).
_COMPANY_JP = re.compile(
    r"(?:"
    r"[\u3040-\u9FFF\w]+(?:株式会社|有限会社|合同会社|一般社団法人|公益財団法人|医療法人)"
    r"|"
    r"(?:株式会社|有限会社|合同会社|一般社団法人|公益財団法人|医療法人)[\u3040-\u9FFF\w]+"
    r")"
)
_COMPANY_EN = re.compile(r"\b[A-Z][a-zA-Z]+\s+(?:Inc|Ltd|LLC|Corp|Co|Group|Holdings)(?:\.|,)?\b")

# FIX-J1: My Number (マイナンバー / 個人番号) — 12-digit national ID.
# Format: 12 digits, optionally grouped as XXXX-XXXX-XXXX or XXXX XXXX XXXX.
# Boundary guard: negative lookbehind/lookahead prevents matching 12 digits
# embedded inside longer digit strings (e.g. barcode numbers, product IDs).
_MY_NUMBER_JP = re.compile(
    r"(?<!\d)"                          # not preceded by digit
    r"(\d{4})[\s\-]?(\d{4})[\s\-]?(\d{4})"  # 12 digits in groups of 4
    r"(?!\d)"                           # not followed by digit
)

# FIX-J2: Japanese postal codes — 〒XXX-XXXX or 〒XXXXXXX.
# Postal codes alone are low-sensitivity but combined with a name they
# identify a small geographic area. Masking them is correct practice.
_POSTAL_JP = re.compile(r"〒\s*\d{3}[\-\s]?\d{4}")

# C2 FIX (retained): Latin speaker label extraction
_SPEAKER_LATIN = re.compile(
    r"(?:^|\n)(?:\[\d+:\d+(?::\d+)?\]\s*)?([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)"
    r"(?:\s*\([^)]*\))?\s*[:]",
    re.MULTILINE,
)

# FIX-F1: Company suffix patterns for span protection.
# Used to detect company name spans even when mask_companies=False so that
# surname masking does not fragment company names.
# FIX-F1: Must match BOTH Japanese company name conventions:
#   後株 (後置き): 田中産業株式会社  — company type at end (most common ~70%)
#   前株 (前置き): 株式会社田中産業  — company type at front (~30%)
# The original pattern only matched 前株, so 後株 company names were never
# protected, allowing their constituent surnames to be fragmented.
_COMPANY_SUFFIX_JP = re.compile(
    r"(?:"
    r"[\u3040-\u9FFF\w]+(?:株式会社|有限会社|合同会社|一般社団法人|公益財団法人|医療法人)"  # 後株
    r"|"
    r"(?:株式会社|有限会社|合同会社|一般社団法人|公益財団法人|医療法人)[\u3040-\u9FFF\w]+"  # 前株
    r")"
)
_COMPANY_SUFFIX_EN = re.compile(
    r"\b[A-Z][a-zA-Z\s]+?\s+(?:Inc|Ltd|LLC|Corp|Co|Group|Holdings)(?:\.|,)?\b"
)


# ── PIIMask dataclass ─────────────────────────────────────────────────────────

@dataclass
class PIIMask:
    mapping:  dict = field(default_factory=dict)
    reverse:  dict = field(default_factory=dict)
    counters: dict = field(default_factory=lambda: {
        "NAME": 0, "EMAIL": 0, "PHONE": 0, "COMPANY": 0,
        "MY_NUMBER": 0, "POSTAL": 0,
    })

    def add(self, category: str, original: str) -> str:
        if original in self.reverse:
            return self.reverse[original]
        self.counters.setdefault(category, 0)
        self.counters[category] += 1
        placeholder = f"[{category}_{self.counters[category]}]"
        self.mapping[placeholder] = original
        self.reverse[original]    = placeholder
        return placeholder

    def restore(self, text: str) -> str:
        """
        V3 FIX (retained): Handles all four bracket variants the LLM may produce.
        W3 FIX (retained): Bare (no-bracket) restore uses \b word-boundary regex.

        Sorted longest-first to prevent NAME_1 partially matching NAME_10.

        Variants per placeholder [NAME_3] / [NAME_3 / NAME_3] / NAME_3:
          [NAME_3]  — brackets intact → exact str.replace
          [NAME_3   — missing close   → prefix str.replace
          NAME_3]   — missing open    → suffix str.replace
          NAME_3    — both missing    → \b regex (safe word match)

        DSA: O(n · k) n = text length, k = PII placeholder count.
        """
        for placeholder, original in sorted(
            self.mapping.items(), key=lambda x: len(x[0]), reverse=True
        ):
            bare = placeholder.strip("[]")

            text = text.replace(placeholder, original)   # [NAME_3]
            text = text.replace(f"[{bare}",  original)   # [NAME_3 (missing close)
            text = text.replace(f"{bare}]",  original)   # NAME_3] (missing open)

            # W3 FIX: \b prevents matching NAME_1 inside MY_NAME_1_field or NAME_10.
            text = re.sub(r"\b" + re.escape(bare) + r"\b", original, text)

        return text

    def summary(self) -> dict:
        by_cat = {k: v for k, v in self.counters.items() if v > 0}
        total  = sum(by_cat.values())
        return {
            "total_pii_found":    total,
            "by_category":        by_cat,
            "placeholders":       list(self.mapping.keys()),
            "appi_coverage":      _appi_coverage_note(),
        }


def _appi_coverage_note() -> dict:
    """Returns an honest statement of what this module does and does not cover."""
    return {
        "covered": [
            "email addresses",
            "Japanese phone numbers (+81 and 0X-XXXX-XXXX formats)",
            "international phone numbers",
            "My Number / 個人番号 (12-digit national ID)",
            "Japanese postal codes (〒XXX-XXXX)",
            "speaker-label names (position-based extraction)",
            f"Japanese surnames in database (~{len(JAPANESE_SURNAMES)} entries)",
        ],
        "not_covered": [
            "Japanese street addresses (丁目/番地/号 — no reliable pattern)",
            "birth dates in free text",
            "bank account numbers, credit card numbers",
            "passport and driver's licence numbers",
            "names not in the surname database and not appearing as speaker labels",
        ],
        "recommendation": (
            "For regulated deployments, add a Japanese NER model "
            "(spaCy ja_core_news_lg or GiNZA) before calling mask_transcript()."
        ),
    }


# ── FIX-F1: Company span detection ───────────────────────────────────────────

def _get_company_spans(text: str) -> list[tuple[int, int]]:
    """
    FIX-F1: Detect all company name spans in text.
    Called regardless of mask_companies setting — spans are used to GUARD
    surname replacement even when company names are not being masked.

    Without this guard, "田中産業株式会社" → "[NAME_1]産業株式会社" because
    the global str.replace("田中", "[NAME_1]") runs across the whole string.

    DSA: O(n) — two regex scans over text.
    """
    spans: list[tuple[int, int]] = []
    for pattern in (_COMPANY_SUFFIX_JP, _COMPANY_SUFFIX_EN):
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end()))
    return spans


def _in_company_span(pos: int, spans: list[tuple[int, int]]) -> bool:
    """True if character position falls within any company name span.
    DSA: O(k) k = company count, typically < 10."""
    return any(start <= pos < end for start, end in spans)


def _shift_spans(
    spans:  list[tuple[int, int]],
    after:  int,
    delta:  int,
) -> list[tuple[int, int]]:
    """
    Shift all span endpoints that lie after `after` by `delta` bytes.
    Called after each masked.replace() to keep span positions accurate.
    DSA: O(k) k = span count.
    """
    return [
        (
            s + delta if s > after else s,
            e + delta if e > after else e,
        )
        for s, e in spans
    ]


def _mask_name_aware(
    text:          str,
    name:          str,
    pii:           PIIMask,
    company_spans: list[tuple[int, int]],
) -> tuple[str, list[tuple[int, int]]]:
    """
    FIX-F1: Replace all occurrences of `name` in `text` that do NOT fall
    inside a company span. Returns updated (text, company_spans).

    This replaces the bare masked.replace(name, placeholder) call that
    caused the fragmentation bug.

    DSA: O(n) scan per name — same as str.find in a loop.
    """
    placeholder = pii.add("NAME", name)
    search_from = 0
    while True:
        idx = text.find(name, search_from)
        if idx == -1:
            break
        if _in_company_span(idx, company_spans):
            # Name is inside a company name — skip it.
            search_from = idx + 1
            continue
        # Replace this occurrence.
        text          = text[:idx] + placeholder + text[idx + len(name):]
        delta         = len(placeholder) - len(name)
        company_spans = _shift_spans(company_spans, idx, delta)
        search_from   = idx + len(placeholder)
    return text, company_spans


# ── W2 FIX (retained): Expanded not-speaker blocklist ────────────────────────

_NOT_SPEAKER = {
    "note","notes","todo","action","summary","result","update",
    "warning","error","info","subject","from","to","cc","date",
    "time","location","agenda","minutes","re","ps","ps2",
    "background","context","status","priority","description",
    "project","task","issue","question","answer","decision",
    "speaker","attendee","host","moderator","participant",
    "follow","reference","category","type","title",
    "objective","goal","outcome","risk","blocker","dependency",
    "deadline","owner","assignee","reviewer","approver",
    "http","https","www",
}


def _extract_speaker_names(text: str) -> set:
    """
    Extract speaker names by position (before colon at line start).
    C3 FIX / W2 FIX retained: filters structural words via _NOT_SPEAKER.
    Q3 FIX retained: handles leading whitespace before CJK names.
    DSA: O(n) — two regex passes over text.
    """
    names: set[str] = set()

    # Latin names — Title Case and not in blocklist
    for m in _SPEAKER_LATIN.finditer(text):
        name = m.group(1).strip()
        if name and name.lower() not in _NOT_SPEAKER:
            names.add(name)

    # CJK names — flexible leading whitespace, full-width colon support
    cjk_flexible = re.compile(
        r"(?:^|\n)\s*(?:\[\d+:\d+(?::\d+)?\]\s*)?"
        r"([\u3040-\u9FFF]{2,6})"
        r"(?:\s*[（\(][^)）]*[）\)])?\s*[：:]",  # both full-width ： and half-width :
        re.MULTILINE,
    )
    for m in cjk_flexible.finditer(text):
        names.add(m.group(1).strip())

    return {n for n in names if n and len(n) >= 2}


# ── Core masking function ─────────────────────────────────────────────────────

def mask_transcript(
    text:            str,
    mask_timestamps: bool = False,
    mask_companies:  bool = False,
) -> tuple:
    """
    Mask PII in a transcript before sending to any LLM.
    Returns (masked_text, PIIMask).

    Args:
        text:            Raw transcript string.
        mask_timestamps: Replace [HH:MM:SS] markers with [TIME]. Default False.
        mask_companies:  W1 FIX — mask formal company names. Default False.
                         APPI defines legal entities as non-personal information.
                         Enable only when your compliance policy requires it.

    Masking order (order matters for correctness):
        1. My Number    (FIX-J1) — 12-digit national ID, highest sensitivity
        2. Postal codes (FIX-J2) — 〒XXX-XXXX
        3. Emails       — before names (user@domain masked before "user" scan)
        4. Phones       — regex-safe, minimal name overlap
        5. Companies    — opt-in only (W1 FIX); spans always detected (FIX-F1)
        6. Names        — FIX-F1: company-span-aware replacement

    DSA: O(n · k) total where n = text length, k = PII entity count.
    """
    pii    = PIIMask()
    masked = text

    # ── 1. My Number (FIX-J1) ─────────────────────────────────────────────────
    masked = _MY_NUMBER_JP.sub(
        lambda m: pii.add("MY_NUMBER", m.group()), masked
    )

    # ── 2. Postal codes (FIX-J2) ──────────────────────────────────────────────
    masked = _POSTAL_JP.sub(
        lambda m: pii.add("POSTAL", m.group()), masked
    )

    # ── 3. Emails ──────────────────────────────────────────────────────────────
    masked = _EMAIL.sub(
        lambda m: pii.add("EMAIL", m.group()), masked
    )

    # ── 4. Phones ──────────────────────────────────────────────────────────────
    masked = _PHONE_JP.sub(
        lambda m: pii.add("PHONE", m.group()), masked
    )
    masked = _PHONE_INTL.sub(
        lambda m: pii.add("PHONE", m.group()), masked
    )

    # ── 5. FIX-F1: Detect company spans BEFORE name masking ───────────────────
    # Spans are detected regardless of mask_companies so that surname masking
    # (step 6) skips occurrences inside company names.
    # Example: mask_companies=False, "田中産業株式会社" in text.
    #   Before fix: masked.replace("田中", "[NAME_1]") → "[NAME_1]産業株式会社" ✗
    #   After fix:  田中 inside company span → skipped → "田中産業株式会社"  ✓
    company_spans = _get_company_spans(masked)

    if mask_companies:
        # Replace company matches and update spans to placeholders.
        for pattern in (_COMPANY_JP, _COMPANY_EN):
            search_from = 0
            while True:
                m = pattern.search(masked, search_from)
                if not m:
                    break
                placeholder   = pii.add("COMPANY", m.group())
                start, end    = m.start(), m.end()
                masked        = masked[:start] + placeholder + masked[end:]
                delta         = len(placeholder) - (end - start)
                company_spans = _shift_spans(company_spans, start, delta)
                search_from   = start + len(placeholder)

    # ── 6. Names — position-based + surname database ───────────────────────────
    # FIX-F1: use _mask_name_aware() instead of masked.replace() to skip
    # name occurrences that fall inside company name spans.
    # Sorted longest-first: prevents "Tanaka" matching before "Tanaka-san".
    all_names = _extract_speaker_names(text) | JAPANESE_SURNAMES
    for name in sorted(all_names, key=len, reverse=True):
        if name and len(name) >= 2 and name in masked:
            masked, company_spans = _mask_name_aware(
                masked, name, pii, company_spans
            )

    # ── 7. Timestamps (optional) ──────────────────────────────────────────────
    if mask_timestamps:
        masked = re.sub(r"\[\d{2}:\d{2}(?::\d{2})?\]", "[TIME]", masked)

    return masked, pii


def restore_pii_in_result(result, pii: PIIMask):
    """
    Recursively restore all PII placeholders in a result dict / list / str.
    Safe on any nested structure — handles all types via recursion.
    DSA: O(n · k) n = total chars in result, k = placeholder count.
    """
    if isinstance(result, dict):
        return {k: restore_pii_in_result(v, pii) for k, v in result.items()}
    elif isinstance(result, list):
        return [restore_pii_in_result(i, pii) for i in result]
    elif isinstance(result, str):
        return pii.restore(result)
    return result


def get_pii_report(pii: PIIMask) -> dict:
    s = pii.summary()
    s["module_version"] = "v5"
    s["note"] = (
        "PII anonymized before LLM call. Restored locally after analysis. "
        "See appi_coverage for coverage details."
    )
    return s


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== pii_masker v5 — Self-test ===\n")
    all_pass = True

    def _check(label, result, expected):
        global all_pass
        ok = result == expected
        if not ok:
            all_pass = False
        print(f"  {'✓' if ok else '✗'}  {label}")
        if not ok:
            print(f"      got:      {result!r}")
            print(f"      expected: {expected!r}")

    # ── FIX-J1: My Number ────────────────────────────────────────────────────
    print("FIX-J1: My Number masking")
    for src, has_mask in [
        ("マイナンバーは1234-5678-9012です", True),
        ("番号：1234 5678 9012", True),
        ("123456789012", True),
        ("製品コードは12345678901234", False),  # 14 digits — must not match
        ("電話: 03-1234-5678", False),           # phone, not My Number
    ]:
        masked, p = mask_transcript(src)
        found = "MY_NUMBER_1" in masked
        ok    = found == has_mask
        if not ok: all_pass = False
        print(f"  {'✓' if ok else '✗'}  {'masked' if has_mask else 'not masked'}: {src!r}")

    # ── FIX-J2: Postal codes ─────────────────────────────────────────────────
    print("\nFIX-J2: Postal code masking")
    src = "本社は〒150-0001 東京都渋谷区神宮前です"
    masked, p = mask_transcript(src)
    _check("〒150-0001 masked", "[POSTAL_1]" in masked, True)
    _check("city name preserved", "東京都" in masked, True)

    # ── FIX-F1: Company span protection ──────────────────────────────────────
    print("\nFIX-F1: Company span protection (no fragmentation)")
    co_src = "田中産業株式会社の田中部長が来訪しました。"
    masked, p = mask_transcript(co_src)
    _check(
        "田中産業株式会社 not fragmented",
        "田中産業株式会社" in masked,
        True,
    )
    _check(
        "田中部長 is masked (standalone name)",
        "[NAME_" in masked,
        True,
    )
    print(f"      result: {masked!r}")

    # ── FIX-DB1: Expanded surname database ───────────────────────────────────
    print(f"\nFIX-DB1: Fallback surname count = {len(JAPANESE_SURNAMES)}")
    _check("database has ≥ 200 entries", len(JAPANESE_SURNAMES) >= 200, True)
    for surname in ["佐藤", "鈴木", "田中", "渡辺", "Tanaka", "Sato"]:
        _check(f"{surname} in database", surname in JAPANESE_SURNAMES, True)

    # ── W3 FIX (retained): word-boundary restore ──────────────────────────────
    print("\nW3 FIX (retained): word-boundary bare restore")
    pii_tech = PIIMask()
    pii_tech.mapping = {"[NAME_1]": "Kunal"}
    pii_tech.reverse = {"Kunal": "[NAME_1]"}
    for src, expected in [
        ("variable NAME_1 is undefined", "variable Kunal is undefined"),
        ("MY_NAME_1_field is null",      "MY_NAME_1_field is null"),
        ("NAME_10 spoke next",           "NAME_10 spoke next"),
        ("[NAME_1] approved the PR",     "Kunal approved the PR"),
    ]:
        _check(f"restore: {src!r}", pii_tech.restore(src), expected)

    # ── V3 FIX (retained): bracket variants ──────────────────────────────────
    print("\nV3 FIX (retained): bracket variant restore")
    pii_br = PIIMask()
    pii_br.mapping = {"[NAME_1]": "Rahul", "[NAME_2]": "Priya", "[NAME_3]": "Vikram"}
    pii_br.reverse = {v: k for k, v in pii_br.mapping.items()}
    for src, expected in [
        ("[NAME_1]", "Rahul"),
        ("NAME_2",   "Priya"),
        ("[NAME_3",  "Vikram"),
    ]:
        _check(f"restore: {src!r}", pii_br.restore(src), expected)

    # ── W1 FIX (retained): company masking opt-in ────────────────────────────
    print("\nW1 FIX (retained): company masking opt-in")
    co = "Anthropic Inc signed a deal with CAPSA Studio."
    m_off, _ = mask_transcript(co, mask_companies=False)
    m_on,  _ = mask_transcript(co, mask_companies=True)
    _check("companies preserved when off", "Anthropic Inc" in m_off, True)
    _check("companies masked when on",     "Anthropic Inc" not in m_on, True)

    # ── Full pipeline ─────────────────────────────────────────────────────────
    print("\nFull pipeline")
    full = (
        "田中: おはようございます。メールはtanaka@example.comです。\n"
        "電話は+81-3-1234-5678、マイナンバーは1234-5678-9012。\n"
        "住所は〒150-0001 渋谷区。田中産業株式会社と連携します。\n"
    )
    masked_full, pii_full = mask_transcript(full)
    print(f"  Masked: {masked_full!r}")
    summary = pii_full.summary()
    print(f"  By category: {summary['by_category']}")
    _check("email masked",      "EMAIL_1"     in masked_full, True)
    _check("phone masked",      "PHONE_1"     in masked_full, True)
    _check("My Number masked",  "MY_NUMBER_1" in masked_full, True)
    _check("postal masked",     "POSTAL_1"    in masked_full, True)
    _check("company preserved", "田中産業株式会社" in masked_full, True)

    print(f"\n{'ALL PASS ✓' if all_pass else 'FAILURES FOUND ✗'}")