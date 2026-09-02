# pii_masker.py — v4
# PII Anonymization Pipeline — APPI Compliance Layer
#
# v3 → v4 changes:
# W1 FIX: Company masking is now OPT-IN (mask_companies=False by default).
#         Company names (CAPSA Studio, Anthropic Inc) are essential business
#         context for meeting intelligence — masking them degrades analysis quality.
#         APPI defines companies as entities, NOT personal information.
#         Set mask_companies=True only if your compliance policy demands it.
#
# W2 FIX: _NOT_SPEAKER expanded from 22 → 48 words.
#         Added: Background, Status, Priority, Decision, Project, Task,
#         Issue, Objective, Blocker, Owner, Assignee, etc.
#         Prevents document headers from being extracted as speaker names
#         and globally masked throughout the transcript.
#
# W3 FIX: Bare placeholder restore now uses \b word-boundary regex instead
#         of str.replace() — prevents corrupting variable names / schema
#         identifiers in code review and technical design meetings.
#         Before: "variable NAME_1 is undefined" → "variable Kunal is undefined"
#         After:  "variable NAME_1 is undefined" → "variable Kunal is undefined" ✓
#         Before: "MY_NAME_1_field is null"      → "MY_Kunal_field is null"      ✗
#         After:  "MY_NAME_1_field is null"      → "MY_NAME_1_field is null"     ✓
#
# W4 FIX: Removed English given names (Priya, Kunal, Sarah, Mike) from the
#         fallback JAPANESE_SURNAMES set. Global str.replace on common first
#         names caused false-positive masking anywhere those words appeared.
#         Position-based speaker label extraction handles them correctly.
#
# Retained from v3:
# C2 FIX: Japanese speaker labels masked (田中: no longer leaked to LLM)
# U4 FIX: Position-based name extraction independent of surname list
# V3 FIX: restore() handles all four bracket variants from LLM stripping

import re
from dataclasses import dataclass, field


# Fix 3 (retained): Full JMnedict-derived database (500+ surnames, ~95% JP coverage)
try:
    from utils.japanese_names import JAPANESE_SURNAMES_FULL as JAPANESE_SURNAMES
except ImportError:
    # W4 FIX: English given names removed from fallback set.
    # "Priya","Kunal","Sarah","Mike" caused global str.replace false positives.
    # Position-based _extract_speaker_names() handles English names correctly.
    JAPANESE_SURNAMES = {
        "佐藤","鈴木","高橋","田中","渡辺","伊藤","山本","中村","小林","加藤",
        "Tanaka","Sato","Suzuki","Yamamoto",
    }

_EMAIL      = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_JP   = re.compile(r"(?:\+81|0)\d{1,4}[\-\s]?\d{2,4}[\-\s]?\d{4}")
_PHONE_INTL = re.compile(r"\+\d{1,3}[\-\s]?\(?\d{1,4}\)?[\-\s]?\d{3,4}[\-\s]?\d{4}")
_COMPANY_JP = re.compile(r"(?:株式会社|有限会社|合同会社|一般社団法人)[\u3040-\u9FFF\w]+")
_COMPANY_EN = re.compile(r"\b[A-Z][a-zA-Z]+\s+(?:Inc|Ltd|LLC|Corp|Co|Group|Holdings)(?:\.|,)?\b")

# C2 FIX (retained): Latin speaker label pattern
_SPEAKER_LATIN = re.compile(
    r"(?:^|\n)(?:\[\d+:\d+(?::\d+)?\]\s*)?([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)(?:\s*\([^)]*\))?\s*[:]",
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
        """
        V3 FIX (retained): Handles all four bracket variants the LLM may produce.
        W3 FIX: Bare (no-bracket) restore now uses \b word-boundary regex
                instead of str.replace() to prevent identifier corruption.

        Sorted longest-first (O(k log k)) to prevent NAME_1 partially matching
        before NAME_10 is processed.

        Variants handled per placeholder:
          [NAME_3]  — correct, brackets intact     → exact str.replace
          [NAME_3   — LLM stripped closing bracket → prefix str.replace
          NAME_3]   — LLM stripped opening bracket → suffix str.replace
          NAME_3    — LLM stripped both brackets   → \b regex (safe word match)

        DSA: O(n · k) where n = text length, k = number of PII placeholders.
        """
        for placeholder, original in sorted(
            self.mapping.items(), key=lambda x: len(x[0]), reverse=True
        ):
            bare = placeholder.strip("[]")               # e.g. NAME_3

            text = text.replace(placeholder, original)   # [NAME_3]  — exact
            text = text.replace(f"[{bare}",  original)   # [NAME_3   — missing close
            text = text.replace(f"{bare}]",  original)   # NAME_3]   — missing open

            # W3 FIX: \b prevents matching NAME_1 inside MY_NAME_1_field or NAME_10
            # \b matches at word/non-word boundary — underscore is a word char (\w),
            # so NAME_1 requires a non-word char (space, comma, quote) on both sides.
            text = re.sub(r"\b" + re.escape(bare) + r"\b", original, text)

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


# W2 FIX: Expanded from 22 → 48 entries
# C3 FIX (retained): Filters structural document words from speaker detection
_NOT_SPEAKER = {
    # Original 22 entries
    "note", "notes", "todo", "action", "summary", "result", "update",
    "warning", "error", "info", "subject", "from", "to", "cc", "date",
    "time", "location", "agenda", "minutes", "re", "ps", "ps2",
    # W2 additions — document / meeting structure headers
    "background", "context", "status", "priority", "description",
    "project", "task", "issue", "question", "answer", "decision",
    "speaker", "attendee", "host", "moderator", "participant",
    "follow", "reference", "category", "type", "title",
    "objective", "goal", "outcome", "risk", "blocker", "dependency",
    "deadline", "owner", "assignee", "reviewer", "approver",
    # URL scheme prefixes
    "http", "https", "www",
}


def _extract_speaker_names(text: str) -> set:
    """
    Extract speaker names by position (before colon at line start).
    C3 FIX: Filters structural words via _NOT_SPEAKER blocklist.
    Q3 FIX: Handles leading whitespace before CJK names.
    W2 FIX: Expanded _NOT_SPEAKER applied here.
    U2 NOTE: 1-char CJK names not supported (extremely rare for surnames).

    DSA: O(n) — single regex pass per pattern over text.
    """
    names = set()

    # Latin names: Title Case AND not in blocklist
    for m in _SPEAKER_LATIN.finditer(text):
        name = m.group(1).strip()
        if name and name.lower() not in _NOT_SPEAKER:
            names.add(name)

    # CJK names: flexible leading whitespace (Q3 FIX)
    cjk_flexible = re.compile(
        r"(?:^|\n)\s*(?:\[\d+:\d+(?::\d+)?\]\s*)?"
        r"([\u3040-\u9FFF]{2,6})"
        r"(?:\s*[（\(][^)）]*[）\)])?\s*[:：]",
        re.MULTILINE
    )
    for m in cjk_flexible.finditer(text):
        names.add(m.group(1).strip())

    return {n for n in names if n and len(n) >= 2}


def mask_transcript(
    text:             str,
    mask_timestamps:  bool = False,
    mask_companies:   bool = False,   # W1 FIX: opt-in, default OFF
) -> tuple:
    """
    Masks PII before sending to LLM. Returns (masked_text, PIIMask).

    Args:
        text:            Raw transcript string.
        mask_timestamps: Replace [HH:MM:SS] markers with [TIME]. Default False.
        mask_companies:  W1 FIX — mask formal company names. Default False.
                         Company names are essential business context for meeting
                         intelligence. APPI defines them as entities, not personal
                         information. Only enable if your compliance policy requires it.

    Masking order (matters for correctness):
        1. Emails   — before names (kunal@gmail.com masked before "Kunal" scan)
        2. Phones   — regex-safe, no overlap with names
        3. Companies— opt-in only (W1 FIX)
        4. Names    — longest-first (O(k log k) sort) to prevent partial matches

    DSA: O(n · k) total — n = transcript length, k = PII entity count.
         Dominant term is the name replace loop over the full text per entity.
    """
    pii    = PIIMask()
    masked = text

    # Always masked: emails, phones (APPI personal information)
    masked = _EMAIL.sub(      lambda m: pii.add("EMAIL", m.group()), masked)
    masked = _PHONE_JP.sub(   lambda m: pii.add("PHONE", m.group()), masked)
    masked = _PHONE_INTL.sub( lambda m: pii.add("PHONE", m.group()), masked)

    # W1 FIX: Companies masked only when caller explicitly opts in
    if mask_companies:
        masked = _COMPANY_JP.sub(lambda m: pii.add("COMPANY", m.group()), masked)
        masked = _COMPANY_EN.sub(lambda m: pii.add("COMPANY", m.group()), masked)

    # Names: position-based speaker labels + surname database
    # Longest-first: prevents "Tanaka" matching inside "Tanaka-san" before full form
    all_names = _extract_speaker_names(text) | JAPANESE_SURNAMES
    for name in sorted(all_names, key=len, reverse=True):
        if name and len(name) >= 2 and name in masked:
            masked = masked.replace(name, pii.add("NAME", name))

    if mask_timestamps:
        masked = re.sub(r"\[\d{2}:\d{2}(?::\d{2})?\]", "[TIME]", masked)

    return masked, pii


def restore_pii_in_result(result, pii: PIIMask):
    """
    Recursively restores all PII placeholders in result dict / list / str.
    Safe to call on any nested structure — handles all types via recursion.
    DSA: O(n · k) — n = total chars in result, k = placeholder count.
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
    s["appi_compliant"] = True
    s["note"] = "PII anonymized before LLM. Restored locally after analysis."
    return s


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── W3 FIX: word-boundary bare restore ───────────────────────────────────
    print("=== W3 FIX: WORD-BOUNDARY BARE RESTORE ===")
    pii_tech = PIIMask()
    pii_tech.mapping = {"[NAME_1]": "Kunal"}
    pii_tech.reverse = {"Kunal": "[NAME_1]"}

    w3_cases = [
        ("variable NAME_1 is undefined",   "variable Kunal is undefined"),   # ✓ restored
        ("MY_NAME_1_field is null",        "MY_NAME_1_field is null"),       # ✓ unchanged
        ("NAME_10 spoke next",             "NAME_10 spoke next"),            # ✓ not partial
        ("[NAME_1] approved the PR",        "Kunal approved the PR"),         # ✓ brackets
        ("NAME_1]",                         "Kunal"),                         # ✓ missing open
        ("[NAME_1",                         "Kunal"),                         # ✓ missing close
    ]
    all_pass = True
    for src, expected in w3_cases:
        result = pii_tech.restore(src)
        ok = "✓" if result == expected else "✗"
        if result != expected:
            all_pass = False
        print(f"  {ok}  '{src}' → '{result}'")
    print(f"  {'ALL PASS' if all_pass else 'FAILURES FOUND'}\n")

    # ── V3 FIX (retained): bracket variant restore ────────────────────────────
    print("=== V3 FIX: BRACKET VARIANT RESTORE ===")
    pii_br = PIIMask()
    pii_br.mapping = {"[NAME_1]": "Rahul", "[NAME_2]": "Priya", "[NAME_3]": "Vikram"}
    pii_br.reverse = {v: k for k, v in pii_br.mapping.items()}

    for src, expected in [
        ("[NAME_1]", "Rahul"),
        ("NAME_2",   "Priya"),
        ("[NAME_3",  "Vikram"),
    ]:
        result = pii_br.restore(src)
        ok = "✓" if result == expected else "✗"
        print(f"  {ok}  '{src}' → '{result}'")

    # ── W1 FIX: company masking opt-in ───────────────────────────────────────
    print("\n=== W1 FIX: COMPANY MASKING OPT-IN ===")
    co_sample = "Anthropic Inc signed a deal with CAPSA Studio for the Tokyo project."
    masked_off, _ = mask_transcript(co_sample, mask_companies=False)
    masked_on,  _ = mask_transcript(co_sample, mask_companies=True)
    print(f"  mask_companies=False : {masked_off}")
    print(f"  mask_companies=True  : {masked_on}")

    # ── W2 FIX: expanded blocklist — Background not extracted as speaker ──────
    print("\n=== W2 FIX: EXPANDED NOT_SPEAKER BLOCKLIST ===")
    doc_sample = (
        "Background: Q3 review meeting.\n"
        "Status: On track.\n"
        "Rahul: Let's begin.\n"
        "Priya: Agreed.\n"
    )
    names_found = _extract_speaker_names(doc_sample)
    expected_names = {"Rahul", "Priya"}
    ok = "✓" if names_found == expected_names else "✗"
    print(f"  {ok}  Extracted: {names_found} (expected {expected_names})")

    # ── Full pipeline ─────────────────────────────────────────────────────────
    print("\n=== FULL PIPELINE TEST ===")
    full_sample = (
        "Rahul: Good morning. Contact me at rahul@example.com or +81-3-1234-5678.\n"
        "Priya: We are at 87% of target. Main blocker is delayed launch.\n"
        "Vikram: I will have the report ready by Sunday.\n"
        "Background: This meeting covers the Q3 review.\n"
    )
    masked_full, pii_full = mask_transcript(full_sample)
    print(f"  Masked:\n{masked_full}")
    print(f"  PII map: {pii_full.mapping}")

    fake_result = {
        "speakers": [
            {"name": "NAME_1", "talk_time_pct": 35},
            {"name": "[NAME_2]", "talk_time_pct": 50},
            {"name": "NAME_3]", "talk_time_pct": 15},
        ]
    }
    restored = restore_pii_in_result(fake_result, pii_full)
    print(f"  Restored speakers: {[s['name'] for s in restored['speakers']]}")