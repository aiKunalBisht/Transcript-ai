#!/usr/bin/env python3
"""
apply_patches.py — TranscriptAI v8.1
Run from your project root:  python apply_patches.py

Patches applied to analysis/analyzer.py:
  1. max_tokens raised        — stops JSON truncation before summary is written
  2. _extractive_summary()   — real no-API summary instead of warning message
  3. _no_api_result()        — uses extractive summary
  4. talk_time char-count    — accurate for English demos (no timestamps)

Companion file required:  analysis/_extractive_summary.py
"""

import pathlib
import sys

TARGET    = pathlib.Path("analysis/analyzer.py")
FN_SOURCE = pathlib.Path("analysis/_extractive_summary.py")

# ── Preflight ─────────────────────────────────────────────────────────────────
for p in (TARGET, FN_SOURCE):
    if not p.exists():
        print(f"ERROR: {p} not found. Run from your project root.")
        sys.exit(1)

print(f"Patching {TARGET} ...\n")
applied = 0


def patch(old: str, new: str, label: str) -> None:
    global applied
    src = TARGET.read_text(encoding="utf-8")
    if old not in src:
        print(f"  \u2717  {label}  (pattern not found \u2014 already applied?)")
        return
    TARGET.write_text(src.replace(old, new, 1), encoding="utf-8")
    print(f"  \u2713  {label}")
    applied += 1


# ─────────────────────────────────────────────────────────────────────────────
# PATCH 1 — Raise max_tokens
# ─────────────────────────────────────────────────────────────────────────────
patch(
    old=(
        "    max_tokens = (\n"
        "         900 if words < 300  else\n"
        "        1100 if words < 800  else\n"
        "        1400 if words < 2000 else\n"
        "        1800\n"
        "    )"
    ),
    new=(
        "    max_tokens = (\n"
        "        1300 if words < 300  else\n"
        "        1700 if words < 800  else\n"
        "        2200 if words < 2000 else\n"
        "        2800\n"
        "    )"
    ),
    label="Patch 1 \u2014 max_tokens raised (900\u21921300 / 1100\u21921700 / 1400\u21922200 / 1800\u21922800)",
)


# ─────────────────────────────────────────────────────────────────────────────
# PATCH 2 — Insert _extractive_summary() before _no_api_result()
# Function code is read from analysis/_extractive_summary.py — no escaping.
# ─────────────────────────────────────────────────────────────────────────────

# Strip the module-level docstring/comment header; keep only the function def
raw = FN_SOURCE.read_text(encoding="utf-8")
lines = raw.splitlines(keepends=True)
fn_start = next(
    i for i, l in enumerate(lines)
    if l.startswith("def _extractive_summary")
)
fn_text = "".join(lines[fn_start:]).rstrip() + "\n\n\n"

patch(
    old='def _no_api_result(text: str, reason: str = "") -> dict:',
    new=fn_text + 'def _no_api_result(text: str, reason: str = "") -> dict:',
    label="Patch 2 \u2014 _extractive_summary() inserted before _no_api_result()",
)


# ─────────────────────────────────────────────────────────────────────────────
# PATCH 3 — _no_api_result() uses extractive summary
# Replaces the three hard-coded warning strings with real transcript content.
# ─────────────────────────────────────────────────────────────────────────────

# OLD block (verbatim from analyzer.py v8.0)
_P3_OLD = (
    "    return {\n"
    '        "meeting_title":  f"{speakers_str} \u2014 Meeting",\n'
    '        "full_summary": (\n'
    '            "\u26a0\ufe0f LLM summary unavailable \u2014 API quota reached. "\n'
    '            "All rule-based analysis (soft rejection, deal outcome, keigo, "\n'
    '            "conversation dynamics) is complete and accurate below."\n'
    "        ),\n"
    '        "summary": [\n'
    '            "\u26a0\ufe0f Summary requires LLM API \u2014 unavailable while quota is reached.",\n'
    '            f"Transcript: {word_count} words | "\n'
    "            f\"{n} speaker{'s' if n > 1 else ''}: {', '.join(names)}.\",\n"
    '            f"Soft rejection risk: {risk_level}. "\n'
    '            f"Full risk analysis, keigo, and deal outcome are operational below.",\n'
    "        ],"
)

# NEW block — uses extractive summary
_P3_NEW = (
    "    # Real extractive summary \u2014 no LLM needed\n"
    "    _ext_full, _ext_bullets = _extractive_summary(text)\n"
    "\n"
    "    return {\n"
    '        "meeting_title": f"{speakers_str} \u2014 Meeting",\n'
    '        "full_summary": (\n'
    "            _ext_full\n"
    "            if _ext_full else\n"
    "            f\"Meeting between {', '.join(names[:3])}. \"\n"
    '            f"Transcript: {word_count} words. "\n'
    '            f"Soft rejection risk: {risk_level}."\n'
    "        ),\n"
    '        "summary": (\n'
    "            _ext_bullets + [\n"
    '                f"Soft rejection risk: {risk_level}. "\n'
    '                f"Full rule-based analysis (keigo, deal outcome, SR) below."\n'
    "            ]\n"
    "            if _ext_bullets else [\n"
    '                f"Transcript: {word_count} words | "\n'
    "                f\"{n} speaker{'s' if n > 1 else ''}: {', '.join(names)}.\",\n"
    '                f"Soft rejection risk: {risk_level}. Full rule-based analysis below.",\n'
    "            ]\n"
    "        ),"
)

patch(old=_P3_OLD, new=_P3_NEW,
      label="Patch 3 \u2014 _no_api_result() uses extractive summary (not warning)")


# ─────────────────────────────────────────────────────────────────────────────
# PATCH 4 — Character-count fallback in _recompute_talk_time_pct()
# English demo transcripts have no timestamps → old code exited early and
# left LLM's inaccurate talk_time_pct values unchanged.
# Char count per speaker is a reliable proxy: Latin syllables ≈ chars,
# JP mora ≈ chars.
# DSA: O(n) line scan + O(speakers) name matching.
# ─────────────────────────────────────────────────────────────────────────────

_P4_OLD = "    if len(timestamped) < 2:\n        return"

# Fullwidth colon used literally — no escaping needed at all
_JP_COLON = "\uff1a"

_P4_NEW = (
    "    if len(timestamped) < 2:\n"
    "        # Char-count fallback: accurate for English/no-timestamp transcripts.\n"
    "        # DSA: O(n) scan + O(speakers) name matching.\n"
    "        _char: dict[str, int] = {}\n"
    f"        _norm = text.replace('{_JP_COLON}', ':')  # fullwidth colon\n"
    "        for _line in _norm.splitlines():\n"
    "            if ':' not in _line:\n"
    "                continue\n"
    "            _idx  = _line.index(':')\n"
    "            _spkr = _line[:_idx].strip()\n"
    "            _cont = _line[_idx + 1:].strip()\n"
    "            if 2 <= len(_spkr) <= 40 and _cont:\n"
    "                _char[_spkr] = _char.get(_spkr, 0) + len(_cont)\n"
    "        if not _char:\n"
    "            return\n"
    "        def _cmatch(nm: str) -> int:\n"
    "            t = nm.lower().strip()\n"
    "            return next((c for n, c in _char.items()\n"
    "                         if t in n.lower() or n.lower() in t), 0)\n"
    "        _raw   = {s['name']: _cmatch(s['name']) for s in speakers}\n"
    "        _total = sum(_raw.values())\n"
    "        if not _total:\n"
    "            return\n"
    "        for s in speakers:\n"
    "            s['talk_time_pct'] = round(_raw[s['name']] * 100 / _total)\n"
    "        _d = 100 - sum(s['talk_time_pct'] for s in speakers)\n"
    "        if _d and speakers:\n"
    "            speakers[0]['talk_time_pct'] += _d\n"
    "        return"
)

patch(old=_P4_OLD, new=_P4_NEW,
      label="Patch 4 \u2014 char-count fallback for English/no-timestamp talk time")


# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 52}")
print(f"Applied {applied}/4 patches.")
if applied == 4:
    print("All done. Verify with:  python analysis/analyzer.py")
elif applied == 0:
    print("No patches applied \u2014 check if already applied or file differs.")
else:
    print(f"{4 - applied} patch(es) skipped \u2014 check output above.")