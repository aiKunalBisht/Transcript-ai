"""
apply_fixes.py — TranscriptAI v8.3 surgical patcher
Run from your project root:  python apply_fixes.py

Makes exactly 2 changes to analysis/analyzer.py:
  P2-FIX: Stage 9 — keigo rule-based pre-check before MeCab
  P6-FIX: _sentiment_backstop_block — FORMAL keigo + frustrated → politely_evasive
"""

import pathlib, sys, re

SRC = pathlib.Path("analysis/analyzer.py")
if not SRC.exists():
    print(f"[ERROR] {SRC} not found. Run from project root.", file=sys.stderr)
    sys.exit(1)

text = SRC.read_text(encoding="utf-8")

# ──────────────────────────────────────────────────────────────────────────────
# PATCH 1 — Stage 9: replace MeCab-only keigo with rule-based + MeCab fallback
# ──────────────────────────────────────────────────────────────────────────────
OLD_STAGE9 = """\
    # Stage 9: MeCab keigo
    if result.get("japan_insights", {}).get("keigo_level", "unknown") == "unknown":
        try:
            from analysis.japanese_tokenizer import (
                get_keigo_level, MECAB_AVAILABLE
            )
            if MECAB_AVAILABLE:
                result["japan_insights"]["keigo_level"]  = get_keigo_level(text)
                result["japan_insights"]["keigo_source"] = "mecab"
        except ImportError:
            pass
        except Exception as e:
            print(f"[TRANSCRIPT_AI] japanese_tokenizer failed: {e}",
                  file=sys.stderr, flush=True)"""

NEW_STAGE9 = """\
    # Stage 9: Keigo detection — rule-based pre-check, MeCab fallback
    # P2-FIX: Deterministic marker scan runs first.
    # DSA: O(k) where k = len(_KEIGO_MARKERS) = 7 — constant per transcript.
    if result.get("japan_insights", {}).get("keigo_level", "unknown") == "unknown":
        _KEIGO_MARKERS = [
            "でございます", "いたします", "おります",
            "拝受", "申し上げ", "ございます", "いただき",
        ]
        _hits = [m for m in _KEIGO_MARKERS if m in text]
        if len(_hits) >= 3:
            result["japan_insights"]["keigo_level"]          = "FORMAL"
            result["japan_insights"]["keigo_source"]         = "rule_based"
            result["japan_insights"]["keigo_markers_matched"] = _hits
        elif len(_hits) >= 1:
            result["japan_insights"]["keigo_level"]          = "SEMI-FORMAL"
            result["japan_insights"]["keigo_source"]         = "rule_based"
            result["japan_insights"]["keigo_markers_matched"] = _hits
        else:
            # MeCab fallback — only when rule-based finds nothing
            try:
                from analysis.japanese_tokenizer import (
                    get_keigo_level, MECAB_AVAILABLE
                )
                if MECAB_AVAILABLE:
                    result["japan_insights"]["keigo_level"]  = get_keigo_level(text)
                    result["japan_insights"]["keigo_source"] = "mecab"
            except ImportError:
                pass
            except Exception as e:
                print(f"[TRANSCRIPT_AI] japanese_tokenizer failed: {e}",
                      file=sys.stderr, flush=True)"""

if OLD_STAGE9 in text:
    text = text.replace(OLD_STAGE9, NEW_STAGE9, 1)
    print("[P2] Stage 9 keigo pre-check — APPLIED")
else:
    print("[P2] Stage 9 block not found — check analyzer.py manually", file=sys.stderr)

# ──────────────────────────────────────────────────────────────────────────────
# PATCH 2 — P6: add keigo→evasive override before D2 sync in _sentiment_backstop_block
# ──────────────────────────────────────────────────────────────────────────────
OLD_D2_SYNC = """\
    _sent_sync = {
        s["speaker"]: s["score"] for s in result.get("sentiment", [])
    }
    for _spk in result.get("speakers", []):
        _spk["sentiment_score"] = _sent_sync.get(
            _spk.get("name", ""),
            _spk.get("sentiment_score", "neutral"),
        )"""

NEW_D2_SYNC = """\
    # P6-FIX: Formal keigo + frustrated/irritated → politely_evasive
    # Frustration in FORMAL Japanese register is deliberate evasion, not leakage.
    # Runs AFTER all SR overrides so it refines, never clobbers, SR-driven labels.
    # DSA: O(speakers) — single pass.
    _keigo_lvl = result.get("japan_insights", {}).get("keigo_level", "unknown")
    if _keigo_lvl in ("FORMAL", "high", "SEMI-FORMAL", "medium"):
        for _s in result.get("sentiment", []):
            if (
                _s.get("label") in ("frustrated", "irritated")
                and float(_s.get("valence", 0.0)) > -0.80   # not rage — controlled
            ):
                _old = _s["label"]
                _s["label"] = "politely_evasive"
                _s.setdefault("secondary_labels", [])
                if _old not in _s["secondary_labels"]:
                    _s["secondary_labels"] = [_old] + _s["secondary_labels"][:3]
                _s["_label_adjusted"] = f"formal_keigo_override:{_old}→politely_evasive"
                print(
                    f"[SENTIMENT] P6: {_s.get('speaker')} "
                    f"{_old} → politely_evasive (keigo={_keigo_lvl})",
                    file=sys.stderr, flush=True,
                )

    # D2: sync sentiment_score into speaker dicts
    _sent_sync = {
        s["speaker"]: s["score"] for s in result.get("sentiment", [])
    }
    for _spk in result.get("speakers", []):
        _spk["sentiment_score"] = _sent_sync.get(
            _spk.get("name", ""),
            _spk.get("sentiment_score", "neutral"),
        )"""

if OLD_D2_SYNC in text:
    text = text.replace(OLD_D2_SYNC, NEW_D2_SYNC, 1)
    print("[P6] Keigo→evasive override — APPLIED")
else:
    print("[P6] D2 sync block not found — check analyzer.py manually", file=sys.stderr)

# Write patched file
SRC.write_text(text, encoding="utf-8")
print(f"\nDone. {SRC} patched in place.")
print("Run:  python -c \"from analysis.analyzer import analyze_transcript; print('OK')\"")