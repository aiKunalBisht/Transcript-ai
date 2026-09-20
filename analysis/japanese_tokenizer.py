# analysis/japanese_tokenizer.py
# Proper Japanese Morphological Analysis
#
# v8.2 changes:
#   N1: Removed 相談する and 確認する from NEMAWASHI_ROOTS.
#       These are authority_deferring / escalation signals, NOT nemawashi.
#       Kayo Miura (Point 2): "「上司に相談します」is simply a normal procedural
#       step when Kenji does not have final authority. Nemawashi is closer to
#       pre-aligning people, conditions, concerns before a formal decision."
#       The sequence detector (nemawashi_sequence.py) handles the actual
#       nemawashi pattern — it cannot be detected from a single phrase.
#
# ── WHY THIS EXISTS ──────────────────────────────────────────────────────────
# Without MeCab, Japanese tokenization is character-level:
#   "検討します" → ["検", "討", "し", "ま", "す"]
# With MeCab (fugashi), it's morpheme-level:
#   "検討します" → ["検討", "し", "ます"]  (base: 検討する)
# This matters because:
#   検討します ≠ 検討しました ≠ 検討した
#   All mean "consider" but character overlap is low.
#   MeCab normalizes all three to base form: 検討する
# ─────────────────────────────────────────────────────────────────────────────

import re

try:
    import fugashi
    _tagger = fugashi.Tagger()
    MECAB_AVAILABLE = True
except ImportError:
    MECAB_AVAILABLE = False
    _tagger = None


def tokenize_japanese(text: str, normalize: bool = True) -> list:
    """
    Tokenizes Japanese text using MeCab (via fugashi).
    Falls back to character-level if MeCab not installed.

    Args:
        text:      Japanese or mixed JA/EN text
        normalize: If True, returns base forms (検討します → 検討する)

    Returns:
        List of tokens (morphemes for JA, words for EN)
    """
    if not MECAB_AVAILABLE:
        return _fallback_tokenize(text)

    tokens = []
    for word in _tagger(text):
        surface = word.surface
        feature = word.feature

        if not surface.strip() or surface in "。、！？「」『』・…":
            continue

        if normalize:
            try:
                base_form = word.feature.lemma
                if not base_form or base_form == "*":
                    base_form = surface
            except AttributeError:
                try:
                    parts = str(feature).split(",")
                    base_form = parts[6] if len(parts) > 6 and parts[6] != "*" else surface
                except Exception:
                    base_form = surface
            tokens.append(base_form)
        else:
            tokens.append(surface)

    return tokens


def _fallback_tokenize(text: str) -> list:
    """Character-level fallback when MeCab is not available."""
    ja_pattern = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]")
    tokens = []
    current_en = []

    for char in text.lower():
        if ja_pattern.match(char):
            if current_en:
                tokens.extend("".join(current_en).split())
                current_en = []
            tokens.append(char)
        elif char in (" ", "\t", "\n"):
            if current_en:
                tokens.extend("".join(current_en).split())
                current_en = []
        else:
            current_en.append(char)

    if current_en:
        tokens.extend("".join(current_en).split())

    cjk     = [t for t in tokens if re.match(r"[\u3040-\u9FFF]", t)]
    bigrams = ["".join(cjk[i:i+2]) for i in range(len(cjk)-1)]
    return tokens + bigrams


def extract_nemawashi_signals(text: str) -> list:
    """
    Extracts nemawashi signals using MeCab morphological analysis.

    ── What counts as a nemawashi signal ────────────────────────────────────
    Only patterns that indicate INDIRECT REJECTION or UNCERTAIN DEFERRAL
    are listed here. Escalation patterns (相談する, 確認する) are NOT
    nemawashi — they are authority_deferring signals detected separately
    by meeting_function_detector.py.

    Nemawashi itself (the pre-alignment sequence) is detected at the
    meeting level by nemawashi_sequence.py, not at the utterance level.

    ── Why 相談する was removed (v8.2 N1 fix) ───────────────────────────────
    「上司に相談します」 = escalation (authority_deferring)
    The speaker lacks decision authority and is sending the matter upward.
    This is a normal procedural step, not nemawashi.
    Nemawashi = the internal coordination that MAY follow this escalation,
    across multiple turns, not the escalation phrase itself.
    ─────────────────────────────────────────────────────────────────────────
    """
    # Nemawashi root forms — base/dictionary forms of genuine deferral signals.
    # Removed from v8.2: 相談する (escalation), 確認する (escalation).
    NEMAWASHI_ROOTS = {
        "検討する":   ("検討する",   "LIKELY_REJECTION", 0.72),
        "難しい":     ("難しい",     "REJECTION",        0.85),
        "善処する":   ("善処する",   "LIKELY_REJECTION", 0.68),
        "承知する":   ("承知する",   "ACKNOWLEDGMENT",   0.60),
        "了解する":   ("了解する",   "ACKNOWLEDGMENT",   0.60),
        # 相談する REMOVED — this is authority_deferring (escalation), not nemawashi
        # 確認する REMOVED — this is authority_deferring (escalation), not nemawashi
        "前向き":     ("前向き",     "UNCERTAIN",        0.55),
        "懸念":       ("懸念",       "HESITATION",       0.45),
    }

    if not MECAB_AVAILABLE:
        # Fallback: string matching against the surface forms
        found = []
        for root, (phrase, intent, conf) in NEMAWASHI_ROOTS.items():
            if root in text or phrase in text:
                found.append({
                    "base_form":  root,
                    "intent":     intent,
                    "confidence": conf,
                    "method":     "string_fallback",
                })
        return found

    found = []
    tokens = tokenize_japanese(text, normalize=True)

    for token in tokens:
        if token in NEMAWASHI_ROOTS:
            phrase, intent, confidence = NEMAWASHI_ROOTS[token]
            found.append({
                "base_form":  token,
                "intent":     intent,
                "confidence": confidence,
                "method":     "mecab",
            })

    return found


def get_keigo_level(text: str) -> str:
    """
    Determines keigo register level using MeCab POS tags.

    MeCab identifies:
      - 丁寧語 (polite): ます、です forms
      - 尊敬語 (respectful): お〜になる、〜られる forms
      - 謙譲語 (humble): お〜する、いたす forms
    """
    if not MECAB_AVAILABLE:
        return _fallback_keigo(text)

    sonkeigo_count  = 0
    kenjougo_count  = 0
    teineigo_count  = 0

    for word in _tagger(text):
        surface     = word.surface
        feature     = word.feature or ""
        feature_str = str(feature) if feature else ""

        if surface in ("ます", "です", "ません", "でした"):
            teineigo_count += 1

        if surface in ("いたし", "いたす", "申し", "申す", "ございます", "おります"):
            kenjougo_count += 1

        if "尊敬" in feature_str or surface in ("くださ", "なさ", "いらっしゃ"):
            sonkeigo_count += 1

    if kenjougo_count >= 1 or sonkeigo_count >= 2:
        return "high"
    elif teineigo_count >= 2 or (sonkeigo_count + kenjougo_count + teineigo_count) >= 2:
        return "medium"
    else:
        return "low"


def _fallback_keigo(text: str) -> str:
    """Marker-based keigo detection without MeCab."""
    high_markers = [
        "ございます", "いただき", "おります", "申し訳", "させていただき",
        "いたします", "いたしました", "承知いたします", "かしこまりました",
    ]
    med_markers = ["です", "ます", "ください", "ありがとう", "おはようございます"]
    high_count  = sum(1 for m in high_markers if m in text)
    med_count   = sum(1 for m in med_markers  if m in text)
    if high_count >= 1: return "high"
    if med_count  >= 2: return "medium"
    return "low"


def semantic_similarity_ja(text_a: str, text_b: str) -> float:
    """
    Computes semantic similarity between two JA/EN texts.
    Uses MeCab base forms if available, character-level otherwise.
    Returns 0.0 (no similarity) to 1.0 (identical meaning).
    """
    tokens_a = set(tokenize_japanese(text_a, normalize=True))
    tokens_b = set(tokenize_japanese(text_b, normalize=True))

    stopwords = {
        "の", "は", "が", "を", "に", "で", "と", "も", "か", "な", "て",
        "し", "た", "です", "ます", "する", "いる", "ある", "こと", "ため",
        "the", "a", "an", "is", "are", "to", "of", "and", "or", "in",
    }
    tokens_a -= stopwords
    tokens_b -= stopwords

    if not tokens_a or not tokens_b:
        return 0.0

    overlap   = tokens_a & tokens_b
    precision = len(overlap) / len(tokens_a)
    recall    = len(overlap) / len(tokens_b)

    if precision + recall == 0:
        return 0.0

    return round(2 * precision * recall / (precision + recall), 3)


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json

    print(f"MeCab available: {MECAB_AVAILABLE}")
    if not MECAB_AVAILABLE:
        print("Install with: pip install fugashi unidic-lite\n")

    print("=== v8.2: 相談する / 確認する NOT in nemawashi roots ===")
    removed = ["相談する", "確認する"]
    for phrase in removed:
        signals = extract_nemawashi_signals(phrase)
        matched = [s for s in signals if s["base_form"] in removed]
        print(f"  '{phrase}' in nemawashi roots: {bool(matched)} (should be False)")

    print("\n=== Genuine nemawashi signals still detected ===")
    genuine = "検討いたします。難しい状況ですが、前向きに考えます。"
    signals = extract_nemawashi_signals(genuine)
    print(f"  Input: {genuine}")
    print(f"  Signals: {json.dumps(signals, ensure_ascii=False, indent=2)}")

    print("\n=== Keigo level ===")
    samples = [
        "はい、承知いたしました。ご確認いただきありがとうございます。",
        "わかった。やっておくよ。",
        "ご連絡いただきありがとうございます。検討いたします。",
    ]
    for s in samples:
        print(f"  '{s[:35]}...' → {get_keigo_level(s)}")