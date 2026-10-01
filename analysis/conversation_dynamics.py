# conversation_dynamics.py — v1.1
#
# v1.1 change:
#   SOFT_REJECTION_PATTERNS import wrapped in try/except.
#   soft_rejection_detector.py no longer exports SOFT_REJECTION_PATTERNS
#   by that name — the import was failing with:
#     ImportError: cannot import name 'SOFT_REJECTION_PATTERNS' from
#     'analysis.soft_rejection_detector'
#   With the fallback, _has_hedge_signal() always returns False and topic
#   stall detection is disabled gracefully rather than crashing the whole
#   no-API fallback path.
#
# v1 original description:
#   Structural / turn-sequence patterns from Japanese business meeting culture
#   that pure keyword matching (soft_rejection_detector.py) can't catch,
#   because they're about WHERE something happens, not WHAT word appears:
#
#   1. Topic stall + cooling-off + circle-back
#   2. Senior-silence pivot
#   3. Closing summarizer

import re

# v1.1 FIX: guard the import — SOFT_REJECTION_PATTERNS was renamed/removed
try:
    from analysis.soft_rejection_detector import SOFT_REJECTION_PATTERNS
except ImportError:
    # Graceful degradation: topic-stall detection disabled, everything else works.
    SOFT_REJECTION_PATTERNS = []

from transcription.speaker_normalizer import normalize_speaker_name, extract_role_hints

_STOPWORDS_JA = {
    "の", "は", "が", "を", "に", "で", "と", "も", "か", "な", "て", "し", "た",
    "です", "ます", "する", "いる", "ある", "こと", "ため", "この", "その",
    "あの", "それ", "これ", "という", "ので", "から", "まで",
}
_STOPWORDS_EN = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or",
    "in", "on", "for", "we", "i", "you", "it", "that", "this", "will",
    "be", "with", "about", "have", "has", "but", "so",
}

_LINE_SPEAKER_PATTERN = re.compile(
    r"^(?:\[\d{2}:\d{2}(?::\d{2})?\]\s*)?([^\n:：\[\]]+)\s*[:：]\s*(.*)$"
)

_CLOSING_MARKERS = [
    "まとめると", "以上です", "ご協力", "お疲れ様でした", "引き続きよろしく",
    "総括すると", "最後に", "本日はありがとうございました",
    "to summarize", "in summary", "overall", "thank you all",
    "appreciate the cooperation", "let's wrap up", "to wrap up",
]


# ── Shared turn parsing ────────────────────────────────────────────────────────

def parse_turns(transcript: str) -> list:
    """
    Splits a transcript into ordered turns: [{index, speaker, raw_label, text}].
    Single forward pass, O(n) in transcript length. A line with no
    "Speaker:" prefix is treated as a continuation of the previous turn.
    """
    turns = []
    for raw_line in transcript.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        m = _LINE_SPEAKER_PATTERN.match(line)
        if not m:
            if turns:
                turns[-1]["text"] += " " + line
            continue

        raw_label, text = m.group(1).strip(), m.group(2).strip()
        if re.match(r"^[0-9]+$", raw_label):
            if turns:
                turns[-1]["text"] += " " + line
            continue

        name = normalize_speaker_name(raw_label)
        if not name:
            if turns:
                turns[-1]["text"] += " " + text
            continue

        turns.append({
            "index":     len(turns),
            "speaker":   name,
            "raw_label": raw_label,
            "text":      text,
        })

    return turns


def _topic_terms(text: str) -> set:
    """Cheap content-word extraction for overlap comparisons.
    Includes katakana (loanwords like コスト/イベント are common topic nouns)."""
    cjk      = set(re.findall(r"[\u4E00-\u9FFF]{2,}", text))
    katakana = set(re.findall(r"[\u30A0-\u30FF]{2,}", text))
    words    = {w.lower() for w in re.findall(r"[A-Za-z]{4,}", text)}
    return (cjk | katakana | words) - _STOPWORDS_JA - _STOPWORDS_EN


def _has_hedge_signal(text: str) -> bool:
    """
    Returns True if the text contains any known soft-rejection phrase.
    When SOFT_REJECTION_PATTERNS is empty (import fallback), always False —
    topic stall detection degrades gracefully.
    """
    return any(p["phrase"] in text for p in SOFT_REJECTION_PATTERNS)


# ── 1. Topic stall + circle-back ──────────────────────────────────────────────

def detect_topic_stalls(transcript: str, lookahead: int = 6) -> list:
    """
    Flags "stall -> pivot to easier topic -> circle back later" pattern.

    For each turn carrying a hedge signal: check the very next turn pivots
    away (low term overlap), then scan up to `lookahead` turns ahead for the
    original topic resurfacing (overlap back above threshold).
    O(n * lookahead), not O(n^2).
    """
    turns  = parse_turns(transcript)
    events = []

    for i, turn in enumerate(turns):
        if not _has_hedge_signal(turn["text"]):
            continue
        stalled_terms = _topic_terms(turn["text"])
        if not stalled_terms or i + 1 >= len(turns):
            continue

        next_terms   = _topic_terms(turns[i + 1]["text"])
        overlap_next = len(stalled_terms & next_terms) / len(stalled_terms)
        if overlap_next > 0.25:
            continue  # conversation just continued normally — no pivot

        for j in range(i + 2, min(i + 1 + lookahead, len(turns))):
            later_terms = _topic_terms(turns[j]["text"])
            if not later_terms:
                continue
            overlap_later = len(stalled_terms & later_terms) / len(stalled_terms)
            if overlap_later >= 0.34:
                events.append({
                    "stalled_at_turn":      i,
                    "stalled_speaker":      turn["speaker"],
                    "stalled_excerpt":      turn["text"][:100],
                    "pivot_turn":           i + 1,
                    "circled_back_turn":    j,
                    "circled_back_speaker": turns[j]["speaker"],
                    "explanation": (
                        "Topic stalled after a hedge signal, the conversation moved "
                        "to a different topic, and the original topic resurfaced "
                        f"{j - i} turns later — the 'cooling off' pattern common in "
                        "Japanese meetings, rather than forcing resolution in sequence."
                    ),
                })
                break

    return events


# ── 2. Senior-silence pivot ────────────────────────────────────────────────────

def detect_senior_silence_pivot(transcript: str, role_hints: dict = None,
                                 min_silence_turns: int = 3,
                                 min_rank: int = 5) -> list:
    """
    Flags: a senior speaker goes quiet for >= min_silence_turns after their
    own turn, and a lower-ranked speaker pivots away during the silence.
    Keyword matching on the junior speaker's line alone would miss this.
    """
    turns = parse_turns(transcript)
    if not turns:
        return []
    role_hints = role_hints if role_hints is not None else extract_role_hints(transcript)
    events     = []

    for i, t in enumerate(turns):
        hint = role_hints.get(t["speaker"], {"role": "", "rank": 0})
        if hint["rank"] < min_rank:
            continue

        next_appearance = None
        for j in range(i + 1, len(turns)):
            if turns[j]["speaker"] == t["speaker"]:
                next_appearance = j
                break
        window_end = next_appearance if next_appearance else len(turns)
        gap        = window_end - i
        if gap < min_silence_turns:
            continue

        stalled_terms = _topic_terms(t["text"])
        if not stalled_terms:
            continue

        for j in range(i + 1, min(window_end, i + 1 + min_silence_turns + 1)):
            junior_hint  = role_hints.get(turns[j]["speaker"], {"rank": 0})
            if junior_hint["rank"] >= hint["rank"]:
                continue
            junior_terms = _topic_terms(turns[j]["text"])
            if not junior_terms:
                continue
            overlap = len(stalled_terms & junior_terms) / len(stalled_terms)
            if overlap < 0.2:
                events.append({
                    "senior_speaker":       t["speaker"],
                    "senior_role":          hint["role"],
                    "silence_starts_turn":  i,
                    "silence_length_turns": gap,
                    "junior_speaker":       turns[j]["speaker"],
                    "pivot_turn":           j,
                    "explanation": (
                        f"{t['speaker']}"
                        f"{' (' + hint['role'] + ')' if hint['role'] else ''} went "
                        f"quiet for {gap} turns after this point, and "
                        f"{turns[j]['speaker']} changed the subject shortly after — "
                        "a known soft-rejection delivery pattern. The junior "
                        "speaker's own words may look neutral in isolation."
                    ),
                })
                break

    return events


# ── 3. Closing summarizer ──────────────────────────────────────────────────────

def infer_closing_summarizer(transcript: str, role_hints: dict = None,
                              min_rank: int = 5) -> dict:
    """
    Flags: a senior speaker who said little through the meeting delivers
    the closing summary. Talk-time share alone is not a reliable
    decision-maker signal when this pattern is present.
    """
    turns = parse_turns(transcript)
    if not turns:
        return {"detected": False}
    role_hints    = role_hints if role_hints is not None else extract_role_hints(transcript)
    last_idx      = len(turns) - 1
    closing_window = turns[max(0, last_idx - 2):]

    for t in reversed(closing_window):
        if not any(m in t["text"] for m in _CLOSING_MARKERS):
            continue

        speaker = t["speaker"]
        hint    = role_hints.get(speaker, {"role": "", "rank": 0})
        if hint["rank"] < min_rank:
            continue

        speaker_turns = [x for x in turns if x["speaker"] == speaker]
        early_cutoff  = last_idx * 0.6
        early_turns   = [x for x in speaker_turns if x["index"] < early_cutoff]

        if len(early_turns) <= 1:
            return {
                "detected":               True,
                "speaker":                speaker,
                "role":                   hint["role"],
                "turn_index":             t["index"],
                "total_turns_by_speaker": len(speaker_turns),
                "explanation": (
                    f"{speaker}"
                    f"{' (' + hint['role'] + ')' if hint['role'] else ''} stayed "
                    "largely quiet through the meeting and delivered the closing "
                    "summary — a structural cue, not noise. Talk-time share alone "
                    "would have under-weighted this speaker's role."
                ),
            }

    return {"detected": False}


# ── Convenience wrapper ────────────────────────────────────────────────────────

def analyze_conversation_dynamics(transcript: str) -> dict:
    """Runs all three checks once, sharing one role_hints pass."""
    role_hints = extract_role_hints(transcript)
    return {
        "role_hints":            role_hints,
        "topic_stalls":          detect_topic_stalls(transcript),
        "senior_silence_pivots": detect_senior_silence_pivot(transcript, role_hints),
        "closing_summarizer":    infer_closing_summarizer(transcript, role_hints),
    }


if __name__ == "__main__":
    import json

    sample = """
田中部長: このプロジェクトの予算についてどう思いますか？
鈴木: 予算については難しいかもしれません。コストが高すぎます。
佐藤: ところで、来週のイベントの会場は決まりましたか？
山田: はい、渋谷の会議室を予約しました。
佐藤: 良かったです。ケータリングも頼みましょう。
鈴木: そういえば、さっきの予算の件ですが、来月もう一度検討しましょう。
田中部長: それでは、まとめると、来月の会議で予算を再検討し、イベント会場も確定しました。ご協力ありがとうございました。
"""
    result = analyze_conversation_dynamics(sample)
    print(json.dumps(result, indent=2, ensure_ascii=False))