# analysis/_extractive_summary.py
# Inserted into analyzer.py by apply_patches.py — do not import directly.
#
# Rule-based extractive summary for the no-API fallback.
# Returns real content instead of a warning message.
#
# Algorithm: score-and-select with speaker diversity.
#   Pass 1 — collect (speaker, utterance) pairs via splitlines()   O(n)
#   Pass 2 — score each utterance: length x content-signal bonus   O(u)
#   Pass 3 — greedy select top max_sentences, prefer diversity      O(u)
# No regex — pure str ops so there are zero escaping issues here.
# DSA: O(n) dominated by line scan.

def _extractive_summary(text: str, max_sentences: int = 3) -> tuple[str, list[str]]:
    _SIGNAL = frozenset({
        # problems / urgency
        "down", "outage", "issue", "problem", "failed", "error",
        "unacceptable", "delay", "breach", "concern", "blocked",
        # commitments
        "will", "commit", "ensure", "guarantee", "provide", "send",
        "deliver", "confirm", "resolve", "fix", "escalate", "respond",
        # deadlines
        "friday", "monday", "tomorrow", "today", "deadline", "within",
        "hours", "days", "week", "urgent",
        # decisions / risk
        "contract", "reconsider", "terminate", "cancel", "agreed",
        "decided", "approved", "rejected", "budget", "plan", "proposal",
        # Hindi commitment / hedge signals
        "dekhte", "sochte", "koshish", "zaroor", "bilkul",
    })

    # Normalise JP fullwidth colon ： so Japanese speaker turns work too
    norm = text.replace("\uff1a", ":")

    turns: list[tuple[str, str]] = []
    for line in norm.splitlines():
        if ":" not in line:
            continue
        idx     = line.index(":")
        speaker = line[:idx].strip()
        content = line[idx + 1:].strip()
        # JP text has few whitespace-separated words — use char count too
        if 2 <= len(speaker) <= 40 and (len(content.split()) >= 5 or len(content) >= 15):
            turns.append((speaker, content))

    if not turns:
        lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 40][:3]
        return " ".join(lines), lines

    def _score(utt: str) -> float:
        words = utt.lower().split()
        hits  = sum(1 for w in words if w in _SIGNAL)
        return min(len(words) / 25.0, 1.2) + hits * 0.35

    scored = sorted(turns, key=lambda t: _score(t[1]), reverse=True)

    # Greedy select with speaker diversity; fill remaining from any speaker
    selected: list[tuple[str, str]] = []
    seen: set[str] = set()
    for spk, utt in scored:
        if len(selected) >= max_sentences:
            break
        if spk not in seen:
            selected.append((spk, utt))
            seen.add(spk)
    for spk, utt in scored:
        if len(selected) >= max_sentences:
            break
        if (spk, utt) not in selected:
            selected.append((spk, utt))

    full = " ".join(
        f"{s}: {u[:160]}{'...' if len(u) > 160 else ''}"
        for s, u in selected
    )
    bulls = [
        f"{s}: {u[:120]}{'...' if len(u) > 120 else ''}"
        for s, u in selected
    ]
    return full, bulls