"""
sentiment_engine.py  —  Fine-Grained Sentiment & Tone Detection Engine
TranscriptAI Pipeline  |  Replaces flat 5-label sentiment with rich multi-signal output

v1.1 changes (this version):
  FIX-JP1: _parse_transcript now handles full-width colon ：
    In v1.0, the speaker-turn parser used pattern [^colon-newline]+ which only
    matched half-width colon. Every Japanese transcript that uses ：
    (the standard in Japanese business meetings) produced an EMPTY utterance
    list. from_raw_transcript() then returned an empty TranscriptSentimentReport
    with no speaker arcs, causing every Japanese speaker in the no-API fallback
    path to receive score="neutral" / label="factual" regardless of content.
    Fix: pattern now matches both ： (U+FF1A) and : (U+003A).

  FIX-JP2: Empty utterance guard added to analyze_transcript().
    When the transcript is Japanese-only and _parse_transcript returns [],
    the engine now logs a warning and returns a minimal safe report rather
    than dividing by zero or silently producing a neutral report with no
    speaker arcs. The caller (analyzer.py _no_api_sentiment_block) can
    detect this via report.utterances == [].

  FIX-JP3: _parse_transcript handles PII-masked speaker labels.
    [NAME_1]: text was not matched because the old parser's character class
    did not correctly stop at the colon — it consumed the separator as part
    of the speaker name group. The new pattern stops at both half-width and
    full-width colons, so [NAME_1]: text is now parsed correctly.

  DOC-JP: Honest coverage statement added.
    The engine's patterns are English and Hinglish. Japanese text produces
    near-zero pattern hits. This is not a bug introduced here — it is a
    documented architectural limitation. See _JAPANESE_COVERAGE_NOTE.

v1.0 retained unchanged:
  All 25-label PATTERN_REGISTRY entries.
  Negation detection with token-index window.
  Tone modifier classification.
  Speaker arc computation with trend detection.
  All data structures (SentimentSignal, ToneProfile, UtteranceResult,
    SpeakerArc, TranscriptSentimentReport).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DSA Architecture
────────────────
  Pattern Registry   dict[label → list[_Pat]]            O(1) label lookup
  Text scan          compiled-regex per label, iterator  O(U · L · P)
  Negation mask      set[int] token-index window         O(T) per utterance
  Score normalize    sum-normalize (softmax-lite)        O(L)
  Top-K select       heapq.nlargest(k, items)            O(L log k)
  Speaker aggregate  defaultdict(list[UtteranceResult])  O(U)
  Arc trend          first-half vs second-half μ         O(U/S)

  Overall: O(U · L · P)
  U ≈ 200 utterances │ L = 25 labels │ P ≈ 4 patterns/label → ~19 K ops
  Sub-millisecond for a typical 1-hour meeting transcript.

Label Taxonomy (25 labels across 4 clusters)
─────────────────────────────────────────────
  POSITIVE  enthusiastic · confident · agreeable · appreciative
            hopeful · relieved · encouraging · satisfied
  NEGATIVE  frustrated · irritated · anxious · disappointed
            dismissive · defensive · skeptical · overwhelmed
            resigned · sarcastic · passive_aggressive · condescending
  NEUTRAL   factual · inquisitive · ambivalent
  COMPLEX   politely_evasive · deflecting

Tone Modifiers (orthogonal to emotion labels)
─────────────────────────────────────────────
  urgency    : low | medium | high
  certainty  : definite | hedged | uncertain
  engagement : active | passive | disengaged

Language Coverage
─────────────────
  English    : Full pattern coverage (primary language of engine).
  Hinglish   : Partial — romanized Hindi markers in key labels.
  Japanese   : Structural parsing fixed (FIX-JP1/JP2/JP3).
               Semantic pattern coverage: NONE. Japanese text produces
               near-zero pattern hits. Sentiment for Japanese speakers
               in the no-API path is controlled by the SR overlay in
               analyzer._no_api_sentiment_block, not by this engine.
               For real Japanese sentiment: fine-tune a Japanese BERT
               (cl-tohoku/bert-base-japanese) on your label taxonomy.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from __future__ import annotations

import logging
import re
import heapq
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

logger = logging.getLogger("transcriptai.sentiment_engine")

# FIX-JP: Honest coverage note — referenced in analyzer.py to decide
# whether to trust the engine output for Japanese-only transcripts.
_JAPANESE_COVERAGE_NOTE = (
    "sentiment_engine patterns are English/Hinglish only. "
    "Japanese utterances will score near-zero on all pattern labels "
    "and default to factual/neutral. Use the SR-overlay path in "
    "_no_api_sentiment_block for Japanese sentiment signals."
)

# FIX-JP1: Regex that matches BOTH half-width ':' and full-width '：'.
# Used in _parse_transcript and any colon-split logic in this module.
_COLON_RE = re.compile(r"[：:]")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TAXONOMY  ── label sets + valence map
# ═══════════════════════════════════════════════════════════════════════════════

POSITIVE_LABELS: frozenset[str] = frozenset({
    "enthusiastic", "confident", "agreeable", "appreciative",
    "hopeful", "relieved", "encouraging", "satisfied",
})
NEGATIVE_LABELS: frozenset[str] = frozenset({
    "frustrated", "irritated", "anxious", "disappointed",
    "dismissive", "defensive", "skeptical", "overwhelmed",
    "resigned", "sarcastic", "passive_aggressive", "condescending",
})
NEUTRAL_LABELS: frozenset[str] = frozenset({"factual", "inquisitive", "ambivalent"})
COMPLEX_LABELS: frozenset[str] = frozenset({"politely_evasive", "deflecting"})

ALL_LABELS: frozenset[str] = (
    POSITIVE_LABELS | NEGATIVE_LABELS | NEUTRAL_LABELS | COMPLEX_LABELS
)

VALENCE_MAP: dict[str, float] = {
    "enthusiastic":       +0.95,
    "confident":          +0.80,
    "agreeable":          +0.65,
    "appreciative":       +0.70,
    "hopeful":            +0.60,
    "relieved":           +0.55,
    "encouraging":        +0.72,
    "satisfied":          +0.68,
    "factual":            +0.02,
    "inquisitive":        +0.08,
    "ambivalent":          0.00,
    "politely_evasive":   -0.20,
    "deflecting":         -0.30,
    "skeptical":          -0.35,
    "resigned":           -0.45,
    "anxious":            -0.45,
    "irritated":          -0.42,
    "overwhelmed":        -0.55,
    "disappointed":       -0.55,
    "defensive":          -0.52,
    "frustrated":         -0.65,
    "dismissive":         -0.68,
    "sarcastic":          -0.72,
    "passive_aggressive": -0.78,
    "condescending":      -0.82,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PATTERN REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

class _Pat(NamedTuple):
    regex: re.Pattern
    weight: float


def _p(pattern: str, weight: float = 0.6, flags: int = re.IGNORECASE) -> _Pat:
    return _Pat(re.compile(pattern, flags), weight)


PATTERN_REGISTRY: dict[str, list[_Pat]] = {

    "enthusiastic": [
        _p(r"\b(amazing|fantastic|brilliant|excellent|outstanding|thrilled|excited)\b", 0.90),
        _p(r"\b(great|awesome|wonderful|incredible|superb|love\s+this)\b", 0.75),
        _p(r"\b(absolutely|definitely|totally|100\s*%)\s+(agree|on\s+board|right|yes)\b", 0.80),
        _p(r"[!]+\s*(yes|yeah|let'?s\s+(do\s+it|go)|perfect|exactly)\b", 0.85),
        _p(r"\b(can'?t\s+wait|looking\s+forward|super\s+excited|pumped)\b", 0.88),
        _p(r"\b(bahut\s+accha|zabardast|ekdum\s+sahi|kya\s+baat\s+hai)\b", 0.82),
    ],

    "confident": [
        _p(r"\b(I'?m\s+sure|no\s+doubt|certainly|without\s+question|I\s+know\s+for\s+sure)\b", 0.85),
        _p(r"\b(guaranteed|will\s+definitely|absolutely\s+will|100\s*\s*percent\s+confident)\b", 0.88),
        _p(r"\b(trust\s+me|you\s+can\s+count\s+on\s+(me|us)|I\s+can\s+promise)\b", 0.82),
        _p(r"\b(we\s+(will|can)\s+(handle|deliver|finish|nail)\s+(this|it))\b", 0.72),
    ],

    "agreeable": [
        _p(r"\b(yes|yeah|yep|sure|of\s+course|absolutely|exactly|correct|agreed)\b", 0.50),
        _p(r"\b(good\s+point|fair\s+enough|makes\s+sense|that'?s\s+(fair|true|right))\b", 0.75),
        _p(r"\b(I\s+agree|on\s+board|sounds\s+good|works\s+for\s+(me|us))\b", 0.80),
        _p(r"\b(no\s+objection|happy\s+to|glad\s+to|I\s+see\s+your\s+point)\b", 0.72),
        _p(r"\b(haan\s+(bilkul)?|theek\s+hai|sahi\s+hai|ji\s+(haan|bilkul))\b", 0.70),
    ],

    "appreciative": [
        _p(r"\b(thank\s+(you|you\s+so\s+much)|thanks|appreciate\s+(it|this|you)|grateful)\b", 0.85),
        _p(r"\b(well\s+done|nice\s+work|great\s+job|kudos|props\s+to)\b", 0.80),
        _p(r"\b(that\s+(really\s+)?helped|this\s+is\s+(very\s+)?helpful|you\s+saved\s+us)\b", 0.78),
        _p(r"\b(couldn'?t\s+have\s+(done\s+it\s+)?without\s+(you|this))\b", 0.88),
    ],

    "hopeful": [
        _p(r"\b(hopefully|I\s+hope|fingers\s+crossed|optimistic|looking\s+forward)\b", 0.75),
        _p(r"\b(should\s+(work|be\s+fine|pan\s+out)|good\s+chance|promising)\b", 0.68),
        _p(r"\b(could\s+potentially|might\s+actually\s+work|there'?s\s+(a\s+)?chance)\b", 0.62),
        _p(r"\b(dekhte\s+hain|ho\s+jayega|sab\s+theek\s+ho\s+jayega)\b", 0.72),
    ],

    "relieved": [
        _p(r"\b(relieved|glad\s+(that|it'?s|we)|phew|finally|at\s+last|thank\s+goodness)\b", 0.85),
        _p(r"\b(turned\s+out\s+(better|fine|okay|well)|didn'?t\s+expect\s+it\s+to\s+work)\b", 0.75),
        _p(r"\b(that'?s\s+a\s+relief|off\s+my\s+(chest|mind|plate)|crisis\s+averted)\b", 0.88),
        _p(r"\b(no\s+longer\s+worried|worry\s+is\s+over|bullet\s+dodged)\b", 0.80),
    ],

    "encouraging": [
        _p(r"\b(you\s+can\s+do\s+(it|this)|keep\s+(going|it\s+up)|almost\s+there)\b", 0.82),
        _p(r"\b(great\s+progress|on\s+the\s+right\s+(track|path)|coming\s+along\s+well)\b", 0.78),
        _p(r"\b(push\s+through|hang\s+in\s+there|don'?t\s+give\s+up|we\s+can\s+do\s+this)\b", 0.85),
        _p(r"\b(I\s+believe\s+in\s+(you|the\s+team)|keep\s+pushing)\b", 0.88),
    ],

    "satisfied": [
        _p(r"\b(happy\s+with|satisfied|pleased\s+(with|about)|content\s+with)\b", 0.80),
        _p(r"\b(this\s+(works|is\s+what\s+we\s+needed)|meets\s+(our\s+)?requirements)\b", 0.75),
        _p(r"\b(good\s+enough|acceptable|that'?ll\s+do|works\s+for\s+(me|us))\b", 0.65),
        _p(r"\b(no\s+complaints|can'?t\s+complain|I'?m\s+okay\s+with\s+this)\b", 0.70),
    ],

    "factual": [
        _p(r"\b(the\s+data\s+(shows|indicates|suggests)|according\s+to|based\s+on)\b", 0.70),
        _p(r"\b(as\s+per|per\s+(the\s+)?report|the\s+numbers\s+(show|indicate))\b", 0.72),
        _p(r"\b(in\s+fact|technically|to\s+be\s+precise|specifically|for\s+the\s+record)\b", 0.65),
        _p(r"\b(to\s+summarize|in\s+summary|the\s+(issue|problem)\s+is\s+(that|this))\b", 0.55),
    ],

    "inquisitive": [
        _p(r"\b(I\s+(was\s+)?wonder(ing)?|curious\s+(about|whether)|what\s+do\s+you\s+think)\b", 0.72),
        _p(r"\b(can\s+you\s+(explain|clarify|elaborate)|could\s+you\s+walk\s+me\s+through)\b", 0.78),
        _p(r"\b(any\s+(thoughts|feedback|insights|ideas)|what'?s\s+your\s+(take|view|opinion))\b", 0.65),
        _p(r"\b(how\s+does\s+that\s+work|why\s+(is|did|would)\s+that|what\s+exactly\s+(is|are))\b", 0.68),
    ],

    "ambivalent": [
        _p(r"\b(not\s+sure|on\s+the\s+fence|could\s+go\s+either\s+way|hard\s+to\s+say)\b", 0.80),
        _p(r"\b(pros\s+and\s+cons|both\s+sides|it\s+depends\s+(on|how|what))\b", 0.72),
        _p(r"\b(I\s+suppose|I\s+guess|kind\s+of|sort\s+of|in\s+a\s+way)\b", 0.50),
        _p(r"\b(might\s+be|could\s+be|possibly|I\s+mean|more\s+or\s+less)\b", 0.42),
    ],

    "frustrated": [
        _p(r"\b(frustrated|frustrating|this\s+is\s+(ridiculous|absurd|insane)|unacceptable)\b", 0.92),
        _p(r"\b(how\s+many\s+times|I'?ve\s+(said|told)\s+(you\s+)?this\s+(before|already|multiple\s+times))\b", 0.88),
        _p(r"\b(keeps?\s+(happening|failing|breaking|going\s+wrong)|still\s+not\s+(working|fixed|done))\b", 0.85),
        _p(r"\b(nothing\s+(works|is\s+working)|what'?s\s+the\s+point|I\s+give\s+up\s+with\s+this)\b", 0.87),
        _p(r"\b(I\s+can'?t\s+(believe|deal\s+with|take)\s+(this|it))\b", 0.83),
        _p(r"\b(yaar\s+kya\s+ho\s+gaya|kuch\s+nahi\s+ho\s+raha|ek\s+kaam\s+nahi\s+hota)\b", 0.82),
    ],

    "irritated": [
        _p(r"\b(annoying|annoyed|(really\s+)?bothers?\s+me|this\s+is\s+irritating)\b", 0.80),
        _p(r"\b(for\s+the\s+last\s+time|as\s+I\s+(said|mentioned)|already\s+covered\s+this)\b", 0.75),
        _p(r"\b(seriously\?+|come\s+on\s*[,!]|give\s+me\s+a\s+break|stop\s+(it|doing\s+that))\b", 0.72),
        _p(r"\b(minor\s+(issue|problem|thing)\s+but|small\s+thing\s+but|trivial\s+but)\b", 0.45),
    ],

    "anxious": [
        _p(r"\b(worried|nervous|anxious|stressed|panic(king)?|scared|afraid|apprehensive)\b", 0.88),
        _p(r"\b(what\s+if\s+(we\s+)?(fail|miss|can'?t)|I'?m\s+not\s+sure\s+we\s+can)\b", 0.80),
        _p(r"\b(running\s+out\s+of\s+time|tight\s+(timeline|deadline)|under\s+pressure)\b", 0.78),
        _p(r"\b(risk(y)?|concern(ed|ing)?|critical\s+issue|this\s+is\s+worrying)\b", 0.65),
        _p(r"\b(could\s+go\s+wrong|might\s+backfire|worst\s+case\s+(scenario)?)\b", 0.72),
    ],

    "disappointed": [
        _p(r"\b(disappointed|disappointing|let\s+down|not\s+what\s+I\s+expected|expected\s+better)\b", 0.90),
        _p(r"\b(thought\s+(we|you|it)\s+would\s+have|was\s+hoping\s+(for|that))\b", 0.72),
        _p(r"\b(fell\s+short|below\s+(expectation|standard|par)|not\s+up\s+to\s+(scratch|par))\b", 0.85),
        _p(r"\b(missed\s+the\s+(mark|target|deadline|goal)|didn'?t\s+meet\s+expectations)\b", 0.82),
    ],

    "dismissive": [
        _p(r"\b(whatever|doesn'?t\s+matter|I\s+don'?t\s+care|not\s+my\s+problem)\b", 0.88),
        _p(r"\b(yeah\s+yeah|ok\s+ok+|moving\s+on|next\s+point|let'?s\s+skip\s+this)\b", 0.78),
        _p(r"\b(that'?s\s+(not\s+important|irrelevant|beside\s+the\s+point|trivial))\b", 0.82),
        _p(r"\b(obviously|everyone\s+knows\s+that|we\s+all\s+know\s+this)\b", 0.65),
        _p(r"\b(chalo\s+chhodo|chhod\s+do\s+yaar|koi\s+baat\s+nahi)\b", 0.75),
    ],

    "defensive": [
        _p(r"\b(that'?s\s+not\s+(my|our)\s+(fault|responsibility|problem))\b", 0.90),
        _p(r"\b(I\s+never\s+said|I\s+didn'?t\s+(say|do|mean)\s+that)\b", 0.85),
        _p(r"\b(you'?re\s+(misunderstanding|misinterpreting|taking\s+it\s+out\s+of\s+context))\b", 0.88),
        _p(r"\b(what\s+I\s+(meant|was\s+saying)\s+was|let\s+me\s+(set|clarify)\s+(the\s+record|this))\b", 0.68),
        _p(r"\b(I\s+followed\s+(the\s+)?procedure|I\s+did\s+exactly\s+what\s+was\s+asked)\b", 0.82),
    ],

    "skeptical": [
        _p(r"\b(I\s+don'?t\s+(think|believe|see\s+how)|I'?m\s+not\s+convinced|doubtful|skeptical)\b", 0.85),
        _p(r"\b(how\s+(would|will|does)\s+that\s+(work|make\s+sense)|that\s+doesn'?t\s+add\s+up)\b", 0.82),
        _p(r"\b(really\?+|are\s+you\s+sure|is\s+that\s+realistic|have\s+(we|you)\s+(tested|verified|confirmed))\b", 0.74),
        _p(r"\b(sounds\s+(too\s+good|unlikely|off)\s+to\s+me|I\s+have\s+(my\s+)?doubts)\b", 0.80),
        _p(r"\b(I\s+'ll\s+believe\s+it\s+when\s+I\s+see\s+it|not\s+sure\s+I\s+buy\s+that)\b", 0.85),
    ],

    "overwhelmed": [
        _p(r"\b(too\s+much|overwhelmed|can'?t\s+(handle|manage|keep\s+up|cope\s+with))\b", 0.90),
        _p(r"\b(swamped|buried\s+in|drowning\s+in|so\s+many\s+(tasks|things|issues|problems))\b", 0.87),
        _p(r"\b(I\s+don'?t\s+know\s+where\s+to\s+start|where\s+do\s+I\s+even\s+begin)\b", 0.85),
        _p(r"\b(everything\s+at\s+once|back\s+to\s+back\s+to\s+back|non[- ]?stop)\b", 0.78),
    ],

    "resigned": [
        _p(r"\b(fine[,.]|whatever[,.]|okay[,.]|alright[,.])\s*(do\s+it|if\s+you)", 0.80),
        _p(r"\b(if\s+you\s+say\s+so|I\s+won'?t\s+push\s+(it|further)|not\s+worth\s+(arguing|fighting))\b", 0.85),
        _p(r"\b(I\s+give\s+up|there'?s\s+no\s+point|can'?t\s+fight\s+(this|it\s+anymore))\b", 0.82),
        _p(r"\b(just\s+(go\s+with\s+it|do\s+it)|does\s+it\s+(even\s+)?matter\s+(anymore)?)\b", 0.75),
    ],

    "sarcastic": [
        _p(r"\b(oh\s+sure|oh\s+great|oh\s+wonderful|right\s+because\s+that'?s)\b", 0.80),
        _p(r"\b(oh\s+wow[,!]|shocking[,!]|what\s+a\s+surprise|who\s+would\s+have\s+thought)\b", 0.78),
        _p(r"\b(yeah\s+no|sure\s+that'?ll\s+(work|happen)|totally|great\s+idea)\s+[.!]", 0.72),
        _p(r"\b(because\s+that\s+(always\s+)?works|right\s*,?\s*I'?m\s+sure\s+that'?ll)\b", 0.75),
    ],

    "passive_aggressive": [
        _p(r"\b(fine\s*[,;]?\s*(do\s+it\s+your\s+(own\s+)?way|whatever\s+you\s+think\s+is\s+best))\b", 0.88),
        _p(r"\b(I\s+(just|only)\s+wanted\s+to\s+(say|mention|note)|not\s+that\s+it\s+matters)\b", 0.82),
        _p(r"\b(by\s+all\s+means|go\s+ahead\s+and|be\s+my\s+guest|sure\s+why\s+not)\b", 0.74),
        _p(r"\b(I\s+guess\s+(we'?ll\s+see|that'?s\s+one\s+way|you\s+know\s+best|if\s+that'?s\s+what\s+you\s+want))\b", 0.82),
        _p(r"\b(no\s+no\s+it'?s\s+fine|I'?m\s+(fine|okay)\s+with\s+(it|whatever))\s+really\b", 0.78),
    ],

    "politely_evasive": [
        _p(r"\b(we'?ll\s+see|let'?s\s+(circle\s+back|revisit|table\s+this|park\s+that))\b", 0.80),
        _p(r"\b(interesting\s+point|something\s+to\s+(consider|think\s+about)|I'?ll\s+look\s+into\s+it)\b", 0.68),
        _p(r"\b(not\s+the\s+right\s+(time|moment|forum)|we\s+can\s+discuss\s+(later|offline|separately))\b", 0.76),
        _p(r"\b(let\s+me\s+get\s+back\s+to\s+you|I'?ll\s+need\s+to\s+(check|verify|confirm)\s+(that|this))\b", 0.72),
        _p(r"\b(it'?s\s+(complicated|nuanced)|there\s+are\s+multiple\s+(factors|considerations))\b", 0.62),
    ],

    "deflecting": [
        _p(r"\b(anyway[,!]|moving\s+on|let'?s\s+get\s+back\s+to|back\s+to\s+(the\s+)?main\s+(topic|point|agenda))\b", 0.68),
        _p(r"\b(that'?s\s+a\s+separate\s+(issue|topic|conversation|discussion))\b", 0.74),
        _p(r"\b(not\s+(really\s+)?related|let'?s\s+not\s+(go\s+there|get\s+into\s+that\s+now))\b", 0.78),
        _p(r"\b(speaking\s+of\s+which|on\s+a\s+different\s+note|that\s+reminds\s+me\s+of)\b", 0.62),
    ],

    "condescending": [
        _p(r"\b(as\s+I\s+(already\s+)?explained|as\s+I\s+(mentioned|told)\s+you\s+(before|earlier|multiple\s+times))\b", 0.90),
        _p(r"\b(let\s+me\s+(simplify|break\s+it\s+down)\s+for\s+you|to\s+put\s+it\s+(simply|in\s+simple\s+terms))\b", 0.82),
        _p(r"\b(you\s+(should|would\s+know|must\s+know|clearly\s+don'?t\s+understand))\b", 0.85),
        _p(r"\b(it'?s\s+(quite\s+)?simple|basic(ally)?|fundamentally|with\s+all\s+due\s+respect\s+,?\s+you'?re)\b", 0.70),
        _p(r"\b(no\s+offense\s+but\s+that'?s|perhaps\s+you\s+should\s+read\s+up\s+on)\b", 0.78),
    ],
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TONE MODIFIER PATTERNS
# ═══════════════════════════════════════════════════════════════════════════════

_URGENCY_HIGH = re.compile(
    r"\b(immediately|urgent(ly)?|ASAP|right\s+now|critical|emergency|"
    r"deadline\s+today|cannot\s+wait|can'?t\s+wait|time\s+sensitive)\b",
    re.IGNORECASE,
)
_URGENCY_MED = re.compile(
    r"\b(soon|shortly|this\s+week|by\s+(Monday|Tuesday|Wednesday|Thursday|Friday|"
    r"end\s+of\s+(day|week))|need\s+to\s+|should\s+|priority|before\s+the\s+deadline)\b",
    re.IGNORECASE,
)
_CERTAINTY_UNCERTAIN = re.compile(
    r"\b(not\s+sure\s+(yet|about|if)|I\s+don'?t\s+know|unclear|TBD|"
    r"yet\s+to\s+be\s+(confirmed|decided)|open\s+question|question\s+mark)\b",
    re.IGNORECASE,
)
_CERTAINTY_HEDGED = re.compile(
    r"\b(might|maybe|perhaps|probably|I\s+think|I\s+believe|"
    r"possibly|could\s+be|sort\s+of|kind\s+of|roughly|approximately|"
    r"I\s+would\s+imagine)\b",
    re.IGNORECASE,
)
_ENGAGEMENT_ACTIVE = re.compile(
    r"\b(let'?s|we\s+should|I'?ll|we'?ll|going\s+to|plan\s+to|"
    r"will\s+(do|be\s+doing|handle)|I\s+want\s+to|I'?m\s+going\s+to)\b",
    re.IGNORECASE,
)
_ENGAGEMENT_DISENGAGED = re.compile(
    r"(?<!\w)(sure|fine|whatever|okay|ok|mm[- ]?hmm|uh[- ]?huh|right|alright)(?!\w)\s*[,.]",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. NEGATION DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

_MIN_RAW_SCORE: float = 0.28

_NEGATION_RE = re.compile(
    r"\b(not|no|never|neither|nor|without|hardly|barely|scarcely|"
    r"don'?t|doesn'?t|didn'?t|won'?t|can'?t|couldn'?t|shouldn'?t|"
    r"wouldn'?t|isn'?t|aren'?t|wasn'?t|weren'?t)\b",
    re.IGNORECASE,
)
_NEGATION_WINDOW: int   = 4
_NEGATION_DAMPEN: float = 0.12


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SentimentSignal:
    label:           str
    score:           float
    valence:         float
    matched_phrases: list[str] = field(default_factory=list)


@dataclass
class ToneProfile:
    urgency:    str
    certainty:  str
    engagement: str

    def to_dict(self) -> dict:
        return {
            "urgency":    self.urgency,
            "certainty":  self.certainty,
            "engagement": self.engagement,
        }


@dataclass
class UtteranceResult:
    speaker:    str
    text:       str
    primary:    str
    secondary:  list[str]
    tone:       ToneProfile
    valence:    float
    confidence: float
    signals:    list[SentimentSignal]

    def to_dict(self) -> dict:
        return {
            "speaker":    self.speaker,
            "text":       self.text,
            "primary":    self.primary,
            "secondary":  self.secondary,
            "tone":       self.tone.to_dict(),
            "valence":    self.valence,
            "confidence": self.confidence,
            "signals": [
                {
                    "label":           s.label,
                    "score":           s.score,
                    "valence":         s.valence,
                    "matched_phrases": s.matched_phrases,
                }
                for s in self.signals
            ],
        }


@dataclass
class SpeakerArc:
    speaker:              str
    dominant:             str
    mean_valence:         float
    trend:                str
    peak_negative:        Optional[UtteranceResult]
    peak_positive:        Optional[UtteranceResult]
    emotion_distribution: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "speaker":          self.speaker,
            "dominant_emotion": self.dominant,
            "mean_valence":     self.mean_valence,
            "trend":            self.trend,
            "peak_negative": (
                {"text": self.peak_negative.text[:100], "valence": self.peak_negative.valence}
                if self.peak_negative else None
            ),
            "peak_positive": (
                {"text": self.peak_positive.text[:100], "valence": self.peak_positive.valence}
                if self.peak_positive else None
            ),
            "emotion_distribution": self.emotion_distribution,
        }


@dataclass
class TranscriptSentimentReport:
    utterances:        list[UtteranceResult]
    speaker_arcs:      dict[str, SpeakerArc]
    meeting_valence:   float
    tension_moments:   list[UtteranceResult]
    consensus_moments: list[UtteranceResult]
    overall_tone:      ToneProfile

    def to_dict(self) -> dict:
        return {
            "overall": {
                "primary":          self._most_common_primary(),
                "secondary":        self._top_secondary(),
                "meeting_valence":  self.meeting_valence,
                "tone":             self.overall_tone.to_dict(),
            },
            "per_speaker": {
                spk: arc.to_dict() for spk, arc in self.speaker_arcs.items()
            },
            "tension_moments": [
                {
                    "speaker": r.speaker, "text": r.text[:100],
                    "valence": r.valence, "primary": r.primary,
                }
                for r in self.tension_moments[:5]
            ],
            "consensus_moments": [
                {
                    "speaker": r.speaker, "text": r.text[:100],
                    "valence": r.valence, "primary": r.primary,
                }
                for r in self.consensus_moments[:5]
            ],
            "utterances": [u.to_dict() for u in self.utterances],
            "source":     "fine_grained_engine_v1.1",
        }

    def _most_common_primary(self) -> str:
        counter: Counter[str] = Counter(u.primary for u in self.utterances)
        return counter.most_common(1)[0][0] if self.utterances else "factual"

    def _top_secondary(self) -> list[str]:
        counter: Counter[str] = Counter(
            label for u in self.utterances for label in u.secondary
        )
        return [label for label, _ in counter.most_common(2)]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class FineSentimentAnalyzer:
    """
    Fine-grained sentiment & tone engine for TranscriptAI.

    DSA Summary
    ───────────
    analyze_utterance()   O(L · P · T)  L=25 labels, P≈4 patterns, T=tokens
    analyze_transcript()  O(U · L · P)  U=utterances
    _negation_mask()      O(T)          deque window pass
    _scan_label()         O(P)          per-label pattern iteration
    heapq.nlargest()      O(L log k)    top-K selection
    _compute_arc()        O(U/S)        per-speaker split

    Stateless — safe to instantiate once and reuse across requests.
    """

    _SCORE_THRESHOLD: float = 0.10
    _PRIMARY_TOPK:    int   = 6
    _TENSION_CUTOFF:  float = -0.40
    _POSITIVE_CUTOFF: float = +0.42
    _FALLBACK_LABEL:  str   = "factual"

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Split text into lowercase word tokens. O(n)"""
        return re.findall(r"\b\w+\b", text.lower())

    def _negation_mask(self, text: str) -> set[int]:
        """Return token-indices falling within a negation window. O(T)"""
        tokens  = self._tokenize(text)
        negated: set[int] = set()
        neg_at  = [i for i, tok in enumerate(tokens) if _NEGATION_RE.match(tok)]
        for ni in neg_at:
            for j in range(ni + 1, min(ni + _NEGATION_WINDOW + 1, len(tokens))):
                negated.add(j)
        return negated

    def _scan_label(
        self, label: str, text: str, neg_mask: set[int]
    ) -> tuple[float, list[str]]:
        """Score one label against text. Returns (raw_score, matched_phrases). O(P)"""
        patterns = PATTERN_REGISTRY.get(label)
        if not patterns:
            return 0.0, []
        raw     = 0.0
        matched: list[str] = []
        for pat, weight in patterns:
            for m in pat.finditer(text):
                tok_idx    = len(re.findall(r"\b\w+\b", text[: m.start()]))
                is_negated = tok_idx in neg_mask
                effective  = weight * _NEGATION_DAMPEN if is_negated else weight
                raw       += effective
                if len(matched) < 5:
                    phrase = m.group(0).strip()
                    if is_negated:
                        phrase = f"[NEG] {phrase}"
                    matched.append(phrase)
        return raw, matched

    def analyze_utterance(
        self, text: str, speaker: str = "unknown"
    ) -> UtteranceResult:
        """Produce a full UtteranceResult for one utterance. O(L · P · T)"""
        if not text or not text.strip():
            return self._empty_result(speaker, text or "")

        neg_mask    = self._negation_mask(text)
        raw_scores: dict[str, float]      = {}
        phrase_map: dict[str, list[str]]  = {}

        for label in ALL_LABELS:
            score, phrases = self._scan_label(label, text, neg_mask)
            if score > 0.0:
                raw_scores[label] = score
                phrase_map[label] = phrases

        raw_scores = {
            lbl: v for lbl, v in raw_scores.items() if v >= _MIN_RAW_SCORE
        }
        total = sum(raw_scores.values()) or 1e-9
        norm: dict[str, float] = {lbl: v / total for lbl, v in raw_scores.items()}

        top = heapq.nlargest(
            self._PRIMARY_TOPK, norm.items(), key=lambda kv: kv[1]
        )
        if not top:
            top = [(self._FALLBACK_LABEL, 1.0)]

        primary   = top[0][0]
        secondary = [
            lbl for lbl, score in top[1:] if score >= self._SCORE_THRESHOLD
        ]
        valence   = sum(norm[lbl] * VALENCE_MAP.get(lbl, 0.0) for lbl in norm)
        valence   = max(-1.0, min(1.0, valence))
        confidence = min(1.0, top[0][1] * 2.5)
        signals    = [
            SentimentSignal(
                label=lbl,
                score=round(score, 3),
                valence=VALENCE_MAP.get(lbl, 0.0),
                matched_phrases=phrase_map.get(lbl, [])[:3],
            )
            for lbl, score in sorted(norm.items(), key=lambda kv: -kv[1])
            if score >= self._SCORE_THRESHOLD
        ]
        tone = self._classify_tone(text)
        return UtteranceResult(
            speaker=speaker, text=text, primary=primary, secondary=secondary,
            tone=tone, valence=round(valence, 3), confidence=round(confidence, 3),
            signals=signals,
        )

    def _classify_tone(self, text: str) -> ToneProfile:
        """Classify urgency/certainty/engagement. O(1) per modifier."""
        if _URGENCY_HIGH.search(text):
            urgency = "high"
        elif _URGENCY_MED.search(text):
            urgency = "medium"
        else:
            urgency = "low"

        if _CERTAINTY_UNCERTAIN.search(text):
            certainty = "uncertain"
        elif _CERTAINTY_HEDGED.search(text):
            certainty = "hedged"
        else:
            certainty = "definite"

        disengaged = len(_ENGAGEMENT_DISENGAGED.findall(text))
        active     = len(_ENGAGEMENT_ACTIVE.findall(text))
        if disengaged >= 2 and disengaged > active:
            engagement = "disengaged"
        elif active > 0:
            engagement = "active"
        else:
            engagement = "passive"

        return ToneProfile(
            urgency=urgency, certainty=certainty, engagement=engagement
        )

    def analyze_transcript(
        self,
        utterances:      list[dict],
        tension_cutoff:  float | None = None,
        positive_cutoff: float | None = None,
    ) -> TranscriptSentimentReport:
        """
        Analyze a full parsed transcript.
        FIX-JP2: Returns a minimal safe report when utterances is empty.
        DSA: O(U · L · P) utterance pass + O(U) aggregation.
        """
        tc = tension_cutoff  if tension_cutoff  is not None else self._TENSION_CUTOFF
        pc = positive_cutoff if positive_cutoff is not None else self._POSITIVE_CUTOFF

        # FIX-JP2: Empty utterance guard.
        # from_raw_transcript() on a Japanese-only transcript with full-width ：
        # previously returned [] here, then divided-by-zero in meeting_valence.
        # The root cause (FIX-JP1) is fixed; this guard is defense-in-depth.
        if not utterances:
            logger.debug(
                "[SentimentEngine] Empty utterance list — transcript may be "
                "Japanese-only with unsupported colon format, or genuinely empty. "
                f"Coverage note: {_JAPANESE_COVERAGE_NOTE}"
            )
            return self._empty_report()

        results:    list[UtteranceResult]              = []
        by_speaker: dict[str, list[UtteranceResult]]  = defaultdict(list)

        for utt in utterances:
            r = self.analyze_utterance(
                utt.get("text", ""), utt.get("speaker", "unknown")
            )
            results.append(r)
            by_speaker[r.speaker].append(r)

        speaker_arcs: dict[str, SpeakerArc] = {
            spk: self._compute_arc(spk, utt_list)
            for spk, utt_list in by_speaker.items()
        }

        n               = len(results)
        meeting_valence = sum(r.valence for r in results) / n
        tension_moments   = [r for r in results if r.valence < tc]
        consensus_moments = [r for r in results if r.valence > pc]
        overall_tone      = self._aggregate_tone([r.tone for r in results])

        return TranscriptSentimentReport(
            utterances=results,
            speaker_arcs=speaker_arcs,
            meeting_valence=round(meeting_valence, 3),
            tension_moments=tension_moments,
            consensus_moments=consensus_moments,
            overall_tone=overall_tone,
        )

    def from_raw_transcript(self, raw_text: str) -> TranscriptSentimentReport:
        """
        Convenience: parse raw transcript then analyze.
        FIX-JP1: _parse_transcript now handles full-width ： colon.
        DSA: O(lines) parse + O(U · L · P) analysis.
        """
        utterances = self._parse_transcript(raw_text)
        if not utterances:
            logger.debug(
                "[SentimentEngine] from_raw_transcript: no utterances parsed. "
                "Check transcript format. "
                f"Coverage note: {_JAPANESE_COVERAGE_NOTE}"
            )
        return self.analyze_transcript(utterances)

    def _compute_arc(
        self, speaker: str, results: list[UtteranceResult]
    ) -> SpeakerArc:
        """Compute sentiment trajectory for one speaker. O(U/S)"""
        dist     = Counter(r.primary for r in results)
        dominant = dist.most_common(1)[0][0]
        valences = [r.valence for r in results]
        mean_v   = sum(valences) / len(valences)

        if len(valences) >= 4:
            mid      = len(valences) // 2
            first_h  = sum(valences[:mid]) / mid
            second_h = sum(valences[mid:]) / (len(valences) - mid)
            delta    = second_h - first_h
            if abs(delta) < 0.10:
                variance = sum((v - mean_v) ** 2 for v in valences) / len(valences)
                trend    = "volatile" if variance > 0.18 else "stable"
            elif delta > 0.10:
                trend = "improving"
            else:
                trend = "declining"
        else:
            trend = "stable"

        total         = len(results)
        emotion_dist  = {
            lbl: round(cnt / total, 3) for lbl, cnt in dist.items()
        }
        return SpeakerArc(
            speaker=speaker,
            dominant=dominant,
            mean_valence=round(mean_v, 3),
            trend=trend,
            peak_negative=min(results, key=lambda r: r.valence),
            peak_positive=max(results, key=lambda r: r.valence),
            emotion_distribution=emotion_dist,
        )

    def _aggregate_tone(self, tones: list[ToneProfile]) -> ToneProfile:
        """Majority vote on each tone modifier. O(U)"""
        if not tones:
            return ToneProfile(urgency="low", certainty="definite", engagement="passive")
        u = Counter(t.urgency    for t in tones).most_common(1)[0][0]
        c = Counter(t.certainty  for t in tones).most_common(1)[0][0]
        e = Counter(t.engagement for t in tones).most_common(1)[0][0]
        return ToneProfile(urgency=u, certainty=c, engagement=e)

    @staticmethod
    def _parse_transcript(text: str) -> list[dict]:
        """
        Parse raw 'Speaker: text' or 'Speaker： text' lines.

        FIX-JP1: Added full-width colon (U+FF1A) support.
        In v1.0 the parser only matched half-width colon U+003A.
        Japanese business meetings universally use full-width colon.
        A transcript of "田中：おはようございます" returned an empty list,
        causing all Japanese speaker arcs to be empty and every speaker to
        score neutral/factual regardless of content.

        FIX-JP3: PII-masked labels like [NAME_1]: text now parse correctly.
        The character class stops at either colon form, so the bracketed
        placeholder is captured as the speaker name and the text after
        the colon is captured as the utterance content.

        DSA: O(lines) — compiled-regex scan.
        """
        # FIX-JP1: Accept both half-width ':' (U+003A) and full-width colon (U+FF1A)
        pattern = re.compile(
            r"^([^：:\n]+)[：:]\s*(.+)$",
            re.MULTILINE,
        )
        return [
            {"speaker": m.group(1).strip(), "text": m.group(2).strip()}
            for m in pattern.finditer(text)
            if m.group(2).strip()
        ]

    @staticmethod
    def _empty_result(speaker: str, text: str) -> UtteranceResult:
        return UtteranceResult(
            speaker=speaker, text=text, primary="factual", secondary=[],
            tone=ToneProfile(urgency="low", certainty="definite", engagement="passive"),
            valence=0.0, confidence=0.0, signals=[],
        )

    @staticmethod
    def _empty_report() -> TranscriptSentimentReport:
        """FIX-JP2: Safe empty report when no utterances are parsed."""
        return TranscriptSentimentReport(
            utterances=[],
            speaker_arcs={},
            meeting_valence=0.0,
            tension_moments=[],
            consensus_moments=[],
            overall_tone=ToneProfile(
                urgency="low", certainty="definite", engagement="passive"
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CONVENIENCE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_sentiment_for_json(report: TranscriptSentimentReport) -> dict:
    """Serialize a TranscriptSentimentReport to a JSON-ready dict."""
    return report.to_dict()


def legacy_to_fine_grained(flat_label: str, transcript_text: str) -> dict:
    """Upgrade a legacy flat sentiment label to fine-grained output."""
    analyzer = FineSentimentAnalyzer()
    report   = analyzer.from_raw_transcript(transcript_text)
    out      = report.to_dict()
    out["legacy_label_replaced"] = flat_label
    out["source"]                = "backstop_upgraded"
    return out


def llm_sentiment_schema_prompt() -> str:
    """Returns the JSON schema fragment for injection into the Groq/Ollama prompt."""
    labels = sorted(ALL_LABELS)
    return f"""
## Sentiment & Tone Output Schema (REQUIRED FORMAT)

Valid primary/secondary labels: {labels}

"sentiment": {{
  "overall": {{
    "primary": "<label>",
    "secondary": ["<label>", "<label>"],
    "valence": <float -1.0 to +1.0>,
    "tone": {{
      "urgency":    "low" | "medium" | "high",
      "certainty":  "definite" | "hedged" | "uncertain",
      "engagement": "active" | "passive" | "disengaged"
    }}
  }},
  "per_speaker": {{
    "<speaker_name>": {{
      "dominant_emotion": "<label>",
      "mean_valence": <float>,
      "trend": "improving" | "declining" | "stable" | "volatile",
      "emotion_distribution": {{ "<label>": <share_0_to_1> }}
    }}
  }},
  "tension_moments":   [{{ "speaker": "<name>", "text": "<excerpt≤80chars>", "valence": <float> }}],
  "consensus_moments": [{{ "speaker": "<name>", "text": "<excerpt≤80chars>", "valence": <float> }}]
}}

RULES:
- Use at least 2 labels per speaker (primary + 1 secondary if detectable).
- valence must reflect aggregated emotional tone, not just positive/negative.
- tension_moments  = utterances with valence < -0.40 (list up to 3).
- consensus_moments = utterances with valence > +0.42 (list up to 3).
- NEVER return a flat string like "frustrated" for the sentiment field.
""".strip()