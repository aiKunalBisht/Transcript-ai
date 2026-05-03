# hindi_analyzer.py
# Hindi / Hinglish Communication Intelligence Layer
#
# Detects communication patterns unique to Hindi business speech:
# - Classic Indian deferral (kal dekhenge, dekhte hain)
# - Hierarchical yes (agreeing to please authority)
# - Indirect no (thoda mushkil, dekhte hain)
# - Face-saving exits (upar se baat karta hoon)
# - Jugaad framing (kuch na kuch ho jayega)
# - Respect deflection (aap jo theek samjhe)
# Handles both Hinglish (Roman script) and Devanagari

import re

# ── PATTERN LIBRARY ───────────────────────────────────────────────────────────

HINDI_PATTERNS = [

    # ── INDIRECT NO / SOFT REJECTION ─────────────────────────────────────────
    {"phrase": "dekhte hain",         "devanagari": "देखते हैं",     "reading": "We'll see — classic Indian soft no",                      "category": "INDIRECT_NO",    "confidence": 0.80, "severity": "HIGH"},
    {"phrase": "dekh lete hain",      "devanagari": "देख लेते हैं",  "reading": "We'll look into it — deferral, likely no action",         "category": "INDIRECT_NO",    "confidence": 0.75, "severity": "HIGH"},
    {"phrase": "thoda mushkil hai",   "devanagari": "थोड़ा मुश्किल है","reading": "A little difficult — Hindi soft rejection",              "category": "INDIRECT_NO",    "confidence": 0.85, "severity": "HIGH"},
    {"phrase": "mushkil lagta hai",   "devanagari": "मुश्किल लगता है","reading": "Seems difficult — indirect refusal",                     "category": "INDIRECT_NO",    "confidence": 0.82, "severity": "HIGH"},
    {"phrase": "kal dekhenge",        "devanagari": "कल देखेंगे",    "reading": "We'll see tomorrow — classic Indian deferral, rarely happens","category": "INDIRECT_NO", "confidence": 0.85, "severity": "HIGH"},
    {"phrase": "kal tak dekh lete",   "devanagari": "कल तक देख लेते","reading": "Will look by tomorrow — very likely won't happen",        "category": "INDIRECT_NO",    "confidence": 0.80, "severity": "HIGH"},
    {"phrase": "baad mein karte hain","devanagari": "बाद में करते हैं","reading": "We'll do it later — indefinite deferral",               "category": "INDIRECT_NO",    "confidence": 0.75, "severity": "MEDIUM"},
    {"phrase": "abhi nahi ho sakta",  "devanagari": "अभी नहीं हो सकता","reading": "Can't happen right now — partial refusal",             "category": "INDIRECT_NO",    "confidence": 0.78, "severity": "MEDIUM"},
    {"phrase": "try karenge",         "devanagari": "ट्राई करेंगे",  "reading": "Will try — no commitment to outcome",                     "category": "INDIRECT_NO",    "confidence": 0.60, "severity": "MEDIUM"},
    {"phrase": "koshish karenge",     "devanagari": "कोशिश करेंगे",  "reading": "Will make an effort — weak commitment",                   "category": "INDIRECT_NO",    "confidence": 0.65, "severity": "MEDIUM"},
    {"phrase": "sochna padega",       "devanagari": "सोचना पड़ेगा",  "reading": "Will have to think — deferral, avoiding direct answer",   "category": "INDIRECT_NO",    "confidence": 0.70, "severity": "MEDIUM"},
    {"phrase": "pata nahi",           "devanagari": "पता नहीं",      "reading": "Don't know — uncertainty or avoiding commitment",         "category": "INDIRECT_NO",    "confidence": 0.55, "severity": "LOW"},

    # ── HIERARCHICAL YES ─────────────────────────────────────────────────────
    {"phrase": "haan haan bilkul",    "devanagari": "हाँ हाँ बिलकुल","reading": "Rapid over-agreement — agreeing to please authority, may not follow through","category": "HIERARCHICAL_YES","confidence": 0.80, "severity": "MEDIUM"},
    {"phrase": "ji bilkul",           "devanagari": "जी बिलकुल",     "reading": "Respectful over-agreement — hierarchical compliance",     "category": "HIERARCHICAL_YES","confidence": 0.70, "severity": "MEDIUM"},
    {"phrase": "aap ka order hai",    "devanagari": "आप का ऑर्डर है","reading": "It's your order — surrendering agency to authority",     "category": "HIERARCHICAL_YES","confidence": 0.85, "severity": "HIGH"},
    {"phrase": "jo aap kahenge",      "devanagari": "जो आप कहेंगे", "reading": "Whatever you say — complete deference, no independent view","category": "HIERARCHICAL_YES","confidence": 0.80, "severity": "HIGH"},
    {"phrase": "aap sahi keh rahe",   "devanagari": "आप सही कह रहे","reading": "You're right — reflexive agreement with authority",        "category": "HIERARCHICAL_YES","confidence": 0.65, "severity": "LOW"},

    # ── RESPECT DEFLECTION ────────────────────────────────────────────────────
    {"phrase": "aap jo theek samjhe", "devanagari": "आप जो ठीक समझे","reading": "Whatever you think is right — surrendering decision, not agreement","category": "RESPECT_DEFLECTION","confidence": 0.85, "severity": "HIGH"},
    {"phrase": "aapki marzi",         "devanagari": "आपकी मर्जी",    "reading": "As you wish — passive acceptance, distancing from outcome","category": "RESPECT_DEFLECTION","confidence": 0.75, "severity": "MEDIUM"},
    {"phrase": "aap decide karo",     "devanagari": "आप डिसाइड करो", "reading": "You decide — abdicating responsibility to senior",        "category": "RESPECT_DEFLECTION","confidence": 0.70, "severity": "MEDIUM"},
    {"phrase": "jo sahib bolein",     "devanagari": "जो साहब बोलें", "reading": "Whatever the boss says — extreme deference",              "category": "RESPECT_DEFLECTION","confidence": 0.90, "severity": "HIGH"},

    # ── JUGAAD / VAGUE OPTIMISM ───────────────────────────────────────────────
    {"phrase": "kuch na kuch ho jayega","devanagari": "कुछ न कुछ हो जाएगा","reading": "Something will work out — no concrete plan, false reassurance","category": "JUGAAD","confidence": 0.80, "severity": "HIGH"},
    {"phrase": "manage ho jayega",    "devanagari": "मैनेज हो जाएगा","reading": "It'll get managed — vague optimism, no ownership",        "category": "JUGAAD",         "confidence": 0.75, "severity": "MEDIUM"},
    {"phrase": "adjust kar lenge",    "devanagari": "एडजस्ट कर लेंगे","reading": "We'll adjust — avoids committing to specific plan",      "category": "JUGAAD",         "confidence": 0.70, "severity": "MEDIUM"},
    {"phrase": "ho jayega",           "devanagari": "हो जाएगा",      "reading": "It'll happen — passive framing, no active commitment",    "category": "JUGAAD",         "confidence": 0.55, "severity": "LOW"},
    {"phrase": "chalta hai",          "devanagari": "चलता है",       "reading": "It's fine / It goes — accepting substandard without objection","category": "JUGAAD",   "confidence": 0.65, "severity": "MEDIUM"},

    # ── FACE-SAVING EXIT ──────────────────────────────────────────────────────
    {"phrase": "upar se baat karta hoon","devanagari": "ऊपर से बात करता हूँ","reading": "Will talk to someone above — deferring up the chain, avoiding direct answer","category": "FACE_SAVING","confidence": 0.85, "severity": "HIGH"},
    {"phrase": "senior se poochhna padega","devanagari": "सीनियर से पूछना पड़ेगा","reading": "Need to ask senior — escalating upward to avoid commitment","category": "FACE_SAVING","confidence": 0.80, "severity": "HIGH"},
    {"phrase": "main approve nahi kar sakta","devanagari": "मैं अप्रूव नहीं कर सकता","reading": "I can't approve — honest boundary or face-saving exit","category": "FACE_SAVING","confidence": 0.75, "severity": "MEDIUM"},
    {"phrase": "upar walon se milna padega","devanagari": "ऊपर वालों से मिलना पड़ेगा","reading": "Will need to meet the higher-ups — bureaucratic deferral","category": "FACE_SAVING","confidence": 0.80, "severity": "HIGH"},
]

# Category display config
CATEGORY_CONFIG = {
    "INDIRECT_NO":       ("Indirect No",        "#B04040", "#FAF0F0"),
    "HIERARCHICAL_YES":  ("Hierarchical Yes",   "#C87030", "#FDF0EA"),
    "RESPECT_DEFLECTION":("Respect Deflection", "#986820", "#FAF0E0"),
    "JUGAAD":            ("Jugaad / Vague Plan", "#486858", "#EDF3EF"),
    "FACE_SAVING":       ("Face-Saving Exit",   "#6A4080", "#F5EEF8"),
}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def _find_speaker(transcript: str, phrase: str) -> str:
    idx = transcript.lower().find(phrase.lower())
    if idx == -1:
        # Also search Devanagari
        for p in HINDI_PATTERNS:
            if p.get("phrase") == phrase and p.get("devanagari") in transcript:
                idx = transcript.find(p["devanagari"])
                break
    if idx == -1:
        return "Unknown"
    preceding = transcript[:idx]
    matches   = re.findall(r"([^\s\[\]:：\n]{2,30}?)\s*[:：]", preceding)
    speakers  = [m.strip() for m in matches if not re.match(r"^\d+$", m.strip())]
    return speakers[-1] if speakers else "Unknown"


def detect_hindi_patterns(transcript: str) -> dict:
    """
    Detects Hindi/Hinglish communication patterns.
    Checks both Roman script (Hinglish) and Devanagari.
    All output is in English.
    """
    text_lower = transcript.lower()
    detected   = []

    for pattern in HINDI_PATTERNS:
        phrase    = pattern["phrase"]
        devanagari = pattern.get("devanagari", "")
        found_at  = ""

        if phrase.lower() in text_lower:
            found_at = phrase
        elif devanagari and devanagari in transcript:
            found_at = devanagari

        if found_at:
            speaker = _find_speaker(transcript, found_at)
            idx     = transcript.lower().find(found_at.lower()) if found_at == phrase else transcript.find(found_at)
            start   = max(0, idx - 100)
            end     = min(len(transcript), idx + len(found_at) + 100)
            context = transcript[start:end].strip()

            detected.append({
                "phrase":      found_at,
                "roman":       phrase,
                "devanagari":  devanagari,
                "reading":     pattern["reading"],
                "category":    pattern["category"],
                "confidence":  pattern["confidence"],
                "severity":    pattern["severity"],
                "speaker":     speaker,
                "context":     context[:120],
            })

    high   = [d for d in detected if d["severity"] == "HIGH"]
    medium = [d for d in detected if d["severity"] == "MEDIUM"]
    low    = [d for d in detected if d["severity"] == "LOW"]

    indirect_no  = [d for d in detected if d["category"] == "INDIRECT_NO"]
    hier_yes     = [d for d in detected if d["category"] == "HIERARCHICAL_YES"]
    face_saving  = [d for d in detected if d["category"] == "FACE_SAVING"]
    jugaad       = [d for d in detected if d["category"] == "JUGAAD"]
    deflection   = [d for d in detected if d["category"] == "RESPECT_DEFLECTION"]

    if len(high) >= 2 or (indirect_no and face_saving):
        risk_level   = "HIGH"
        risk_summary = "Multiple strong deferral or avoidance signals detected. Commitments made in this conversation are unlikely to be followed through."
    elif len(high) >= 1 or len(medium) >= 2:
        risk_level   = "MEDIUM"
        risk_summary = "Indirect refusal or significant deference detected. Verify commitments explicitly in writing."
    elif detected:
        risk_level   = "LOW"
        risk_summary = "Some soft signals present. Communication is mostly cooperative but watch for follow-through."
    else:
        risk_level   = "NONE"
        risk_summary = "No indirect communication patterns detected."

    return {
        "detected":         detected,
        "high_signals":     high,
        "medium_signals":   medium,
        "low_signals":      low,
        "total_signals":    len(detected),
        "indirect_no":      indirect_no,
        "hierarchical_yes": hier_yes,
        "face_saving":      face_saving,
        "jugaad":           jugaad,
        "deflection":       deflection,
        "risk_level":       risk_level,
        "risk_summary":     risk_summary,
        "cultural_note":    "In Indian business culture, direct refusal is often avoided to maintain relationships and respect hierarchy. Phrases like 'dekhte hain' or 'kal dekhenge' frequently mean no. Always confirm commitments in writing.",
    }


if __name__ == "__main__":
    sample = """
Rahul: Kya aap Friday tak yeh report de sakte ho?
Priya: Haan haan bilkul, dekhte hain. Thoda mushkil hai lekin koshish karenge.
Rahul: Confirm kar sakte ho?
Priya: Aap jo theek samjhe. Main upar se baat karta hoon. Kuch na kuch ho jayega.
"""
    result = detect_hindi_patterns(sample)
    print(f"Risk: {result['risk_level']} — {result['risk_summary']}")
    for d in result["detected"]:
        print(f"  [{d['severity']}] '{d['phrase']}' ({d['category']}) — {d['reading']}")