---
title: TranscriptAI
emoji: 🎙️
colorFrom: pink
colorTo: yellow
sdk: streamlit
sdk_version: 1.55.0
app_file: app.py
pinned: true
license: mit
short_description: Speech & Meeting Intelligence — English · Hindi · Japanese
---

<div align="center">

# 🎙️ TranscriptAI

**Speech & Meeting Intelligence Platform**

[![Python](https://img.shields.io/badge/Python-3.10%2B-C45C74?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-486858?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20UI-D96080?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-Free%20Tier-B87830?style=flat-square)](https://console.groq.com)
[![Eval](https://img.shields.io/badge/Eval%20Score-93%25-486858?style=flat-square)](https://github.com/aiKunalBisht/Transcript-ai)
[![License](https://img.shields.io/badge/License-MIT-A8897C?style=flat-square)](LICENSE)

**[🚀 Live Demo](https://huggingface.co/spaces/KunalTheBeast/TranscriptAI) · [GitHub](https://github.com/aiKunalBisht/Transcript-ai)**

*Turn any meeting or speech into structured intelligence — summaries, action items, tone analysis, and communication risk signals.*

</div>

---

## What this is

TranscriptAI converts meeting recordings and transcripts into structured business intelligence. It goes beyond basic summarization — it understands **how** people communicate, not just **what** they say.

Three language-specific NLP layers detect the indirect communication patterns that matter in each culture:

| Language | What gets detected |
|---|---|
| **English** | Hedging ("we'll circle back"), power imbalance, escalation signals, commitment strength |
| **Hindi / Hinglish** | Indirect no ("dekhte hain"), hierarchical yes, jugaad framing, face-saving exits |
| **Japanese** | Soft rejection patterns (nemawashi), keigo formality level, code-switching |

Each layer produces output in **English**, regardless of input language.

---

## Features

### Core Analysis — every language
- **Full narrative summary** — 2–4 sentence prose paragraph describing what was discussed, decided, and what the overall outcome was
- **Key bullet points** — 3–8 concise bullets scaled to transcript length
- **Previous session panel** — shows last analysis for meeting continuity tracking
- **Meeting Health Score** — 0–100 score from 4 signals: Sentiment (30), Action Clarity (25), Communication Risk (25), AI Confidence (20)
- **Action items** — owner, deadline, confidence score, hallucination flag
- **Speaker tone intelligence** — 6-level color-coded scale: 🔴 Aggressive → 🟠 Assertive → 🟡 Neutral → 🟢 Cooperative → 🔵 Deferential → 🟣 Hesitant
- **Per-speaker sentiment** — with cultural normalization (neutral is professional in Japanese/formal Hindi contexts)
- **Talk time distribution** — sorted by speaking percentage
- **JSON export** — full structured output

### English Communication Intelligence
- **Commitment strength meter** — strong ("I will") vs weak ("I'll try") vs hedged ("we'll see")
- **Escalation signals** — "I'm going to have to escalate", "reconsider the contract"
- **Power imbalance** — "this is unacceptable", "you need to understand", "non-negotiable"
- **Corporate hedging** — "circle back", "take under advisement", "touch base"
- **Passive aggression** — "fine", "whatever works for you", "with all due respect"

### Hindi Communication Intelligence
- **Indirect no** — `dekhte hain`, `thoda mushkil hai`, `koshish karenge` (we'll try — weak commitment)
- **Hierarchical yes** — `haan haan bilkul`, `jo aap kahenge` (agreeing to please authority)
- **Face-saving exits** — `upar se baat karta hoon` (deferring up the chain)
- **Jugaad framing** — `kuch na kuch ho jayega` (vague optimism, no concrete plan)
- **Respect deflection** — `aap jo theek samjhe` (surrendering the decision)
- Detects both **Hinglish (Roman script)** and **Devanagari**

### Japanese Communication Intelligence
- **Keigo formality** — MeCab morphological analysis overrides LLM classification
- **16 nemawashi patterns** — soft rejection and deferral with confidence scores and cultural explanations
- **Code-switch counting** — deterministic JA↔EN detection via Unicode ranges
- **Cross-script normalization** — 田中 and Tanaka are the same speaker

### Production Features
- **APPI-compliant PII masking** — names, phones, emails anonymized before LLM; restored locally after
- **Hallucination guard** — 100% rule-based token overlap verification
- **Groq → Ollama → Mock fallback** — with explicit UX feedback per provider
- **Meeting trends dashboard** — soft rejection trends, hallucination drift, workload analysis
- **Live streaming** — see summary generate token by token (Groq)
- **FastAPI REST endpoint** — for CRM integration
- **MD5 result caching** — 24-hour TTL
- **JSONL observability logging** — with drift detection

---

## Language Routing

```
detect_language(transcript)
       │
       ├── Japanese / Mixed-JA ──→ Japanese NLP layer
       │                            Keigo · Nemawashi · Soft rejection
       │
       ├── English ─────────────→ English NLP layer
       │                            Hedging · Power · Commitment · Escalation
       │
       ├── Hindi / Hinglish ─────→ Hindi NLP layer
       │                            Deferral · Hierarchy · Indirect No
       │
       └── Mixed EN+HI ──────────→ Both layers, two-column view
```

All outputs are always in English.

---

## Evaluation — 93% Overall Score

Custom evaluation system with bilingual ground truth. Cultural corrections applied — standard NLP evaluation incorrectly labels Japanese professional neutral speech as "positive" (Western bias).

| Metric | Score | Grade |
|---|---|---|
| **Overall** | **93%** | EXCELLENT |
| Action Items F1 | 1.0 | EXCELLENT |
| Sentiment (soft) | 1.0 | EXCELLENT |
| Summary Semantic | 0.525 | FAIR |
| Hallucination Risk | LOW | ✅ |

**Iteration history:**

| Version | Score | Key improvement |
|---|---|---|
| v1 | 30% | Baseline — exact matching only |
| v2 | 55% | Fuzzy names, rule-based code-switch, semantic similarity |
| v3 | 75% | Cultural ground truth, JA tokenization, soft sentiment scoring |
| v4 | 83% | Hallucination guard, speaker sort fix, nemawashi filter |
| v5 | **93%** | Sentiment rules, optimal bullet matching, tone intelligence |

---

## Meeting Health Score

Every analysis produces a 0–100 score:

```
Score = Sentiment (30pts) + Action Clarity (25pts) + Communication Risk (25pts) + AI Confidence (20pts)
```

| Score | Label | Color |
|---|---|---|
| 80–100 | Productive Meeting | 🟢 Green |
| 60–79 | Mostly Aligned | 🟡 Amber |
| 40–59 | Needs Follow-up | 🟠 Orange |
| 0–39 | High Risk | 🔴 Red |

---

## Data Residency & APPI Compliance

**With Ollama (fully local):**
- Zero data leaves your machine
- Suitable for regulated industries

**With Groq (cloud, APPI-compliant):**
- PII masking runs locally before any data is sent
- Names → `[NAME_1]`, phones → `[PHONE_1]`, emails → `[EMAIL_1]`
- LLM processes anonymized text only
- PII restored locally after analysis

**PII restore is robust** — handles all 4 LLM bracket-stripping variants: `[NAME_3]`, `NAME_3`, `[NAME_3`, `NAME_3]`

---

## Architecture

```
transcription/
  pii_masker.py          APPI anonymization — runs before any LLM call (v3)
  speaker_normalizer.py  Cross-script identity (田中 ↔ Tanaka ↔ Director)
  audio_processor.py     Whisper transcription (Groq free / local)

analysis/
  analyzer.py            Groq → Ollama → Mock · trilingual detection · tone schema
  english_analyzer.py    English NLP — hedging, power, escalation, commitment
  hindi_analyzer.py      Hindi NLP — deferral, hierarchy, jugaad, face-saving
  hallucination_guard.py 100% rule-based claim verification
  soft_rejection.py      16-pattern Japanese soft rejection detector
  semantic_validator.py  Three-tier similarity engine
  japanese_tokenizer.py  MeCab morphological analysis

utils/
  logger.py              JSONL logging + trend analysis engine
  cache.py               MD5 result caching (24-hour TTL)
  evaluator.py           ROUGE + semantic + F1 + optimal bullet matching (v3)
  language_intelligence.py  Language-aware feature routing

api.py                   FastAPI REST endpoints
app.py                   Streamlit UI — translucent navbar · 7 tabs · health score
async_processor.py       ThreadPoolExecutor job queue
```

**Processing order (sequence matters):**

```
1. PII masking           local — before LLM
2. LLM analysis          Groq / Ollama / Mock
3. PII restoration       local — BEFORE speaker normalization
4. Speaker normalization now sees real names, not [NAME_1]
5. Tone classification   6-level color-coded scale
6. Hallucination guard   rule-based, not LLM
7. NLP layer routing     Japanese / English / Hindi based on language
8. Cache + log           local JSONL
```

---

## Cost at Scale

| Volume | Provider | Monthly cost |
|---|---|---|
| Up to 500 analyses/day | Groq free tier | **$0** |
| 10,000/month | Groq paid | **~$2** |
| Unlimited | Ollama (local) | **$0** |

---

## Installation

```bash
git clone https://github.com/aiKunalBisht/Transcript-ai.git
cd Transcript-ai
pip install -r requirements.txt
```

**Run:**
```bash
export GROQ_API_KEY=your_key_here
python -m streamlit run app.py
```

**Fully local:**
```bash
ollama pull qwen3:8b
python -m streamlit run app.py
```

**Optional capabilities:**
```bash
pip install fugashi unidic-lite    # MeCab Japanese tokenizer
pip install scikit-learn           # TF-IDF semantic similarity
pip install sentence-transformers  # True semantic understanding
```

---

## REST API

```bash
python api.py
# Docs: http://localhost:8000/docs
```

```python
import requests
r = requests.post("http://localhost:8000/analyze", json={
    "transcript": "Rahul: Aaj hum Q3 discuss karenge. Priya: Dekhte hain.",
    "language": "hi",
    "mask_pii": True
})
print(r.json()["result"]["soft_rejections"]["risk_level"])  # HIGH
```

---

## Known Limitations

| Limitation | Current | Path |
|---|---|---|
| Names not as speaker labels | May not be masked if not in surname DB | spaCy NER |
| Speaker diarization | Silence-gap heuristic ~70% | pyannote.audio |
| Audio transcription on HF | Not available (Whisper not in requirements) | Groq Whisper API |
| Confidence scores | Heuristic, not probabilistic | Labeled dataset + calibration |
| Evaluation | 3 synthetic test cases | External validation |

---

## Codebase

| Metric | Value |
|---|---|
| Python files | 19 |
| Lines of code | 6,000+ |
| English NLP patterns | 40+ |
| Hindi NLP patterns | 30+ |
| Japanese soft rejection patterns | 16 |
| Japanese surname database | 500+ |
| Supported input formats | TXT · VTT · JSON · MP4 · MP3 · WAV · M4A |
| Evaluation score | **93% (EXCELLENT)** |

---

<div align="center">

Built by [Kunal Bisht](https://github.com/aiKunalBisht) · Pithoragarh, India

</div>