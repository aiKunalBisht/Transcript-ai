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

# TranscriptAI

**Meeting intelligence that understands not just what was said — but what was meant.**

![CI](https://github.com/aiKunalBisht/Transcript-ai/actions/workflows/ci.yml/badge.svg)


[![Live Demo](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-FF9D00?style=for-the-badge)](https://huggingface.co/spaces/KunalTheBeast/TranscriptAI)
[![GitHub](https://img.shields.io/badge/GitHub-Source-181717?style=for-the-badge&logo=github)](https://github.com/aiKunalBisht/Transcript-ai)
[![Eval Score](https://img.shields.io/badge/Eval%20Score-93%25-brightgreen?style=for-the-badge)]()
[![License MIT](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)]()

Trilingual · English · Hindi · Japanese

</div>

---

## The Problem

Most meeting tools extract *what* was said. They miss everything underneath.

Every language and culture has indirect communication patterns — polite rejections, soft commitments, face-saving agreements — that a generic summarizer will log as action items that never get done.

TranscriptAI is built to catch exactly those signals.

---

## Live Demo

**[→ Try it on Hugging Face](https://huggingface.co/spaces/KunalTheBeast/TranscriptAI)**

No setup. No API key. Paste any transcript and get structured intelligence in seconds.

**Example — what a generic tool misses:**

| What was said | Generic AI output | TranscriptAI output |
|---|---|---|
| Indirect verbal agreement | ✅ Action item logged | ⚠️ Soft commitment — low follow-through probability |
| Japanese polite consideration phrase | ✅ Action item logged | 🔴 72% rejection confidence — request written confirmation |
| Corporate hedge — "we'll circle back" | 📝 Meeting note | 🌀 No concrete next step — escalation recommended |
| Enthusiastic but hierarchical yes | ✅ Agreement confirmed | 🟠 Agreeing to please, not necessarily to act |

---

## Output

For every transcript, TranscriptAI produces:

- **Summary** — concise narrative paragraph plus key bullet points scaled to meeting length
- **Action items** — extracted with owner, deadline, and commitment strength rating
- **Communication risk signals** — indirect rejections, hedging language, power imbalance markers
- **Speaker tone profile** — 6-level colour-coded scale with intensity score per speaker
- **Meeting health score** — 0 to 100 composite across sentiment, action clarity, risk, and AI confidence
- **Session trends** — risk drift, hallucination rate, and workload patterns across meetings

---

## Language Engines

Three independent NLP modules, auto-detected from transcript content.

### English
Commitment strength grading distinguishes "I will deliver" from "I will try" from "we will see." Detects escalation signals, power imbalance language, passive aggression, and corporate hedging. Over 40 patterns across 4 categories.

### Hindi
Identifies indirect refusals, hierarchical agreement (saying yes to please rather than commit), face-saving exits, and vague reassurances. Handles both Roman script and Devanagari. Over 30 patterns.

### Japanese
16 nemawashi soft-rejection patterns with per-pattern confidence scores. Keigo formality detection via MeCab morphological analysis. Cross-script speaker normalization — the same person written in kanji and in romanization resolves to a single speaker identity.

---

## Architecture

```
transcription/
  pii_masker.py           Local anonymization — runs before any LLM call
  speaker_normalizer.py   Cross-script speaker identity resolution
  audio_processor.py      Whisper transcription pipeline

analysis/
  analyzer.py             LLM orchestration — Groq → Ollama → Mock fallback
  english_analyzer.py     English NLP engine
  hindi_analyzer.py       Hindi NLP engine
  soft_rejection.py       Japanese nemawashi detector
  hallucination_guard.py  Rule-based output verification
  japanese_tokenizer.py   MeCab morphological analysis

utils/
  evaluator.py            ROUGE-L + F1 + semantic similarity scoring
  cache.py                MD5 result caching — 24h TTL
  logger.py               JSONL observability and trend analysis

app.py                    Streamlit UI — 7 tabs, health score, trend dashboard
api.py                    FastAPI REST endpoints
```

**Processing pipeline — order is strict:**

```
1. PII Mask       local, before LLM          (privacy compliance)
2. LLM Analysis   Groq / Ollama / Mock
3. PII Restore    local, before normalization
4. Normalize      cross-script speaker deduplication
5. Tone Classify  per-speaker 6-level scoring
6. NLP Layer      language-specific signal detection
7. Cache + Log    MD5 cache write, JSONL append
```

---

## Evaluation

Standard NLP metrics carry Western assumptions. Formal neutral speech in Japanese or indirect communication in South Asian business contexts scores poorly on metrics calibrated for direct English. This project uses a custom evaluation framework with cultural corrections applied at each version iteration.

| Version | Score | Primary Change |
|---------|-------|----------------|
| v1 | 30% | Baseline — exact string matching |
| v2 | 55% | Fuzzy matching, semantic similarity |
| v3 | 75% | Cultural ground truth, Japanese tokenization |
| v4 | 83% | Hallucination guard, soft rejection filter |
| v5 | **93%** | Tone intelligence, optimal bullet assignment |

| Metric | Result |
|--------|--------|
| Action Item F1 | 1.0 — Excellent |
| Sentiment (cultural) | 1.0 — Excellent |
| Hallucination Risk | Low |
| Overall | **93%** |

---

## Production Features

**Privacy**
PII anonymization runs locally before any transcript reaches an LLM. Names, phone numbers, and email addresses are masked on input and restored on output. No personal data is transmitted.

**Reliability**
Three-tier LLM fallback — Groq (1–2s, free tier) → Ollama (local, zero cost) → Mock (always available). MD5 result caching with 24-hour TTL means repeat queries return in under one second.

**Observability**
Every analysis is written to a local JSONL log. A built-in trends dashboard tracks soft rejection rates, hallucination drift, and workload distribution across sessions.

**Integration**
FastAPI REST endpoint at `/analyze` for direct integration with CRM systems, Slack bots, or downstream pipelines.

---

## Quick Start

```bash
git clone https://github.com/aiKunalBisht/Transcript-ai.git
cd Transcript-ai
pip install -r requirements.txt
```

**Cloud — Groq (recommended, free tier)**
```bash
export GROQ_API_KEY=your_key_here    # console.groq.com
python -m streamlit run app.py
```

**Local — fully offline, zero data leaves your machine**
```bash
ollama pull qwen3:8b
python -m streamlit run app.py
```

**Optional dependencies**
```bash
pip install fugashi unidic-lite        # MeCab Japanese tokenizer
pip install scikit-learn               # TF-IDF semantic similarity
pip install sentence-transformers      # Neural semantic scoring
```

---

## REST API

```bash
python api.py
# Interactive docs at http://localhost:8000/docs
```

```python
import requests

response = requests.post("http://localhost:8000/analyze", json={
    "transcript": "Alex: Can we get this delivered by Friday?\nJordan: We will see what we can do.",
    "language": "en",
    "mask_pii": True
})

result = response.json()["result"]
print(result["soft_rejections"]["risk_level"])    # HIGH
print(result["soft_rejections"]["risk_summary"])  # Commitment unlikely to be followed through
```

---

## Known Limitations

| Limitation | Planned Improvement |
|------------|---------------------|
| Speaker diarization ~70% accuracy | pyannote.audio integration |
| Audio upload unavailable on HF Spaces | Groq Whisper API — next release |
| Confidence scores are heuristic | Labeled dataset and calibration |
| Demo uses synthetic test cases | Real-world transcript validation ongoing |

---

## Project Scale

19 Python files · 6,000+ lines · 90+ functions
86 linguistic patterns across 3 languages · 500+ Japanese surname entries
Supported formats: TXT · VTT · JSON · MP4 · MP3 · WAV · M4A

---

<div align="center">

Built by [Kunal Bisht](https://github.com/aiKunalBisht) — Pithoragarh, India

[Hugging Face](https://huggingface.co/KunalTheBeast) · [LinkedIn](https://linkedin.com/in/kunalhere) · [GitHub](https://github.com/aiKunalBisht)

</div>