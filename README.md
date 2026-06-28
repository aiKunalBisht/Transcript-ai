---
title: TranscriptAI
emoji: 🧠
colorFrom: pink
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

<img src="https://img.shields.io/badge/🧠-TranscriptAI-D96080?style=for-the-badge&labelColor=1a0a0f" alt="TranscriptAI"/>

**Multilingual Meeting Intelligence · Japanese · Hindi · English · Mixed**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Space-FF4B4B?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/KunalTheBeast/TranscriptAI)
[![GitHub](https://img.shields.io/badge/Source-GitHub-3C2416?style=flat-square&logo=github)](https://github.com/aiKunalBisht/Transcript-ai)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Accuracy](https://img.shields.io/badge/Accuracy-93.8%25-D96080?style=flat-square)](https://huggingface.co/spaces/KunalTheBeast/TranscriptAI)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

_Turns any meeting transcript into structured business intelligence in ~3 seconds._

</div>

_Turns any meeting transcript into structured business intelligence in ~3 seconds._

## The Problem No Generic AI Tool Solves

Generic meeting summarisers extract what was **said**. They miss what was **meant**.

| What was said                                  | Generic AI                      | TranscriptAI                                              |
| ---------------------------------------------- | ------------------------------- | --------------------------------------------------------- |
| 検討いたします                                 | "Action: We will consider it"   | ⚠ Soft rejection — 72% confidence. Follow up explicitly.  |
| 難しいかもしれません                           | "It may be difficult" — neutral | 🚨 HIGH rejection signal — 90% confidence. Deal at risk.  |
| 前向きに検討                                   | "Positive consideration"        | ⚠ Uncertain — outcome not guaranteed (55%)                |
| 承知いたしました                               | "Acknowledged"                  | 🏯 High keigo register — senior speaker or formal context |
| 善処します                                     | "Action: We will handle it"     | 🚨 Classic nemawashi dodge — no real commitment made      |
| パートナーシップは継続しないことを決定しました | "Decision made"                 | ⛔ CRITICAL — Explicit contract termination. Irrevocable. |

Japanese enterprise also mandates APPI compliance — raw meeting data cannot be sent to foreign cloud LLMs. Most tools fail this requirement by design. TranscriptAI masks PII locally before any LLM call.

---

## What It Does

```
Input: Any transcript (JP · HI · EN · Mixed) or Audio file (MP3/MP4/WAV)
Output: Structured business intelligence in ~3 seconds
```

- **Trilingual analysis** — Japanese / Hindi (Hinglish) / English / Mixed in a single meeting
- **Keigo detection** — MeCab morphological analysis extracts formality register (High / Medium / Low)
- **20 soft rejection patterns** — nemawashi, 難しいですね, ぜひ検討, and more with confidence scores
- **CRITICAL risk tier** — explicit contract termination detection, separate from soft refusals
- **8 Hindi indirect patterns** — देखते हैं, थोड़ा मुश्किल है, कोशिश करेंगे and more
- **APPI-compliant PII masking** — 500+ Japanese surnames, local NER, before any LLM call
- **Hallucination guard** — 100% rule-based token overlap; LLM never validates its own output
- **Meeting Health Score** — 0–100 across sentiment, action clarity, communication risk, AI confidence
- **議事録 export** — structured Japanese business minutes in standard enterprise format
- **Cultural insights export** — 根回し risk, 稟議 approval status, nemawashi pattern breakdown
- **Groq Whisper** — MP4 / MP3 / WAV audio transcription
- **Vector cache** — ChromaDB semantic similarity for instant return on similar transcripts
- **2-key Groq rotation** — auto-failover on 429 rate limits

| What was said                                  | Generic AI                      | TranscriptAI                                |
| ---------------------------------------------- | ------------------------------- | ------------------------------------------- |
| 検討いたします                                 | "Action: We will consider it"   | ⚠ Soft rejection — 72% confidence           |
| 難しいかもしれません                           | "It may be difficult" — neutral | 🚨 HIGH rejection signal — deal at risk     |
| パートナーシップは継続しないことを決定しました | "Decision made"                 | ⛔ CRITICAL — Explicit contract termination |

## Accuracy History

| Version       | Key change                                                  | Accuracy  |
| ------------- | ----------------------------------------------------------- | --------- |
| v1            | Hard exact matching, English-only                           | 22–30%    |
| v2            | Fuzzy speaker names, TF-IDF similarity                      | ~45%      |
| v3            | MeCab keigo override, bilingual ground truth                | ~60%      |
| v4            | Hallucination guard, nemawashi patterns, APPI masking       | 75–85%    |
| **v5 (live)** | 2-key rotation, vector cache, MLflow, bypass_cache eval fix | **93.8%** |

Every rebuild was driven by evaluation metric failures traced through the pipeline — not intuition. When action F1 was 0.4 at v2, tracing showed the owner field extracted role titles ("Director") instead of first names. One prompt instruction fixed it.

---

## Evaluation Metrics · v5 Live

| Test case                   | Overall   | ROUGE-1 | Action F1 | Sentiment |
| --------------------------- | --------- | ------- | --------- | --------- |
| Sales call · JA/EN mixed    | **94.5%** | 0.694   | 1.0       | 1.0       |
| Internal meeting · Japanese | **93.8%** | 0.703   | 1.0       | 1.0       |
| Client conflict · EN/JA     | **93.8%** | 0.703   | 1.0       | 1.0       |

---

## Architecture — 11-Stage Pipeline

```
Input transcript
    │
    ▼
 1  Vector cache check       utils/vector_cache.py          ChromaDB cosine similarity
 2  MD5 exact cache          utils/cache.py                 Instant return if identical
 3  PII masking              transcription/pii_masker.py    APPI — BEFORE LLM
 4  LLM analysis             analysis/analyzer.py           Groq key1 → key2 → mock
 5  PII restoration          transcription/pii_masker.py    BEFORE speaker normalization
 6  Speaker normalization    transcription/speaker_normalizer.py
 7  MeCab keigo override     analysis/japanese_tokenizer.py
 8  Code-switch count        utils/evaluator.py             Rule-based Unicode ranges
 9  Hallucination guard      analysis/hallucination_guard.py
10  Soft rejection detection analysis/soft_rejection_detector.py
11  Cache + log              utils/vector_cache.py + utils/logger.py
```

**Critical ordering constraint:** PII must be masked before the LLM sees the text. PII must be restored before speaker normalization — otherwise the normalizer receives `[NAME_1]` and cannot resolve `田中 ↔ Tanaka ↔ Director` as the same person.

---

## Technology Decisions

| Decision      | Chosen             | Over               | Reason                                                                                |
| ------------- | ------------------ | ------------------ | ------------------------------------------------------------------------------------- |
| Vector DB     | **ChromaDB**       | Pinecone, Weaviate | Free, local, HF Spaces compatible, APPI compliant                                     |
| LLM inference | **Groq**           | OpenAI, Anthropic  | 10–20× faster (LPU vs GPU). Free tier. JSON mode.                                     |
| Japanese NLP  | **MeCab + IPADIC** | spaCy ja, Fugashi  | Morpheme-level auxiliary verb detection — keigo is invisible to word-level tokenizers |
| Web framework | **FastAPI**        | Flask, Django      | Native async with `asyncio.to_thread()`. Auto Swagger.                                |
| Eval tracking | **MLflow**         | W&B, Neptune       | Free, local SQLite, APPI compliant                                                    |

---

## Quick Start

```bash
git clone https://github.com/aiKunalBisht/Transcript-ai.git
cd Transcript-ai
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here

# FastAPI (docs at localhost:7860/docs)
uvicorn main:app --reload --port 7860
```

**Local inference (zero cost, full APPI data residency):**

```bash
ollama pull qwen3:8b
# App auto-detects Ollama — no config needed
```

---

## Accuracy

```
main.py                           FastAPI server + route handlers
analysis/
  analyzer.py                     LLM orchestration + 2-key Groq rotation
  soft_rejection_detector.py      20 JP nemawashi + 8 HI indirect + CRITICAL termination
  japanese_tokenizer.py           MeCab keigo extraction
  hallucination_guard.py          Rule-based token overlap validation
  english_analyzer.py             English NLP patterns
  hindi_analyzer.py               Hindi/Hinglish indirect signals
agents/
  gijiroku_formatter.py           議事録 Japanese business minutes generator
  cultural_insights_formatter.py  Cultural context layer for JP meetings
  slide_architect.py              PPTX slide plan generation
exporters/
  pptx_builder.py                 PowerPoint export builder
transcription/
  pii_masker.py                   APPI compliance — local PII masking
  audio_processor.py              Groq Whisper audio transcription
utils/
  html_renderer.py                Results renderer — health score, tabs, insights
  vector_cache.py                 ChromaDB semantic cache
  logger.py                       JSONL trends + drift detection
templates/
  base.html                       Sidebar, navigation, session persistence
  index.html                      Main analysis page
  export.html                     Export page — PPTX, 議事録, MD, JSON, TXT
static/
  style.css                       Full responsive CSS — mobile, tablet, desktop
tests/
  test_data.py                    3 bilingual ground truth test cases
```

---

## REST API

```bash
# Analyze a transcript
POST /analyze-text
  transcript: str
  language:   str | null   # auto-detect if null
  mask_pii:   bool         # default true

# Exports
POST /export/pptx
POST /export/gijiroku
POST /export/cultural-insights
POST /export/markdown
POST /export/json
POST /export/txt

# Health
GET /health
```

---

## Performance

| Metric                        | Score |
| ----------------------------- | ----- |
| Lighthouse Performance        | 94    |
| Lighthouse Accessibility      | 100   |
| CLS (Cumulative Layout Shift) | 0     |
| Speed Index                   | 1.2s  |

---

<div align="center">

**Built by [Kunal Bisht](https://linkedin.com/in/kunalhere)**
AI Engineer · LLM Systems & RAG Pipelines · Multilingual NLP
Uttarakhand, India · Open to Remote / Relocation

[![LinkedIn](https://img.shields.io/badge/LinkedIn-kunalhere-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/kunalhere)
[![Email](https://img.shields.io/badge/Email-kunalbisht909@gmail.com-D96080?style=flat-square)](mailto:kunalbisht909@gmail.com)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Try%20TranscriptAI-FF4B4B?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/KunalTheBeast/TranscriptAI)

</div>
