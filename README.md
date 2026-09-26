---
title: TranscriptAI
emoji: 🎙️
colorFrom: pink
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

<div align="center">

<img src="https://img.shields.io/badge/🎙️-TranscriptAI-D96080?style=for-the-badge&labelColor=1a0a0f" alt="TranscriptAI"/>

**Multilingual Meeting Intelligence · Japanese · Hindi · English · Mixed**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Space-FF4B4B?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/KunalTheBeast/TranscriptAI)
[![GitHub](https://img.shields.io/badge/Source-GitHub-3C2416?style=flat-square&logo=github)](https://github.com/aiKunalBisht/Transcript-ai)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-21%20passing-22C55E?style=flat-square)](https://github.com/aiKunalBisht/Transcript-ai/actions)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

_Turns meeting transcripts and audio into structured business intelligence — Japanese, Hindi, English, and mixed._

</div>

---

## The Problem No Generic AI Tool Solves

Generic meeting summarisers extract what was **said**. They miss what was **meant**.

| What was said                                  | Generic AI                      | TranscriptAI                                              |
| ---------------------------------------------- | ------------------------------- | --------------------------------------------------------- |
| 検討いたします                                 | "Action: We will consider it"   | ⚠ Soft rejection — 72% confidence. Follow up explicitly.  |
| 難しいかもしれません                           | "It may be difficult" — neutral | 🚨 HIGH rejection signal — 90% confidence. Deal at risk.  |
| 前向きに検討                                   | "Positive consideration"        | ⚠ Uncertain — outcome not guaranteed (55%)                |
| 承知いたしました                               | "Acknowledged"                  | 🏯 はい trap — understanding, NOT approval                |
| 善処します                                     | "Action: We will handle it"     | 🚨 Classic nemawashi dodge — no real commitment           |
| パートナーシップは継続しないことを決定しました | "Decision made"                 | ⛔ CRITICAL — Explicit contract termination. Irrevocable. |
| dekhte hain                                    | "We'll see"                     | ⚠ Hindi deferral — classic avoidance signal               |
| kal pakka                                      | "Definitely tomorrow"           | ⚠ Fake urgency — indefinite future in disguise            |

---

## What It Does

```
Input:  Transcript (JP · HI · EN · Mixed) or Audio (MP3/MP4/WAV/M4A)
Output: Structured business intelligence
```

### Core Intelligence

- **8-state meeting outcome verdict** — 🟢 Approved · 🔴 Rejected · 🔵 Conditional · 🟣 Deferred · 🟡 Pending · ⚪ Informational · 🟠 At Risk · ⚫ Unclear
- **Rejection detection across 3 tiers** — CRITICAL (explicit termination) → HIGH (performance-failure framing) → MEDIUM/LOW (soft hedging)
- **Japanese soft-rejection patterns** — nemawashi, 難しいですね, ぜひ検討, 対応しかねます with confidence scores
- **Hindi indirect communication patterns** — देखते हैं, थोड़ा मुश्किल, kuch na kuch ho jayega
- **Keigo formality detection** — MeCab morphological analysis, not word-level guessing
- **はい trap detection** — 承知しました flagged as understanding, not approval
- **Approval gate detection** — 稟議が必要です — deal not closed yet

### Privacy-First Processing

- **Local PII masking** before any data leaves the server — 500+ Japanese surnames, phones, emails
- **Bidirectional masking** — `[NAME_1]` → `Tanaka` restored after analysis; raw names never sent to LLM
- **Fully local mode** via Ollama — zero cloud exposure when required

> **Note:** Local PII masking reduces personal data exposure before LLM inference. This is a privacy-by-design engineering choice, not a legal APPI compliance certification.

### Analysis Quality

- **Hallucination guard** — rule-based token overlap; LLM never validates its own output
- **Register-based sentiment** — scores how a speaker treats the other party, not word valence (professional apologies = neutral, not negative)
- **Cross-script speaker normalization** — 田中 ↔ Tanaka ↔ Director resolved to same identity
- **Meeting health score** — 0–100 across sentiment, action clarity, communication risk, AI confidence. Capped at 35 for HIGH risk, 22 for CRITICAL/termination

### Exports

- **議事録** — Japanese formal business minutes in standard enterprise structure
- **PPTX deck** — 6-slide presentation with said-vs-meant, risk watch, decisions, next steps
- **Cultural insights** — nemawashi risk, 稟議 approval status, keigo level breakdown
- **Markdown / JSON / TXT** — for downstream workflows

### Observability & Tracking

- **Evaluation page** — run 3 bilingual ground-truth test cases on demand, view scores
- **MLflow integration** — experiment tracking at `http://127.0.0.1:5000`, auto-logs per run
- **JSONL audit log** — append-only log with schema drift detection

---

## Evaluation

> **Benchmark scope:** 3 curated bilingual test scenarios (TC001–TC003). These scores reflect internal development benchmarks — not a general accuracy claim. A larger evaluation dataset is in progress.

| Test Case                   | Score          | ROUGE-1 | Action F1 | Sentiment |
| --------------------------- | -------------- | ------- | --------- | --------- |
| Sales call · JA/EN mixed    | **94.5 / 100** | 0.694   | 1.0       | 1.0       |
| Internal meeting · Japanese | **93.8 / 100** | 0.703   | 1.0       | 1.0       |
| Client conflict · EN/JA     | **93.8 / 100** | 0.703   | 1.0       | 1.0       |

### Accuracy Progression

| Version       | Key change                                           | Score          |
| ------------- | ---------------------------------------------------- | -------------- |
| v1            | Hard exact matching, English-only                    | 22–30%         |
| v2            | Fuzzy speaker names, TF-IDF similarity               | ~45%           |
| v3            | MeCab keigo override, bilingual ground truth         | ~60%           |
| v4            | Hallucination guard, nemawashi patterns, PII masking | 75–85%         |
| **v5 (live)** | 2-key rotation, vector cache, eval fix               | **93.8 / 100** |

Each improvement was driven by evaluation metric failures traced through the pipeline — not intuition. When action F1 was 0.4 at v2, tracing revealed the LLM was extracting `"Director"` instead of `"Tanaka"` as the action item owner. One prompt rule fixed it; F1 jumped to 0.87.

---

## Token Efficiency (v3.2)

| Stage                 | Before           | After            | Saved             |
| --------------------- | ---------------- | ---------------- | ----------------- |
| `_GROUNDING_RULES`    | 424 tokens       | 94 tokens        | -330 (-87%)       |
| Rules block           | 291 tokens       | 110 tokens       | -181 (-62%)       |
| Schema block          | 280 tokens       | 55 tokens        | -225 (-80%)       |
| Sandwich repeat       | 75 tokens        | 0 tokens         | -75 (-100%)       |
| **Per-request total** | **2,728 tokens** | **1,521 tokens** | **-1,207 (-44%)** |

Additional optimizations active:

- Model routing — short EN-only transcripts use `llama-3.1-8b-instant` (separate quota bucket)
- Dynamic schema — `japan_insights` block included only for JP/mixed transcripts
- Transcript truncation — 1,200 word cap (first 60% + last 40%) for long meetings
- Reduced `max_tokens` — capped at 550–1,100 depending on transcript length

---

## Architecture — 11-Stage Pipeline

```
Input transcript / audio
    │
    ▼
 1  Vector cache check       utils/vector_cache.py              ChromaDB cosine similarity (≥95% → instant return)
 2  MD5 exact cache          utils/cache.py                     Hash match → return in <1ms
 3  PII masking              transcription/pii_masker.py        Masks ALL PII before LLM sees text
 4  LLM analysis             analysis/analyzer.py               Groq 70B → 8B → Ollama → Mock
 5  PII restoration          transcription/pii_masker.py        Restores [NAME_1] → Tanaka BEFORE normalization
 6  Speaker normalization    transcription/speaker_normalizer.py 田中 ↔ Tanaka ↔ Director → unified
 7  MeCab keigo override     analysis/japanese_tokenizer.py     Morpheme-level formality, overrides LLM guess
 8  Code-switch count        utils/evaluator.py                 Rule-based Unicode range detection
 9  Hallucination guard      analysis/hallucination_guard.py    Token overlap + semantic similarity
10  Rejection + outcome      analysis/soft_rejection_detector.py + deal_outcome_detector.py
11  Cache + log              utils/vector_cache.py + utils/logger.py
```

**Critical ordering:** PII masked before step 4 (LLM). PII restored before step 6 (normalization). Reversing either order breaks the pipeline.

**Design rationale — deterministic vs LLM:**

| Task                            | Approach                           | Why                                                       |
| ------------------------------- | ---------------------------------- | --------------------------------------------------------- |
| PII detection                   | Deterministic (regex + dictionary) | Regex is more reliable than LLM for structured patterns   |
| Language/script detection       | Deterministic (Unicode ranges)     | Rule is faster and always correct                         |
| Keigo formality                 | Deterministic (MeCab morphemes)    | Auxiliary verb detection requires morpheme-level analysis |
| Semantic intent / summarization | LLM                                | Too complex for rules                                     |
| Hallucination check             | Deterministic (token overlap)      | LLM must not validate its own output                      |

---

## Technology Stack

| Layer               | Choice                   | Why                                                                                   |
| ------------------- | ------------------------ | ------------------------------------------------------------------------------------- |
| LLM inference       | **Groq (llama-3.3-70b)** | Fast, free tier, JSON mode                                                            |
| Local fallback      | **Ollama (qwen3:8b)**    | Zero cloud exposure option                                                            |
| Vector DB           | **ChromaDB**             | Local, HF Spaces compatible                                                           |
| Japanese NLP        | **MeCab + IPADIC**       | Morpheme-level auxiliary verb detection — keigo is invisible to word-level tokenizers |
| Web framework       | **FastAPI**              | Native async, auto Swagger                                                            |
| Frontend            | **Alpine.js + Jinja2**   | No bundler, no build step, HF Spaces compatible                                       |
| Experiment tracking | **MLflow**               | Local SQLite, free                                                                    |
| PPTX                | **python-pptx**          | Full slide control, pure Python                                                       |
| Audio               | **Groq Whisper**         | Free tier, multilingual                                                               |

---

## Quick Start

```bash
git clone https://github.com/aiKunalBisht/Transcript-ai.git
cd Transcript-ai
pip install -r requirements.txt

# Required
export GROQ_API_KEY=your_key_here

# Optional — 2-key round-robin (use keys from DIFFERENT accounts for separate quotas)
export GROQ_API_KEY_2=your_second_key_here

uvicorn main:app --reload --port 7860
# Open http://localhost:7860
# API docs at http://localhost:7860/docs
```

**Fully local — zero cloud exposure:**

```bash
ollama pull qwen3:8b
# App auto-detects Ollama when no Groq key is set
```

---

## Project Structure

```
main.py                           FastAPI server — routes, module loading, speaker label detection
analysis/
  analyzer.py                     LLM orchestration — provider chain, prompt, token optimization
  soft_rejection_detector.py      3-tier rejection detection — CRITICAL / HIGH / MEDIUM / LOW
  deal_outcome_detector.py        8-state meeting outcome verdict
  hallucination_guard.py          Rule-based token overlap verification
  conversation_dynamics.py        Topic stalls, senior silence pivots, closing summarizer
  japanese_tokenizer.py           MeCab morphological keigo detection
  english_analyzer.py             EN hedging and commitment-strength patterns
  hindi_analyzer.py               Hindi/Hinglish indirect communication patterns
  semantic_validator.py           Sentence-transformer semantic similarity
agents/
  gijiroku_formatter.py           議事録 Japanese business minutes generator
  cultural_insights_formatter.py  Cultural context export
  slide_architect.py              PPTX slide plan
exporters/
  pptx_builder.py                 python-pptx builder
transcription/
  pii_masker.py                   PII masking — 500+ JP surnames
  audio_processor.py              Groq Whisper audio transcription
  speaker_normalizer.py           Cross-script identity resolution
rags/
  meeting_store.py                ChromaDB meeting store for historical retrieval
  rag_retriever.py                Semantic retrieval over past meetings
utils/
  html_renderer.py                Results HTML — health score, outcome badge, 5-tab layout
  evaluator.py                    Ground-truth scoring — ROUGE, F1, sentiment, MLflow
  vector_cache.py                 ChromaDB semantic cache
  logger.py                       JSONL audit log with drift detection
templates/
  base.html                       Layout, sidebar, nav, Alpine.js reactive state
  index.html                      Main analysis page
  export.html                     Export page — PPTX, 議事録, MD, JSON, TXT
  evaluate.html                   Evaluation page — idle until run clicked
tests/
  test_core.py                    21 pytest tests
  test_data.py                    3 bilingual ground-truth test cases (TC001–TC003)
```

---

## REST API

```bash
POST /analyze-text        transcript: str, language: str|null, mask_pii: bool
POST /transcribe          file: UploadFile (audio or text)
POST /export/pptx         result: dict → PPTX binary
POST /export/gijiroku     result: dict → markdown string
POST /export/cultural-insights
POST /export/markdown
POST /export/json
POST /export/txt
GET  /evaluate            Evaluation page (idle — run on demand)
POST /evaluate/run        Runs 3 ground-truth cases, returns scored HTML
GET  /health              Module availability report
```

---

## Known Limitations

- **Small evaluation dataset** — 3 scenarios. Benchmark scores are internal development metrics, not general accuracy claims.
- **Authentication is demo-grade** — not suitable for production deployment of sensitive meeting data.
- **HF Spaces storage is ephemeral** — ChromaDB cache and MLflow logs reset on container restart.
- **Hallucination guard is rule-based** — token overlap is a proxy measure, not a perfect grounding check.
- **RAG retrieval is lightly tested** — historical meeting search is functional but not load-tested.

---

## Lighthouse Scores

| Metric                        | Score |
| ----------------------------- | ----- |
| Performance                   | 94    |
| Accessibility                 | 100   |
| CLS (Cumulative Layout Shift) | 0.000 |
| Speed Index                   | 1.2s  |

---

<div align="center">

**Built by [Kunal Bisht](https://linkedin.com/in/kunalhere)**
AI/ML Engineer · LLM Pipelines · RAG · Multilingual NLP · FastAPI
Bangalore, Karnataka, India · Open to Remote / Relocation

[![LinkedIn](https://img.shields.io/badge/LinkedIn-kunalhere-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/kunalhere)
[![GitHub](https://img.shields.io/badge/GitHub-aiKunalBisht-3C2416?style=flat-square&logo=github)](https://github.com/aiKunalBisht)
[![Email](https://img.shields.io/badge/Email-kunalbisht909@gmail.com-D96080?style=flat-square)](mailto:kunalbisht909@gmail.com)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Try%20TranscriptAI-FF4B4B?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/spaces/KunalTheBeast/TranscriptAI)

</div>
