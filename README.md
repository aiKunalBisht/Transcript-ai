---
title: TranscriptAI
emoji: 🧠
colorFrom: pink
colorTo: red
sdk: streamlit
sdk_version: "1.32.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# TranscriptAI 🧠

**Multilingual Meeting Intelligence Platform — Japanese · Hindi · English · Mixed**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace-D96080?style=flat-square)](https://huggingface.co/spaces/KunalTheBeast/TranscriptAI)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-3C2416?style=flat-square&logo=github)](https://github.com/aiKunalBisht/Transcript-ai)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?style=flat-square)](https://fastapi.tiangolo.com)
[![Lighthouse](https://img.shields.io/badge/Lighthouse-94%2F100-D96080?style=flat-square)](https://huggingface.co/spaces/KunalTheBeast/TranscriptAI)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> Turns any meeting transcript into structured business intelligence in ~3 seconds.
> Built specifically for the gap generic AI tools miss: Japanese business indirection,
> APPI compliance, and trilingual meetings no off-the-shelf tool handles.

---

## Why This Exists

Generic meeting summarisers extract what was _said_. They miss what was _meant_.

| What was said                 | Generic AI                      | TranscriptAI                                              |
| ----------------------------- | ------------------------------- | --------------------------------------------------------- |
| 検討いたします                | "Action: We will consider it"   | ⚠ Soft rejection — 72% confidence. Follow up explicitly.  |
| 難しいかもしれません          | "It may be difficult" — neutral | 🚨 HIGH rejection signal — 90% confidence. Deal at risk.  |
| 前向きに検討                  | "Positive consideration"        | ⚠ Uncertain — outcome not guaranteed (55%)                |
| 承知いたしました + いたします | "Acknowledged"                  | 🏯 High keigo register — senior speaker or formal context |
| 善処します                    | "Action: We will handle it"     | 🚨 Classic nemawashi dodge — no real commitment made      |

Japanese enterprise also mandates APPI compliance — raw meeting data containing personal information cannot be sent to foreign cloud LLMs. Most tools fail this requirement by design. TranscriptAI masks PII locally before any LLM call.

---

## What It Does

- **Trilingual analysis** — Japanese / Hindi (Hinglish) / English / Mixed in a single meeting
- **Keigo detection** — MeCab morphological analysis extracts formality register (High / Medium / Low)
- **16 nemawashi soft rejection patterns** with confidence scores and cultural explanations
- **8 Hindi indirect patterns** — देखते हैं, थोड़ा मुश्किल है, कोशिश करेंगे and more
- **APPI-compliant PII masking** — 500+ Japanese surnames, 100 kanji↔romaji pairs, local NER — before any LLM call
- **Hallucination guard** — 100% rule-based token overlap check; LLM never validates its own output
- **2-key Groq rotation** — auto-failover on 429, 24h reset, 60 RPD total
- **Vector cache** — ChromaDB semantic similarity for instant return on similar transcripts
- **Meeting Health Score** — 0–100 across sentiment, action clarity, communication risk, AI confidence
- **Speaker tone intelligence** — 6-level tone classification per speaker
- **Trends dashboard** — soft rejection drift, provider usage, hallucination rates over time
- **Groq Whisper transcription** — MP4 / MP3 / WAV audio input
- **FastAPI REST layer** — async `/analyze` and `/analyze/batch` for CRM integration
- **MLflow eval tracking** — every evaluation run logged with ROUGE, F1, nemawashi precision

---

## Accuracy History — v1 to v5

| Version       | Key change                                                           | Accuracy  |
| ------------- | -------------------------------------------------------------------- | --------- |
| v1            | Hard exact matching, English-only assumptions                        | 22–30%    |
| v2            | Fuzzy speaker names, TF-IDF similarity, rule-based code-switch       | ~45%      |
| v3            | MeCab keigo override, bilingual ground truth, soft sentiment scoring | ~60%      |
| v4            | Hallucination guard, 16 nemawashi patterns, APPI PII masking         | 75–85%    |
| **v5 (live)** | 2-key rotation, vector cache, MLflow, bypass_cache eval fix          | **93.8%** |

Every rebuild was driven by evaluation metric failures traced through the pipeline — not intuition. When action F1 was 0.4 at v2, tracing showed the owner field extracted role titles ("Director") instead of first names. One prompt instruction fixed it. That is the entire methodology.

---

## Evaluation Metrics (v5 · Live)

| Test case                   | Overall   | ROUGE-1 | Action F1 | Sentiment |
| --------------------------- | --------- | ------- | --------- | --------- |
| Sales call · JA/EN mixed    | **94.5%** | 0.694   | 1.0       | 1.0       |
| Internal meeting · Japanese | **93.8%** | 0.703   | 1.0       | 1.0       |
| Client conflict · EN/JA     | **93.8%** | 0.703   | 1.0       | 1.0       |

Evaluation uses custom bilingual ground truth with `sentiment_acceptable` maps — Japanese professional neutral speech is neither positive nor negative by Western definitions. Every eval run is logged to MLflow with full metric breakdown.

---

## Architecture — 11-Stage Pipeline

```
Input transcript
    │
    ▼
 1  Vector cache check       utils/vector_cache.py       ChromaDB cosine similarity
 2  MD5 exact cache          utils/cache.py              Instant return if identical
 3  PII masking              transcription/pii_masker.py APPI — BEFORE LLM
 4  LLM analysis             analysis/analyzer.py        Groq key1 → key2 → mock
 5  PII restoration          transcription/pii_masker.py BEFORE speaker normalization
 6  Speaker normalization    transcription/speaker_normalizer.py
 7  MeCab keigo override     analysis/japanese_tokenizer.py
 8  Code-switch count        utils/evaluator.py          Rule-based Unicode ranges
 9  Hallucination guard      analysis/hallucination_guard.py
10  Soft rejection detection analysis/soft_rejection_detector.py
11  Cache + log              utils/vector_cache.py + utils/logger.py
```

**Critical ordering constraint:** PII must be masked before the LLM sees the text. PII must be restored before speaker normalization — otherwise the normalizer receives `[NAME_1]` and cannot resolve `田中 ↔ Tanaka ↔ Director` as the same person.

---

## Technology Decisions

| Decision      | Chosen                     | Over                      | Reason                                                                                                |
| ------------- | -------------------------- | ------------------------- | ----------------------------------------------------------------------------------------------------- |
| Vector DB     | **ChromaDB**               | Pinecone, Weaviate, FAISS | Free, local, HF Spaces compatible, APPI compliant. Pinecone costs $70+/mo and sends data to US cloud. |
| LLM inference | **Groq**                   | OpenAI, Anthropic         | 10–20x faster (LPU vs GPU). Free tier. JSON mode. $0 at demo scale.                                   |
| Japanese NLP  | **MeCab + IPADIC**         | spaCy ja, Fugashi         | Morpheme-level auxiliary verb detection. keigo is invisible to word-level tokenizers.                 |
| Web framework | **FastAPI**                | Flask, Django             | Native async with `asyncio.to_thread()`. Auto Swagger. Pydantic validation.                           |
| LLM client    | **Direct requests.post()** | LangChain (primary)       | Zero cold-start penalty. Exact 429 detection for key rotation. LangChain kept as fallback only.       |
| Frontend      | **Streamlit**              | React, Gradio             | HF Spaces native. Single HTML render via `build_results_html()` — 1 WebSocket message vs 50+.         |
| Eval tracking | **MLflow**                 | W&B, Neptune              | Free, local SQLite, APPI compliant. `mlflow ui` → localhost:5000.                                     |

---

## Quick Start

```bash
git clone https://github.com/aiKunalBisht/Transcript-ai.git
cd Transcript-ai
pip install -r requirements.txt

# Set your free Groq key (get one at console.groq.com)
export GROQ_API_KEY=your_key_here

# Run Streamlit UI
python -m streamlit run app.py

# Or run FastAPI (docs at localhost:8000/docs)
uvicorn api:app --reload --port 8000
```

**HuggingFace Spaces:** Add `GROQ_API_KEY` in Space → Settings → Repository secrets. The app handles key rotation, caching, and fallback automatically.

**Local inference (zero cost, APPI data residency):**

```bash
ollama pull qwen3:8b
# App auto-detects Ollama — no config needed
```

---

## Project Structure

```
app.py                          Streamlit UI v7.2
api.py                          FastAPI REST v2 (async)
analysis/
  analyzer.py                   LLM orchestration + 2-key Groq rotation
  soft_rejection_detector.py    16 JP nemawashi + 8 HI indirect patterns
  japanese_tokenizer.py         MeCab keigo extraction
  hallucination_guard.py        Rule-based token overlap validation
  english_analyzer.py           English NLP patterns
  hindi_analyzer.py             Hindi/Hinglish indirect signals
transcription/
  pii_masker.py                 APPI compliance — local PII masking
  audio_processor.py            Groq Whisper audio transcription
utils/
  html_renderer.py              CSS-only tabs, single-call results renderer
  evaluator.py                  ROUGE + F1 + MLflow eval (v5)
  vector_cache.py               ChromaDB semantic cache
  logger.py                     JSONL trends + drift detection
tests/
  test_data.py                  3 bilingual ground truth test cases
  sample_transcripts.py         Demo transcripts (trilingual, conflict, hinglish)
```

---

## Performance

| Metric                        | Score |
| ----------------------------- | ----- |
| Lighthouse Performance        | 94    |
| Lighthouse Accessibility      | 100   |
| Lighthouse Best Practices     | 100   |
| CLS (Cumulative Layout Shift) | 0     |
| Speed Index                   | 1.2s  |
| LCP                           | 1.3s  |

**10K user scalability:** Results rendered as a single HTML string via `build_results_html()` — one `st.markdown()` call instead of 50+. Tab switching is pure CSS (radio button trick) — no JavaScript, no Streamlit re-render on tab click.

---

## REST API

```bash
# Health check
GET http://localhost:8000/health

# Analyze a transcript
POST http://localhost:8000/analyze
{
  "transcript": "田中: 検討いたします。Rahul: Can you confirm by Friday?",
  "language": null,
  "mask_pii": true,
  "include_soft_rejections": true
}

# Batch (up to 10, parallel via asyncio.gather)
POST http://localhost:8000/analyze/batch

# Soft rejection pattern dictionary
GET http://localhost:8000/patterns/soft-rejections
```

Interactive docs: `http://localhost:8000/docs`

---

## Built By

**Kunal Bisht** — AI Engineer · LLM Systems & RAG Pipelines · Multilingual NLP

I build AI to turn real problems into actual solutions — not proof-of-concepts that never ship. TranscriptAI started because I kept forgetting my meetings and hated taking notes. It became a trilingual intelligence platform rebuilt five times until accuracy went from 22% to 93%.

[![GitHub](https://img.shields.io/badge/GitHub-aiKunalBisht-3C2416?style=flat-square&logo=github)](https://github.com/aiKunalBisht)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-kunalhere-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/kunalhere)
[![Email](https://img.shields.io/badge/Email-kunalbisht909@gmail.com-D96080?style=flat-square)](mailto:kunalbisht909@gmail.com)

📍 Bengaluru, Karnataka, India · Open to Remote / Relocation
