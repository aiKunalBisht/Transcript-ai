<div align="center">

# 🌸 TranscriptAI · 議事録分析エンジン

**Enterprise meeting intelligence for the Japanese business market**

[![Python](https://img.shields.io/badge/Python-3.10%2B-C45C74?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-486858?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20UI-D96080?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-Free%20Tier-B87830?style=flat-square)](https://console.groq.com)
[![License](https://img.shields.io/badge/License-MIT-A8897C?style=flat-square)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Titankunal/Transcript-ai?style=flat-square&color=D96080)](https://github.com/Titankunal/Transcript-ai/stargazers)

**[Live Demo](https://transcript-ai-qkuqcld42yym54zxmyhhby.streamlit.app) · [API Docs](https://transcript-ai-qkuqcld42yym54zxmyhhby.streamlit.app) · [GitHub](https://github.com/Titankunal/Transcript-ai)**

*"Boring projects. Infinite scale."*

</div>

---

## What this is

TranscriptAI converts meeting recordings and transcripts into structured business intelligence. It is built specifically for the Japanese enterprise market — not adapted, not translated, but purpose-built.

Most meeting AI tools treat Japanese text as English with different characters. They miss everything that matters: the keigo register that signals organizational hierarchy, the nemawashi phrases that mean "No" while saying "We'll consider it", the code-switching patterns that indicate how internationally-oriented a team is. TranscriptAI treats these as first-class features, not afterthoughts.

---

## Why Japanese business is different

| What was said | What a generic AI extracts | What TranscriptAI extracts |
|---|---|---|
| 検討いたします | "We will consider it" — Action item assigned | ⚠ LIKELY REJECTION (72% confidence) — follow up explicitly |
| 難しいかもしれません | "It may be difficult" — neutral sentiment | 🚨 HIGH REJECTION signal (90% confidence) — deal at risk |
| 前向きに検討 | "Positive consideration" — positive sentiment | ⚠ UNCERTAIN (55%) — outcome not guaranteed |
| 承知いたしました + いたします | Acknowledged | 🏯 HIGH keigo register — senior speaker or formal context |

This is the difference between a generic summarizer and a Japanese business intelligence tool.

---

## Features

**Core analysis** — for every language
- Dynamic summary scaled to transcript length (3 to 8+ bullets)
- Action items with owner, deadline, confidence score, and hallucination flag
- Per-speaker sentiment with cultural normalization (neutral is professional in Japan)
- Speaker breakdown with talk-time percentage and tone classification
- JSON export with full structured output

**Japan intelligence layer** — Japanese and mixed transcripts
- Keigo register detection via MeCab morphological analysis (overrides LLM classification)
- 16 nemawashi soft rejection patterns with confidence scores and cultural explanations
- Deterministic JA↔EN code-switch counting via Unicode range detection
- Cross-script speaker normalization (田中 and Tanaka are the same person)

**Hindi intelligence layer** — Hindi transcripts
- 8 indirect communication patterns (देखते हैं, थोड़ा मुश्किल है, etc.)
- Formality level detection
- Same risk scoring as the Japanese layer

**Production features**
- APPI-compliant PII masking — names, phones, emails anonymized before LLM
- Hallucination guard — 100% rule-based token overlap, LLM never validates itself
- Three-tier semantic similarity — sentence-transformers → TF-IDF → token overlap
- Groq → Ollama → Mock fallback hierarchy with explicit UX feedback
- Meeting trends dashboard — soft rejection trends, hallucination drift, workload analysis
- Live streaming — see summary generate token by token (Groq)
- MP4/MP3/WAV transcription via Groq Whisper (free tier)
- FastAPI REST endpoint for CRM integration
- Async job queue (ThreadPoolExecutor)
- MD5 result caching (24-hour TTL)
- JSONL observability logging with drift detection

---

## Data residency and APPI compliance

Japanese enterprises in finance, healthcare, and government cannot send meeting data to cloud LLMs under APPI (Act on the Protection of Personal Information).

TranscriptAI is designed for this constraint from the ground up:

**With Ollama (fully local):**
- Every component runs on your machine
- Zero data leaves your infrastructure
- Suitable for regulated industries with strict data residency requirements

**With Groq (cloud, APPI-compliant):**
- PII masking runs locally before any data is sent
- Names → `[NAME_1]`, phones → `[PHONE_1]`, emails → `[EMAIL_1]`
- The LLM processes anonymized text only
- PII is restored locally after analysis
- The LLM never sees raw personal data

**Position-based NER:** Any name appearing as a speaker label (`田中:`, `Tanaka:`) is masked regardless of whether it is in the surname database — covering uncommon surnames automatically.

**Honest limitation:** Names not appearing as speaker labels and not in the surname database require a full NER model (spaCy `ja_core_news_sm`) for complete coverage. This is documented, not hidden.

---

## Cost at scale

Every enterprise buyer asks this question. Here is the honest answer:

| Volume | Provider | Monthly cost |
|---|---|---|
| Up to 500 analyses/day | Groq free tier | **$0** |
| 10,000 analyses/month | Groq paid | **~$2** |
| Unlimited | Ollama (local) | **$0** (electricity only) |
| Any volume | vLLM self-hosted | **$0** (server cost only) |

The JSON schema is identical across all providers. Switching from Groq to a self-hosted vLLM instance requires changing one environment variable. Nothing downstream changes.

---

## Architecture

```
transcription/
  pii_masker.py          APPI anonymization — runs before any LLM call
  speaker_normalizer.py  Cross-script identity (田中 ↔ Tanaka ↔ Director)
  audio_processor.py     Whisper transcription (Groq free / local)

analysis/
  analyzer.py            Groq → Ollama → Mock with retry and UX feedback
  hallucination_guard.py 100% rule-based claim verification
  soft_rejection.py      16-pattern nemawashi + 8-pattern Hindi detection
  semantic_validator.py  Three-tier similarity engine
  japanese_tokenizer.py  MeCab morphological analysis

utils/
  logger.py              JSONL logging + trend analysis engine
  cache.py               MD5 result caching (24-hour TTL)
  evaluator.py           ROUGE + semantic + F1 evaluation
  japanese_names.py      500+ surname database (JMnedict-derived)
  language_intelligence.py  Language-aware feature routing

api/
  app.py                 Streamlit UI (sakura/peach palette, 7 tabs)
  api.py                 FastAPI REST endpoints
  async_processor.py     ThreadPoolExecutor job queue
```

**The order that matters:**

```
1. PII masking          (local — before LLM)
2. LLM analysis         (Groq / Ollama / Mock)
3. PII restoration      (local — BEFORE speaker normalization)
4. Speaker normalization (now sees real names, not [NAME_1])
5. Hallucination guard  (rule-based, not LLM)
6. Soft rejection       (pattern matching)
7. Cache + log          (local JSONL)
```

Step 3 before step 4 is critical. If PII is restored after normalization, the normalizer receives `[NAME_1]` and cannot resolve cross-script identities.

---

## Provider configuration

```bash
# Groq — 3 seconds, free tier, recommended for cloud deployment
export GROQ_API_KEY=your_key_here     # get free at console.groq.com
export TRANSCRIPT_AI_PROVIDER=auto

# Ollama — fully local, no API key, APPI data residency guarantee
ollama pull qwen3:8b
export TRANSCRIPT_AI_PROVIDER=ollama

# Auto — tries Groq first, falls back to Ollama, then mock
export TRANSCRIPT_AI_PROVIDER=auto

# All infrastructure URLs are configurable
export OLLAMA_URL=http://localhost:11434/api/generate
export OLLAMA_MODEL=qwen3:8b
export GROQ_MODEL=llama-3.1-8b-instant
```

**Dynamic token budget** (prevents Ollama timeouts):

| Transcript length | Max output tokens |
|---|---|
| < 300 words | 700 |
| 300–800 words | 1,200 |
| 800–2,000 words | 1,800 |
| 2,000+ words | 2,000 |

---

## Evaluation

Custom evaluation system with bilingual ground truth. Cultural corrections applied — standard NLP evaluation incorrectly labels Japanese professional neutral speech as "positive" (Western bias). Ground truth uses `sentiment_acceptable` maps with soft scoring.

| Test case | Baseline (v1) | Current (v4) | Grade |
|---|---|---|---|
| Sales call · JA/EN mixed | 30.8% | **75.7%** | GOOD |
| Internal meeting · Japanese heavy | 22.2% | **81.6%** | GOOD |
| Client complaint · tense | 55.9% | **85.8%** | GOOD |

**Iteration history:**
- v1 → hard exact matching, no cultural awareness: 22–30%
- v2 → fuzzy names, rule-based code-switch, semantic similarity: +15–20%
- v3 → cultural ground truth, JA tokenization, soft sentiment: +10–15%
- v4 → hallucination guard bonus, bilingual action items, speaker fix: +8–12%

Each improvement was driven by what the evaluation metrics revealed — not guesswork.

---

## REST API

```bash
pip install fastapi uvicorn
python api.py
# Interactive docs: http://localhost:8000/docs
```

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Status, modules, provider |
| POST | `/analyze` | Single transcript |
| POST | `/analyze/batch` | Up to 10 concurrent |
| GET | `/patterns/soft-rejections` | Full pattern dictionary |

```python
import requests

response = requests.post("http://localhost:8000/analyze", json={
    "transcript": "田中: Q3の進捗を報告します。鈴木: 検討いたします。",
    "language": "ja",
    "mask_pii": True,
    "include_soft_rejections": True
})

result = response.json()
print(result["result"]["soft_rejections"]["risk_level"])    # MEDIUM
print(result["result"]["japan_insights"]["keigo_level"])    # high
print(result["pii_items_found"])                            # 2
print(result["processing_time_ms"])                         # ~3000
```

---

## Installation

```bash
git clone https://github.com/Titankunal/Transcript-ai.git
cd Transcript-ai
pip install -r requirements.txt
```

**Optional — each unlocks a capability tier:**

```bash
pip install fugashi unidic-lite       # MeCab Japanese tokenizer
pip install scikit-learn              # TF-IDF semantic similarity
pip install sentence-transformers     # True semantic understanding (~500MB)
pip install openai-whisper            # Local audio transcription
```

**Run:**

```bash
# With Groq (recommended)
export GROQ_API_KEY=your_key
python -m streamlit run app.py

# Fully local
ollama pull qwen3:8b
python -m streamlit run app.py
```

---

## Scaling path

```
Current:   ThreadPoolExecutor (3 workers) + FastAPI async
Scale-1:   Redis Queue (RQ) + multiple FastAPI workers
Scale-2:   vLLM for batched LLM inference
Scale-3:   Kubernetes with horizontal pod autoscaling
```

The JSON schema is the stable contract at every scale level. CRM integrations, dashboards, and downstream systems do not change when infrastructure scales.

---

## Known limitations

| Limitation | Current behavior | Production path |
|---|---|---|
| Names not as speaker labels | May not be masked if not in surname DB | spaCy `ja_core_news_sm` NER |
| Speaker diarization | Silence-gap heuristic (~70% accuracy) | pyannote.audio + HuggingFace |
| Confidence scores not calibrated | Scores are heuristic, not probabilistic | Labeled dataset + calibration |
| Evaluation on synthetic test cases | 3 cases written by the author | External validation on real transcripts |
| No learning loop | System does not improve from feedback | User correction collection + fine-tuning |

---

## Codebase

| Metric | Value |
|---|---|
| Python files | 17 |
| Lines of code | 5,200+ |
| Functions | 90+ |
| Ground truth test cases | 3 bilingual |
| Nemawashi patterns | 16 |
| Hindi indirect patterns | 8 |
| Japanese surname database | 500+ entries |
| Kanji↔Romaji mappings | 100 pairs |
| Supported input formats | TXT · VTT · JSON · MP4 · MP3 · WAV · M4A |

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built by [Kunal Bisht](https://github.com/Titankunal) · Pithoragarh, Uttarakhand, India

*Boring projects. Infinite scale.*

</div>