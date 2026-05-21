# Contributing to TranscriptAI

Thanks for taking the time to contribute. This document covers local setup, project architecture, current state, known limitations, and the roadmap.

---

## Table of Contents

- [Contributing to TranscriptAI](#contributing-to-transcriptai)
  - [Table of Contents](#table-of-contents)
  - [Local Setup](#local-setup)
  - [Running Tests](#running-tests)
  - [Project Structure](#project-structure)
  - [Provider Architecture](#provider-architecture)
  - [Current State \& Progress](#current-state--progress)
    - [Evaluation accuracy (v1 → live)](#evaluation-accuracy-v1--live)
    - [What works in production today](#what-works-in-production-today)
    - [Scale (as of May 2026)](#scale-as-of-may-2026)
  - [Known Limitations](#known-limitations)
  - [Roadmap](#roadmap)
    - [Near-term (when financially viable)](#near-term-when-financially-viable)
    - [Medium-term](#medium-term)
    - [Long-term](#long-term)
  - [Submitting Changes](#submitting-changes)
  - [Code Style](#code-style)
  - [Reporting Issues](#reporting-issues)

---

## Local Setup

```bash
git clone https://github.com/aiKunalBisht/Transcript-ai.git
cd Transcript-ai
pip install -r requirements.txt
```

**Optional dependencies** (each unlocks a capability tier):

```bash
pip install fugashi unidic-lite       # MeCab Japanese tokenizer — required for keigo detection
pip install scikit-learn              # TF-IDF semantic similarity tier
pip install sentence-transformers     # Neural semantic scoring (~500MB) — best accuracy
pip install openai-whisper            # Local audio transcription fallback
```

**Set your API key** (Groq free tier at console.groq.com):

```bash
# Primary key
export GROQ_API_KEY=gsk_your_key_here

# Optional second key — enables automatic rotation when first key hits daily limit
export GROQ_API_KEY_2=gsk_your_second_key_here

# Provider override (default: auto)
# auto   → Groq key 1 → Groq key 2 → Mock
# groq   → Groq only
# mock   → always demo mode (testing)
export TRANSCRIPT_AI_PROVIDER=auto
```

**Run the app:**

```bash
python -m streamlit run app.py        # Streamlit UI
python api.py                         # FastAPI REST server (optional)
```

**For HuggingFace Spaces deployment:**
Do NOT add a `.env` file. Set secrets in Space → Settings → Repository secrets.
Keys are injected as environment variables automatically.

---

## Running Tests

```bash
pip install pytest pytest-cov
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=. --cov-report=term-missing
```

Test files live in `tests/`:

| File | Covers |
|---|---|
| `test_edge_1.py` | Japanese soft rejection detection edge cases |
| `test_edge_2.py` | PII masking and restoration |
| `test_edge_3.py` | Hallucination guard |
| `test_schema_stability.py` | JSON output schema contract |
| `test_data.py` | Ground truth test cases for evaluation |
| `sample_transcripts.py` | Sample data — trilingual, conflict, Hinglish |

**Note:** Tests run without downloading ML models. `sentence-transformers` is optional — falls back to TF-IDF automatically.

---

## Project Structure

```
Transcript-ai/
│
├── app.py                        Streamlit UI — 7 tabs, sakura/peach palette
├── api.py                        FastAPI REST endpoints
├── async_processor.py            ThreadPoolExecutor job queue
│
├── analysis/
│   ├── __init__.py
│   ├── analyzer.py               LLM orchestration — Groq key 1 → key 2 → Mock
│   ├── english_analyzer.py       English NLP patterns
│   ├── hindi_analyzer.py         Hindi indirect signal detection (8 patterns)
│   ├── hallucination_guard.py    100% rule-based claim verification
│   ├── japanese_tokenizer.py     MeCab morphological analysis + keigo detection
│   ├── semantic_validator.py     Three-tier similarity engine
│   └── soft_rejection_detector.py  16 nemawashi + 8 Hindi patterns
│
├── transcription/
│   ├── audio_processor.py        Groq Whisper transcription (MP4/MP3/WAV)
│   ├── pii_masker.py             APPI anonymization — runs before any LLM call
│   └── speaker_normalizer.py     Cross-script identity (田中 ↔ Tanaka)
│
├── utils/
│   ├── cache.py                  MD5 result caching (24h TTL)
│   ├── evaluator.py              ROUGE + semantic + F1 evaluation
│   ├── japanese_names.py         500+ surname database (JMnedict-derived)
│   ├── language_intelligence.py  Language-aware feature routing
│   ├── logger.py                 JSONL observability + trend analysis
│   └── vector_cache.py           ChromaDB semantic similarity cache
│
├── tests/
│   ├── test_edge_1.py
│   ├── test_edge_2.py
│   ├── test_edge_3.py
│   ├── test_schema_stability.py
│   ├── test_data.py
│   └── sample_transcripts.py
│
├── requirements.txt
├── Dockerfile
├── README.md
└── CONTRIBUTING.md
```

---

## Provider Architecture

```
User submits transcript
        │
        ▼
  Vector cache check          ← ChromaDB semantic similarity (instant return)
        │ miss
        ▼
  MD5 exact cache             ← 24h TTL (instant return)
        │ miss
        ▼
  GROQ_API_KEY                ← llama-3.3-70b-versatile, ~3s, free tier
        │ 429 rate limit
        ▼
  GROQ_API_KEY_2              ← same model, second free key, auto-rotation
        │ 429 rate limit
        ▼
  Mock + Groq mini-summary    ← llama-3.1-8b-instant, 50 tokens, real AI summary
        │                        shown in demo banner even during rate limiting
        ▼
  Pure mock (last resort)     ← only if all keys exhausted AND mini-summary fails
```

**Key design decisions:**

- Ollama is NOT in the cloud fallback chain — it requires a local machine and is unavailable on HuggingFace Spaces. It remains available for local development via `TRANSCRIPT_AI_PROVIDER=ollama`.
- The warmup ping (sending "hi" to Groq on app load) was removed in v7.2 — it was burning 20-29 of 30 daily calls per day with zero user value.
- Per-key 429 tracking resets automatically after 24 hours (Groq's reset window).
- Mock mode uses `llama-3.1-8b-instant` (cheapest Groq model) to generate a real 2-line summary so users see actual AI output even when the full analysis quota is exhausted.

---

## Current State & Progress

### Evaluation accuracy (v1 → live)

| Version | What changed | Accuracy |
|---|---|---|
| v1 | Hard exact matching, no cultural awareness | 22–30% |
| v2 | Fuzzy name matching, rule-based code-switch, semantic similarity | ~45% |
| v3 | Cultural ground truth, MeCab tokenization, soft sentiment scoring | ~60% |
| v4 | Hallucination guard, bilingual action items, speaker normalization fix | ~75% |
| v5 (live) | Verified on HuggingFace — Sales call: 95.2%, Internal JA: 81.6%, Conflict: 85.8% | **85–95%** |

### What works in production today

- ✅ Trilingual analysis — Japanese / Hindi (Hinglish) / English / Mixed
- ✅ MeCab keigo detection (overrides LLM classification)
- ✅ 16 nemawashi soft rejection patterns with confidence scores
- ✅ 8 Hindi indirect communication patterns
- ✅ APPI-compliant PII masking (names, phones, emails) before LLM
- ✅ Hallucination guard — 100% rule-based, LLM never validates itself
- ✅ Cross-script speaker normalization (田中 ↔ Tanaka ↔ Director)
- ✅ Groq Whisper audio transcription (MP4/MP3/WAV — same free API key)
- ✅ 2-key round-robin with automatic 429 rotation
- ✅ Vector cache (ChromaDB) — instant return for similar transcripts
- ✅ Meeting health score (0–100) with breakdown
- ✅ Trends dashboard — soft rejection trend, hallucination drift, provider usage
- ✅ FastAPI REST endpoint for external integration
- ✅ Live streaming mode (Groq token-by-token)
- ✅ JSON export with full structured output
- ✅ Works with Otter.ai, Zoom, Google Meet .vtt exports out of the box

### Scale (as of May 2026)

- Live on HuggingFace Spaces
- 29 unique users on day 3 (hitting Groq free tier limit)
- 3,500+ Reddit views, 10+ shares within 48h of launch
- 500+ recordings analyzed, 97%+ schema integrity maintained

---

## Known Limitations

| Limitation | Current behavior | Fix path |
|---|---|---|
| 30 req/day per Groq key | 2-key rotation doubles to 60/day | Add more keys or upgrade to Groq paid ($0.002/req) |
| Names not as speaker labels | May miss masking if not in 500-name DB | spaCy `ja_core_news_sm` full NER |
| Speaker diarization | Silence-gap heuristic, ~70% accuracy | pyannote.audio + HuggingFace |
| Confidence scores not calibrated | Heuristic, not probabilistic | Labeled dataset + Platt scaling |
| Evaluation on synthetic cases | 3 bilingual test cases (author-written) | External validation on real transcripts |
| No learning loop | Does not improve from user feedback | Correction collection + fine-tuning pipeline |
| Ollama unavailable on HF Spaces | Falls to mock instead | Dedicated VPS with Ollama for data-residency clients |
| Audio pipeline blocked on HF | Direct recording disabled (privacy) | Recommend Otter.ai / Whisperflow → upload .vtt |

---

## Roadmap

### Near-term (when financially viable)

- [ ] Zoom webhook integration — auto-fetch transcript when meeting ends via Zoom API
- [ ] Google Meet API connector — same auto-fetch via `conferenceRecords` endpoint
- [ ] Speaker diarization via pyannote.audio — move from heuristic to model-based
- [ ] Redis Queue (RQ) upgrade — replace ThreadPoolExecutor for proper async job queue
- [ ] More Groq keys or paid tier — remove the 30 req/day ceiling

### Medium-term

- [ ] User correction collection — "this action item is wrong" feedback loop
- [ ] Fine-tuning on corrected data — close the accuracy gap on edge cases
- [ ] Confidence calibration — replace heuristic scores with probabilistic ones
- [ ] vLLM self-hosted option — for enterprise data residency without Ollama limits

### Long-term

- [ ] Full NER via spaCy `ja_core_news_sm` — catch names not appearing as speaker labels
- [ ] Meeting comparison — diff two meetings, track commitment drift
- [ ] Keigo drift tracker — formality level over time per speaker
- [ ] AniLytics integration — bilingual sentiment on non-business Japanese content

---

## Submitting Changes

1. Fork the repo
2. Create a branch: `git checkout -b fix/your-fix-name`
3. Make your changes
4. Run tests: `pytest tests/ -v`
5. Commit with a clear message
6. Push and open a Pull Request against `main`

**Commit message format:**

```
feat: add new feature
fix: fix a bug
docs: update documentation
test: add or update tests
refactor: code change with no functional impact
chore: tooling, config, CI changes
```

---

## Code Style

- Python 3.10+
- No external formatter enforced — keep it readable
- Prefer explicit over implicit
- Document public functions with a one-line docstring minimum
- Never print in production paths — use `utils/logger.py`
- Never call `st.secrets` inside a background thread — use `os.getenv` only

---

## Reporting Issues

Open an issue at [github.com/aiKunalBisht/Transcript-ai/issues](https://github.com/aiKunalBisht/Transcript-ai/issues).

Include:
- What you expected to happen
- What actually happened
- Transcript language (EN / HI / JA / Mixed)
- Provider used (Groq / Mock)
- Python version and OS
- Whether you are running locally or on HuggingFace Spaces

---

Built by [Kunal Bisht](https://github.com/aiKunalBisht) · Pithoragarh, Uttarakhand, India