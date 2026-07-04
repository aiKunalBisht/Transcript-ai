# Contributing to TranscriptAI

Thanks for taking the time to contribute. This document covers local setup, project architecture, current state, known limitations, and the roadmap.

> **v3.2 note:** TranscriptAI migrated from Streamlit to FastAPI + Jinja2 + Alpine.js in v3.0. The pipeline logic was untouched during the migration — only the serving layer changed. This doc reflects the current v3.2 architecture (156 commits, 93.8% accuracy).

---

## Table of Contents

- [Contributing to TranscriptAI](#contributing-to-transcriptai)
  - [Table of Contents](#table-of-contents)
  - [Local Setup](#local-setup)
  - [Running Tests](#running-tests)
  - [Project Structure](#project-structure)
  - [Provider Architecture](#provider-architecture)
  - [The 11-Stage Pipeline](#the-11-stage-pipeline)
  - [Current State \& Progress](#current-state--progress)
    - [Evaluation accuracy (v1 → live)](#evaluation-accuracy-v1--live)
    - [What works in production today](#what-works-in-production-today)
    - [Scale (as of July 2026)](#scale-as-of-july-2026)
  - [Known Limitations](#known-limitations)
  - [Roadmap](#roadmap)
    - [P0 — Near-term](#p0--near-term)
    - [P1 — Medium-term](#p1--medium-term)
    - [P2 — Long-term](#p2--long-term)
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
pip install sentence-transformers     # Neural semantic scoring (paraphrase-multilingual-MiniLM-L12-v2, lazy-loaded)
pip install openai-whisper            # Local audio transcription fallback
```

**Set your API key** (Groq free tier at console.groq.com):

```bash
# Primary key
export GROQ_API_KEY=gsk_your_key_here

# Optional second key — enables automatic rotation when first key hits daily limit
export GROQ_API_KEY_2=gsk_your_second_key_here

# Optional — local Ollama endpoint (zero cloud exposure, data-residency mode)
export OLLAMA_URL=http://localhost:11434/api/generate
export OLLAMA_MODEL=qwen3:8b

# Provider override (default: auto)
# auto   → Groq key 1 → Groq key 2 → Ollama → Mock
# groq   → Groq only
# ollama → Ollama only (local dev, not available on HF Spaces)
# mock   → always demo mode (testing)
export TRANSCRIPT_AI_PROVIDER=auto
```

**Run the app (FastAPI + Jinja2 + Alpine.js — no Streamlit, no build step):**

```bash
uvicorn main:app --host 0.0.0.0 --port 7860 --reload   # local dev
# or, matching the production container:
uvicorn main:app --host 0.0.0.0 --port 7860 --workers 1
```

Legacy `app.py` (Streamlit) is kept in the repo as a frozen reference only — it is not run in production and should not be modified.

**For HuggingFace Spaces deployment:**
Do NOT add a `.env` file. Set secrets in Space → Settings → Repository secrets.
Keys are injected as environment variables automatically. The Space uses `sdk: docker` (not `sdk: streamlit`) — see the Dockerfile.

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

| File                       | Covers                                                                                                                |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `test_core.py`             | 21 smoke tests across core modules (PII masking, hallucination guard, soft rejection, caching, speaker normalization) |
| `test_data.py`             | 3 bilingual ground-truth test cases (TC001 sales call, TC002 internal JP update, TC003 client complaint)              |
| `sample_transcripts.py`    | Sample data — trilingual, conflict, Hinglish                                                                          |
| `test_schema_stability.py` | JSON output schema contract                                                                                           |

**⚠️ Known test-coverage gap:** all 21 current tests are smoke tests only (`assert isinstance(result, dict)`), not correctness assertions. Meaningful regression tests are the top-priority open item — see [Known Limitations](#known-limitations).

**Note:** Tests run without downloading ML models. `sentence-transformers` is optional — falls back to TF-IDF automatically.

---

## Project Structure

```
Transcript-ai/
│
├── main.py                        FastAPI server — routes, optional module loading (431 lines)
├── api.py / api/api.py            FastAPI REST sub-application
├── api/async_processor.py         Background job queue (built, not yet wired into main.py)
├── app.py                         Legacy Streamlit UI — frozen reference only, do not edit
│
├── analysis/
│   ├── analyzer.py                LLM orchestration — Groq key1 → key2 → Ollama → Mock (1,140 lines)
│   ├── english_analyzer.py        40+ English hedging/commitment patterns
│   ├── hindi_analyzer.py          8-category Hindi/Hinglish indirect signal detection
│   ├── hallucination_guard.py     Rule-based token overlap + Jaccard verification
│   ├── japanese_tokenizer.py      MeCab morphological analysis + keigo detection
│   ├── semantic_validator.py      Sentence-transformer semantic similarity (lazy-loaded)
│   ├── soft_rejection_detector.py 3-tier pattern matching — termination/performance-failure/soft refusal (628 lines)
│   ├── deal_outcome_detector.py   8-state meeting outcome verdict system (327 lines)
│   └── conversation_dynamics.py   Topic stalls, senior silence pivots, closing summarizer (306 lines)
│
├── agents/
│   ├── gijiroku_formatter.py      議事録 (Japanese business minutes) generator
│   ├── slide_architect.py         Deterministic + LLM-narrative PPTX slide planner
│   └── cultural_insights_formatter.py  Standalone cultural context export
│
├── exporters/
│   └── pptx_builder.py            python-pptx slide deck builder (670 lines)
│
├── rags/
│   ├── meeting_store.py           ChromaDB meeting store for cross-meeting retrieval
│   └── rag_retriever.py           Semantic + metadata-filtered retrieval
│
├── transcription/
│   ├── audio_processor.py         Groq Whisper transcription (MP4/MP3/WAV)
│   ├── pii_masker.py              APPI anonymization — runs before any LLM call
│   └── speaker_normalizer.py      Cross-script identity (田中 ↔ Tanaka)
│
├── utils/
│   ├── html_renderer.py           Full results HTML — health score, 5-tab layout, insights
│   ├── cache.py                   MD5 result caching (content-addressable, 24h TTL)
│   ├── evaluator.py               ROUGE-1 + LCS action-item F1 + semantic + sentiment eval
│   ├── japanese_names.py          200+ surname database (JMnedict-derived)
│   ├── language_intelligence.py   Language-aware feature routing
│   ├── logger.py                  JSONL observability + schema-drift detection
│   └── vector_cache.py            ChromaDB HNSW semantic similarity cache
│
├── templates/                     Jinja2 — base.html, index.html, export.html, evaluate.html
├── static/style.css               Washi/sakura design system
│
├── tests/
│   ├── test_core.py
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
  Vector cache check          ← ChromaDB HNSW cosine similarity (≥95% match → instant return)
        │ miss
        ▼
  MD5 exact cache             ← content-addressable, 24h TTL (instant return)
        │ miss
        ▼
  GROQ_API_KEY                ← llama-3.3-70b-versatile, ~1.5-4s, free tier
        │ 429 rate limit
        ▼
  GROQ_API_KEY_2              ← same model, second free key, auto-rotation
        │ 429 rate limit
        ▼
  Ollama (qwen3:8b)           ← local inference, zero cloud exposure — local dev only,
        │                        NOT available on HuggingFace Spaces
        │ unavailable / fails
        ▼
  Mock + Groq mini-summary    ← llama-3.1-8b-instant, 50 tokens, real AI summary
        │                        shown in demo banner even during rate limiting
        ▼
  Pure mock (last resort)     ← only if all providers exhausted AND mini-summary fails
```

**Key design decisions:**

- Ollama now sits in the fallback chain (after both Groq keys) for local/dev deployments and data-residency clients. It remains unavailable on HuggingFace Spaces and is skipped automatically in that environment.
- The warmup ping (sending "hi" to Groq on app load) was removed in v7.2 — it was burning 20-29 of 30 daily calls per day with zero user value.
- Per-key 429 tracking (`_KEY_EXHAUSTED` dict, keyed by the first 12 characters of each key) resets automatically after 24 hours (Groq's reset window). A key marked exhausted is skipped without wasting a call.
- Mock mode uses `llama-3.1-8b-instant` (cheapest Groq model) to generate a real 2-line summary so users see actual AI output even when the full analysis quota is exhausted.
- System/user prompt separation (FIX-9) ensures the transcript is always treated as data, never as instructions — this closes a prompt-injection and misread-as-question vector.

---

## The 11-Stage Pipeline

Every transcript passes through eleven sequential, fault-isolated stages (each wrapped in its own try/except so a failure in one stage never aborts the rest):

| #   | Module                                      | Description                                                                                            |
| --- | ------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| 1   | `utils/vector_cache.py`                     | ChromaDB cosine similarity check — instant return if ≥95% match                                        |
| 2   | `utils/cache.py`                            | MD5 exact cache — instant return if identical transcript seen before                                   |
| 3   | `transcription/pii_masker.py`               | APPI compliance — masks names, phones, emails **before** LLM sees text                                 |
| 4   | `analysis/analyzer.py`                      | LLM analysis — Groq key1 → key2 → Ollama → Mock fallback chain                                         |
| 5   | `transcription/pii_masker.py`               | PII restoration — replaces `[NAME_1]` tokens with real names in result                                 |
| 6   | `transcription/speaker_normalizer.py`       | Unifies 田中 ↔ Tanaka ↔ Director as the same speaker across all result fields                          |
| 7   | `analysis/japanese_tokenizer.py`            | MeCab override — replaces the LLM's keigo guess with morphological analysis                            |
| 8   | `utils/evaluator.py`                        | Rule-based code-switch count using Unicode range detection                                             |
| 9   | `analysis/hallucination_guard.py`           | Token overlap + semantic verification — flags action items not grounded in transcript                  |
| 10  | `analysis/soft_rejection_detector.py`       | 3-tier pattern matching: termination (CRITICAL), performance-failure (HIGH), soft refusal (MEDIUM/LOW) |
| 11  | `utils/vector_cache.py` + `utils/logger.py` | Stores result in both caches and appends to JSONL audit log                                            |

`analysis/deal_outcome_detector.py` (8-state verdict) and `analysis/conversation_dynamics.py` (topic stalls, senior silence pivots) run alongside this pipeline and are exposed as separate result fields (`deal_outcome`, `conversation_dynamics`, `role_hints`).

📌 **Critical ordering constraint:** PII must be masked **before** stage 4 (LLM call). PII must be **restored** before stage 6 (speaker normalization) — the normalizer needs real names to resolve cross-script identity (田中 ↔ Tanaka), and receiving `[NAME_1]` tokens instead would break it.

---

## Current State & Progress

### Evaluation accuracy (v1 → live)

| Version   | What changed                                                                                    | Accuracy        |
| --------- | ----------------------------------------------------------------------------------------------- | --------------- |
| v1        | Hard exact matching, no cultural awareness                                                      | 22–30%          |
| v2        | Fuzzy name matching, rule-based code-switch, semantic similarity                                | ~45%            |
| v3        | Cultural ground truth, MeCab tokenization, FastAPI + Jinja2 migration                           | ~65%            |
| v4        | Hallucination guard, bilingual action items, APPI PII masking                                   | 75–85%          |
| v5 (live) | 2-key rotation, vector cache, `bypass_cache` eval fix, system/user prompt separation            | 93.8%           |
| v3.2      | Soft rejection rewrite — CRITICAL termination tier, approval gate, deal outcome 8-state verdict | 93.8% (current) |

Every accuracy jump was traced to a specific pipeline failure via MLflow-logged evaluation runs — not estimated. Full trace history: v2→v3 (action item F1 0.22→0.87, fixed owner field extracting role titles instead of first names), v3→v4 (hallucination risk 0.42→0.08, added the guard), v4→v5 (overall score ~82%→93.8%, `bypass_cache` fix revealed all 3 test cases were returning the same cached result).

### What works in production today

- ✅ Trilingual analysis — Japanese / Hindi (Hinglish) / English / Mixed
- ✅ MeCab keigo detection (overrides LLM classification)
- ✅ 20 Japanese nemawashi soft rejection patterns + 8 Hindi + 40+ English hedging patterns, each with confidence scores
- ✅ 3-tier soft rejection system: CRITICAL termination, HIGH performance-failure/approval-gate, MEDIUM/LOW soft refusal
- ✅ 8-state deal outcome verdict (REJECTED → APPROVED → CONDITIONAL → DEFERRED → PENDING → INFORMATIONAL → AT_RISK → UNCLEAR)
- ✅ APPI-compliant PII masking (names, phones, emails, company names) before any LLM call
- ✅ Hallucination guard — rule-based token overlap + Jaccard + semantic (sentence-transformer) verification
- ✅ Cross-script speaker normalization (田中 ↔ Tanaka ↔ Director), including role-only-label fix
- ✅ Conversation dynamics — topic stalls, senior silence pivots, closing summarizer detection
- ✅ Groq Whisper audio transcription (MP4/MP3/WAV — same free API key)
- ✅ 2-key round-robin with automatic 429 rotation + Ollama local fallback
- ✅ Vector cache (ChromaDB HNSW) — instant return for semantically similar transcripts
- ✅ Meeting health score (0–100) with breakdown, risk-based hard caps
- ✅ Evaluate page wired to MLflow experiment tracking
- ✅ FastAPI REST endpoint + async job-queue module (built, not yet wired to frontend)
- ✅ Six export formats: PPTX, 議事録 (gijiroku), Cultural Insights, Markdown, JSON, plain text
- ✅ Meeting store (`rags/`) for cross-meeting semantic retrieval
- ✅ Works with Otter.ai, Zoom, Google Meet `.vtt` exports out of the box

### Scale (as of July 2026)

- Live on HuggingFace Spaces, v3.2, 156 commits
- Codebase: 11,646 lines of Python across 43 modules + 1,231 lines of HTML
- Lighthouse: Performance 94, Accessibility 100, CLS 0.000
- Action Item F1: 1.0 on ground-truth test cases; Sentiment accuracy: 100%
- Average response time: ~3 seconds (Groq LPU)

---

## Known Limitations

| Limitation                                | Current behavior                                                                                                        | Fix path                                                                                                                      |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| Async navigation drops in-flight analysis | Blocking `fetch()` — navigating away cancels the request even though the server finished                                | `api/async_processor.py` job queue exists but isn't wired into `main.py`/frontend polling — **P0**                            |
| All 21 tests are smoke tests              | Assert `isinstance(result, dict)` only — no correctness assertions                                                      | Add meaningful regression tests (e.g. assert CRITICAL risk_level, assert neutral sentiment for deferential register) — **P0** |
| 30 req/day per Groq key                   | 2-key rotation doubles to 60/day; Ollama fallback for local/data-residency use                                          | Add more keys or upgrade to Groq paid tier                                                                                    |
| Names not as speaker labels               | May miss masking if not in the 200+-name DB                                                                             | Full NER via spaCy `ja_core_news_sm`                                                                                          |
| Speaker diarization                       | Silence-gap heuristic, ~70% accuracy                                                                                    | pyannote.audio + HuggingFace                                                                                                  |
| Confidence scores not calibrated          | Heuristic, not probabilistic                                                                                            | Labeled dataset + Platt scaling                                                                                               |
| Evaluation on synthetic cases             | 3 bilingual test cases (author-written); `eval_25/30/60_recordings.json` planned but empty                              | External validation on real transcripts                                                                                       |
| Yes-trap signals inflate risk_level       | 承知しました / はい、承知しました sit in `SOFT_PATTERNS` and drive up risk even for normal, politely-attentive meetings | Separate `yes_trap_signals` from rejection `risk_level`                                                                       |
| No learning loop                          | Does not improve from user feedback                                                                                     | Correction collection + fine-tuning pipeline                                                                                  |
| Ollama unavailable on HF Spaces           | Falls to mock instead                                                                                                   | Dedicated VPS with Ollama for data-residency clients                                                                          |
| Audio pipeline blocked on HF              | Direct recording disabled (privacy)                                                                                     | Recommend Otter.ai / Whisperflow → upload `.vtt`                                                                              |

---

## Roadmap

### P0 — Near-term

- [ ] Wire `api/async_processor.py` job queue into `main.py` + `sessionStorage`-based frontend polling — fixes the async navigation bug
- [ ] Meaningful unit tests — assert correct output _values_, not just "didn't crash"
- [ ] Separate `yes_trap_signals` from rejection `risk_level` to stop false-positive risk inflation

### P1 — Medium-term

- [ ] Real-time streaming analysis — WebSocket → Web Speech API → live meeting intelligence
- [ ] Multi-meeting trend dashboard — aggregate over `rags/meeting_store.py` for client-level trajectory
- [ ] Speaker diarization via pyannote.audio — move from silence-gap heuristic to model-based "who spoke when"
- [ ] User correction collection — feedback loop for fine-tuning on corrected data
- [ ] Confidence calibration — replace heuristic scores with probabilistic ones (Platt scaling)

### P2 — Long-term

- [ ] Google Docs / Notion export via MCP connectors (already available)
- [ ] Ringi-sho (稟議書) document generator — full formatted output from deal outcome + approval chain
- [ ] Full NER via spaCy `ja_core_news_sm` — catch names not appearing as speaker labels
- [ ] Aho-Corasick migration for pattern matching, if the pattern set grows past ~500 (currently 60+, naive `O(n·m·p)` search is well under 1ms and not yet worth the complexity)
- [ ] Trie-based PII name matching for large-scale name databases
- [ ] Keigo drift tracker — formality level over time per speaker
- [ ] vLLM self-hosted option for enterprise data residency without Ollama's limits

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

- Python 3.11 (3.10 removed from the CI test matrix — test runner incompatibility)
- No external formatter enforced — keep it readable
- Prefer explicit over implicit
- Document public functions with a one-line docstring minimum
- Never print in production paths — use `utils/logger.py`
- Every non-essential module import goes in a `try/except ImportError` block at the top of `main.py`, setting an `_AVAILABLE` flag — features degrade gracefully instead of 500-ing
- Every pipeline stage in `analyze_transcript()` gets its own `try/except` — a failure in one stage must never abort the rest
- Company names (Fujitsu, NTT Data, Hitachi, etc.) must never appear in user-facing copy

---

## Reporting Issues

Open an issue at [github.com/aiKunalBisht/Transcript-ai/issues](https://github.com/aiKunalBisht/Transcript-ai/issues).

Include:

- What you expected to happen
- What actually happened
- Transcript language (EN / HI / JA / Mixed)
- Provider used (Groq / Ollama / Mock)
- Python version and OS
- Whether you are running locally or on HuggingFace Spaces

---

Built by [Kunal Bisht](https://github.com/aiKunalBisht) · Pithoragarh, Uttarakhand, India
