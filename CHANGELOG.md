# Changelog

All notable changes to TranscriptAI are documented here.

Format: `[version] — date` → what changed and why. Where a change was driven by a specific data-structure/algorithm choice, that's called out — most of TranscriptAI's accuracy and latency wins trace directly to a DSA decision, not a bigger model.

---

## [v3.3] — 2026-09 (current)

**Auth hardening · Provider expansion · Reliability · Repo hygiene.**

### Added

- **NVIDIA NIM provider support** — `analyzer.py` now supports NIM as a primary inference provider ahead of Groq, with automatic model selection (`NIM_MODEL`, `NIM_MODEL_FAST`) and full backwards compatibility; provider chain is `NIM → Groq 70B → Groq 8B → Ollama → Mock`
- **RPM-aware retry vs daily-quota detection** — `analyzer.py` distinguishes rate-per-minute limits (retry after backoff) from daily quota exhaustion (rotate key immediately); previously both triggered the same key-rotation path, burning the backup key on retryable errors
- **Deterministic extractive fallback** — when all LLM providers fail, the pipeline now returns a structured result extracted deterministically from the transcript rather than raising an exception; extracted summaries use sentence scoring, action items use regex, sentiment uses word-list heuristics
- **Firebase Firestore user storage** — `utils/firebase_client.py` replaces the previous ephemeral session-only storage; users persist across HuggingFace redeploys; `upsert_user_firebase` and `get_user_by_email_firebase` added
- **bcrypt password verification** — `/login` POST route now hashes passwords on signup (`bcrypt.hashpw`) and verifies on sign-in (`bcrypt.checkpw`); previous implementation accepted any email+password combination without verification
- **slowapi rate limiting** — `POST /analyze-text` limited to 10 requests/minute per IP to prevent free-tier quota exhaustion
- **Per-user ChromaDB collections** — vector cache now isolates cached transcripts by `user_id`; user A cannot retrieve user B's cached results
- **`PIPELINES.md`** — dedicated 15-stage pipeline architecture document; removed pipeline detail from README to keep it focused
- **15-stage hybrid pipeline** — documented stages: normalization → PII masking → vector cache → MD5 cache → prompt construction → NIM/LLM inference → PII restoration → schema repair → speaker normalization → MeCab keigo override → action-item backfill → grounding validation → soft rejection → nemawashi sequence → deal outcome

### Fixed

- **SESSION_SECRET hardcoded fallback removed** — `_setup_auth()` previously fell back to `"transcript-ai-secret-session-key-2026"` when the env var was unset; now raises `RuntimeError` at startup so the misconfiguration is visible immediately rather than silently running with a known-public secret
- **CORS wildcard + credentials contradiction** — `allow_origins=["*"]` combined with `allow_credentials=True` is rejected by browsers; replaced with an explicit `_ALLOWED_ORIGINS` list sourced from the `ALLOWED_ORIGINS` env var
- **RAG `ChatOpenAI` → `ChatGroq`** — `rags/rag_retriever.py` imported `ChatGroq` but instantiated `ChatOpenAI` (which was never imported), causing a `NameError` at runtime on the LangChain path; fixed to use `ChatGroq` with correct Groq parameters; `streamlit` secrets fallback also removed (app migrated to FastAPI in v3.0)
- **MeCab system dependencies added to Dockerfile** — `mecab`, `libmecab-dev`, `mecab-ipadic-utf8` apt packages were missing; `mecab-python3` pip installation silently succeeded but MeCab calls failed at runtime on HuggingFace Spaces

### Removed

- `apply_patches.py`, `apply_sentiment_trust_patches.py`, `patch_main.py`, `setup_migration.py` — iterative patch scripts; logic integrated into main modules
- `groq_key_exhausted.json` — runtime state file committed by mistake
- `eval_25_recordings.json`, `eval_30_recordings.json`, `eval_60_recordings.json` — empty scaffolding files; replaced by `tests/eval_dataset.json` (30 labeled examples)
- Streamlit secrets fallback in `rag_retriever.py`

### Changed

- `health` endpoint: `"appi_compliant"` key renamed to `"pii_masking"` — matches README language change from "APPI compliant" to "privacy-first local PII masking"
- `requirements.txt` restored and extended with `bcrypt`, `langchain-groq`, `langchain-core`, `firebase-admin`; `mlflow` retained for local experiment tracking

### DSA notes

- **Provider chain as ordered list** `O(P)` scan (`P ≤ 5`) — not worth a priority queue at this scale; each provider is tried once in order, with short-circuit on success
- **RPM vs quota detection** uses response status + error-message substring matching — `O(1)` per check; no external state needed
- **Per-user ChromaDB isolation** uses `collection_name = f"cache_{user_id}"` — hashed user ID as collection key gives `O(1)` lookup with zero cross-user data risk

---

## [v3.2] — 2026-07

**Soft rejection rewrite + deal outcome detector.**

### Added

- `analysis/deal_outcome_detector.py` — 8-state meeting outcome verdict (`REJECTED → APPROVED → CONDITIONAL → DEFERRED → PENDING → INFORMATIONAL → AT_RISK → UNCLEAR`), computed with a strict priority order (most-specific/highest-certainty signal wins; `CONDITIONAL` checked before plain `APPROVED` so "we accept on the condition that…" doesn't read as a clean yes)
- `CRITICAL` tier in `soft_rejection_detector.py` for explicit contract termination — previously these transcripts (e.g. 契約を更新しないことを決定いたしました) matched zero soft-hedge patterns and returned `risk_level: NONE`, which was backwards for a meeting that just ended a relationship
- Tier 1b "approval gate" detection (HIGH risk — decision pending external authority)
- Backward-compatible `detected` / `risk_summary` alias keys on the return dict, so `cultural_insights_formatter.py` and `slide_architect.py` didn't need touching

### Fixed

- Role-only speaker merge bug — `normalize_speaker_name()` stripped role-only labels (`Director`, `Manager`) to `''`, and the substring-match fallback then treated `''` as a substring of every name, silently merging two distinct speakers into one. Fix: explicit empty-string guard before the substring loop in `_best_match()`.

### DSA notes

- **Zero-collision pattern design**: before shipping 72 new acceptance/conditional/deferred/informational phrases, every one was checked for substring overlap against all 6 existing rejection pattern sets. Result: zero collisions — a transcript can never trigger both an acceptance and a rejection pattern on the same phrase.
- Pattern matching stays naive `O(n·m·p)` substring search (Python's `in`) rather than Aho-Corasick — at ~60 patterns and 500–3,000-word transcripts this is <0.5ms; Aho-Corasick's `O(n + Σm)` only pays for itself past ~500 patterns.

### Accuracy

| Version               | Score | Root cause traced                                                                                                            |
| --------------------- | ----- | ---------------------------------------------------------------------------------------------------------------------------- |
| v4                    | ~82%  | —                                                                                                                            |
| v5 (bypass_cache fix) | 93.8% | All 3 eval test cases were returning the same cached vector-similarity hit — `bypass_cache=True` now forces independent runs |

---

## [v3.1] — 2026-06

**Performance hardening.**

### Changed

- Lighthouse Performance: 55 → 94; Accessibility: 100; CLS: 0.05 → 0.000
- Removed SVG repaint animation on the health-ring `stroke-dasharray` (was the CLS source)
- Added `font-display: swap` for the Noto Sans JP import + `preconnect` hints for `fonts.googleapis.com` (−400ms / −200ms FCP)
- Sentence-transformer, MeCab, and LangChain imports moved to lazy-load-on-first-use (−1200ms / −300ms / −800ms cold start respectively) — a sentinel + function pattern (`_MODEL = None` → populate on first call) rather than eager module-level imports

---

## [v3.0] — 2026-06

**FastAPI + Jinja2 + Alpine.js migration.** Not a preference change — Streamlit's CSP was blocking the onclick tab system, and its single-thread session model raced with `asyncio.to_thread()` LLM calls.

### Changed

- Full serving-layer migration from Streamlit (`app.py`, kept as frozen legacy reference) to FastAPI + Jinja2 templates + Alpine.js — no SPA framework, no bundler, no build step (required for HuggingFace Spaces' no-build-step constraint)
- `sdk: docker` replaces `sdk: streamlit` in the HF Space config
- Every blocking call (LLM, MeCab, ChromaDB) wrapped in `asyncio.to_thread()` so the event loop is never blocked
- CSS-only radio-button tab system (no inline JS, CSP-safe)
- MeCab keigo detection added — LLM's surface-text keigo guess was unreliable without morphological analysis

### Accuracy

~65% (up from ~45% in v2), driven by the MeCab morphological override at pipeline stage 7.

---

## [v2.0] — 2026-05

**Language expansion.**

### Added

- Japanese and Hindi analysis layers
- Initial nemawashi soft-rejection pattern set
- Fuzzy speaker-name matching via TF-IDF similarity

### Fixed

- Action-item attribution was collapsing to `"Director"` (role title) instead of the speaker's actual first name — traced via eval Action-Item F1 (0.22 → 0.87 after the fix)

### Accuracy

~45–50%, up from the v1 baseline.

---

## [v1.0] — 2026-05-14

**First stable release.**

### Added

- Trilingual meeting intelligence — English, Hindi, Japanese
- 16 nemawashi soft-rejection patterns with per-pattern confidence scores
- 8 Hindi indirect communication patterns (Devanagari + Roman script)
- 40+ English commitment-strength and hedging patterns
- Keigo formality detection via MeCab morphological analysis
- Cross-script speaker normalization (田中 ↔ Tanaka ↔ Director) — hash-map dedup keyed by normalized name, `O(n)` single pass
- PII masking — names, phones, emails anonymized before LLM, via a bidirectional dict (`PIIMask`: `mapping` placeholder→value, `reverse` value→placeholder) giving `O(1)` lookups in both directions
- Rule-based hallucination guard — Jaccard token-overlap verification, no LLM self-validation
- Three-tier LLM fallback: Groq → Ollama → Mock
- MD5 result caching with 24-hour TTL
- JSONL observability logging with drift detection
- FastAPI REST endpoints: `/analyze`, `/analyze/batch`, `/health`
- Async job queue via ThreadPoolExecutor
- Deployed on Hugging Face Spaces: [KunalTheBeast/TranscriptAI](https://huggingface.co/spaces/KunalTheBeast/TranscriptAI)

### Evaluation (v1 → v5 iteration history)

| Version | Score     | Primary Change                                                      |
| ------- | --------- | ------------------------------------------------------------------- |
| v1      | ~22–30%   | Baseline — exact string matching, no cultural awareness             |
| v2      | ~45–50%   | Fuzzy names, rule-based code-switch, semantic similarity            |
| v3      | ~65–75%   | Cultural ground truth, JA tokenization, soft sentiment              |
| v4      | ~75–85%   | Hallucination guard bonus, bilingual action items, speaker fix      |
| v5      | **93.8%** | 2-key rotation, vector cache, `bypass_cache` fix, tone intelligence |

---

## [Unreleased]

Planned for future releases — grouped by priority, each tied to a specific algorithmic change:

### P0

- Wire `api/async_processor.py` job queue into `main.py` + `sessionStorage` polling — replaces the current blocking `fetch()` that loses in-flight analysis on navigation, with a submit-job/poll-status pattern (`O(1)` job-status dict lookup)
- Expand behavioral test coverage beyond current smoke tests — assert correct values not just return types
- Separate `yes_trap_signals` (承知しました, はい、承知しました) from rejection `risk_level` — currently inflates risk for normal, politely-attentive meetings

### P1

- `pyannote.audio` speaker diarization (replace the ~70%-accurate silence-gap heuristic with model-based "who spoke when")
- Labeled dataset + Platt scaling for calibrated confidence scores
- External validation on real-world transcripts (the 25/30/60-recording eval sets are scaffolded but empty)
- User correction loop for fine-tuning

### P2

- Trie-based PII name matching (`O(n·L)` vs. the current sort-then-scan `O(n·k)`) — ~30-50x speedup once the surname DB scales past a few hundred entries
- Aho-Corasick multi-pattern matching (`O(n + Σm)`) if the rejection pattern set grows past ~500
- Redis Queue + multi-worker FastAPI (Scale-1 path)
