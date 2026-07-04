# Changelog

All notable changes to TranscriptAI are documented here.

Format: `[version] — date` → what changed and why. Where a change was driven by a specific data-structure/algorithm choice, that's called out — most of TranscriptAI's accuracy and latency wins trace directly to a DSA decision, not a bigger model.

---

## [v3.2] — 2026-07 (current · 156 commits · 93.8% accuracy)

**Soft rejection rewrite + deal outcome detector.**

### Added

- `analysis/deal_outcome_detector.py` — 8-state meeting outcome verdict (`REJECTED → APPROVED → CONDITIONAL → DEFERRED → PENDING → INFORMATIONAL → AT_RISK → UNCLEAR`), computed with a strict priority order (most-specific/highest-certainty signal wins; `CONDITIONAL` checked before plain `APPROVED` so "we accept on the condition that…" doesn't read as a clean yes)
- `CRITICAL` tier in `soft_rejection_detector.py` for explicit contract termination — previously these transcripts (e.g. 契約を更新しないことを決定いたしました) matched zero soft-hedge patterns and returned `risk_level: NONE`, which was backwards for a meeting that just ended a relationship
- Tier 1b "approval gate" detection (HIGH risk — decision pending external authority)
- Backward-compatible `detected` / `risk_summary` alias keys on the return dict, so `cultural_insights_formatter.py` and `slide_architect.py` didn't need touching

### Fixed

- Role-only speaker merge bug — `normalize_speaker_name()` stripped role-only labels (`Director`, `Manager`) to `''`, and the substring-match fallback then treated `''` as a substring of every name, silently merging two distinct speakers into one. Fix: explicit empty-string guard before the substring loop in `_best_match()`.

### DSA notes

- **Zero-collision pattern design**: before shipping 72 new acceptance/conditional/deferred/informational phrases, every one was checked for substring overlap against all 6 existing rejection pattern sets (20 soft, 17 EN-high, 15 JP-high, 19 EN-termination, 15 JP-termination, approval-gate). Result: zero collisions — a transcript can never trigger both an acceptance and a rejection pattern on the same phrase.
- Pattern matching stays naive `O(n·m·p)` substring search (Python's `in`) rather than Aho-Corasick — at ~60 patterns and 500–3,000-word transcripts this is <0.5ms; Aho-Corasick's `O(n + Σm)` only pays for itself past ~500 patterns.

### Accuracy

| Version               | Score | Root cause traced                                                                                                              |
| --------------------- | ----- | ------------------------------------------------------------------------------------------------------------------------------ |
| v4                    | ~82%  | —                                                                                                                              |
| v5 (bypass_cache fix) | 93.8% | All 3 eval test cases were returning the _same_ cached vector-similarity hit — `bypass_cache=True` now forces independent runs |

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
- APPI-compliant PII masking — names, phones, emails anonymized before LLM, via a bidirectional dict (`PIIMask`: `mapping` placeholder→value, `reverse` value→placeholder) giving `O(1)` lookups in both directions
- Rule-based hallucination guard — Jaccard token-overlap verification, no LLM self-validation (avoids the model grading its own homework)
- Three-tier LLM fallback: Groq → Ollama → Mock (explicit UX feedback at each tier), implemented as an ordered-list priority scan (`O(P)`, `P≤3` — not worth a heap at this scale)
- Dynamic token budget by transcript length (prevents Ollama timeouts)
- MD5 result caching with 24-hour TTL — content-addressable storage, same principle as Git object storage
- JSONL observability logging with drift detection
- Meeting trends dashboard — soft rejection trends, hallucination rate, workload analysis
- Live token streaming (Groq)
- MP4/MP3/WAV/M4A transcription via Groq Whisper (free tier)
- FastAPI REST endpoints: `/analyze`, `/analyze/batch`, `/health`, `/patterns/soft-rejections`
- Async job queue via ThreadPoolExecutor
- Streamlit UI with 7 tabs and sakura/peach palette
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
- Meaningful regression tests — current 21 tests only assert `isinstance(result, dict)`; none assert a correct _value_
- Separate `yes_trap_signals` (承知しました, はい、承知しました) from rejection `risk_level` — currently inflates risk for normal, politely-attentive meetings

### P1

- `pyannote.audio` speaker diarization (replace the ~70%-accurate silence-gap heuristic with model-based "who spoke when")
- Audio upload on Hugging Face Spaces (Groq Whisper API integration)
- Labeled dataset + Platt scaling for calibrated confidence scores
- External validation on real-world transcripts (the 25/30/60-recording eval sets are scaffolded but empty)
- User correction loop for fine-tuning

### P2

- Trie-based PII name matching (`O(n·L)` vs. the current sort-then-scan `O(n·k)`) — ~30-50x speedup once the surname DB scales past a few hundred entries
- Aho-Corasick multi-pattern matching (`O(n + Σm)`) if the rejection pattern set grows past ~500
- Redis Queue + multi-worker FastAPI (Scale-1 path)
