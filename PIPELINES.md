# TranscriptAI — Comprehensive Pipeline Architecture & Engineering Deep Dive

TranscriptAI processes multilingual business dialogue (Japanese, English, Hindi, and Code-Switched speech) into structured, culturally decoded intelligence. The system avoids relying solely on an end-to-end LLM; instead, it implements a **15-stage hybrid pipeline** combining deterministic NLP, morphological analysis, vector search, rule engines, and LLM inference.

This document details every pipeline in the system from input ingestion to final export.

---

## High-Level Pipeline Architecture

```
Raw Audio / Text File / Form Input
       │
       ▼
[ Pipeline 1: Ingestion & Audio Transcription ]
       │  (Groq Whisper large-v3 / text normalization / speaker labeling)
       ▼
[ Pipeline 2: APPI Local PII Masking ] ◄── BEFORE any network / LLM call
       │  (500+ JP surnames, Latin speaker labels, emails, phones -> [NAME_1])
       ▼
[ Pipeline 3: Two-Tier Cache Lookup ]
       ├─ Vector Semantic Cache (ChromaDB + MiniLM cosine ≥ 0.98) ──► Hit: Return (<100ms)
       └─ MD5 Content Cache (24h TTL) ──────────────────────────────► Hit: Return (<1ms)
       │ (Miss)
       ▼
[ Pipeline 4: Prompt Engineering & Model Routing ]
       │  (Truncation cap 1200w, dynamic schema, token budget 1300–2800)
       ▼
[ Pipeline 5: LLM Inference Cascade & Key Rotation ]
       │  (NIM / Groq 70B ──► Groq 8B ──► Ollama Local ──► Deterministic Extractive)
       ▼
[ Pipeline 6: Local PII Restoration ]
       │  (Word-boundary regex restore with bracket corruption tolerance)
       ▼
[ Pipeline 7: Output Validation, Schema Repair & Speaker Normalization ]
       │  (Field backfilling, talk-time recalculation, 田中 ↔ Tanaka ↔ Director merge)
       ▼
[ Pipeline 8: Action Item Extraction & Rule-Based Backfill ]
       │  (LLM items + deterministic regex imperative backfill)
       ▼
[ Pipeline 9: Hallucination Verification Guard ]
       │  (Deterministic Jaccard token overlap + MiniLM semantic rescue; NO LLM self-grading)
       ▼
[ Pipeline 10: Japanese Business-Language & Cultural Analysis ]
       ├─ MeCab morphological keigo detection (fugashi IPADIC override)
       ├─ Rule-based Unicode code-switching counter
       ├─ 3-tier Soft Rejection & "はい trap" detector
       └─ 3-phase temporal Nemawashi sequence detector
       ▼
[ Pipeline 11: Deal Outcome & Conversation Dynamics ]
       ├─ 8-state priority state machine (Approved, Rejected, Conditional, etc.)
       ├─ Meeting Health Score (0–100 composite index)
       └─ Senior silence, stalls, and conversational pivots
       ▼
[ Pipeline 12: Caching, Observability & MLflow Logging ]
       │  (ChromaDB vector store upsert, JSONL audit log, MLflow tracking)
       ▼
[ Pipeline 13: Export & Multi-Modal Serving ]
       ├─ 6-slide executive PPTX deck (python-pptx)
       ├─ Formal Japanese Business Minutes (議事録 / Gijiroku)
       ├─ Markdown / JSON / TXT
       └─ FastAPI REST API & Jinja2 / Alpine.js UI
```

---

## Pipeline 1: Ingestion & Audio Transcription

* **Source Files:** [`transcription/audio_processor.py`](file:///c:/All%20Projects/voice%20analizer/transcription/audio_processor.py), [`main.py`](file:///c:/All%20Projects/voice%20analizer/main.py#L447-L487)
* **Goal:** Convert raw audio files or unstructured text documents into clean, speaker-attributed text.

### Ingestion Flow
1. **File Validation:**
   * Supported audio formats: `.mp3`, `.wav`, `.m4a`, `.mp4`, `.ogg`, `.webm`.
   * Size limit check: Max 25 MB (configured for Groq Whisper limits).
   * Supported text formats: `.txt`, `.vtt`, `.json`.
2. **Speech-to-Text (`transcribe_audio`):**
   * Calls the Groq Whisper API endpoint (`https://api.groq.com/openai/v1/audio/transcriptions`) with model `whisper-large-v3`.
   * Requests verbose JSON (`response_format="verbose_json"`), returning full text, segment-level timestamps (`start`, `end`), and language identification.
3. **Timestamp & Turn Formatting (`format_transcript_with_timestamps`):**
   * Segments are formatted as `[MM:SS] Speaker: Content`.
   * Consecutive speech turns from the same speaker within 1.5 seconds are merged to avoid fragmented sentences.
4. **Speaker Attribution Heuristics (`_ensure_speaker_labels`):**
   * If the input is raw text, regex patterns scan the first 40 lines for speaker labels (`Tanaka:`, `[00:01] Sato:`, `【佐藤】:`).
   * If fewer than 2 labeled turns are found, text is split by paragraph breaks and labeled synthetically (`Speaker 1:`, `Speaker 2:`) so downstream speaker attribution algorithms have stable anchor points.

---

## Pipeline 2: APPI Local PII Masking & Restoration

* **Source File:** [`transcription/pii_masker.py`](file:///c:/All%20Projects/voice%20analizer/transcription/pii_masker.py)
* **Goal:** Comply with Japan's Act on the Protection of Personal Information (APPI) by anonymizing personal data locally on the server before transmitting text across network boundaries.

### Masking Architecture (Stage A3)
```
Raw Transcript
     │
     ├─► Step 1: Scan 500+ Japanese Surnames (JMnedict table) ──┐
     ├─► Step 2: Scan Latin Speaker Labels (regex ^[A-Z][a-z]+:) ├─► Generate [NAME_N]
     ├─► Step 3: Scan JP/Intl Phone Numbers (\+81|0\d...) ───────┼─► Generate [PHONE_N]
     ├─► Step 4: Scan Email Addresses ([a-zA-Z0-9._%+-]+@...) ────┼─► Generate [EMAIL_N]
     └─► Step 5: (Optional) Corporate Prefixes (株式会社, etc.) ────┴─► Generate [COMPANY_N]
     │
     ▼
Masked Transcript (Safe to send to external LLMs / Vector DB)
```

1. **Bidirectional Dictionary (`PIIMask`):**
   * Maintains `mapping: dict` (`[NAME_1]` $\rightarrow$ `"Tanaka"`) and `reverse: dict` (`"Tanaka"` $\rightarrow$ `[NAME_1]`).
   * Lookup complexity: $O(1)$ in both directions.
2. **False Positive Prevention:**
   * English first names (e.g., "Kunal", "Sarah", "Mike") are excluded from global string replacement; they are only masked when extracted by positional speaker label regexes (`^[A-Z][a-zA-Z]+:`). This prevents words that double as common English nouns from corrupting the text.
   * `_NOT_SPEAKER` blocklist (48 words: `Background`, `Status`, `Priority`, `Decision`, `Assignee`, etc.) prevents markdown section headers from being converted into names.

### Restoration Architecture (Stage A4)
After the LLM completes inference, the placeholders in the output JSON must be restored:
* **Sorting Rule:** Placeholders are sorted by length descending ($O(k \log k)$) so `[NAME_10]` is replaced before `[NAME_1]` can partially match inside it.
* **LLM Bracket Corruption Tolerance:** Small LLMs frequently strip or alter markdown brackets. The restoration engine checks four structural variants for every placeholder:
  1. `[NAME_1]` (standard)
  2. `[NAME_1` (missing closing bracket)
  3. `NAME_1]` (missing opening bracket)
  4. `\bNAME_1\b` (word-boundary regex matching bare placeholder without modifying variable names like `MY_NAME_1_VAR`).

---

## Pipeline 3: Two-Tier Caching & Semantic Retrieval

* **Source Files:** [`utils/vector_cache.py`](file:///c:/All%20Projects/voice%20analizer/utils/vector_cache.py), [`utils/cache.py`](file:///c:/All%20Projects/voice%20analizer/utils/cache.py)
* **Goal:** Eliminate duplicate LLM inference calls, lower costs to \$0 for recurring meetings, and achieve $<100$ms response times.

### Tier 1: ChromaDB Semantic Vector Cache
1. **Embedding:** The first 2,000 characters of the **masked transcript** (ensuring no raw PII is written to disk) are embedded using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384-dimensional dense vectors).
2. **Distance Metric:** Cosine similarity via HNSW indexing (`metadata={"hnsw:space": "cosine"}`).
3. **Threshold Logic:**
   * $\text{Cosine Similarity} \ge 0.98$: Exact / near-identical match $\rightarrow$ instant cache hit.
   * $0.82 \le \text{Cosine Similarity} < 0.98$: Semantic match $\rightarrow$ if transcript length and speaker count correlate, returns the cached result.
   * $\text{Cosine Similarity} < 0.82$: Cache miss $\rightarrow$ proceed to LLM pipeline.
4. **Data Isolation:** ChromaDB collections are partitioned per user ID (`transcripts_{user_id}`) to prevent cross-tenant data leakage.

### Tier 2: MD5 Exact Hash Cache
* Computes MD5 hash of `f"{user_id}_{language}_{transcript}"`.
* Persists to local disk JSON store with a 24-hour Time-To-Live (TTL). Returns in $<1$ms on identical re-runs.

---

## Pipeline 4: Prompt Engineering & Token Optimization

* **Source Files:** [`prompts/analysis_prompt.py`](file:///c:/All%20Projects/voice%20analizer/prompts/analysis_prompt.py), [`analysis/analyzer.py`](file:///c:/All%20Projects/voice%20analizer/analysis/analyzer.py#L182-L350)
* **Goal:** Enforce strict JSON output, ground factual claims, and minimize token usage to operate sustainably on free-tier rate limits.

### Dynamic Schema Composition
To reduce prompt token bloat by 44% (from 2,728 tokens down to 1,521 tokens):
* **Language-Conditional Schema Injection:** The `japan_insights` schema block (keigo, nemawashi, ringi status) is omitted entirely for English-only transcripts and injected only when Japanese or mixed characters are detected.
* **Truncation Window:** Transcripts exceeding 1,200 words are compressed by retaining the first 60% (meeting setup, agenda, key debates) and the last 40% (conclusions, action assignments, deadlines), dropping low-signal middle pleasantries.
* **Dynamic Max Token Budget:**
  * $<300$ words: 1,300 tokens
  * $300 - 800$ words: 1,700 tokens
  * $800 - 2000$ words: 2,200 tokens
  * $>2000$ words: 2,800 tokens

---

## Pipeline 5: LLM Inference Cascade & Key Rotation

* **Source File:** [`analysis/analyzer.py`](file:///c:/All%20Projects/voice%20analizer/analysis/analyzer.py#L900-L1400)
* **Goal:** Guarantee high-availability inference without 500 error crashes when encountering rate limits or provider downtime.

```
Request Triggered
       │
       ▼
Primary Provider: Groq / NVIDIA NIM (Llama-3.3-70B-Versatile / Nemotron-49B)
       │
       ├─ HTTP 200 ──► Success, proceed to parsing
       ├─ HTTP 429 (RPM Limit) ──► Sleep retry-after duration, retry on same key
       └─ HTTP 429 (Daily Quota Exhausted) ──► Rotate to GROQ_API_KEY_2
              │
              ▼ (All 70B keys exhausted)
       Fallback 1: Groq Fast Model (Llama-3.1-8B-Instant)
              │
              ▼ (Groq API completely unavailable)
       Fallback 2: Local Ollama Instance (qwen2.5:7b / qwen3:8b)
              │
              ▼ (Ollama offline / timed out)
       Fallback 3: Deterministic Extractive Fallback (_no_api_result)
```

### Deterministic Extractive Fallback (`_extractive_summary.py`)
If cloud and local LLMs are completely unreachable, the pipeline does not throw an unhandled exception. It executes an extractive fallback:
* Computes sentence importance using word frequency scoring.
* Identifies action items using regex verb-phrase patterns (`must`, `will`, `agreed to`, `べき`, `確認する`).
* Extracts speaker talk-time via character count heuristics.
* Sets `_provider: "no_api"` with transparent warning banners in the UI.

---

## Pipeline 6: Output Validation, Schema Repair & Speaker Normalization

* **Source Files:** [`analysis/analyzer.py`](file:///c:/All%20Projects/voice%20analizer/analysis/analyzer.py#L1400-L1810), [`transcription/speaker_normalizer.py`](file:///c:/All%20Projects/voice%20analizer/transcription/speaker_normalizer.py)
* **Goal:** Guarantee structural integrity of the JSON payload and resolve cross-script identity duplicates.

### 1. Partial JSON Recovery & Schema Filling
* If the LLM generation cut off prematurely, `_extract_partial()` salvages completed summary bullets and action items before parsing fails.
* `_validate_and_fill()` checks required top-level keys (`summary`, `action_items`, `speakers`, `sentiment`, `japan_insights`, `deal_outcome`). Any missing key is initialized with schema-valid defaults rather than allowing `KeyError` crashes in the frontend.

### 2. Talk-Time Recalculation (`_recompute_talk_time_pct`)
* LLMs are poor at estimating percentage talk-time. The pipeline scans the transcript text deterministically, aggregates character lengths per speaker turn, and normalizes talk-time percentages to sum to 100%.

### 3. Cross-Script Speaker Normalization (`unify_speakers_in_result`)
* Solves the bilingual speaker fragmentation problem:
  ```
  "田中"  (Kanji) ──┐
  "Tanaka" (Romaji) ──┼──► Canonical Entity: "Tanaka" (Merged talk time & actions)
  "Director" (Role) ──┘
  ```
* Strips Japanese honorifics (`-san`, `様`, `さん`, `部長`, `課長`).
* Compares Latin and Japanese forms. Merges role titles (e.g. "Director") with named entities when context indicates identity.

---

## Pipeline 7: Action Item Extraction & Rule-Based Backfill

* **Source Files:** [`analysis/action_item_extractor.py`](file:///c:/All%20Projects/voice%20analizer/analysis/action_item_extractor.py), [`analysis/analyzer.py`](file:///c:/All%20Projects/voice%20analizer/analysis/analyzer.py#L2090-L2110)
* **Goal:** Maximize action-item recall by pairing LLM semantic comprehension with high-precision deterministic pattern matching.

### The Two-Stage Extraction Strategy
1. **LLM Extraction:** The LLM extracts high-level commitments with assigned owners and deadlines.
2. **Deterministic Pattern Backfill:**
   * Scans dialogue for imperative and commitment markers:
     * **English:** `will prepare`, `need to confirm`, `follow up with`, `agreed to send`, `by [Day/Time]`.
     * **Japanese:** `〜までに`, `確認します`, `送付いたします`, `担当します`, `作成予定`.
   * For every regex hit, checks whether the task is already present in the LLM output (using 30-character normalized string matching).
   * Any missed task is appended as an action item, guaranteeing that explicit procedural commitments are not overlooked.

---

## Pipeline 8: Hallucination Verification Guard

* **Source Files:** [`analysis/hallucination_guard.py`](file:///c:/All%20Projects/voice%20analizer/analysis/hallucination_guard.py), [`analysis/semantic_validator.py`](file:///c:/All%20Projects/voice%20analizer/analysis/semantic_validator.py)
* **Goal:** Verify that every extracted action item, owner, deadline, and summary bullet is grounded in the source transcript **without circular LLM self-validation**.

```
Extracted Action Item (Task, Owner, Deadline)
       │
       ├─► Check 1: Jaccard Token Overlap Score (_overlap_score)
       │            Extracts transcript tokens + speaker labels (Tanaka:)
       │            Custom _ja_tokenize: splits Latin words + CJK glyphs + bigrams
       │            Computes: |Claim ∩ Transcript| / |Claim| (excluding stopwords)
       │
       └─► Check 2: Cross-Language Semantic Rescue (semantic_grounding_score)
                    If token overlap is low (e.g. English task from Japanese speech),
                    computes cosine similarity using paraphrase-multilingual-MiniLM-L12-v2.
       │
       ▼
Unified Effective Task Score = max(token_overlap, semantic_score)
       │
       ▼
Confidence = 0.60 * task_score + 0.30 * owner_score + 0.10 * deadline_score
       │
       ├─ Effective Task Score < 0.20 ──► Set hallucination_flag = True (Flagged)
       └─ Effective Task Score ≥ 0.20 ──► Set hallucination_flag = False (Verified)
```

---

## Pipeline 9: Japanese Business-Language & Cultural Decoding

* **Source Files:** [`analysis/japanese_tokenizer.py`](file:///c:/All%20Projects/voice%20analizer/analysis/japanese_tokenizer.py), [`analysis/soft_rejection_detector.py`](file:///c:/All%20Projects/voice%20analizer/analysis/soft_rejection_detector.py), [`analysis/nemawashi_sequence.py`](file:///c:/All%20Projects/voice%20analizer/analysis/nemawashi_sequence.py), [`utils/evaluator.py`](file:///c:/All%20Projects/voice%20analizer/utils/evaluator.py#L58-L84)
* **Goal:** Decode indirect communication norms that generic LLMs misclassify as agreement or neutral talk.

### 1. MeCab Keigo Morphological Formality Detection
* Generic tokenizers split Japanese at character level or arbitrary subwords.
* TranscriptAI uses `fugashi` (MeCab wrapper with IPADIC):
  * Normalizes inflected verbs to base lemmas (`検討します` $\rightarrow$ `検討する`).
  * Counts auxiliary honorific morphemes:
    * **Sonkeigo / Kenjougo (Respectful/Humble):** `いらっしゃる`, `おっしゃる`, `参る`, `致す`, `いただく`.
    * **Teineigo (Polite):** `です`, `ます`, `ございます`.
  * Classifies formality into `high`, `moderate`, or `plain`, overriding LLM guesses.

### 2. Tiered Soft Rejection & "はい Trap" Detection
Contains 60+ patterns across three calibrated severity tiers:
* **Tier 1 — CRITICAL (Explicit Termination):**
  * `継続しないことを決定いたしました` ("Decided not to continue"), `契約を更新しない` ("Will not renew").
  * Immediate contract termination; overrides all positive sentiment.
* **Tier 1b & 1c — HIGH (Contract Threat & Approval Gates):**
  * Conditional threats: `再検討せざるを得ない` ("Will be forced to reconsider"), `not acceptable`, `unless resolved by`.
  * Approval gates: `稟議が必要です` ("Ringi board approval required"), `上司の判断を仰ぐ`.
* **Tier 2 & 3 — MEDIUM / LOW (Soft Hedging):**
  * `検討いたします` ("We will consider it" — 72% rejection confidence).
  * `難しいかもしれません` ("It may be difficult" — 90% polite rejection).
  * `善処します` ("We will handle appropriately" — non-committal avoidance).
* **The "はい Trap":**
  * Flags `承知いたしました` and `はい` as **listening acknowledgment**, not contractual approval.

### 3. Three-Phase Nemawashi Sequence State Machine
Detects informal behind-the-scenes consensus building through multi-turn conversational dynamics:
* **Phase 1 (Trigger):** A speaker displays authority deferral (`上司に相談して...`).
* **Phase 2 (Gap):** Offline stakeholder alignment (implicit gap in dialogue).
* **Phase 3 (Resolution):** The same speaker returns in a subsequent turn with **quantified specificity markers**:
  * Deadlines with quantities (`2時間以内`, `within 24 hours`).
  * Specific deliverables (`書面でご回答`, `written confirmation`).
  * Named stakeholders or methods.
* Returns a structured `NemawashiResult` with confidence rating and cultural explanation.

---

## Pipeline 10: Deal Outcome & Conversation Dynamics

* **Source Files:** [`analysis/deal_outcome_detector.py`](file:///c:/All%20Projects/voice%20analizer/analysis/deal_outcome_detector.py), [`analysis/conversation_dynamics.py`](file:///c:/All%20Projects/voice%20analizer/analysis/conversation_dynamics.py), [`utils/html_renderer.py`](file:///c:/All%20Projects/voice%20analizer/utils/html_renderer.py)
* **Goal:** Synthesize multi-speaker conversation patterns into clear executive decisions and risk ratings.

### 1. 8-State Deal Outcome State Machine
Evaluates dialogue using a strict priority order:
1. `REJECTED` (Tier 1 termination phrases matched)
2. `CONDITIONAL` (Conditional terms detected before clean acceptance, e.g. "accept provided that...")
3. `APPROVED` (Explicit acceptance: `契約を締結します`, `agree to move forward`)
4. `DEFERRED` (Postponed decision, e.g. `持ち帰らせてください`)
5. `PENDING` (Awaiting internal approval gate / Ringi)
6. `INFORMATIONAL` (Status update or knowledge sharing with no commercial decision)
7. `AT_RISK` (Unresolved complaints or deadline ultimatums)
8. `UNCLEAR` (Default if dialogue inconclusive)

### 2. Meeting Health Score Formula (0–100)
A composite index reflecting overall meeting health:
$$\text{Health Score} = \text{Base Score} - (\text{Rejection Penalty}) - (\text{Hallucination Penalty}) - (\text{Action Ambiguity})$$
* **Hard Ceilings:**
  * If risk tier is `CRITICAL`, score is hard-capped at **22/100**.
  * If risk tier is `HIGH`, score is hard-capped at **35/100**.

### 3. Conversation Dynamics
* **Senior Silence:** Flags when designated senior decision makers speak $<10\%$ of talk time.
* **Topic Stalls:** Identifies agenda items debated for $>5$ turns without reaching an action item.
* **Closing Summarizer:** Identifies which speaker took authority to summarize next steps.

---

## Pipeline 11: RAG Cross-Meeting Retrieval

* **Source Files:** [`rags/meeting_store.py`](file:///c:/All%20Projects/voice%20analizer/rags/meeting_store.py), [`rags/rag_retriever.py`](file:///c:/All%20Projects/voice%20analizer/rags/rag_retriever.py)
* **Goal:** Provide grounded, cross-meeting conversational memory without hallucinations.

### 1. Storage & Chunking Architecture
* Transcript text is chunked using an overlapping sliding window:
  * Chunk Size: 200 words
  * Overlap: 40 words
  * Stride: 160 words
* Chunk 0 is enriched by appending the LLM executive summary.
* Embedded via `paraphrase-multilingual-MiniLM-L12-v2` and stored in ChromaDB with metadata filters (`language`, `soft_risk`, `date`, `keigo_level`).

### 2. Retrieval & Grounded Generation
* When a user queries historical meetings (`ask_about_meetings`):
  1. Embeds the user question.
  2. Retrieves top-$K$ ($K=3$) matching chunks from ChromaDB, deduplicating chunks from the same meeting.
  3. Builds a strict grounding prompt injecting retrieved excerpts.
  4. Calls `llama-3.1-8b-instant` to generate a cited response (*"Meeting 1 indicates..."*), strictly instructing the model to reply *"Information not in records"* if excerpts do not contain the answer.

---

## Pipeline 12: MLOps Evaluation & Continuous Benchmarking

* **Source Files:** [`utils/evaluator.py`](file:///c:/All%20Projects/voice%20analizer/utils/evaluator.py), [`tests/test_data.py`](file:///c:/All%20Projects/voice%20analizer/tests/test_data.py)
* **Goal:** Benchmark pipeline accuracy against ground truth on every code commit.

### Metric Calculation
* **Summary Score:** Weighted combination of token ROUGE-1 F1 and MiniLM sentence embedding cosine similarity.
* **Action-Item F1:** Precision and Recall matched against ground truth using semantic task matching ($\ge 0.65$), fuzzy owner matching, and deadline alignment.
* **Sentiment Soft Accuracy:** Exact match scores 1.0; culturally acceptable variants score 0.5.
* **Code-Switching Validation:** Asserts predicted count matches exact regex-counted language switches.
* **MLflow Tracking:** Logs run parameters, overall scores (current live baseline: **93.8%**), and sub-metrics to `./mlruns` or remote tracking servers.

---

## Pipeline 13: Export & Serving Layer

* **Source Files:** [`exporters/pptx_builder.py`](file:///c:/All%20Projects/voice%20analizer/exporters/pptx_builder.py), [`agents/gijiroku_formatter.py`](file:///c:/All%20Projects/voice%20analizer/agents/gijiroku_formatter.py), [`agents/slide_architect.py`](file:///c:/All%20Projects/voice%20analizer/agents/slide_architect.py), [`main.py`](file:///c:/All%20Projects/voice%20analizer/main.py)
* **Goal:** Distribute structured intelligence into standard enterprise formats and fast web interfaces.

### 1. 6-Slide Executive PowerPoint Deck (`pptx_builder.py`)
Generates native `.pptx` presentations using `python-pptx`:
* Slide 1: Title & Executive Meeting Verdict
* Slide 2: Said vs. Meant (Cultural decoding table)
* Slide 3: Risk Watch & Soft Rejection Analysis
* Slide 4: Key Decisions & Deal Outcome
* Slide 5: Action Items & Responsibilities
* Slide 6: Communication Dynamics & Next Steps

### 2. Formal Japanese Business Minutes (議事録 / Gijiroku)
Generates markdown minutes adhering to standard Tokyo enterprise structure:
* `日時` (Date & Time), `場所` (Location), `出席者` (Attendees)
* `決定事項` (Decisions reached)
* `協議内容` (Discussion points categorized by topic)
* `保留事項・次回検討` (Pending items / Ringi gates)
* `次回日程` (Next meeting schedule)

### 3. Serving Architecture
* **FastAPI Server:** Non-blocking async event loop wrapping CPU-bound tasks in `asyncio.to_thread()`.
* **Rate Limiting:** SlowAPI IP-based rate limiting (10 requests/min on `/analyze-text`).
* **Frontend:** Server-side rendered Jinja2 templates styled with Vanilla CSS and animated with Alpine.js reactive components (no heavy Node.js SPA build step required, enabling instant deploys on Hugging Face Spaces Docker SDK).
