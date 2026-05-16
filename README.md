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
short_description: Meeting intelligence — English, Hindi & Japanese
---

<div align="center">

# TranscriptAI

**Meeting intelligence that understands not just what was said — but what was meant.**

[![GitHub](https://img.shields.io/badge/GitHub-Source-181717?style=for-the-badge&logo=github)](https://github.com/aiKunalBisht/Transcript-ai)
[![CI](https://github.com/aiKunalBisht/Transcript-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/aiKunalBisht/Transcript-ai/actions)
[![Eval Score](https://img.shields.io/badge/Eval%20Score-93%25-brightgreen?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)]()

English · Hindi · Japanese

</div>

---

## What it does

Most meeting tools extract *what* was said. They miss everything underneath.

Every language carries indirect communication — polite rejections, soft commitments, face-saving agreements — that a generic summarizer logs as action items that never get done.

TranscriptAI is built to catch those signals.

**Paste any transcript → get structured intelligence in seconds. No API key needed.**

---

## Output

| Field | What you get |
|---|---|
| Summary | Concise narrative + key bullets scaled to meeting length |
| Action items | Owner · deadline · commitment strength rating |
| Risk signals | Soft rejections · hedging · power imbalance markers |
| Speaker tone | 6-level colour-coded scale per speaker |
| Meeting health | 0–100 composite score across sentiment, clarity, and AI confidence |
| Trends | Risk drift and workload patterns across sessions |

---

## Language Engines

### English
Commitment strength grading — distinguishes "I will deliver" from "I will try" from "we will see." Detects hedging, escalation signals, and passive aggression. 40+ patterns across 4 categories.

### Hindi
Indirect refusals, hierarchical agreement (saying yes to please rather than commit), face-saving exits. Handles Devanagari and Roman script. 30+ patterns.

### Japanese
**16 nemawashi soft-rejection patterns** — each scored with per-pattern confidence. Keigo formality detection via MeCab morphological analysis (丁寧語・尊敬語・謙譲語). Cross-script speaker normalization — 田中 and Tanaka and Director resolve to one identity.

---

## Accuracy

| Version | Score | Change |
|---|---|---|
| v1 | 22% | Baseline — exact string matching |
| v2 | 45% | Fuzzy matching, semantic similarity |
| v3 | 65% | Cultural ground truth, Japanese tokenization |
| v4 | 83% | Hallucination guard, soft rejection filter |
| v5 | **93%** | Tone intelligence, optimal speaker assignment |

---

## How it works

```
1. PII Mask     → local, before LLM          (names, phones, emails anonymized)
2. LLM Analysis → Groq / Ollama / Mock
3. PII Restore  → local, before output
4. Normalize    → cross-script speaker deduplication
5. Tone Score   → per-speaker 6-level classification
6. NLP Layer    → language-specific signal detection
7. Cache + Log  → MD5 cache write, JSONL observability
```

**LLM fallback chain:** Groq (1–2s, free) → Ollama (local, offline) → Mock (always available)

**Hallucination guard:** rule-based token overlap. LLM output is never used to validate itself.

---

## Quick Start

```bash
git clone https://github.com/aiKunalBisht/Transcript-ai.git
cd Transcript-ai
pip install -r requirements.txt
export GROQ_API_KEY=your_key    # free at console.groq.com
python -m streamlit run app.py
```

**Fully offline:**
```bash
ollama pull qwen3:8b
python -m streamlit run app.py
```

---

## REST API

```bash
python api.py
# Docs at http://localhost:8000/docs
```

```python
import requests
result = requests.post("http://localhost:8000/analyze", json={
    "transcript": "Alex: Can we ship Friday?\nJordan: We will see what we can do.",
    "language": "en",
    "mask_pii": True
}).json()["result"]

print(result["soft_rejections"]["risk_level"])    # HIGH
print(result["soft_rejections"]["risk_summary"])  # Commitment unlikely to be followed through
```

---

## Known Limitations

| Limitation | Planned |
|---|---|
| Speaker diarization ~70% | pyannote.audio integration |
| Audio upload unavailable on HF | Groq Whisper API next release |
| Confidence scores are heuristic | Labeled dataset + calibration |

---

<div align="center">

Built by [Kunal Bisht](https://github.com/aiKunalBisht) — Pithoragarh, India

[GitHub](https://github.com/aiKunalBisht) · [LinkedIn](https://linkedin.com/in/kunalhere)

</div>