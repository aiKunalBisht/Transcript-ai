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
**Japanese Business Intelligence Platform — Speech & Meeting Analyzer**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace-D96080?style=flat-square)](https://huggingface.co/spaces/KunalTheBeast/TranscriptAI)
[![GitHub](https://img.shields.io/badge/GitHub-Repo-3C2416?style=flat-square&logo=github)](https://github.com/aiKunalBisht/Transcript-ai)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> Turns any meeting transcript into structured business intelligence in ~3 seconds.
> Works with Otter.ai, Zoom, Google Meet, Whisperflow exports out of the box.

---

## How we got here — v1 to v5

| Version | What changed | Accuracy |
|---|---|---|
| v1 | Hard exact matching, no cultural awareness | 22–30% |
| v2 | Fuzzy names, rule-based code-switch, TF-IDF similarity | ~45% |
| v3 | MeCab keigo detection, bilingual ground truth, soft sentiment | ~60% |
| v4 | Hallucination guard, 16 nemawashi patterns, APPI PII masking | 75–85% |
| **v5 (live)** | 2-key Groq rotation, vector cache, health score, trends | **85–95%** |

---

## Why Japanese business is different

| What was said | Generic AI | TranscriptAI |
|---|---|---|
| 検討いたします | "We will consider it" — action item | ⚠ Likely soft rejection (72%) |
| 難しいかもしれません | "May be difficult" — neutral | 🚨 High rejection signal (90%) |
| 承知いたしました | Acknowledged | 🏯 High keigo — senior speaker |

---

## Features

- **Trilingual** — Japanese / Hindi (Hinglish) / English / Mixed
- **Keigo detection** via MeCab morphological analysis
- **16 nemawashi soft rejection patterns** with confidence scores
- **8 Hindi indirect patterns** (देखते हैं, थोड़ा मुश्किल है)
- **APPI-compliant PII masking** before any LLM call
- **Hallucination guard** — 100% rule-based, LLM never validates itself
- **2-key Groq rotation** — auto-failover on 429 rate limit
- **Vector cache** — instant return for similar transcripts
- **Meeting health score** (0–100) with breakdown
- **Trends dashboard** — soft rejection drift, provider usage over time
- **Groq Whisper** — MP4/MP3/WAV transcription (same free key)
- **FastAPI REST** endpoint for CRM integration

---

## Quick start

```bash
git clone https://github.com/aiKunalBisht/Transcript-ai.git
cd Transcript-ai
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here   # free at console.groq.com
python -m streamlit run app.py
```

**HuggingFace Spaces:** Add `GROQ_API_KEY` in Space → Settings → Repository secrets.

---

## Evaluation

| Test case | v1 baseline | v5 live |
|---|---|---|
| Sales call · JA/EN mixed | 30.8% | **95.2%** |
| Internal meeting · Japanese | 22.2% | **81.6%** |
| Client conflict · EN/JA | 55.9% | **85.8%** |

---

## Built by

[Kunal Bisht](https://github.com/aiKunalBisht) · Benglore, Karnataka, India