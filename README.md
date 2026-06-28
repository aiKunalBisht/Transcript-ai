---
title: TranscriptAI
emoji: 🧠
colorFrom: pink
colorTo: red
sdk: docker
pinned: false
---

<div align="center">

# 🧠 TranscriptAI

**Multilingual Meeting Intelligence · Japanese · Hindi · English · Mixed**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![Accuracy](https://img.shields.io/badge/Accuracy-93.8%25-D96080?style=flat-square)](https://huggingface.co/spaces/KunalTheBeast/TranscriptAI)
[![GitHub](https://img.shields.io/badge/Source-GitHub-3C2416?style=flat-square&logo=github)](https://github.com/aiKunalBisht/Transcript-ai)

*Turns any meeting transcript into structured business intelligence in ~3 seconds.*

</div>

---

## What It Does

Generic meeting summarisers extract what was **said**. They miss what was **meant**.

| What was said | Generic AI | TranscriptAI |
|---|---|---|
| 検討いたします | "Action: We will consider it" | ⚠ Soft rejection — 72% confidence |
| 難しいかもしれません | "It may be difficult" — neutral | 🚨 HIGH rejection signal — deal at risk |
| パートナーシップは継続しないことを決定しました | "Decision made" | ⛔ CRITICAL — Explicit contract termination |

- **Trilingual** — Japanese / Hindi / English / Mixed in a single meeting
- **Keigo detection** — MeCab morphological analysis, formality register High / Medium / Low
- **20 soft rejection patterns** + CRITICAL explicit termination tier
- **APPI-compliant PII masking** — before any LLM call, local NER
- **議事録 export** — structured Japanese business minutes
- **Cultural insights export** — 根回し risk, 稟議 status, nemawashi breakdown
- **Meeting Health Score** — 0–100, caps at 22 for contract terminations
- **Groq Whisper** — MP4 / MP3 / WAV audio input

---

## Quick Start

```bash
git clone https://github.com/aiKunalBisht/Transcript-ai.git
cd Transcript-ai
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
uvicorn main:app --reload --port 7860
```

Get a free Groq key at [console.groq.com](https://console.groq.com).

**HuggingFace Spaces:** Add `GROQ_API_KEY` in Space → Settings → Repository secrets.

---

## Accuracy

| Version | Accuracy |
|---|---|
| v1 — hard exact matching | 22–30% |
| v3 — MeCab keigo, bilingual ground truth | ~60% |
| v4 — hallucination guard, APPI masking | 75–85% |
| **v5 (live)** | **93.8%** |

---

## Architecture

```
Input → Vector cache → PII mask → LLM → PII restore
     → Speaker norm → MeCab keigo → Hallucination guard
     → Soft rejection detection → Export
```

Critical: PII masked **before** LLM. Restored **before** speaker normalization.

---

<div align="center">

Built by **[Kunal Bisht](https://linkedin.com/in/kunalhere)** · AI Engineer · Uttarakhand, India

[![LinkedIn](https://img.shields.io/badge/LinkedIn-kunalhere-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/kunalhere)
[![Email](https://img.shields.io/badge/Email-kunalbisht909@gmail.com-D96080?style=flat-square)](mailto:kunalbisht909@gmail.com)

</div>