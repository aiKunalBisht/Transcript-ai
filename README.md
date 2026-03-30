# 🎙️ TranscriptAI — Japanese Business Intelligence

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Ollama](https://img.shields.io/badge/Ollama-qwen3%3A8b-black?style=flat-square&logoColor=white)](https://ollama.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Titankunal/Transcript-ai?style=flat-square&color=facc15)](https://github.com/Titankunal/Transcript-ai/stargazers)

**AI-powered meeting transcript analyzer optimized for Japanese business culture.**
Extract action items, sentiment, speaker breakdowns, and Japan-specific insights — 100% free, runs locally.

## 🚀 [Live Demo → transcript-ai-qkuqcld42yym54zxmyhhby.streamlit.app](https://transcript-ai-qkuqcld42yym54zxmyhhby.streamlit.app/)

</div>

---

## 📸 Screenshots

### Main Interface — Transcript Input
![Main Interface](screenshots/main.png)
*Dark purple UI with file upload, paste input, auto language detection (Mixed JA/EN detected), and analysis history sidebar*

### Analysis Results — Meeting Summary
![Summary Tab](screenshots/summary.png)
*3-bullet TL;DR summary with tabs for Action Items, Sentiment, Speakers, and Japan Insights*

### Japan Business Intelligence Tab
![Japan Insights](screenshots/japan.png)
*Keigo register detection, 4 nemawashi signals in Japanese, and 10 code-switches detected mid-conversation*

---

## 🤔 Why This Project?

> *"The most valuable AI projects aren't the flashy ones — they're the boring ones that solve real problems businesses have struggled with for decades."*

Every day, thousands of business meetings happen across Japanese companies. Sales calls, client check-ins, internal standups — all generating hours of conversation that get **manually summarized, poorly documented, or simply forgotten.**

This project solves that. But here's what makes it different from a generic meeting summarizer:

### 🇯🇵 The Japanese Business Culture Problem

Japanese business communication is uniquely challenging for AI:

| Challenge | What it means | How TranscriptAI handles it |
|---|---|---|
| **Keigo (敬語)** | Formal honorific speech changes meaning entirely | Detects register level: high / medium / low |
| **Nemawashi (根回し)** | Indirect consensus-building disguised as agreement | Extracts phrases like 同意します, 了解しました, 素晴らしい |
| **Code-switching** | JA↔EN mixing mid-sentence in modern offices | Counts switches, handles bilingual context |
| **Indirect refusal** | "We will consider it" often means "No" | Flags ambiguous sentiment per speaker |

### 📈 Why It Scales Into Any Industry

| Industry | Use case |
|---|---|
| 💼 **Sales / CRM** | Auto-log call summaries after every client call |
| 🏥 **Healthcare** | Summarize doctor-patient consultations |
| ⚖️ **Legal** | Extract action items from depositions |
| 🎓 **HR** | Analyze interview transcripts for candidate sentiment |
| 💰 **Finance** | Flag risk signals in earnings call transcripts |

One JSON schema. Infinite industries.

---

## ✨ Features

- 📄 **Multi-format input** — Upload `.txt`, `.vtt`, `.json` or paste directly
- 🌐 **Auto language detection** — Japanese, English, or mixed JA/EN
- ✅ **Action item extraction** — Owner + deadline pulled from conversation
- 😊 **Per-speaker sentiment** — Positive / Neutral / Negative with reasoning
- 📋 **Meeting summary** — 3-bullet TL;DR
- 👤 **Speaker breakdown** — Talk time % and tone per participant
- 🇯🇵 **Japan insights** — Keigo level, nemawashi signals, code-switch count
- 💾 **Export as JSON** — Feed results into any downstream system
- 🕐 **Analysis history** — Last 5 analyses saved in sidebar
- 🔒 **100% local** — Your data never leaves your machine

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    STREAMLIT UI                         │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ File Upload │  │  Paste Text  │  │ Language Sel. │  │
│  └──────┬──────┘  └──────┬───────┘  └───────┬───────┘  │
└─────────┼────────────────┼──────────────────┼──────────┘
          └────────────────┼──────────────────┘
                           ▼
              ┌────────────────────────┐
              │   Language Detection   │
              │   (langdetect)         │
              │   JA / EN / Mixed      │
              └───────────┬────────────┘
                          ▼
              ┌────────────────────────┐
              │   LLM Core             │
              │   Ollama + qwen3:8b    │
              │   Structured JSON out  │
              └───────────┬────────────┘
                          ▼
        ┌─────────────────────────────────────┐
        │           OUTPUT PIPELINE           │
        │                                     │
        │  ✅ Action Items  😊 Sentiment      │
        │  📋 Summary       👤 Speakers       │
        │                                     │
        └─────────────────┬───────────────────┘
                          ▼
        ┌─────────────────────────────────────┐
        │     🇯🇵 JAPAN INTELLIGENCE LAYER    │
        │                                     │
        │  敬語 Keigo Detector                │
        │  根回し Nemawashi Signal Extractor  │
        │  JA↔EN Code-Switch Handler          │
        │                                     │
        └─────────────────────────────────────┘
```

---

## 🚀 Run Locally

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed

```bash
git clone https://github.com/Titankunal/Transcript-ai.git
cd Transcript-ai
pip install -r requirements.txt
ollama pull qwen3:8b
python -m streamlit run app.py
```

Open **http://localhost:8501** 🎉

---

## 🔄 Swap the AI Provider

`analyzer.py` is designed for one-line provider swaps. The JSON schema stays identical.

**→ Claude (Anthropic)**
```python
import anthropic
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
message = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1024,
    messages=[{"role": "user", "content": build_prompt(text, language)}]
)
return json.loads(message.content[0].text)
```

**→ Gemini (Google — free tier)**
```python
import google.generativeai as genai
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content(build_prompt(text, language))
return json.loads(response.text)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| AI Model | Ollama + qwen3:8b |
| Language Detection | langdetect |
| Deployment | Streamlit Community Cloud (free) |

---

## 🤝 Contributing

Contributions welcome — especially Japanese business culture patterns, new language support, and CRM integrations.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ by [Kunal Bisht](https://github.com/Titankunal)

*"Boring projects. Infinite scale."*

⭐ **If this helped you, star the repo!** ⭐

</div>
