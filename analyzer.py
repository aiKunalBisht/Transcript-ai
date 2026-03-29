import json
import os
import re
import requests

# ── CONFIG ──────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"        # change to "qwen3:30b" for higher quality (needs good GPU)
IS_CLOUD = not os.path.exists("/usr/bin/ollama") and os.environ.get("HOME", "").startswith("/home")
# ────────────────────────────────────────────────────────────────────────────


def build_prompt(text: str, language: str) -> str:
    lang_hint = (
        "The transcript may contain Japanese (日本語) and English mixed together."
        if language in ("ja", "mixed")
        else "The transcript is in English."
    )

    return f"""You are an expert meeting analyst specializing in Japanese business culture.
{lang_hint}

Analyze the following meeting transcript and return ONLY a valid JSON object — no explanation, no markdown, no backticks.

The JSON must follow this exact schema:
{{
  "summary": ["bullet point 1", "bullet point 2", "bullet point 3"],
  "action_items": [
    {{"task": "description", "owner": "person name", "deadline": "date or 'Not specified'"}}
  ],
  "sentiment": [
    {{"speaker": "name", "score": "positive|neutral|negative", "label": "brief reason"}}
  ],
  "speakers": [
    {{"name": "name", "talk_time_pct": 50, "tone": "formal|casual|mixed"}}
  ],
  "japan_insights": {{
    "keigo_level": "high|medium|low",
    "nemawashi_signals": ["signal 1", "signal 2"],
    "code_switch_count": 0
  }}
}}

Rules:
- summary: exactly 3 concise bullet points
- action_items: list every concrete task mentioned, even implied ones
- sentiment: one entry per unique speaker
- speakers: talk_time_pct values must sum to 100
- japan_insights.nemawashi_signals: list actual phrases from the transcript that show indirect agreement or hesitation (e.g. そうですね, 検討します, なるほど). Empty list if none found.
- japan_insights.code_switch_count: count how many times the language switches between Japanese and English
- Return ONLY the JSON. No other text.

TRANSCRIPT:
{text}
"""


def _mock_response(text: str) -> dict:
    """
    Returns realistic demo data when Ollama is unavailable (e.g. Streamlit Cloud).
    Swap analyze_transcript() body with a real LLM call to go live.
    """
    return {
        "summary": [
            "The team discussed Q3 project progress and confirmed key deadlines.",
            "Action items were assigned with clear ownership and timelines.",
            "Both speakers demonstrated collaborative tone with mixed JA/EN communication."
        ],
        "action_items": [
            {"task": "Review financial section of Q3 report", "owner": "Tanaka", "deadline": "Thursday"},
            {"task": "Send delay notification email to client", "owner": "Sato", "deadline": "Today"},
            {"task": "Schedule follow-up sync meeting", "owner": "Tanaka", "deadline": "Tomorrow morning"}
        ],
        "sentiment": [
            {"speaker": "Tanaka", "score": "positive", "label": "Professional and collaborative tone"},
            {"speaker": "Sato", "score": "positive", "label": "Cooperative and solution-oriented"}
        ],
        "speakers": [
            {"name": "Tanaka", "talk_time_pct": 50, "tone": "formal"},
            {"name": "Sato", "talk_time_pct": 50, "tone": "mixed"}
        ],
        "japan_insights": {
            "keigo_level": "medium",
            "nemawashi_signals": ["そうですね", "検討します", "ありがとうございます"],
            "code_switch_count": 6
        }
    }


def analyze_transcript(text: str, language: str = "en") -> dict:
    """
    Calls local Ollama to analyze a transcript.
    Falls back to mock data if Ollama is unreachable (e.g. on Streamlit Cloud).

    To swap provider:
      - Gemini:  replace this function body with google.generativeai call
      - Claude:  replace with anthropic.Anthropic().messages.create(...)
      - OpenAI:  replace with openai.chat.completions.create(...)
    The return schema stays identical — nothing else in the app changes.
    """
    prompt = build_prompt(text, language)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 1024,
                },
                "think": False
            },
            timeout=300
        )
        response.raise_for_status()
        raw = response.json().get("response", "")

        # Strip qwen3 <think>...</think> blocks and markdown fences
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
        raw = re.sub(r"```(?:json)?", "", raw).strip()

        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in model response")

        result = json.loads(match.group())
        return _validate_and_fill(result)

    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        # Ollama not available — return mock demo data for cloud deployment
        return _mock_response(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned invalid JSON: {e}\n\nRaw output:\n{raw[:500]}")


def _validate_and_fill(data: dict) -> dict:
    """Ensure all required keys exist with sensible defaults."""
    data.setdefault("summary", ["No summary available."])
    data.setdefault("action_items", [])
    data.setdefault("sentiment", [])
    data.setdefault("speakers", [])
    data.setdefault("japan_insights", {})

    ji = data["japan_insights"]
    ji.setdefault("keigo_level", "unknown")
    ji.setdefault("nemawashi_signals", [])
    ji.setdefault("code_switch_count", 0)

    speakers = data["speakers"]
    if speakers:
        total = sum(s.get("talk_time_pct", 0) for s in speakers)
        if total > 0 and total != 100:
            for s in speakers:
                s["talk_time_pct"] = round(s.get("talk_time_pct", 0) * 100 / total)

    return data


# ── QUICK TEST ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    Tanaka: おはようございます。今日のプロジェクト更新について話しましょう。
    Sato: Good morning. Yes, the Q3 report is almost ready.
    Tanaka: そうですね。Deadline is next Friday, right?
    Sato: Correct. I will handle the financial section. Can you review by Thursday?
    Tanaka: 検討します。Also, we need to inform the client about the delay.
    Sato: Understood. I will send them an email today.
    Tanaka: ありがとうございます。Let's sync again tomorrow morning.
    """
    result = analyze_transcript(sample, language="mixed")
    print(json.dumps(result, indent=2, ensure_ascii=False))