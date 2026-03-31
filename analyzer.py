# analyzer.py
# Core AI analysis pipeline for TranscriptAI
# Calls Ollama locally, falls back to mock data on cloud deployment.

import json
import os
import re
import requests

# ── CONFIG ───────────────────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "qwen3:8b"   # swap to "qwen3:30b" for higher quality (needs good GPU)
# ─────────────────────────────────────────────────────────────────────────────


def _summary_instruction(text: str) -> str:
    """
    Dynamically decide how many summary bullets to request
    based on transcript length. Short = 3, medium = 5, long = 7+.
    """
    words = len(text.split())
    if words < 200:
        return "summary: write 3 concise bullet points covering the key topics"
    elif words < 600:
        return "summary: write 5 concise bullet points covering ALL key topics discussed"
    elif words < 1200:
        return "summary: write 7 concise bullet points — cover every major topic, decision, and outcome"
    else:
        return (
            "summary: write as many bullet points as needed (minimum 8) to fully cover "
            "every major topic, decision, concern, and outcome discussed. "
            "Do NOT compress multiple topics into one bullet."
        )


def build_prompt(text: str, language: str) -> str:
    lang_hint = (
        "The transcript may contain Japanese (日本語) and English mixed together. "
        "Detect and handle both languages. Extract Japanese phrases as-is for nemawashi signals."
        if language in ("ja", "mixed")
        else "The transcript is in English."
    )

    summary_rule = _summary_instruction(text)

    return f"""You are an expert meeting analyst specializing in Japanese business culture.
{lang_hint}

Analyze the following meeting transcript and return ONLY a valid JSON object.
No explanation, no markdown, no backticks — just the raw JSON.

The JSON must follow this exact schema:
{{
  "summary": ["bullet point 1", "bullet point 2", "...as many as needed"],
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
    "nemawashi_signals": ["actual phrase from transcript"],
    "code_switch_count": 0
  }}
}}

Rules:
- {summary_rule}
- action_items: list EVERY concrete task, request, or commitment mentioned — even implied ones
- sentiment: one entry per unique speaker found in transcript
- speakers: talk_time_pct values must sum to exactly 100
- japan_insights.nemawashi_signals: extract ACTUAL phrases from the transcript showing
  indirect agreement, hesitation, or soft refusal (e.g. そうですね, 検討します, 難しいかもしれません,
  前向きに検討, なるほど, 承知しました). Empty list [] if none found.
- japan_insights.code_switch_count: count how many times language switches between JA and EN
- Return ONLY the JSON object. No other text whatsoever.

TRANSCRIPT:
{text}
"""


def _mock_response(text: str) -> dict:
    """
    Returns realistic demo data when Ollama is unavailable (e.g. Streamlit Cloud).
    Number of summary bullets scales with transcript length.
    """
    words = len(text.split())

    if words < 200:
        summary = [
            "The team discussed Q3 project progress and confirmed key deadlines.",
            "Action items were assigned with clear ownership and timelines.",
            "Both speakers demonstrated collaborative tone with mixed JA/EN communication."
        ]
    elif words < 600:
        summary = [
            "The meeting opened with a Q3 progress review showing strong KPI performance at 98% of target.",
            "Concerns were raised about the new feature release schedule and potential delays.",
            "Budget adjustments were discussed and require sign-off from the technical team.",
            "Action items were clearly assigned with deadlines ranging from today to next Friday.",
            "Next meeting scheduled for Friday at 15:00 — minutes to be prepared by Tanaka."
        ]
    else:
        summary = [
            "The meeting opened with a Q3 progress review showing strong KPI performance at 98% of target.",
            "Concerns were raised about the new feature release schedule and a possible buffer needed.",
            "鈴木 expressed indirect hesitation about the April 1st deadline using nemawashi language.",
            "Budget adjustments were discussed — final confirmation required from the technical team by Monday.",
            "Customer feedback volume has increased, prompting a proposal to expand the support team.",
            "Support manual revision was identified as a parallel workstream to the team expansion.",
            "All action items were formally assigned with clear owners and deadlines.",
            "Next sync confirmed for Friday 15:00 — Tanaka responsible for meeting minutes."
        ]

    return {
        "summary": summary,
        "action_items": [
            {"task": "Confirm release schedule buffer", "owner": "Suzuki", "deadline": "Monday"},
            {"task": "Review financial section of Q3 report", "owner": "Tanaka", "deadline": "Thursday"},
            {"task": "Send delay notification email to client", "owner": "Sato", "deadline": "Today"},
            {"task": "Draft support manual revision", "owner": "Suzuki", "deadline": "Friday"},
            {"task": "Prepare meeting minutes", "owner": "Tanaka", "deadline": "Friday 15:00"}
        ],
        "sentiment": [
            {"speaker": "Tanaka", "score": "positive", "label": "Professional and collaborative tone"},
            {"speaker": "Suzuki", "score": "neutral", "label": "Cautiously cooperative, indirect hesitation"}
        ],
        "speakers": [
            {"name": "Tanaka", "talk_time_pct": 55, "tone": "formal"},
            {"name": "Suzuki", "talk_time_pct": 45, "tone": "formal"}
        ],
        "japan_insights": {
            "keigo_level": "high",
            "nemawashi_signals": [
                "そうですね", "検討いたします", "難しいかもしれません",
                "前向きに対応したいと思います", "承知しました"
            ],
            "code_switch_count": 4
        }
    }


def analyze_transcript(text: str, language: str = "en") -> dict:
    """
    Analyzes a transcript using local Ollama.
    Automatically falls back to mock data if Ollama is unreachable.

    To swap LLM provider (one function replacement):
      - Gemini:  google.generativeai call
      - Claude:  anthropic.Anthropic().messages.create(...)
      - OpenAI:  openai.chat.completions.create(...)
    JSON schema stays identical — nothing else in the app changes.
    """
    prompt = build_prompt(text, language)

    # Increase token budget for longer transcripts
    words = len(text.split())
    max_tokens = min(2048, max(1024, words * 3))

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": max_tokens,
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
        result = _validate_and_fill(result)

        # Fix 1: override LLM code-switch count with deterministic rule-based count
        try:
            from evaluator import count_code_switches
            result["japan_insights"]["code_switch_count"] = count_code_switches(text)
            result["japan_insights"]["code_switch_source"] = "rule_based"
        except ImportError:
            pass  # evaluator.py not present — use LLM count

        return result

    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return _mock_response(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model returned invalid JSON: {e}\n\nRaw:\n{raw[:500]}")


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

    # Normalize talk_time_pct to sum to 100
    speakers = data["speakers"]
    if speakers:
        total = sum(s.get("talk_time_pct", 0) for s in speakers)
        if total > 0 and total != 100:
            for s in speakers:
                s["talk_time_pct"] = round(s.get("talk_time_pct", 0) * 100 / total)

    return data


# ── QUICK TEST ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
    田中: おはようございます、田中です。本日はお時間をいただきありがとうございます。
    鈴木: こちらこそ、よろしくお願いいたします。鈴木です。
    田中: まず、Q4の進捗についてご報告させていただきます。売上KPIは現時点で目標の98%に達しており、ほぼ計画通りです。
    鈴木: そうですね、順調に進んでいるようで安心しました。ただ、新機能のリリーススケジュールについては、少し懸念がございます。
    田中: Yes, I understand your concern. The release is scheduled for April 1st, but we may need a buffer.
    鈴木: 検討いたします。技術チームとも相談してみますが、難しいかもしれません。できれば前向きに対応したいと思います。
    田中: Understood. では、リリース日を鈴木さんの方でサインオフをいただければ、我々は準備を進めます。
    鈴木: 承知しました。来週の月曜日までに確認いたします。
    田中: ありがとうございます。次に、顧客からのフィードバック対応についてですが、サポートチームの増員が必要だと考えています。
    鈴木: そうですね、確認してみます。サポートマニュアルの改訂も同時に進めた方が良いかもしれません。
    田中: 同感です。鈴木さん、マニュアルのドラフト作成をお願いできますか？来週の金曜日までにレビュー用に提出していただければ。
    鈴木: かしこまりました。対応いたします。
    田中: では、次回のミーティングは来週金曜日の15:00に設定しましょう。議事録は田中が担当します。
    鈴木: 承知いたしました。本日はありがとうございました。
    """
    result = analyze_transcript(sample, language="mixed")
    print(json.dumps(result, indent=2, ensure_ascii=False))