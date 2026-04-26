# analyzer.py — v7
# LangChain orchestration layer added.
#
# Provider hierarchy via LangChain:
#   Groq (ChatGroq)   → ~3s,  free tier, cloud
#   Ollama (ChatOllama) → ~60s, fully local, APPI data residency
#   Mock              → instant, demo mode
#
# LangChain chosen for:
#   - Provider-agnostic interface (swap model in one line)
#   - Built-in retry + timeout handling
#   - StrOutputParser for clean text extraction
#   Falls back to direct requests.post() if langchain not installed

import json
import os
import re
import time
import requests

# LangChain — primary orchestration layer
try:
    from langchain_groq import ChatGroq
    try:
        from langchain_ollama import OllamaLLM as LangChainOllama
    except ImportError:
        from langchain_community.llms import Ollama as LangChainOllama
    from langchain_core.messages import HumanMessage
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# ── CONFIG ────────────────────────────────────────────────────────────────────
PROVIDER     = os.getenv("TRANSCRIPT_AI_PROVIDER", "auto")
# "auto"   → try Groq first, fall back to Ollama, then mock
# "groq"   → Groq only
# "ollama" → Ollama only
# "mock"   → always mock (testing)

# C1 FIX: All URLs configurable via env vars — not hardcoded
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
# Auto-detect available Ollama model if qwen3:8b not found
def _get_ollama_model() -> str:
    configured = os.getenv("OLLAMA_MODEL", "qwen3:8b")
    if configured != "qwen3:8b":
        return configured  # user explicitly set a model
    try:
        import requests as _req
        r = _req.get(OLLAMA_URL.replace("/api/generate", "/api/tags"), timeout=2)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            if models and not any("qwen3:8b" in m for m in models):
                # Use first available model
                return models[0]
    except Exception:
        pass
    return configured

OLLAMA_MODEL = _get_ollama_model()
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
MAX_RETRIES  = int(os.getenv("TRANSCRIPT_AI_MAX_RETRIES", "2"))
# ─────────────────────────────────────────────────────────────────────────────


def _summary_instruction(text: str) -> str:
    words = len(text.split())
    if words < 200:   return "summary: 3 concise bullet points"
    elif words < 600: return "summary: 5 bullet points covering ALL key topics"
    elif words < 1200: return "summary: 7 bullet points covering every topic and decision"
    else:              return "summary: as many bullets as needed (min 8) — never compress"


def _extract_speaker_hint(text: str) -> str:
    """Pre-extract speakers to hint LLM — prevents missing speakers."""
    import re
    pattern = re.compile(r"^\s*([A-Za-z\u3040-\u9FFF][^\n:：\[\]]{0,30}?)\s*[:：]", re.MULTILINE)
    found = pattern.findall(text)
    # Clean and deduplicate
    seen, clean = set(), []
    for name in found:
        n = re.sub(r"\s*\([^)]*\)", "", name).strip()
        if n and n.lower() not in seen and not re.match(r"^[0-9]+$", n):
            seen.add(n.lower())
            clean.append(n)
    return ", ".join(clean[:10]) if clean else "Not detected"


def build_prompt(text: str, language: str) -> str:
    lang_hint = (
        "Transcript contains Japanese and English. Extract Japanese phrases as-is."
        if language in ("ja", "mixed") else "Transcript is in English."
    )
    speakers_hint = _extract_speaker_hint(text)
    return f"""You are an expert meeting analyst for Japanese business culture.
{lang_hint}

Return ONLY valid JSON. No markdown, no backticks, no explanation.

{{
  "summary": ["bullet 1", "..."],
  "action_items": [{{"task": "...", "owner": "FIRST NAME ONLY — no role titles", "deadline": "..."}}],
  "sentiment": [{{"speaker": "FIRST NAME ONLY", "score": "positive|neutral|negative", "label": "..."}}],
  "speakers": [{{"name": "FIRST NAME ONLY", "talk_time_pct": 50, "tone": "formal|casual|mixed"}}],
  "japan_insights": {{
    "keigo_level": "high|medium|low",
    "nemawashi_signals": ["actual phrase"],
    "code_switch_count": 0
  }}
}}

Rules:
- {_summary_instruction(text)}
- owner/speaker/name: FIRST NAME ONLY. No roles. No (Director). No (PM).
- List ALL speakers found in transcript — do not skip any
- action_items: every explicit task or commitment
- talk_time_pct must sum to 100
- Return ONLY JSON.

SPEAKERS FOUND IN TRANSCRIPT (include ALL of these):
{speakers_hint}

TRANSCRIPT:
{text}
"""


def _get_groq_key() -> str:
    # Q3 FIX: Standardized to GROQ_API_KEY only
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
    return key


def _call_groq(prompt: str, max_tokens: int) -> str:
    api_key = _get_groq_key()
    if not api_key:
        raise ValueError("NO_GROQ_KEY")
    r = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},  # structured output guarantee
        },
        timeout=30
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def stream_transcript_groq(text: str, language: str = "en"):
    """
    Streams analysis token-by-token via Groq API.
    Use with Streamlit: st.write_stream(stream_transcript_groq(text, lang))

    Yields text chunks as they arrive — user sees summary building in real time.
    Note: streaming returns raw text not JSON, so we stream the summary only.
    """
    api_key = _get_groq_key()
    if not api_key:
        yield "⚠️ No Groq API key. Add GROQ_API_KEY for streaming."
        return

    # Streaming-friendly prompt — ask for readable text not JSON
    stream_prompt = f"""You are an expert meeting analyst for Japanese business culture.
Analyze this transcript and write a clear meeting summary with key points, action items, and any notable Japanese business communication patterns observed.
Be concise and professional.

TRANSCRIPT:
{text[:3000]}"""  # cap for streaming

    try:
        import requests
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model":    GROQ_MODEL,
                "messages": [{"role": "user", "content": stream_prompt}],
                "temperature": 0.3,
                "max_tokens":  1000,
                "stream":   True,
            },
            stream=True,
            timeout=60
        )
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: ") and line != "data: [DONE]":
                    import json as _json
                    try:
                        chunk = _json.loads(line[6:])
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        continue
    except Exception as e:
        yield f"Stream error: {str(e)[:80]}"


def _call_ollama(prompt: str, max_tokens: int) -> str:
    r = requests.post(
        OLLAMA_URL,
        json={
            "model":   OLLAMA_MODEL,
            "prompt":  prompt,
            "stream":  False,
            "format":  "json",   # structured output — Ollama enforces valid JSON
            "options": {"temperature": 0.2, "num_predict": max_tokens},
            "think":   False
        },
        timeout=90   # 90s max — if slower, use Groq
    )
    r.raise_for_status()
    return r.json().get("response", "")


def _call_groq_langchain(prompt: str, max_tokens: int) -> str:
    """
    Primary LLM call via LangChain ChatGroq.
    LangChain provides provider-agnostic interface,
    built-in retry logic, and clean output parsing.
    """
    api_key = _get_groq_key()
    if not api_key:
        raise ValueError("NO_GROQ_KEY")

    llm = ChatGroq(
        api_key=api_key,
        model=GROQ_MODEL,
        temperature=0.2,
        max_tokens=max_tokens,
    )
    parser  = StrOutputParser()
    chain   = llm | parser
    result  = chain.invoke([HumanMessage(content=prompt)])
    return result


def _call_ollama_langchain(prompt: str, max_tokens: int) -> str:
    """
    Ollama call via LangChain for fully local inference.
    Zero data leaves the machine — APPI data residency guarantee.
    """
    llm = LangChainOllama(
        base_url=OLLAMA_URL.replace("/api/generate", ""),
        model=OLLAMA_MODEL,
        temperature=0.2,
        num_predict=max_tokens,
        format="json",
    )
    return llm.invoke(prompt)


def _parse(raw: str) -> dict:
    """
    C4 FIX: Robust JSON parsing — handles nested braces, multiple objects,
    provider-specific wrappers. Uses JSONDecoder.raw_decode() for correctness.
    """
    # Strip thinking blocks and markdown fences
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()

    # Find first valid JSON object using raw_decode (stops at first complete object)
    decoder = json.JSONDecoder()
    for i, char in enumerate(raw):
        if char == "{":
            try:
                obj, _ = decoder.raw_decode(raw, i)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue

    raise ValueError(f"No valid JSON object found in response (first 200 chars): {raw[:200]}")


def _try_providers(prompt: str, max_tokens: int) -> tuple[str, str]:
    """
    Fix: Fallback hierarchy — Groq → Ollama → raises
    Returns (raw_response, provider_name)
    """
    providers_to_try = []

    if PROVIDER == "auto":
        if _get_groq_key():
            providers_to_try = [("groq", _call_groq), ("ollama", _call_ollama)]
        else:
            providers_to_try = [("ollama", _call_ollama)]
    elif PROVIDER == "groq":
        providers_to_try = [("groq", _call_groq)]
    elif PROVIDER == "ollama":
        providers_to_try = [("ollama", _call_ollama)]

    last_error = None
    for name, caller in providers_to_try:
        retries = MAX_RETRIES if name == "groq" else 0

        for attempt in range(retries + 1):
            try:
                # Try LangChain first if available, fall back to raw requests
                if LANGCHAIN_AVAILABLE and name == "groq":
                    try:
                        raw = _call_groq_langchain(prompt, max_tokens)
                        return raw, f"{name}_langchain"
                    except Exception:
                        raw = caller(prompt, max_tokens)  # fallback to raw
                elif LANGCHAIN_AVAILABLE and name == "ollama":
                    try:
                        raw = _call_ollama_langchain(prompt, max_tokens)
                        return raw, f"{name}_langchain"
                    except Exception:
                        raw = caller(prompt, max_tokens)
                else:
                    raw = caller(prompt, max_tokens)
                return raw, name   # SUCCESS
            except ValueError as e:
                if "NO_GROQ_KEY" in str(e):
                    break  # no key — skip Groq, try next
                last_error = e
                if attempt < retries:
                    time.sleep(1)
            except requests.exceptions.Timeout:
                last_error = TimeoutError(f"{name} timed out")
                break   # 2.1 FIX: timeout = don't retry, move to next provider
            except requests.exceptions.ConnectionError:
                last_error = ConnectionError(f"{name} offline")
                break
            except Exception as e:
                last_error = e
                if attempt < retries:
                    time.sleep(1)

    raise last_error or RuntimeError("All providers failed")


def _mock_response(text: str, reason: str = "") -> dict:
    """
    U2 FIX: Mock response is now clearly labeled as DEMO DATA.
    Does NOT use fake hardcoded names (Priya, Sato, Tanaka).
    Extracts actual speaker names from transcript for honest placeholders.
    User is clearly informed this is not a real analysis.
    """
    # Extract real speaker names from transcript for honest placeholders
    speaker_names = []
    colon_pat = re.compile(r"(?:^|\n)\s*([A-Za-z\u3040-\u9FFF][^\n:]{0,20}?)\s*[:：]", re.MULTILINE)
    for m in colon_pat.finditer(text):
        raw = m.group(1).strip()
        clean = re.sub(r"\s*\([^)]*\)", "", raw).strip()
        if clean and len(clean) >= 2 and not re.match(r"^\d+$", clean) and clean not in speaker_names:
            speaker_names.append(clean)

    if not speaker_names:
        speaker_names = ["Speaker A", "Speaker B"]

    n = len(speaker_names)
    pct = round(100 / n)
    speakers = [{"name": name, "talk_time_pct": pct, "tone": "formal"} for name in speaker_names]
    sentiment = [{"speaker": name, "score": "neutral", "label": "Demo mode — not analyzed"} for name in speaker_names]

    words = len(text.split())
    summary_count = 3 if words < 200 else 5 if words < 600 else 7

    return {
        "summary": [f"⚠️ DEMO MODE: Real analysis unavailable ({reason or 'AI offline'})."] +
                   [f"Transcript has {words} words and {n} detected speakers."] +
                   ["Connect to Groq (free) or Ollama to see real analysis."] * (summary_count - 2),
        "action_items": [
            {"task": "⚠️ Demo mode — connect AI provider for real action items",
             "owner": speaker_names[0] if speaker_names else "Unknown",
             "deadline": "N/A"}
        ],
        "sentiment": sentiment,
        "speakers":  speakers,
        "japan_insights": {
            "keigo_level": "unknown",
            "nemawashi_signals": [],
            "code_switch_count": 0
        },
        "_mock_reason":   reason,
        "_demo_mode":     True,
        "_demo_warning":  "This is demo data. Real transcript not analyzed. Add GROQ_API_KEY for real analysis."
    }


def analyze_transcript(text: str, language: str = "en") -> dict:
    """
    Full analysis pipeline v6:
    1. Cache check (instant return for repeated transcripts)
    2. Provider fallback: Groq → Ollama → Mock
    3. Schema enforcement + parsing
    4. Speaker normalization (Fix 1)
    5. MeCab keigo (Fix: wired in)
    6. Rule-based code-switch
    7. Semantic validation (Fix: TF-IDF rescues false flags)
    8. Hallucination guard (rule-based)
    9. Soft rejection detection
    10. Logging
    """
    start_time = time.time()

    # Step 1: Cache check
    try:
        from utils.cache import get_cached, set_cache
        cached = get_cached(text, language)
        if cached:
            cached["_from_cache"] = True
            return cached
    except ImportError:
        get_cached = set_cache = None

    prompt     = build_prompt(text, language)
    # Q1 FIX: Dynamic token budget based on actual transcript length
    # 1200 was insufficient for long meetings (60min = ~8000 words needs ~3000 tokens)
    words = len(text.split())
    if words < 300:
        max_tokens = 700
    elif words < 800:
        max_tokens = 1200
    elif words < 2000:
        max_tokens = 1800
    else:
        max_tokens = 2000   # cap — Ollama on CPU can't handle more in <90s
    # Groq handles up to 4000 tokens fine — limit only matters for Ollama
    provider_used = "unknown"
    last_error    = None

    try:
        raw, provider_used = _try_providers(prompt, max_tokens)
        result = _parse(raw)
        result = _validate_and_fill(result)

    except Exception as e:
        last_error    = str(e)[:80]
        provider_used = "mock"
        result        = _mock_response(text, reason=last_error)
        result        = _validate_and_fill(result)

    # Speaker normalization (runs on original unmasked text passed to analyzer)
    # C2 NOTE: When PII masking is enabled in app.py, this receives masked text.
    # The Kanji↔Romaji matching in speaker_normalizer is therefore only active
    # when PII masking is OFF. With masking ON, normalization still deduplicates
    # by placeholder ([NAME_1] appearing in both sentiment and speakers lists).
    # Full cross-script resolution happens AFTER PII restore in app.py.
    try:
        from transcription.speaker_normalizer import unify_speakers_in_result
        result = unify_speakers_in_result(result, text)
    except ImportError:
        pass

    # Fix: Wire MeCab keigo (overrides LLM classification)
    try:
        from analysis.japanese_tokenizer import get_keigo_level, MECAB_AVAILABLE
        if MECAB_AVAILABLE:
            result["japan_insights"]["keigo_level"] = get_keigo_level(text)
            result["japan_insights"]["keigo_source"] = "mecab"
    except ImportError:
        pass

    # Rule-based code-switch
    try:
        from utils.evaluator import count_code_switches
        result["japan_insights"]["code_switch_count"] = count_code_switches(text)
        result["japan_insights"]["code_switch_source"] = "rule_based"
    except ImportError:
        pass

    # Fix 2: Hallucination guard + semantic rescue
    try:
        from analysis.hallucination_guard import verify_result
        result = verify_result(result, text)
        # Semantic validation rescues false flags
        from analysis.semantic_validator import validate_action_items_semantic
        result["action_items"] = validate_action_items_semantic(
            result.get("action_items", []), text
        )
    except ImportError:
        pass

    # Soft rejection detection
    try:
        from analysis.soft_rejection_detector import detect_soft_rejections
        result["soft_rejections"] = detect_soft_rejections(text)
    except ImportError:
        pass

    duration_ms = (time.time() - start_time) * 1000
    result["_provider"]    = provider_used
    result["_duration_ms"] = round(duration_ms, 1)
    if last_error:
        result["_last_error"] = last_error

    # Step 10: Log
    try:
        from utils.logger import log_analysis
        log_analysis(len(text), language, provider_used, duration_ms, result, last_error)
    except ImportError:
        pass

    # Cache successful results
    if set_cache and "mock" not in provider_used:
        try:
            set_cache(text, language, result)
        except Exception:
            pass

    return result


def _validate_and_fill(data: dict) -> dict:
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


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        os.environ["TRANSCRIPT_AI_PROVIDER"] = sys.argv[1]
    print(f"Provider config: {PROVIDER}")

    sample = """
    Kunal (Lead Engineer): 皆さん、お疲れ様です。We are moving to Groq API.
    Tanaka (Director): セキュリティはどうですか？APPI complianceが重要です。
    Sato (PM): 承知いたしました。PII maskingで対応済みです。検討いたします。
    Priya (Backend Dev): Redis async processor also added for scaling.
    """
    result = analyze_transcript(sample, language="mixed")
    print(f"Provider: {result.get('_provider')} | {result.get('_duration_ms')}ms")
    print(f"Speakers: {[s['name'] for s in result.get('speakers', [])]}")
    print(f"Keigo: {result['japan_insights']['keigo_level']} ({result['japan_insights'].get('keigo_source','llm')})")
    print(f"Cache: {result.get('_from_cache', False)}")
    print(json.dumps(result, indent=2, ensure_ascii=False))