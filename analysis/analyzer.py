# analyzer.py — v8.0
# LangChain orchestration layer + true system/user prompt separation +
# anti-hallucination grounding for degenerate / single-message transcripts.
#
# v8.0 — Fine-grained sentiment engine + prompt extraction.
#   E1: System prompt extracted to prompts/analysis_prompt.py.
#       build_prompt() now delegates all text to that module — edit
#       the prompt there, not here.
#   E2: Sentiment upgraded from flat 5-label to 25-label fine-grained
#       taxonomy via analysis/sentiment_engine.py (FineSentimentAnalyzer).
#       Both paths (LLM and no-API) now return per-speaker valence,
#       secondary_labels, tone modifiers, and trend.
#   E3: _normalize_speaker_format() upgraded to 7-pass normalize_format()
#       (CRLF, smart quotes, em-dash, timestamps, multi-space, speaker
#       newline, triple-blank collapse).
#   E4: _validate_and_fill() extended with fine-grained field validation
#       (label ∈ 25-label set, secondary_labels, valence clamp, tone keys).
#
# v7.9.2 — Action item extraction fixes (D5/D6/D7). All retained.
# v7.9.1 — No-API path fixes (D2/D3/D4). All retained.
# v7.9   — B1–B9 fixes. All retained.
# v7.8.1 — A5/A6 speaker hint fixes. All retained.
# v7.8   — A1–A4 PII masking. All retained.
# v7.7   — FIX-21/22/23. All retained.

import datetime
import json
import os
import pathlib
import re
import sys
import time
import requests
from dotenv import load_dotenv
load_dotenv()

# ── Prompt module (E1) ────────────────────────────────────────────────────────
from prompts.analysis_prompt import (
    GROUNDING_RULES        as _GROUNDING_RULES,
    GROUNDING_RULES_SHORT  as _GROUNDING_RULES_SHORT,
    summary_instruction    as _summary_instruction,
    language_hint          as _language_hint,
    japan_schema_str       as _japan_schema_str,
    build_system_prompt,
    build_user_prompt,
    POSITIVE_VALENCE_THRESHOLD,
    NEGATIVE_VALENCE_THRESHOLD,
)

# ── Sentiment engine (E2) ─────────────────────────────────────────────────────
try:
    from analysis.sentiment_engine import (
        FineSentimentAnalyzer,
        build_sentiment_for_json,
        ALL_LABELS as _ALL_FINE_GRAINED,
    )
    _SENTIMENT_ENGINE = FineSentimentAnalyzer()   # stateless singleton
    _SENTIMENT_ENGINE_AVAILABLE = True
except ImportError:
    _SENTIMENT_ENGINE_AVAILABLE = False
    _ALL_FINE_GRAINED = frozenset()
    _SENTIMENT_ENGINE = None  # type: ignore[assignment]


LANGCHAIN_AVAILABLE = None

def _ensure_langchain():
    global LANGCHAIN_AVAILABLE, ChatGroq, LangChainOllama, HumanMessage, SystemMessage, StrOutputParser
    if LANGCHAIN_AVAILABLE is not None:
        return LANGCHAIN_AVAILABLE
    try:
        from langchain_groq import ChatGroq as _ChatGroq
        try:
            from langchain_ollama import OllamaLLM as _Ollama
        except ImportError:
            from langchain_community.llms import Ollama as _Ollama
        from langchain_core.messages import HumanMessage as _HM, SystemMessage as _SM
        from langchain_core.output_parsers import StrOutputParser as _SOP
        ChatGroq        = _ChatGroq
        LangChainOllama = _Ollama
        HumanMessage    = _HM
        SystemMessage   = _SM
        StrOutputParser = _SOP
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
    return LANGCHAIN_AVAILABLE


# ── CONFIG ────────────────────────────────────────────────────────────────────
_env  = os.getenv("ENVIRONMENT", "production").lower()
_auto = "ollama" if _env in ("local", "dev", "development") else "auto"
PROVIDER     = os.getenv("TRANSCRIPT_AI_PROVIDER", _auto)
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")

def _get_ollama_model() -> str:
    configured = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
    if configured != "qwen3:8b":
        return configured
    try:
        import requests as _req
        r = _req.get(OLLAMA_URL.replace("/api/generate", "/api/tags"), timeout=2)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            if models and not any("qwen3:8b" in m for m in models):
                return models[0]
    except Exception:
        pass
    return configured

OLLAMA_MODEL    = _get_ollama_model()
GROQ_URL        = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL      = os.getenv("GROQ_MODEL",      "openai/gpt-oss-120b")
GROQ_MODEL_FAST = os.getenv("GROQ_MODEL_FAST", "openai/gpt-oss-20b")
MAX_RETRIES     = int(os.getenv("TRANSCRIPT_AI_MAX_RETRIES", "1"))
# ─────────────────────────────────────────────────────────────────────────────


# ── Token budget helpers ──────────────────────────────────────────────────────
_MAX_TRANSCRIPT_WORDS = 1_200

def _truncate_transcript(text: str) -> str:
    """Hard cap: keep first 60% + last 40%. DSA: O(n) — split/slice/join."""
    words = text.split()
    if len(words) <= _MAX_TRANSCRIPT_WORDS:
        return text
    keep_start = int(_MAX_TRANSCRIPT_WORDS * 0.60)
    keep_end   = int(_MAX_TRANSCRIPT_WORDS * 0.40)
    return (
        " ".join(words[:keep_start])
        + "\n\n[...middle section omitted — transcript too long...]\n\n"
        + " ".join(words[-keep_end:])
    )


def _select_model(text: str, language: str, has_japanese: bool) -> str:
    """Route short English-only transcripts to faster 8B model."""
    if has_japanese:                return GROQ_MODEL
    if language in ("ja", "mixed"): return GROQ_MODEL
    if _detect_hinglish(text):      return GROQ_MODEL
    if len(text.split()) > 600:     return GROQ_MODEL
    return GROQ_MODEL_FAST


# ── A5 FIX: speaker hint — handles [NAME_1]: masked tokens ───────────────────
def _extract_speaker_hint(text: str) -> str:
    """
    Extract speaker identifiers for the LLM system prompt.
    A5 FIX: alternation matches both [NAME_1]: placeholders and real names.
    DSA: O(n) — single regex scan.
    """
    pattern = re.compile(
        r"^\s*"
        r"(\[[A-Z]+_\d+\]"
        r"|[A-Za-z\u3040-\u9FFF][^\n:：\[\]]{0,30}?)"
        r"\s*[:：]",
        re.MULTILINE,
    )
    found = pattern.findall(text)
    seen, clean = set(), []
    for name in found:
        n = re.sub(r"\s*\([^)]*\)", "", name).strip()
        if n and n.lower() not in seen and not re.match(r"^[0-9]+$", n):
            seen.add(n.lower())
            clean.append(n)
    return ", ".join(clean[:10]) if clean else "Not detected"


_HINGLISH_MARKERS = {
    "hai","hain","nahi","kya","aur","toh","bhi","se","ko","ka","ki","ke",
    "mein","par","pe","hoga","hogi","karenge","karke","bataungi","padega",
    "sab","log","ek","aaj","hum","main","tum","yeh","woh","karo","karna",
}

def _detect_hinglish(text: str) -> bool:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        return False
    return sum(1 for w in words if w in _HINGLISH_MARKERS) >= 3


# ── E3: normalize_format — upgraded 7-pass FIX-22 ────────────────────────────
def normalize_format(text: str) -> str:
    """
    FIX-22 (upgraded): Normalize raw transcript text before PII masking.

    7 sequential O(n) passes:
      1. CRLF → LF
      2. Smart quotes → straight quotes
      3. Em/en dashes in speaker labels → hyphen
      4. Strip stray timestamps at line start ([00:03:45] / (00:03:45))
      5. Collapse multiple spaces/tabs to single space
      6. Ensure speaker turns start on their own line
      7. Collapse 3+ blank lines to 1 blank line

    DSA: O(n · passes) ≈ O(7n) → linear in transcript length.
    """
    t = text
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = t.replace("\u2018", "'").replace("\u2019", "'")
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = re.sub(r"[\u2013\u2014]", "-", t)
    t = re.sub(r"^[\[(]?\d{1,2}:\d{2}(:\d{2})?[\])]?\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"[ \t]{2,}", " ", t)
    t = re.sub(r"([.!?])\s+([A-Z][^:\n]{1,40}:)", r"\1\n\2", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ── A6 FIX: degenerate detection — handles [NAME_1]: tokens ──────────────────
def _is_degenerate_transcript(text: str, speaker_hint: str) -> bool:
    """
    A6 FIX: turn_count regex matches [NAME_1]: after PII masking.
    DSA: O(n) — single regex findall.
    """
    words = text.split()
    if not words:
        return False
    detected_speakers = [
        s.strip() for s in speaker_hint.split(",")
        if s.strip() and s.strip().lower() != "not detected"
    ]
    turn_count = len(re.findall(
        r"(?:^|\n)\s*"
        r"(?:\[[A-Z]+_\d+\]"
        r"|[A-Za-z\u3040-\u9FFF][^\n:：]{0,30}?)"
        r"[:：]",
        text, re.MULTILINE,
    ))
    if turn_count >= 2 or len(detected_speakers) >= 2:
        return False
    if len(words) > 40:
        return False
    return True


# ── FIX-20: Timestamp-gap talk_time helpers ────────────────────────────────────

def _parse_turn_timestamp(ts_str: str) -> datetime.time | None:
    if not ts_str:
        return None
    for fmt in ("%I:%M %p", "%I:%M%p", "%H:%M:%S", "%H:%M"):
        try:
            return datetime.datetime.strptime(ts_str.strip(), fmt).time()
        except ValueError:
            continue
    return None


def _extract_turns_with_timestamps(text: str) -> list[tuple[str, datetime.time | None]]:
    turns: list[tuple[str, datetime.time | None]] = []
    lk_pat = re.compile(
        r"^([A-Za-z][A-Za-z\s\.]{1,35}?)\s+(?:\([^)]+\)\s+)?(\d{1,2}:\d{2}\s*[AP]M)\s*$",
        re.MULTILINE | re.IGNORECASE,
    )
    std_pat = re.compile(
        r"^([A-Za-z\u3040-\u9FFF][^\n:：]{0,30}?)\s*[:：]\s*(?:\[?(\d{1,2}:\d{2}(?::\d{2})?)\]?)?",
        re.MULTILINE,
    )
    lk_matches = list(lk_pat.finditer(text))
    if lk_matches:
        for m in lk_matches:
            name = re.sub(r"\s+", " ", m.group(1)).strip()
            if not name or len(name) < 2 or "view" in name.lower():
                continue
            name = re.sub(r"(?i)\s*sent the following.*", "", name).strip()
            if name:
                turns.append((name, _parse_turn_timestamp(m.group(2))))
    else:
        for m in std_pat.finditer(text):
            name = m.group(1).strip()
            ts   = _parse_turn_timestamp(m.group(2)) if m.group(2) else None
            if name and len(name) >= 2:
                turns.append((name, ts))
    return turns


def _recompute_talk_time_pct(text: str, speakers: list[dict]) -> None:
    """
    FIX-20: Override talk_time_pct with timestamp-gap weights.
    Gap ≤60s → 1.0 | ≤300s → 0.5 | >300s → 0.01
    DSA: O(t) where t = timestamped turns.
    """
    turns       = _extract_turns_with_timestamps(text)
    timestamped = [(n, ts) for n, ts in turns if ts is not None]
    if len(timestamped) < 2:
        return

    weights: dict[str, float] = {}
    first_name, first_ts = timestamped[0]
    weights[first_name]  = weights.get(first_name, 0.0) + 1.0
    prev_sec = first_ts.hour * 3600 + first_ts.minute * 60 + first_ts.second

    for name, ts in timestamped[1:]:
        curr_sec = ts.hour * 3600 + ts.minute * 60 + ts.second
        gap      = curr_sec - prev_sec
        if gap < 0:
            gap += 86400
        w = 1.0 if gap <= 60 else (0.5 if gap <= 300 else 0.01)
        weights[name] = weights.get(name, 0.0) + w
        prev_sec      = curr_sec

    total_weight = sum(weights.values())
    if not total_weight:
        return

    def _match(llm_name: str) -> float:
        target = llm_name.lower().strip()
        for full_name, w in weights.items():
            if target in full_name.lower().split() or target == full_name.lower():
                return w
        return 0.0

    raw       = {s["name"]: _match(s["name"]) for s in speakers}
    raw_total = sum(raw.values())
    if not raw_total:
        return
    for s in speakers:
        s["talk_time_pct"] = round(raw[s["name"]] * 100 / raw_total)
    delta = 100 - sum(s["talk_time_pct"] for s in speakers)
    if delta and speakers:
        speakers[0]["talk_time_pct"] += delta


def build_prompt(text: str, language: str) -> tuple[str, str]:
    """
    Build (system_prompt, user_prompt) from masked transcript.
    text must be the masked transcript — never raw PII.

    E1: All prompt *text* now lives in prompts/analysis_prompt.py.
        Edit that file to change what the LLM is asked to produce.
        Detection logic (Japanese/Hinglish/speakers/degenerate) stays here.
    """
    has_japanese = bool(re.search(r"[\u3040-\u9fff\u4e00-\u9fff]", text))
    has_hinglish = _detect_hinglish(text)

    try:
        from utils.speaker_detector import detect_speakers as _ds
        _sr           = _ds(text)
        speakers_hint = ", ".join(_sr["names"][:12]) if _sr["names"] else "Not detected"
    except ImportError:
        speakers_hint = _extract_speaker_hint(text)

    is_degenerate = _is_degenerate_transcript(text, speakers_hint)
    include_japan = has_japanese or language in ("ja", "mixed")

    system_prompt = build_system_prompt(
        lang_hint     = _language_hint(has_japanese, has_hinglish, language),
        speakers_hint = speakers_hint,
        summary_instr = _summary_instruction(len(text.split())),
        japan_schema  = _japan_schema_str(include_japan),
    )
    user_prompt = build_user_prompt(text, is_degenerate=is_degenerate)
    return system_prompt, user_prompt


# ── FIX-15/16: Persistent key exhaustion ─────────────────────────────────────
_EXHAUSTED_FILE = (
    pathlib.Path(os.getenv("TRANSCRIPT_AI_STATE_DIR", ".")) / "groq_key_exhausted.json"
)

def _load_key_exhausted() -> dict[str, float]:
    try:
        if _EXHAUSTED_FILE.exists():
            data = json.loads(_EXHAUSTED_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {k: float(v) for k, v in data.items()}
    except Exception as e:
        print(f"[GROQ] Could not load key exhaustion state: {e}", file=sys.stderr, flush=True)
    return {}

def _save_key_exhausted(mapping: dict[str, float]) -> None:
    try:
        _EXHAUSTED_FILE.parent.mkdir(parents=True, exist_ok=True)
        _EXHAUSTED_FILE.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[GROQ] Could not persist key exhaustion state: {e}", file=sys.stderr, flush=True)

_KEY_EXHAUSTED: dict[str, float] = _load_key_exhausted()
_KEY_INDEX:     dict[str, int]   = {"n": 0}

def _groq_quota_has_reset(exhausted_at: float) -> bool:
    if exhausted_at == 0:
        return True
    utc            = datetime.timezone.utc
    now_date       = datetime.datetime.now(utc).date()
    exhausted_date = datetime.datetime.fromtimestamp(exhausted_at, utc).date()
    return now_date > exhausted_date

def _all_groq_keys() -> list[str]:
    keys = []
    for var in ["GROQ_API_KEY", "GROQ_API_KEY_2"]:
        k = os.getenv(var, "").strip()
        if not k:
            try:
                import streamlit as st
                k = (st.secrets.get(var, "") or "").strip()
            except Exception:
                pass
        if k:
            keys.append(k)
    return keys

def _available_groq_keys() -> list[str]:
    return [k for k in _all_groq_keys() if _groq_quota_has_reset(_KEY_EXHAUSTED.get(k[:12], 0))]

def _get_groq_key() -> str:
    available = _available_groq_keys()
    return available[0] if available else ""

def _mark_key_exhausted(key: str) -> None:
    _KEY_EXHAUSTED[key[:12]] = time.time()
    _save_key_exhausted(_KEY_EXHAUSTED)
    print(f"[GROQ] Key {key[:8]}... exhausted (429). Rotating.", file=sys.stderr, flush=True)


def _call_groq(system_prompt: str, user_prompt: str, max_tokens: int,
               model: str = "") -> str:
    """
    Round-robin across available Groq keys.
    B3 FIX: response_format json_object guaranteed.
    DSA: O(k) where k = configured API keys.
    """
    if not _all_groq_keys():
        raise ValueError("NO_GROQ_KEY")

    keys = _available_groq_keys()
    if not keys:
        raise ValueError("ALL_KEYS_EXHAUSTED")

    active_model = model if model else GROQ_MODEL
    last_error: Exception | None = None
    start = _KEY_INDEX["n"] % len(keys)

    for i in range(len(keys)):
        key = keys[(start + i) % len(keys)]
        try:
            r = requests.post(
                GROQ_URL,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model":    active_model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_prompt},
                    ],
                    "temperature":     0.1,
                    "max_tokens":      max_tokens,
                    "response_format": {"type": "json_object"},
                },
                timeout=30,
            )
            if r.status_code == 429:
                _mark_key_exhausted(key)
                last_error = ValueError(f"Key {key[:8]}... rate-limited (429)")
                continue
            if not r.ok:
                msg = f"Key {key[:8]}... HTTP {r.status_code}: {r.text[:120]}"
                print(f"[GROQ] {msg}", file=sys.stderr, flush=True)
                last_error = requests.exceptions.HTTPError(msg)
                continue
            _KEY_INDEX["n"] += 1
            return r.json()["choices"][0]["message"]["content"]
        except requests.exceptions.HTTPError as e:
            if "429" in str(e):
                _mark_key_exhausted(key)
            else:
                print(f"[GROQ] HTTPError key {key[:8]}...: {e}", file=sys.stderr, flush=True)
            last_error = e
            continue

    if last_error and "rate-limited" in str(last_error).lower():
        raise ValueError("ALL_KEYS_EXHAUSTED")
    raise last_error or ValueError("ALL_KEYS_EXHAUSTED")


def stream_transcript_ollama(text: str, language: str = "en"):
    """Streams analysis via Ollama — local mode."""
    system = (
        "You are an expert meeting analyst. "
        + _GROUNDING_RULES
        + "Write a concise professional meeting summary covering key points "
        "and action items based ONLY on the transcript you receive."
    )
    prompt = f"{_GROUNDING_RULES_SHORT}\n<transcript>\n{text[:3000]}\n</transcript>"
    try:
        r = requests.post(
            OLLAMA_URL,
            json={
                "model":   OLLAMA_MODEL,
                "system":  system,
                "prompt":  prompt,
                "stream":  True,
                "options": {"temperature": 0.1, "num_predict": 600},
            },
            stream=True,
            timeout=120,
        )
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode("utf-8"))
                    token = chunk.get("response", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
                except Exception:
                    continue
    except Exception as e:
        yield f"Ollama stream error: {str(e)[:80]}"


def stream_transcript_groq(text: str, language: str = "en"):
    """Routes to Ollama in local mode, Groq otherwise."""
    if PROVIDER == "ollama":
        yield from stream_transcript_ollama(text, language)
        return

    api_key = _get_groq_key()
    if not api_key:
        yield "⚠️ No Groq API key. Add GROQ_API_KEY for streaming."
        return

    stream_system = (
        "You are an expert meeting analyst for Japanese business culture.\n\n"
        + _GROUNDING_RULES
        + "\nWrite a clear, concise, professional meeting summary covering key points, "
        "action items, and any notable Japanese business communication patterns "
        "observed — based ONLY on what is explicitly in the transcript you receive."
    )
    stream_user = (
        f"{_GROUNDING_RULES_SHORT}\n"
        f"<transcript>\n{text[:3000]}\n</transcript>"
    )
    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model":       GROQ_MODEL,
                "messages":    [
                    {"role": "system", "content": stream_system},
                    {"role": "user",   "content": stream_user},
                ],
                "temperature": 0.1,
                "max_tokens":  1000,
                "stream":      True,
            },
            stream=True,
            timeout=60,
        )
        r.raise_for_status()
        for line in r.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        continue
    except Exception as e:
        yield f"Stream error: {str(e)[:80]}"


def _call_ollama(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    """
    FIX-13: Ollama native system/prompt split.
    FIX-23: "think" only for thinking-capable model families.
    """
    payload: dict = {
        "model":   OLLAMA_MODEL,
        "prompt":  user_prompt,
        "system":  system_prompt,
        "stream":  False,
        "format":  "json",
        "options": {"temperature": 0.1, "num_predict": max_tokens, "top_p": 0.85},
    }
    _THINKING_MODELS = ("qwen3", "qwq", "deepseek-r", "marco-o1")
    if any(m in OLLAMA_MODEL.lower() for m in _THINKING_MODELS):
        payload["think"] = False
    print(f"[OLLAMA] Calling {OLLAMA_MODEL} at {OLLAMA_URL}", file=sys.stderr, flush=True)
    r = requests.post(OLLAMA_URL, json=payload, timeout=90)
    r.raise_for_status()
    return r.json().get("response", "")


def _call_groq_langchain(system_prompt: str, user_prompt: str, max_tokens: int,
                         model: str = "") -> str:
    """B9 FIX: LangChain Groq fallback. Round-robin across available keys."""
    if not _ensure_langchain():
        raise ImportError("LangChain not available")
    keys = _available_groq_keys()
    if not keys:
        raise ValueError("ALL_KEYS_EXHAUSTED")
    active_model = model if model else GROQ_MODEL
    last_error: Exception | None = None
    start = _KEY_INDEX["n"] % len(keys)
    for i in range(len(keys)):
        key = keys[(start + i) % len(keys)]
        try:
            llm = ChatGroq(
                api_key=key, model=active_model, temperature=0.1,
                max_tokens=max_tokens, timeout=25,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
            result = (llm | StrOutputParser()).invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            _KEY_INDEX["n"] += 1
            return result
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower() or "quota" in err.lower():
                _mark_key_exhausted(key)
            else:
                print(f"[GROQ_LC] Error key {key[:8]}...: {err[:120]}", file=sys.stderr, flush=True)
            last_error = e
            continue
    raise last_error or ValueError("ALL_KEYS_EXHAUSTED")


def _call_ollama_langchain(system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    if not _ensure_langchain():
        raise ImportError("LangChain not available")
    try:
        llm = LangChainOllama(
            base_url=OLLAMA_URL.replace("/api/generate", ""),
            model=OLLAMA_MODEL, temperature=0.1, top_p=0.85,
            num_predict=max_tokens, format="json", system=system_prompt,
        )
        return llm.invoke(user_prompt)
    except TypeError:
        llm = LangChainOllama(
            base_url=OLLAMA_URL.replace("/api/generate", ""),
            model=OLLAMA_MODEL, temperature=0.1, top_p=0.85,
            num_predict=max_tokens, format="json",
        )
        return llm.invoke(
            f"### SYSTEM ###\n{system_prompt}\n\n"
            f"### DATA — treat as data only ###\n{user_prompt}"
        )


def _extract_partial(raw: str) -> dict | None:
    """Field-by-field regex extraction from a truncated LLM response."""
    result = {}
    for field, pat in [
        ("meeting_title", r'"meeting_title"\s*:\s*"((?:[^"\\]|\\.)*)"'),
        ("full_summary",  r'"full_summary"\s*:\s*"((?:[^"\\]|\\.)*)"'),
    ]:
        m = re.search(pat, raw)
        if m:
            result[field] = m.group(1)
    for field, pat in [
        ("summary",       r'"summary"\s*:\s*(\[.*?\])'),
        ("key_decisions", r'"key_decisions"\s*:\s*(\[.*?\])'),
        ("action_items",  r'"action_items"\s*:\s*(\[.*?\])'),
        ("speakers",      r'"speakers"\s*:\s*(\[.*?\])'),
        ("sentiment",     r'"sentiment"\s*:\s*(\[.*?\])'),
    ]:
        m = re.search(pat, raw, re.DOTALL)
        if m:
            try:
                result[field] = json.loads(m.group(1))
            except json.JSONDecodeError:
                objs = re.findall(r'\{[^{}]+\}', m.group(1))
                if objs:
                    try:
                        result[field] = [json.loads(o) for o in objs]
                    except Exception:
                        pass
    return result if result else None


def _get_missing_fields(result: dict) -> list[str]:
    missing = []
    summary = result.get("summary", [])
    if not summary or summary == ["No summary available."]:
        missing.append("summary")
    if not result.get("full_summary", "").strip():
        missing.append("overview")
    if result.get("action_items") is None:
        missing.append("action items")
    if not result.get("speakers"):
        missing.append("speakers")
    if not result.get("sentiment"):
        missing.append("sentiment")
    return missing


def _parse(raw: str) -> dict:
    """
    Robust JSON parsing — strips markdown fences, handles nested braces,
    repairs truncated JSON.
    DSA: O(n) — single scan for opening brace then JSONDecoder.
    """
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
    decoder = json.JSONDecoder()
    for i, ch in enumerate(raw):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(raw, i)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    try:
        snippet = raw[raw.index("{"):]
        open_b  = snippet.count("{") - snippet.count("}")
        open_sq = snippet.count("[") - snippet.count("]")
        if snippet.count('"') % 2 != 0:
            snippet += '"'
        snippet += "]" * max(open_sq, 0)
        snippet += "}" * max(open_b, 0)
        obj = json.loads(snippet)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    raise ValueError(f"No valid JSON in response (first 200): {raw[:200]}")


def _try_providers(system_prompt: str, user_prompt: str, max_tokens: int,
                   model: str = "") -> tuple[str, str]:
    """
    FIX-21: Provider cascade always runs Groq → Ollama → raises.
    B9 FIX: LangChain Groq fallback restored within the groq provider slot.
    """
    providers_to_try: list = []
    if PROVIDER == "auto":
        providers_to_try = (
            [("groq", _call_groq), ("ollama", _call_ollama)]
            if _all_groq_keys() else
            [("ollama", _call_ollama)]
        )
    elif PROVIDER == "groq":
        providers_to_try = [("groq", _call_groq)]
    elif PROVIDER == "ollama":
        providers_to_try = [("ollama", _call_ollama)]

    last_error: Exception | None = None

    for name, caller in providers_to_try:
        retries = MAX_RETRIES if name == "groq" else 0
        for attempt in range(retries + 1):
            try:
                if name == "groq":
                    raw = caller(system_prompt, user_prompt, max_tokens, model)
                    return raw, "groq"
                elif _ensure_langchain() and name == "ollama":
                    try:
                        raw = _call_ollama_langchain(system_prompt, user_prompt, max_tokens)
                        return raw, f"{name}_langchain"
                    except Exception:
                        raw = caller(system_prompt, user_prompt, max_tokens)
                else:
                    raw = caller(system_prompt, user_prompt, max_tokens)
                return raw, name

            except ValueError as e:
                last_error = e
                if "NO_GROQ_KEY" in str(e) or "ALL_KEYS_EXHAUSTED" in str(e):
                    print("[PROVIDERS] Groq exhausted — falling back.", file=sys.stderr, flush=True)
                    break
                if attempt < retries:
                    import random
                    time.sleep(min((2 ** attempt) + random.uniform(0, 1), 8))

            except requests.exceptions.Timeout:
                last_error = TimeoutError(f"{name} timed out")
                print(f"[PROVIDERS] {name} timed out.", file=sys.stderr, flush=True)
                break

            except requests.exceptions.ConnectionError as e:
                last_error = ConnectionError(f"{name} offline — {str(e)[:80]}")
                print(f"[PROVIDERS] {name} connection error: {e}", file=sys.stderr, flush=True)
                break

            except Exception as e:
                last_error = e
                if attempt < retries:
                    import random
                    time.sleep(min((2 ** attempt) + random.uniform(0, 1), 8))

    raise last_error or RuntimeError("All providers failed")


def _groq_demo_summary(text: str) -> str:
    """Real 2-line summary for mock banner using fast 8B model."""
    key = _get_groq_key()
    if not key:
        return ""
    try:
        r = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": GROQ_MODEL_FAST,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "In exactly 2 sentences, summarize the transcript you are given. "
                            "Be factual and concise. No lists, no markdown. "
                            + _GROUNDING_RULES_SHORT
                        ),
                    },
                    {"role": "user", "content": f"<transcript>\n{text[:1200]}\n</transcript>"},
                ],
                "temperature": 0.1,
                "max_tokens":  120,
            },
            timeout=12,
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        if r.status_code == 429:
            _mark_key_exhausted(key)
    except Exception:
        pass
    return ""


def _mock_response(text: str, reason: str = "") -> dict:
    """Structured mock when all providers fail. Real speaker detection, no placeholders."""
    speaker_names: list[str] = []
    colon_pat = re.compile(
        r"(?:^|\n)\s*(?!https?:)([A-Za-z][A-Za-z\s\.\']{1,35}?)\s*[:：]",
        re.MULTILINE,
    )
    for m in colon_pat.finditer(text):
        raw_n = m.group(1).strip()
        clean = re.sub(r"\s*\([^)]*\)", "", raw_n).strip()
        if (clean and len(clean) >= 2
                and not re.match(r"^\d+$", clean)
                and clean.lower() not in {"http", "https", "view", "sorry", "would", "could"}
                and clean not in speaker_names):
            speaker_names.append(clean)
    if not speaker_names:
        speaker_names = ["Speaker A", "Speaker B"]

    n   = len(speaker_names)
    pct = round(100 / n)
    speakers  = [{"name": nm, "talk_time_pct": pct, "tone": "neutral",
                  "tone_label": "Demo mode — full analysis unavailable", "tone_intensity": 3}
                 for nm in speaker_names]
    sentiment = [{"speaker": nm, "score": "neutral", "label": "factual",
                  "secondary_labels": [], "valence": 0.0,
                  "tone": {"urgency": "low", "certainty": "definite", "engagement": "passive"},
                  "risk_to_relationship": "none",
                  "label": "Demo mode — full analysis unavailable"}
                 for nm in speaker_names]

    words        = len(text.split())
    summary_cnt  = 3 if words < 200 else 5 if words < 600 else 7
    demo_summary = _groq_demo_summary(text) if _get_groq_key() else ""

    if demo_summary:
        full_summary_text = (
            f"{demo_summary}\n\n"
            f"⚠️ Full structured analysis unavailable — daily API limit reached. "
            f"Results will be available again within 24 hours."
        )
        summary_bullets = [
            f"📋 AI Summary: {demo_summary[:180]}",
            f"👥 {n} speaker{'s' if n > 1 else ''} detected: {', '.join(speaker_names[:4])}",
            f"📝 Transcript: {words} words — full analysis resumes when API quota resets.",
        ]
    else:
        full_summary_text = (
            f"⚠️ Daily API limit reached — full analysis unavailable for up to 24 hours. "
            f"Transcript: {words} words, {n} detected speaker{'s' if n > 1 else ''}. "
            f"Real analysis resumes automatically."
        )
        summary_bullets = (
            ["⚠️ API rate limit reached — demo mode active."]
            + [f"Transcript: {words} words · {n} speaker{'s' if n > 1 else ''} detected."]
            + ["Full analysis resumes within 24 hours."] * (summary_cnt - 2)
        )

    return {
        "meeting_title": (
            " ".join(demo_summary.split()[:8]) + ("…" if len(demo_summary.split()) > 8 else "")
            if demo_summary else f"Demo Analysis — {n} Speaker{'s' if n > 1 else ''}"
        ),
        "full_summary":  full_summary_text,
        "summary":       summary_bullets,
        "action_items": [{"task": "Full action items unavailable — API limit reached.",
                          "owner": speaker_names[0] if speaker_names else "Unknown", "deadline": "N/A"}],
        "sentiment":         sentiment,
        "speakers":          speakers,
        "japan_insights":    {"keigo_level": "unknown", "nemawashi_signals": [], "code_switch_count": 0},
        "conversation_dynamics": {},
        "role_hints":        {},
        "_mock_reason":      reason,
        "_demo_mode":        True,
        "_demo_warning":     "API rate limit reached. Demo data shown. Full analysis resumes in 24h.",
        "_has_ai_summary":   bool(demo_summary),
    }


# ── E2: Fine-grained sentiment helpers (no-API path) ─────────────────────────

def _sr_confidence_to_label(sig: dict) -> str:
    """
    Map a soft-rejection signal to the nearest fine-grained sentiment label.
    DSA: dict lookup O(1).
    """
    phrase = (sig.get("phrase", "") + sig.get("explanation", "")).lower()
    _MAP   = {
        "reconsider": "dismissive",  "contract":    "anxious",
        "apolog":     "defensive",   "complaint":   "frustrated",
        "ultimatum":  "frustrated",  "written":     "anxious",
        "down":       "frustrated",  "unacceptable":"frustrated",
        "terminate":  "dismissive",  "delay":       "disappointed",
    }
    for keyword, lbl in _MAP.items():
        if keyword in phrase:
            return lbl
    return "frustrated"


def _no_api_sentiment_block(
    text: str,
    names: list[str],
    sr: dict,
    speakers: list[dict],
) -> list[dict]:
    """
    E2: Fine-grained sentiment for the no-API path.

    Two layers:
      Layer 1 — FineSentimentAnalyzer: 25-label analysis from transcript text.
      Layer 2 — Soft-rejection overlay: critical SR signals override neutral.

    Falls back to flat neutral if sentiment_engine not installed.
    D2 FIX: syncs sentiment_score into speaker dicts.
    D3 FIX: strict speaker name matching — no Unknown fallback.
    DSA: O(U·L·P) for engine + O(n_speakers · n_signals) for SR overlay.
    """
    risk_level = sr.get("risk_level", "NONE")

    # SR speaker→signals map (D3 FIX: exclude Unknown)
    sr_signals: dict[str, list] = {}
    for sig in sr.get("detected", []):
        spk = sig.get("speaker", "Unknown")
        if spk != "Unknown":
            sr_signals.setdefault(spk, []).append(sig)

    # Layer 1: engine
    if _SENTIMENT_ENGINE_AVAILABLE:
        report  = _SENTIMENT_ENGINE.from_raw_transcript(text)
        arc_map = report.speaker_arcs
        overall_tone = {
            "urgency":    report.overall_tone.urgency,
            "certainty":  report.overall_tone.certainty,
            "engagement": report.overall_tone.engagement,
        }
    else:
        arc_map = {}
        overall_tone = {"urgency": "low", "certainty": "definite", "engagement": "passive"}

    sentiment = []
    for nm in names:
        # Find matching arc
        arc = None
        for arc_spk, arc_obj in arc_map.items():
            if arc_spk.lower() in nm.lower() or nm.lower() in arc_spk.lower():
                arc = arc_obj
                break

        if arc:
            valence = arc.mean_valence
            score   = ("positive" if valence > POSITIVE_VALENCE_THRESHOLD
                       else "negative" if valence < NEGATIVE_VALENCE_THRESHOLD
                       else "neutral")
            label   = arc.dominant
            secondary = [
                lbl for lbl, _ in sorted(
                    arc.emotion_distribution.items(), key=lambda kv: -kv[1]
                ) if lbl != label
            ][:2]
            trend = arc.trend
        else:
            score = "neutral"; label = "factual"; secondary = []
            valence = 0.0; trend = "stable"

        # Layer 2: SR override
        matched_sr = []
        for spk_key, sigs in sr_signals.items():
            if spk_key.lower() in nm.lower() or nm.lower() in spk_key.lower():
                matched_sr.extend(sigs)

        if matched_sr:
            top = max(matched_sr, key=lambda s: s.get("confidence", 0.5))
            if top.get("confidence", 0) >= 0.65:
                score   = "negative"
                label   = _sr_confidence_to_label(top)
                valence = min(valence, -0.45)
        elif risk_level in ("CRITICAL", "HIGH") and score == "neutral":
            score = "negative"; label = "anxious"; valence = -0.45

        risk = ("high"   if risk_level in ("CRITICAL", "HIGH") and score == "negative"
                else "none" if score == "positive"
                else "low")

        sentiment.append({
            "speaker":              nm,
            "score":                score,
            "label":                label,
            "secondary_labels":     secondary,
            "tone":                 overall_tone,
            "valence":              round(valence, 3),
            "trend":                trend,
            "risk_to_relationship": risk,
        })

    # D2 FIX: sync into speaker dicts
    sent_lookup = {s["speaker"]: s["score"] for s in sentiment}
    for spk in speakers:
        spk["sentiment_score"] = sent_lookup.get(spk["name"], "neutral")

    return sentiment


# ── E2: Fine-grained sentiment backstop (post-LLM) ───────────────────────────

def _sentiment_backstop_block(result: dict, text: str) -> None:
    """
    Stage 12b — post-LLM sentiment backstop. Mutates result in place.

    Tier 1: Enrich flat/partial LLM output with local engine fields
            (valence, secondary_labels, tone, trend).
    Tier 2: Blend valence when LLM and local engine disagree by > 0.40.
    Tier 3: SR hard evidence overrides residual neutrals.
    D2 FIX: sync sentiment_score into speaker dicts (all tiers).
    DSA: O(U·L·P) for engine + O(speakers · arcs) for enrichment.
    """
    _sr       = result.get("soft_rejections", {})
    _sr_level = _sr.get("risk_level", "NONE")

    # Tier 1 + 2: enrich if any speaker is missing fine-grained fields
    needs_enrichment = any(
        "valence" not in s or "secondary_labels" not in s
        for s in result.get("sentiment", [])
    )

    if needs_enrichment and _SENTIMENT_ENGINE_AVAILABLE:
        local_report = _SENTIMENT_ENGINE.from_raw_transcript(text)
        arc_map = local_report.speaker_arcs
        overall_tone = {
            "urgency":    local_report.overall_tone.urgency,
            "certainty":  local_report.overall_tone.certainty,
            "engagement": local_report.overall_tone.engagement,
        }

        for s in result.get("sentiment", []):
            spk_name = s.get("speaker", "")
            arc = next(
                (a for key, a in arc_map.items()
                 if key.lower() in spk_name.lower() or spk_name.lower() in key.lower()),
                None,
            )
            if arc:
                s.setdefault("secondary_labels", [
                    lbl for lbl, _ in sorted(
                        arc.emotion_distribution.items(), key=lambda kv: -kv[1]
                    ) if lbl != arc.dominant
                ][:2])
                s.setdefault("valence", arc.mean_valence)
                s.setdefault("tone",    overall_tone)
                s.setdefault("trend",   arc.trend)
                s.setdefault("label",   arc.dominant)

                # Tier 2: valence blend
                llm_val   = float(s.get("valence", 0.0))
                local_val = arc.mean_valence
                if abs(llm_val - local_val) > 0.40:
                    s["valence"]        = round((llm_val + local_val) / 2, 3)
                    s["valence_source"] = "backstop_blended"

    # Tier 3: SR hard override
    _contract_risk = _sr.get("contract_risk_detected", False)
    _term_detected = _sr.get("termination_detected",   False)

    if _sr_level in ("CRITICAL", "HIGH"):
        for _s in result.get("sentiment", []):
            if _s.get("score") == "neutral":
                _s["score"]   = "negative"
                _s["valence"] = min(float(_s.get("valence", 0.0)), -0.45)
                if _contract_risk or _term_detected:
                    _s["label"]                = (
                        f"⚠️ Relationship at risk — {_s.get('label', 'see soft rejection signals')}"
                    )
                    _s["risk_to_relationship"] = "high"
                else:
                    _s["label"]                = (
                        f"Tension detected — {_s.get('label', 'elevated risk signals present')}"
                    )
                    _s["risk_to_relationship"] = "medium"
                print(
                    f"[SENTIMENT] Backstop T3: neutral→negative for {_s.get('speaker')} "
                    f"(SR level: {_sr_level})",
                    file=sys.stderr, flush=True,
                )

    # D2 FIX: sync sentiment_score into speaker dicts
    _sent_sync = {s["speaker"]: s["score"] for s in result.get("sentiment", [])}
    for _spk in result.get("speakers", []):
        _spk["sentiment_score"] = _sent_sync.get(
            _spk.get("name", ""), _spk.get("sentiment_score", "neutral")
        )


# ─────────────────────────────────────────────────────────────────────────────

def _no_api_result(text: str, reason: str = "") -> dict:
    """
    Full analysis without any LLM API call.
    Real data for: speakers, sentiment (E2 fine-grained), keigo, deal outcome,
                   conversation dynamics, action items (rule-based).
    Honestly empty for: summary, key_decisions (require LLM understanding).
    """
    import re as _re

    # ── Speaker extraction ────────────────────────────────────────────────────
    try:
        from utils.speaker_detector import detect_speakers as _detect_spk
        _spk_result = _detect_spk(text)
        names = _spk_result["names"]
        _turns = _spk_result["turns_per_speaker"]
    except ImportError:
        hint  = _extract_speaker_hint(text)
        names = [n.strip() for n in hint.split(",")
                 if n.strip() and n.strip().lower() != "not detected"]
        _turns = {nm: 1 for nm in names}

    if not names:
        names = ["Speaker A"]
    n       = len(names)
    total_t = sum(_turns.values()) or 1

    speakers = [
        {
            "name":           nm,
            "talk_time_pct":  round(_turns.get(nm, 1) * 100 / total_t),
            "tone":           "neutral",
            "tone_label":     "Tone analysis unavailable — API limit reached",
            "tone_intensity": 3,
        }
        for nm in names
    ]
    _diff = 100 - sum(s["talk_time_pct"] for s in speakers)
    if _diff and speakers:
        speakers[0]["talk_time_pct"] += _diff

    # ── Soft rejection ────────────────────────────────────────────────────────
    sr = {}
    try:
        from analysis.soft_rejection_detector import detect_soft_rejections
        sr = detect_soft_rejections(text)
    except Exception as _e:
        print(f"[NO_API] soft_rejection_detector: {_e}", file=sys.stderr, flush=True)

    risk_level = sr.get("risk_level", "NONE")

    # ── E2: Fine-grained sentiment (replaces old flat block) ──────────────────
    sentiment = _no_api_sentiment_block(text, names, sr, speakers)

    # ── D4 FIX: rule-based comm risk ─────────────────────────────────────────
    _RISK_TO_COMM_SCORE = {
        "CRITICAL": 25, "HIGH": 20, "MEDIUM": 13,
        "LOW": 6, "MINIMAL": 2, "NONE": 0,
    }
    _comm_risk_rule_based = _RISK_TO_COMM_SCORE.get(risk_level, 0)

    # ── Japan insights ────────────────────────────────────────────────────────
    japan_insights: dict = {
        "keigo_level": "unknown", "nemawashi_signals": [], "code_switch_count": 0,
    }
    try:
        from analysis.japanese_tokenizer import get_keigo_level, MECAB_AVAILABLE
        if MECAB_AVAILABLE:
            japan_insights["keigo_level"]  = get_keigo_level(text)
            japan_insights["keigo_source"] = "mecab"
    except Exception:
        pass

    jp_sigs = [
        s["phrase"]
        for s in sr.get("medium_signals", []) + sr.get("low_signals", [])
        if _re.search(r"[\u3040-\u9fff]", s.get("phrase", ""))
    ]
    japan_insights["nemawashi_signals"] = jp_sigs[:5]

    try:
        from utils.evaluator import count_code_switches
        japan_insights["code_switch_count"]  = count_code_switches(text)
        japan_insights["code_switch_source"] = "rule_based"
    except Exception:
        pass

    # ── Deal outcome ──────────────────────────────────────────────────────────
    deal_outcome = {}
    try:
        from analysis.deal_outcome_detector import detect_deal_outcome
        deal_outcome = detect_deal_outcome(text)
    except Exception as _e:
        print(f"[NO_API] deal_outcome_detector: {_e}", file=sys.stderr, flush=True)

    # ── Conversation dynamics ─────────────────────────────────────────────────
    conversation_dynamics: dict = {}
    role_hints: dict = {}
    try:
        from analysis.conversation_dynamics import analyze_conversation_dynamics
        conversation_dynamics = analyze_conversation_dynamics(text)
        role_hints = conversation_dynamics.get("role_hints", {})
    except Exception as _e:
        print(f"[NO_API] conversation_dynamics: {_e}", file=sys.stderr, flush=True)

    word_count   = len(text.split())
    speakers_str = " & ".join(names[:2])

    # ── D5 FIX: rule-based action items ──────────────────────────────────────
    _rule_action_items: list[dict] = []
    try:
        from analysis.action_item_extractor import extract_action_items
        _rule_action_items = extract_action_items(text)
        if _rule_action_items:
            print(f"[NO_API] Rule-based action items: {len(_rule_action_items)} found",
                  file=sys.stderr, flush=True)
    except ImportError:
        pass
    except Exception as _e:
        print(f"[NO_API] action_item_extractor failed: {_e}", file=sys.stderr, flush=True)

    return {
        "meeting_title":  f"{speakers_str} — Meeting",
        "full_summary": (
            "⚠️ LLM summary unavailable — API quota reached. "
            "All rule-based analysis (soft rejection, deal outcome, keigo, "
            "conversation dynamics) is complete and accurate below."
        ),
        "summary": [
            "⚠️ Summary requires LLM API — unavailable while quota is reached.",
            f"Transcript: {word_count} words | "
            f"{n} speaker{'s' if n > 1 else ''}: {', '.join(names)}.",
            f"Soft rejection risk: {risk_level}. "
            f"Full risk analysis, keigo, and deal outcome are operational below.",
        ],
        "key_decisions":         [],
        "action_items":          _rule_action_items,
        "sentiment":             sentiment,
        "speakers":              speakers,
        "japan_insights":        japan_insights,
        "soft_rejections":       sr,
        "deal_outcome":          deal_outcome,
        "conversation_dynamics": conversation_dynamics,
        "role_hints":            role_hints,
        "_no_api":               True,
        "_no_api_reason":        reason,
        "_no_api_warning": (
            "Summary, action items, and key decisions require LLM. "
            "All other analysis is rule-based and fully accurate."
        ),
        "_comm_risk_rule_based": _comm_risk_rule_based,
    }


def analyze_transcript(text: str, language: str = "en",
                       bypass_cache: bool = False,
                       user_id: str | None = None) -> dict:
    """
    Full analysis pipeline v8.0

    Stage order:
     0.   E3     normalize_format (7-pass, upgraded from FIX-22)
     A3:         PII mask full transcript
     1.          Vector cache check
     2.          MD5 exact cache
     3.          Truncate masked text
     4.          LLM extraction (Groq → Ollama → no-API)
     A4:         PII restoration
     5.          _validate_and_fill (defaults + E4 fine-grained validation)
     6.          Speaker fallback
     7.          talk_time_pct recompute (FIX-20)
     8.          Speaker normalizer
     9.          MeCab keigo
    10.          Code-switch count
    10b. D6     Post-LLM rule-based action item backfill
    11.          Hallucination guard
    12.          Soft rejection detection
    12b. E2     Sentiment backstop (fine-grained enrichment + SR override + D2 sync)
    13.          Deal outcome detection
    14.          Conversation dynamics
    15.          Log + cache store
    """
    start_time = time.time()

    # Stage 0: normalize format (E3 — upgraded 7-pass)
    text = normalize_format(text)

    # A3 FIX: PII mask
    pii         = None
    masked_text = text
    _restore_fn = None
    try:
        from transcription.pii_masker import (
            mask_transcript       as _mask_fn,
            restore_pii_in_result as _restore_fn,
        )
        masked_text, pii = _mask_fn(text)
        print(f"[PII] Masked {pii.counters} entities.", file=sys.stderr, flush=True)
        _hint = _extract_speaker_hint(_truncate_transcript(masked_text))
        print(f"[PII] Speaker hint on masked text: '{_hint}'", file=sys.stderr, flush=True)
    except ImportError:
        print("[PII] pii_masker not found — transcript unmasked.", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[PII] mask_transcript failed: {e}", file=sys.stderr, flush=True)

    # Stage 1: vector cache
    vector_cache_available = False
    store_result           = None
    try:
        from utils.vector_cache import get_cached_result, store_result, is_available
        vector_cache_available = is_available()
        if vector_cache_available and not bypass_cache:
            cached = get_cached_result(text, language, user_id=user_id,
                                       masked_transcript=masked_text)
            if cached:
                cached["_from_vector_cache"] = True
                return cached
    except ImportError:
        vector_cache_available = False
    except Exception as e:
        print(f"[TRANSCRIPT_AI] vector_cache read failed: {e}", file=sys.stderr, flush=True)
        store_result = None; vector_cache_available = False

    # Stage 2: MD5 cache
    get_cached = None
    set_cache  = None
    try:
        from utils.cache import get_cached, set_cache
        if not bypass_cache:
            cached = get_cached(text, language, user_id=user_id)
            if cached:
                cached["_from_cache"] = True
                return cached
    except ImportError:
        pass
    except Exception as e:
        print(f"[TRANSCRIPT_AI] MD5 cache read failed: {e}", file=sys.stderr, flush=True)
        get_cached = set_cache = None

    # Stage 3: truncate masked text
    text_for_llm   = _truncate_transcript(masked_text)
    system_prompt, user_prompt = build_prompt(text_for_llm, language)

    words          = len(text_for_llm.split())
    selected_model = _select_model(
        text_for_llm, language,
        bool(re.search(r"[぀-鿿]", text_for_llm))
    )
    max_tokens = (
         900 if words < 300  else
        1100 if words < 800  else
        1400 if words < 2000 else
        1800
    )

    provider_used = "unknown"
    last_error    = None
    captured_raw  = None

    try:
        # Stage 4: LLM call
        raw, provider_used = _try_providers(
            system_prompt, user_prompt, max_tokens, selected_model
        )
        captured_raw = raw

        result = _parse(raw)

        # A4 FIX: restore PII
        if pii is not None and _restore_fn is not None:
            result = _restore_fn(result, pii)
            print("[PII] Restored PII in LLM result.", file=sys.stderr, flush=True)

        # Stage 5: validate and fill (includes E4 fine-grained validation)
        result = _validate_and_fill(result)

        missing = _get_missing_fields(result)
        if missing:
            present = [f for f in ["summary", "action items", "speakers", "sentiment"]
                       if f not in missing]
            result["_partial"]         = True
            result["_missing_fields"]  = missing
            result["_partial_warning"] = (
                f"⚠️ Partial analysis — API limit reached mid-generation.\n"
                f"Generated: {', '.join(present) or 'none'}.\n"
                f"Unavailable: {', '.join(missing)}."
            )

        # Stage 6: speaker fallback
        if not result.get("speakers"):
            _names = [
                s.strip() for s in _extract_speaker_hint(text).split(",")
                if s.strip() and s.strip().lower() != "not detected"
            ]
            if _names:
                _n = len(_names)
                result["speakers"] = [
                    {"name": nm, "talk_time_pct": round(100 / _n),
                     "tone": "neutral", "tone_label": "Professional tone", "tone_intensity": 3}
                    for nm in _names
                ]
                result["sentiment"] = [
                    {"speaker": nm, "score": "neutral", "label": "factual",
                     "secondary_labels": [], "valence": 0.0,
                     "tone": {"urgency": "low", "certainty": "definite", "engagement": "passive"},
                     "risk_to_relationship": "none"}
                    for nm in _names
                ]

    except Exception as e:
        err_str = str(e)
        if "NO_GROQ_KEY"          in err_str: provider_used = "mock_no_key"
        elif "ALL_KEYS_EXHAUSTED" in err_str or "429" in err_str: provider_used = "mock_rate_limit"
        elif isinstance(e, TimeoutError)    or "timed out"  in err_str.lower(): provider_used = "mock_timeout"
        elif isinstance(e, ConnectionError) or "offline"    in err_str.lower(): provider_used = "mock_offline"
        else: provider_used = "mock"

        last_error = err_str[:120]
        print(
            f"[TRANSCRIPT_AI] All providers failed → fallback. "
            f"reason={provider_used} err={last_error}",
            file=sys.stderr, flush=True,
        )

        salvaged = _extract_partial(captured_raw) if captured_raw else None
        if salvaged:
            if pii is not None and _restore_fn is not None:
                salvaged = _restore_fn(salvaged, pii)
            result  = _validate_and_fill(salvaged)
            missing = _get_missing_fields(result)
            present = [f for f in ["summary", "action items", "speakers", "sentiment"]
                       if f not in missing]
            result["_partial"]         = True
            result["_missing_fields"]  = missing
            result["_partial_warning"] = (
                f"⚠️ API limit reached mid-generation. Showing what was generated.\n"
                f"Generated: {', '.join(present) or 'none'}.\n"
                f"Unavailable (API limit): {', '.join(missing)}.\n"
                f"Full analysis available when quota resets."
            )
            provider_used = f"partial_{provider_used}"
            print(f"[TRANSCRIPT_AI] Salvaged partial result: {present}", file=sys.stderr, flush=True)
        else:
            result        = _no_api_result(text, reason=last_error)
            result        = _validate_and_fill(result)
            provider_used = "no_api"

    # Stage 7: talk_time_pct on raw text
    _recompute_talk_time_pct(text, result.get("speakers", []))

    # Stage 8: speaker normalizer
    try:
        from transcription.speaker_normalizer import unify_speakers_in_result
        result = unify_speakers_in_result(result, text)
    except ImportError:
        pass
    except Exception as e:
        print(f"[TRANSCRIPT_AI] speaker_normalizer failed: {e}", file=sys.stderr, flush=True)

    # Stage 9: MeCab keigo
    if result.get("japan_insights", {}).get("keigo_level", "unknown") == "unknown":
        try:
            from analysis.japanese_tokenizer import get_keigo_level, MECAB_AVAILABLE
            if MECAB_AVAILABLE:
                result["japan_insights"]["keigo_level"]  = get_keigo_level(text)
                result["japan_insights"]["keigo_source"] = "mecab"
        except ImportError:
            pass
        except Exception as e:
            print(f"[TRANSCRIPT_AI] japanese_tokenizer failed: {e}", file=sys.stderr, flush=True)

    # Stage 10: code-switch count
    if result.get("japan_insights", {}).get("code_switch_source") != "rule_based":
        try:
            from utils.evaluator import count_code_switches
            result["japan_insights"]["code_switch_count"]  = count_code_switches(text)
            result["japan_insights"]["code_switch_source"] = "rule_based"
        except ImportError:
            pass
        except Exception as e:
            print(f"[TRANSCRIPT_AI] count_code_switches failed: {e}", file=sys.stderr, flush=True)

    # Stage 10b: D6 FIX — post-LLM action item backfill
    if not result.get("_no_api") and "mock" not in provider_used:
        try:
            from analysis.action_item_extractor import extract_action_items as _ae
            _rule_items = _ae(text)
            existing_tasks = {
                re.sub(r"\s+", "", i.get("task", ""))[:30].lower()
                for i in result.get("action_items", [])
            }
            for item in _rule_items:
                key = re.sub(r"\s+", "", item["task"])[:30].lower()
                if key not in existing_tasks:
                    item["hallucination_flag"]   = False
                    item["verification_skipped"] = False
                    result.setdefault("action_items", []).append(item)
                    existing_tasks.add(key)
        except Exception:
            pass

    # Stage 11: hallucination guard
    try:
        from analysis.hallucination_guard import verify_result
        result = verify_result(result, text)
        from analysis.semantic_validator import validate_action_items_semantic
        result["action_items"] = validate_action_items_semantic(
            result.get("action_items", []), text
        )
    except ImportError:
        for item in result.get("action_items", []):
            item.setdefault("hallucination_flag", False)
            item["verification_skipped"] = True
    except Exception as e:
        print(f"[TRANSCRIPT_AI] hallucination_guard failed: {e}", file=sys.stderr, flush=True)
        for item in result.get("action_items", []):
            item["hallucination_flag"]   = True
            item["flag_reason"]          = "Hallucination guard failed — output unverified"
            item["verification_skipped"] = True
        result["_hallucination_guard_error"] = str(e)

    # Stage 12: soft rejection
    if "soft_rejections" not in result:
        try:
            from analysis.soft_rejection_detector import detect_soft_rejections
            result["soft_rejections"] = detect_soft_rejections(text)
        except ImportError:
            pass
        except Exception as e:
            print(f"[TRANSCRIPT_AI] soft_rejection_detector failed: {e}", file=sys.stderr, flush=True)

    # Stage 12b: E2 sentiment backstop (fine-grained enrichment + SR override + D2 sync)
    _sentiment_backstop_block(result, text)

    # Stage 13: deal outcome
    if "deal_outcome" not in result:
        try:
            from analysis.deal_outcome_detector import detect_deal_outcome
            result["deal_outcome"] = detect_deal_outcome(text)
        except ImportError:
            pass
        except Exception as e:
            print(f"[TRANSCRIPT_AI] deal_outcome_detector failed: {e}", file=sys.stderr, flush=True)

    # Stage 14: conversation dynamics
    if "conversation_dynamics" not in result or not result["conversation_dynamics"]:
        try:
            from analysis.conversation_dynamics import analyze_conversation_dynamics
            dynamics                        = analyze_conversation_dynamics(text)
            result["conversation_dynamics"] = dynamics
            result["role_hints"]            = dynamics["role_hints"]
        except ImportError:
            pass
        except Exception as e:
            print(f"[TRANSCRIPT_AI] conversation_dynamics failed: {e}", file=sys.stderr, flush=True)

    duration_ms            = (time.time() - start_time) * 1000
    result["_provider"]    = provider_used
    result["_duration_ms"] = round(duration_ms, 1)
    if last_error:
        result["_last_error"] = last_error

    try:
        from utils.logger import log_analysis
        log_analysis(len(text), language, provider_used, duration_ms, result, last_error)
    except ImportError:
        pass
    except Exception as e:
        print(f"[TRANSCRIPT_AI] log_analysis failed: {e}", file=sys.stderr, flush=True)

    # Stage 15: cache store
    if "mock" not in provider_used:
        if vector_cache_available and store_result:
            try:
                store_result(text, language, result, user_id=user_id,
                             masked_transcript=masked_text)
            except Exception as e:
                print(f"[TRANSCRIPT_AI] vector_cache store failed: {e}", file=sys.stderr, flush=True)
        if set_cache:
            try:
                set_cache(text, language, result, user_id=user_id)
            except Exception as e:
                print(f"[TRANSCRIPT_AI] MD5 cache store failed: {e}", file=sys.stderr, flush=True)

    return result


def _fallback_meeting_title(data: dict) -> str:
    source = ""
    bullets = data.get("summary")
    if isinstance(bullets, list) and bullets:
        source = bullets[0]
    if not source:
        source = data.get("full_summary", "")
    source = re.sub(r"^[📋👥📝⚠️\s]+", "", str(source)).strip()
    words  = source.split()
    if words:
        return " ".join(words[:8]) + ("…" if len(words) > 8 else "")
    return "Meeting Analysis"


def _validate_and_fill(data: dict) -> dict:
    """
    Fill missing keys with safe defaults and normalise LLM output.

    v8.0 (E4): Added fine-grained field validation —
      label       validated against 25-label set
      secondary_labels  cleaned to valid labels only
      valence     clamped to [-1.0, +1.0]; score derived from valence if missing
      tone        dict validated with correct keys and value sets

    Retained from v7.9:
      B4: tone strip+lower, intensity clamp
      B5: sentiment score strip+lower
      A2: japan_insights None guard
    """
    data.setdefault("meeting_title", "")
    if not data["meeting_title"].strip():
        data["meeting_title"] = _fallback_meeting_title(data)
    data.setdefault("full_summary", "")
    data.setdefault("summary", ["No summary available."])
    data.setdefault("key_decisions", [])
    data.setdefault("action_items", [])
    data.setdefault("conversation_dynamics", {})
    data.setdefault("role_hints", {})

    # B5 + E4: sentiment field validation
    data.setdefault("sentiment", [])
    _VALID_SCORES     = {"positive", "neutral", "negative"}
    _VALID_RISK       = {"high", "medium", "low", "none"}
    _VALID_URGENCY    = {"low", "medium", "high"}
    _VALID_CERTAINTY  = {"definite", "hedged", "uncertain"}
    _VALID_ENGAGEMENT = {"active", "passive", "disengaged"}

    for s in data.get("sentiment", []):
        # B5: coarse score
        raw_score  = s.get("score", "").strip().lower()
        s["score"] = raw_score if raw_score in _VALID_SCORES else "neutral"
        s.setdefault("label", "factual")

        # E4: fine-grained label
        raw_label  = s.get("label", "").strip().lower().replace(" ", "_").replace("-", "_")
        s["label"] = raw_label if (_ALL_FINE_GRAINED and raw_label in _ALL_FINE_GRAINED) else "factual"

        # E4: secondary_labels
        raw_sec = s.get("secondary_labels", [])
        s["secondary_labels"] = [
            lbl.strip().lower().replace(" ", "_").replace("-", "_")
            for lbl in (raw_sec if isinstance(raw_sec, list) else [])
            if lbl.strip().lower().replace(" ", "_").replace("-", "_")
            in (_ALL_FINE_GRAINED or set())
        ][:2]

        # E4: valence clamp
        try:
            s["valence"] = max(-1.0, min(1.0, float(s.get("valence", 0.0))))
        except (TypeError, ValueError):
            s["valence"] = 0.0

        # E4: derive score from valence when LLM left it neutral but valence disagrees
        if s["score"] == "neutral" and abs(s["valence"]) > 0.01:
            if s["valence"] > POSITIVE_VALENCE_THRESHOLD:
                s["score"] = "positive"
            elif s["valence"] < NEGATIVE_VALENCE_THRESHOLD:
                s["score"] = "negative"

        # E4: tone dict validation
        raw_tone = s.get("tone") if isinstance(s.get("tone"), dict) else {}
        s["tone"] = {
            "urgency":    raw_tone.get("urgency",    "low")     if raw_tone.get("urgency")    in _VALID_URGENCY    else "low",
            "certainty":  raw_tone.get("certainty",  "definite") if raw_tone.get("certainty")  in _VALID_CERTAINTY  else "definite",
            "engagement": raw_tone.get("engagement", "passive")  if raw_tone.get("engagement") in _VALID_ENGAGEMENT else "passive",
        }

        # risk_to_relationship
        raw_risk = s.get("risk_to_relationship", "").strip().lower()
        if raw_risk not in _VALID_RISK:
            s["risk_to_relationship"] = (
                "high" if s["score"] == "negative" else
                "none" if s["score"] == "positive" else "low"
            )
        else:
            s["risk_to_relationship"] = raw_risk

    # A2 FIX: japan_insights None guard
    if not isinstance(data.get("japan_insights"), dict):
        data["japan_insights"] = {}
    ji = data["japan_insights"]
    ji.setdefault("keigo_level",       "unknown")
    ji.setdefault("nemawashi_signals", [])
    ji.setdefault("code_switch_count", 0)

    for spk in data.get("speakers", []):
        for bad_key in list(spk.keys()):
            if "talk" in bad_key and "pct" in bad_key and bad_key != "talk_time_pct":
                spk["talk_time_pct"] = spk.pop(bad_key)

    _JP_RE = re.compile(r"[぀-鿿゠-ヿ･-ﾟ]")
    _NEMAWASHI_FP = {
        "ありがとうございます","ありがとう","おはようございます","こんにちは","こんばんは",
        "お疲れ様でした","お疲れ様です","よろしくお願いします","よろしくお願いいたします",
        "承知しました","了解しました","分かりました","かしこまりました",
        "素晴らしい","なるほど","検討しました","はい","いいえ",
        "それでは月曜日にお会いしましょう","またお会いしましょう","失礼します",
    }

    def _is_fp(s: str) -> bool:
        stripped = re.sub(r"[。、！？…「」『』\s]", "", s)
        for fp in _NEMAWASHI_FP:
            fp_s = re.sub(r"[。、！？…「」『』\s]", "", fp)
            if stripped == fp_s:
                return True
            if fp_s and len(fp_s) / max(len(stripped), 1) > 0.85:
                return True
        return False

    ji["nemawashi_signals"] = [
        s for s in ji.get("nemawashi_signals", [])
        if isinstance(s, str) and _JP_RE.search(s) and not _is_fp(s)
    ]

    # B4 FIX: speaker tone normalisation
    data.setdefault("speakers", [])
    _VALID_TONES = {"aggressive", "assertive", "neutral", "cooperative", "deferential", "hesitant"}
    speakers = data["speakers"]
    if speakers:
        for s in speakers:
            raw_tone   = s.get("tone", "").strip().lower()
            s["tone"]  = raw_tone if raw_tone in _VALID_TONES else "neutral"
            s.setdefault("tone_label",     "Professional tone")
            s.setdefault("tone_intensity", 3)
            try:
                s["tone_intensity"] = max(1, min(5, int(s["tone_intensity"])))
            except (TypeError, ValueError):
                s["tone_intensity"] = 3

        total = sum(s.get("talk_time_pct", 0) for s in speakers)
        if total > 0 and total != 100:
            for s in speakers:
                s["talk_time_pct"] = round(s.get("talk_time_pct", 0) * 100 / total)
        if sum(s.get("talk_time_pct", 0) for s in speakers) == 0:
            equal = round(100 / len(speakers))
            for s in speakers:
                s["talk_time_pct"] = equal

    return data


if __name__ == "__main__":
    if len(sys.argv) > 1:
        os.environ["TRANSCRIPT_AI_PROVIDER"] = sys.argv[1]

    print(f"Provider:        {PROVIDER}")
    print(f"GROQ_MODEL:      {GROQ_MODEL}")
    print(f"GROQ_MODEL_FAST: {GROQ_MODEL_FAST}")
    print(f"MAX_RETRIES:     {MAX_RETRIES}")
    print(f"Ollama URL:      {OLLAMA_URL}")
    print(f"Sentiment engine:{_SENTIMENT_ENGINE_AVAILABLE}")

    print(f"\n[KEY STATE] from {_EXHAUSTED_FILE}:")
    for prefix, ts in _KEY_EXHAUSTED.items():
        dt    = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).isoformat()
        reset = _groq_quota_has_reset(ts)
        print(f"  {prefix}... exhausted at {dt} | reset: {reset}")
    if not _KEY_EXHAUSTED:
        print("  (no keys marked exhausted)")
    print(f"  Available: {len(_available_groq_keys())} / {len(_all_groq_keys())} keys")

    print("\n--- B1/B2: model IDs ---")
    for name, val in [("GROQ_MODEL", GROQ_MODEL), ("GROQ_MODEL_FAST", GROQ_MODEL_FAST)]:
        ok = "groq/" not in val and "meta-llama/" not in val
        print(f"  {'✓' if ok else '✗'}  {name} = '{val}'")

    print("\n--- B4: tone normalisation ---")
    _VALID_TONES = {"aggressive", "assertive", "neutral", "cooperative", "deferential", "hesitant"}
    for raw, expected in [("Cooperative","cooperative"),(" hesitant ","hesitant"),
                          ("ASSERTIVE","assertive"),("unknown_tone","neutral"),("","neutral")]:
        result = raw.strip().lower()
        result = result if result in _VALID_TONES else "neutral"
        print(f"  {'✓' if result == expected else '✗'}  '{raw}' → '{result}'")

    print("\n--- B5: sentiment normalisation ---")
    _VALID_SCORES = {"positive", "neutral", "negative"}
    for raw, expected in [("Positive","positive"),("NEGATIVE","negative"),
                          ("neutral","neutral"),("happy","neutral"),("","neutral")]:
        result = raw.strip().lower()
        result = result if result in _VALID_SCORES else "neutral"
        print(f"  {'✓' if result == expected else '✗'}  '{raw}' → '{result}'")

    print("\n--- E2: fine-grained sentiment (no-API path) ---")
    _test_transcript = (
        "Client: This is completely unacceptable. The system has been down for 6 hours.\n"
        "Kenji: 大変申し訳ございません。We are working on it as fast as possible.\n"
        "Client: I need a written commitment.\n"
        "Kenji: 上司に相談して、2時間以内に書面でご回答します。\n"
        "Client: If this isn't resolved by Friday we will reconsider the entire contract.\n"
        "Kenji: 誠に申し訳ございません。全力で対応いたします。We will not let that happen."
    )
    _r = _no_api_result(_test_transcript)
    for s in _r.get("sentiment", []):
        print(f"  {s['speaker']:<10} score={s['score']:<9} label={s.get('label','?'):<22} "
              f"valence={s.get('valence',0):+.3f}  secondary={s.get('secondary_labels',[])}")

    print("\n--- E4: validate_and_fill fine-grained fields ---")
    _dummy = {"sentiment": [
        {"speaker": "A", "score": "Positive", "label": "enthusiastic", "valence": 0.9,
         "secondary_labels": ["confident"], "tone": {"urgency": "high", "certainty": "definite", "engagement": "active"}},
        {"speaker": "B", "score": "neutral",  "label": "NOT_A_LABEL",  "valence": -0.7,
         "secondary_labels": ["fake_label"], "tone": "bad_tone"},
    ]}
    _filled = _validate_and_fill(_dummy)
    for s in _filled["sentiment"]:
        print(f"  {s['speaker']}: score={s['score']}, label={s['label']}, "
              f"valence={s['valence']}, tone={s['tone']}")

    print("\n--- A5/A6: speaker hint + degenerate detection ---")
    masked_sample = "[NAME_1]: Good morning.\n[NAME_2]: Let us start.\n"
    h_masked = _extract_speaker_hint(masked_sample)
    degen    = _is_degenerate_transcript("[NAME_1]: Short.", "[NAME_1]")
    print(f"  {'✓' if '[NAME_1]' in h_masked else '✗'}  masked hint: '{h_masked}'")
    print(f"  {'✓' if degen else '✗'}  single-speaker degenerate={degen}")