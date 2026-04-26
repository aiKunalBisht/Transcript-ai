"""
health_check.py
===============
Run this to verify every component is working.
Paste the full output back to Claude for diagnosis.

Usage:
    python health_check.py
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import os
import json
import subprocess
import importlib
from pathlib import Path

# Load .env file before any os.getenv() calls
try:
    from dotenv import load_dotenv
    loaded = load_dotenv()
    if loaded:
        print("  📁 .env file loaded")
    else:
        print("  ⚠️  No .env file found (or already loaded via environment)")
except ImportError:
    # Try manual parse as fallback — no python-dotenv needed
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and not os.getenv(key):
                    os.environ[key] = val
        print("  📁 .env file loaded manually")

PASS = "✅"
FAIL = "❌"
WARN = "⚠️"

results = []

def check(name, passed, detail=""):
    icon = PASS if passed else FAIL
    results.append((icon, name, detail))
    print(f"  {icon}  {name}")
    if detail:
        print(f"       {detail}")

def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


# ── 1. Python version ────────────────────────────────────────
section("1. Python Environment")
v = sys.version_info
check("Python version",
      v.major == 3 and v.minor >= 10,
      f"Python {v.major}.{v.minor}.{v.micro} — need 3.10+")


# ── 2. Required packages ─────────────────────────────────────
section("2. Package Installation")

packages = {
    "streamlit":              ("streamlit",              True),
    "fastapi":                ("fastapi",                True),
    "requests":               ("requests",               True),
    "langchain":              ("langchain",              True),
    "langchain_groq":         ("langchain_groq",         True),
    "langchain_community":    ("langchain_community",    True),
    "chromadb":               ("chromadb",               True),
    "sentence_transformers":  ("sentence_transformers",  True),
    "sklearn":                ("sklearn",                True),
    "langdetect":             ("langdetect",             True),
    "fugashi":                ("fugashi",                False),  # optional
    "unidic":                 ("unidic",                 False),  # optional
    "numpy":                  ("numpy",                  True),
}

missing_required = []
for pkg_name, (import_name, required) in packages.items():
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "?")
        check(f"{pkg_name}", True, f"v{ver}")
    except ImportError:
        if required:
            missing_required.append(pkg_name)
            check(f"{pkg_name}", False, "NOT INSTALLED — run: pip install " + pkg_name)
        else:
            results.append((WARN, pkg_name, "optional — not installed"))
            print(f"  {WARN}  {pkg_name} (optional)")

if missing_required:
    print(f"\n  Run this to fix:")
    print(f"  pip install {' '.join(missing_required)}")


# ── 3. Project files ─────────────────────────────────────────
section("3. Project Files")

required_files = [
    "app.py", "analysis/analyzer.py", "utils/__init__.py",
    "transcription/pii_masker.py", "analysis/hallucination_guard.py",
    "analysis/soft_rejection_detector.py", "analysis/semantic_validator.py",
    "transcription/speaker_normalizer.py", "analysis/japanese_tokenizer.py",
    "utils/language_intelligence.py", "utils/logger.py", "utils/cache.py",
    "rags/meeting_store.py", "rags/rag_retriever.py",
    "utils/evaluator.py", "tests/test_data.py",
    "tests/generate_test_data.py", "tests/test_schema_stability.py",
    "requirements.txt", "Dockerfile", ".gitignore",
]

for f in required_files:
    exists = Path(f).exists()
    check(f, exists, "" if exists else "FILE MISSING")


# ── 4. Environment variables ─────────────────────────────────
section("4. Environment Variables")

groq_key = os.getenv("GROQ_API_KEY", "")
check("GROQ_API_KEY set",
      bool(groq_key),
      f"Key starts with: {groq_key[:8]}..." if groq_key else "NOT SET — mock mode will be used")

provider = os.getenv("TRANSCRIPT_AI_PROVIDER", "auto")
check("TRANSCRIPT_AI_PROVIDER",
      True,
      f"Current value: '{provider}'")


# ── 5. Ollama connection ──────────────────────────────────────
section("5. Ollama (Local AI)")

try:
    import requests as req
    r = req.get("http://localhost:11434/api/tags", timeout=3)
    if r.status_code == 200:
        models = [m["name"] for m in r.json().get("models", [])]
        has_qwen3_8b   = any("qwen3:8b" in m for m in models)
        has_any_qwen   = any("qwen" in m for m in models)
        check("Ollama running", True, f"Models: {models[:4]}")
        if has_qwen3_8b:
            check("qwen3:8b available", True, "Ready")
        elif has_any_qwen:
            qwen_model = next(m for m in models if "qwen" in m)
            check("qwen3:8b available", False,
                  f"You have '{qwen_model}' — run: ollama pull qwen3:8b  OR  set OLLAMA_MODEL={qwen_model} in .env")
        else:
            check("qwen3:8b available", False, "Run: ollama pull qwen3:8b")
    else:
        check("Ollama running", False, f"Status {r.status_code}")
except Exception as e:
    check("Ollama running", False, f"Not reachable — start Ollama app or: ollama serve")


# ── 6. Groq API connection ────────────────────────────────────
section("6. Groq API")

if not groq_key:
    results.append((WARN, "Groq API", "Skipped — no key set"))
    print(f"  {WARN}  Groq API (skipped — no GROQ_API_KEY)")
else:
    try:
        import requests as req
        r = req.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_key}"},
            json={"model": "llama-3.1-8b-instant",
                  "messages": [{"role": "user", "content": "Reply with: OK"}],
                  "max_tokens": 5},
            timeout=15
        )
        if r.status_code == 200:
            reply = r.json()["choices"][0]["message"]["content"]
            check("Groq API connection", True, f"Response: '{reply.strip()}'")
        elif r.status_code == 401:
            check("Groq API connection", False, "Invalid API key")
        elif r.status_code == 429:
            check("Groq API connection", False, "Rate limited — try again in a minute")
        else:
            check("Groq API connection", False, f"Status {r.status_code}: {r.text[:80]}")
    except Exception as e:
        check("Groq API connection", False, str(e)[:80])


# ── 7. Core module imports ────────────────────────────────────
section("7. Core Module Imports")

modules = [
    ("analysis.analyzer",               "analyze_transcript"),
    ("transcription.pii_masker",         "mask_transcript"),
    ("analysis.hallucination_guard",     "verify_result"),
    ("analysis.soft_rejection_detector", "detect_soft_rejections"),
    ("analysis.semantic_validator",      "semantic_similarity"),
    ("transcription.speaker_normalizer", "normalize_speaker_name"),
    ("utils.language_intelligence",      "get_features"),
    ("utils.logger",                     "get_trends"),
    ("utils.cache",                      "get_cached"),
    ("rags.meeting_store",               "store_meeting"),
    ("rags.rag_retriever",               "ask_about_meetings"),
    ("utils.evaluator",                  "evaluate"),
]

for mod_name, func_name in modules:
    try:
        mod  = importlib.import_module(mod_name)
        has  = hasattr(mod, func_name)
        check(f"{mod_name}.{func_name}()",
              has,
              "" if has else f"Function {func_name} not found in module")
    except Exception as e:
        check(f"{mod_name}", False, str(e)[:80])


# ── 8. Quick analysis test ────────────────────────────────────
section("8. Quick Analysis Test (mock mode)")

try:
    os.environ["TRANSCRIPT_AI_PROVIDER"] = "mock"
    from analysis.analyzer import analyze_transcript

    sample = "田中: おはようございます。Q3の報告をします。\n鈴木: 承知しました。検討いたします。"
    result = analyze_transcript(sample, "ja")

    has_summary  = len(result.get("summary", [])) > 0
    has_actions  = "action_items" in result
    has_japan    = "japan_insights" in result
    has_provider = "_provider" in result

    check("Analysis returns summary",     has_summary,  f"{len(result.get('summary',[]))} bullets")
    check("Analysis returns action_items",has_actions,  f"{len(result.get('action_items',[]))} items")
    check("Analysis returns japan_insights", has_japan, f"keigo: {result.get('japan_insights',{}).get('keigo_level','?')}")
    check("Provider tracked",             has_provider, f"Provider: {result.get('_provider','?')}")

    # Restore original provider
    if groq_key:
        os.environ["TRANSCRIPT_AI_PROVIDER"] = "auto"

except Exception as e:
    check("Quick analysis test", False, str(e)[:120])


# ── 9. Soft rejection test ────────────────────────────────────
section("9. Soft Rejection Detector")

try:
    from analysis.soft_rejection_detector import detect_soft_rejections
    test_text = "田中: 検討いたします。難しいかもしれません。善処します。"
    sr = detect_soft_rejections(test_text)
    count = sr.get("total_signals", 0)
    risk  = sr.get("risk_level", "?")
    check("Detects signals",  count >= 2, f"{count} signals, risk: {risk}")
    check("Returns risk level", risk in ("HIGH","MEDIUM","LOW","NONE","MINIMAL"),
          f"Risk: {risk}")
except Exception as e:
    check("Soft rejection detector", False, str(e)[:80])


# ── 10. ChromaDB test ─────────────────────────────────────────
section("10. ChromaDB Storage")

try:
    from rags.meeting_store import store_meeting, get_meeting_count, CHROMADB_AVAILABLE
    if not CHROMADB_AVAILABLE:
        results.append((WARN, "ChromaDB", "Not installed — pip install chromadb"))
        print(f"  {WARN}  ChromaDB not installed")
    else:
        ok  = store_meeting(
            "health_check_test",
            "田中: テストです。",
            {"summary":["test"],"action_items":[],"speakers":[{"name":"田中"}],
             "japan_insights":{"keigo_level":"low"},"soft_rejections":{"risk_level":"NONE"},
             "verification":{"risk_label":"LOW"}},
            "ja"
        )
        cnt = get_meeting_count()
        check("ChromaDB store", ok, f"Total meetings stored: {cnt}")
except Exception as e:
    check("ChromaDB", False, str(e)[:80])


# ── 11. PII masking test ──────────────────────────────────────
section("11. PII Masking")

try:
    from transcription.pii_masker import mask_transcript, restore_pii_in_result
    text   = "田中: 090-1234-5678に電話してください。Email: tanaka@company.co.jp"
    masked, pii = mask_transcript(text)
    phones_masked = "[PHONE_" in masked
    emails_masked = "[EMAIL_" in masked
    names_masked  = "田中" not in masked or "[NAME_" in masked
    restored = pii.restore(masked)
    check("Phone masking",  phones_masked, f"Masked: {masked[:60]}")
    check("Email masking",  emails_masked)
    check("PII restore",    "090-1234-5678" in restored, "Restored correctly")
except Exception as e:
    check("PII masking", False, str(e)[:80])


# ── 12. Logs directory ────────────────────────────────────────
section("12. Directories & Storage")

for d in ["logs", "cache", "chroma_db"]:
    p = Path(d)
    check(f"{d}/ directory",
          p.exists(),
          f"{'exists' if p.exists() else 'will be created on first use'}")


# ── SUMMARY ───────────────────────────────────────────────────
section("SUMMARY")

passed  = sum(1 for icon,_,_ in results if icon == PASS)
failed  = sum(1 for icon,_,_ in results if icon == FAIL)
warned  = sum(1 for icon,_,_ in results if icon == WARN)
total   = passed + failed

print(f"  {PASS} Passed:  {passed}")
print(f"  {FAIL} Failed:  {failed}")
print(f"  {WARN} Warnings: {warned}")
print()

if failed == 0:
    print("  🎉 All checks passed! Run: python -m streamlit run app.py")
else:
    print("  Failed checks:")
    for icon, name, detail in results:
        if icon == FAIL:
            print(f"    ❌ {name}: {detail}")
    print()
    print("  Fix the failures above, then re-run this script.")

print()
print("  Paste this full output to Claude for diagnosis.")
print("=" * 55)