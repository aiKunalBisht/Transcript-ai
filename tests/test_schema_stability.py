"""
test_schema_stability.py
========================
Measures real JSON schema stability across multiple analysis runs.
Run this to get the honest number for your resume.

Usage:
    python test_schema_stability.py

Outputs:
    - Pass/fail per run
    - Schema stability percentage (the real number)
    - Field-level stability breakdown
"""

import json
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env so GROQ_API_KEY is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    from pathlib import Path
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip() and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                k = k.strip(); v = v.strip().strip('"').strip("'")
                if k and not os.getenv(k):
                    os.environ[k] = v

from analysis.analyzer import analyze_transcript

REQUIRED_FIELDS = ["summary", "action_items", "sentiment", "speakers", "japan_insights"]
REQUIRED_ACTION_FIELDS = ["task", "owner", "deadline"]
REQUIRED_JAPAN_FIELDS  = ["keigo_level", "nemawashi_signals", "code_switch_count"]

# 10 diverse test transcripts covering different languages/formats
TEST_TRANSCRIPTS = [
    ("田中: Q3の進捗を報告します。売上は95%です。鈴木: 承知しました。", "ja"),
    ("Sarah: Let's review the timeline. Mike: I'll have it done by Friday.", "en"),
    ("Tanaka: Good morning. 本日はよろしくお願いします。Sato: こちらこそ。", "mixed"),
    ("田中: 検討いたします。難しいかもしれません。鈴木: 承知しました。", "ja"),
    ("Priya: The API is ready. Kenji: 了解しました。I'll test it today.", "mixed"),
    ("鈴木: 予算の件ですが、社内で確認が必要です。田中: いつ頃ご回答いただけますか？", "ja"),
    ("Mike: The client wants a decision by Monday. Sarah: I'll escalate today.", "en"),
    ("Tanaka (Director): セキュリティについて懸念があります。Sato: 対応いたします。", "mixed"),
    ("田中: 来週月曜までにレポートをお願いします。佐藤: かしこまりました。", "ja"),
    ("Sarah: Good work everyone. Kenji: Thank you. 引き続きよろしくお願いします。", "mixed"),
]


def check_schema(result: dict) -> dict:
    """Check if result has all required fields with correct types."""
    issues = []

    # Top-level fields
    for field in REQUIRED_FIELDS:
        if field not in result:
            issues.append(f"Missing top-level field: {field}")
        elif field in ("summary", "action_items", "sentiment", "speakers"):
            if not isinstance(result[field], list):
                issues.append(f"{field} is not a list")

    # Japan insights sub-fields
    ji = result.get("japan_insights", {})
    for field in REQUIRED_JAPAN_FIELDS:
        if field not in ji:
            issues.append(f"Missing japan_insights.{field}")

    # Action items structure
    for i, item in enumerate(result.get("action_items", [])):
        for field in REQUIRED_ACTION_FIELDS:
            if field not in item:
                issues.append(f"action_items[{i}] missing {field}")

    # Speakers structure
    for i, spk in enumerate(result.get("speakers", [])):
        if "name" not in spk:
            issues.append(f"speakers[{i}] missing name")
        if "talk_time_pct" not in spk:
            issues.append(f"speakers[{i}] missing talk_time_pct")

    # Summary is non-empty
    if isinstance(result.get("summary"), list) and len(result["summary"]) == 0:
        issues.append("summary is empty list")

    return {
        "valid":  len(issues) == 0,
        "issues": issues,
    }


def run_stability_test(n_runs: int = None) -> dict:
    """Run schema stability test across all transcripts."""
    transcripts = TEST_TRANSCRIPTS * (2 if n_runs and n_runs > len(TEST_TRANSCRIPTS) else 1)
    if n_runs:
        transcripts = transcripts[:n_runs]

    print(f"Running schema stability test across {len(transcripts)} analyses...")
    print("=" * 55)

    results       = []
    field_failures = {}

    for i, (transcript, lang) in enumerate(transcripts):
        print(f"  Run {i+1:2d}/{len(transcripts)} [{lang}]... ", end="", flush=True)
        start = time.time()

        try:
            result = analyze_transcript(transcript, lang)
            check  = check_schema(result)
            elapsed = round((time.time() - start) * 1000)

            results.append({
                "run":      i + 1,
                "language": lang,
                "valid":    check["valid"],
                "issues":   check["issues"],
                "ms":       elapsed,
                "provider": result.get("_provider", "unknown"),
            })

            if check["valid"]:
                print(f"✅ ({result.get('_provider','?')}, {elapsed}ms)")
            else:
                print(f"❌ ISSUES: {check['issues'][:1]}")
                for issue in check["issues"]:
                    field_failures[issue] = field_failures.get(issue, 0) + 1

        except Exception as e:
            results.append({
                "run": i+1, "language": lang, "valid": False,
                "issues": [f"Exception: {str(e)[:60]}"], "ms": 0, "provider": "error"
            })
            print(f"💥 Exception: {str(e)[:40]}")

    # Calculate stats
    total    = len(results)
    passed   = sum(1 for r in results if r["valid"])
    pct      = round(passed / total * 100, 1)
    avg_ms   = round(sum(r["ms"] for r in results) / total)
    providers = {}
    for r in results:
        p = r["provider"]
        providers[p] = providers.get(p, 0) + 1

    print("\n" + "=" * 55)
    print(f"SCHEMA STABILITY: {passed}/{total} = {pct}%")
    print(f"Average latency:  {avg_ms}ms")
    print(f"Provider breakdown: {providers}")

    if field_failures:
        print(f"\nField-level failures:")
        for issue, count in sorted(field_failures.items(), key=lambda x: -x[1]):
            print(f"  {count}x {issue}")

    print("=" * 55)
    print(f"\n✅ RESUME CLAIM: '{pct}% JSON schema stability across {total} test runs'")

    return {
        "total":     total,
        "passed":    passed,
        "stability_pct": pct,
        "avg_ms":    avg_ms,
        "providers": providers,
        "failures":  field_failures,
        "runs":      results,
    }


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    report = run_stability_test(n)
    
    # Save report
    with open("schema_stability_report.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull report saved to schema_stability_report.json")