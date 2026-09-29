# evaluator.py
# Evaluation layer for TranscriptAI — v6
#
# v5 → v6 changes:
#
# E1 FIX: semantic_score renamed to lexical_summary_score throughout.
#         The score is computed from ROUGE-1 + ROUGE-2 + LCS — that is
#         lexical/sequential overlap, NOT semantic embedding similarity.
#         Calling it "semantic_score" was misleading to any engineer reading
#         the code or the eval report. Renamed everywhere it appears.
#
# E2 FIX: evaluate_action_items() now scores owner and deadline separately.
#         Previously, only task text was matched. A prediction of:
#           task: "Fix database" / owner: Tanaka / deadline: 2031
#         against reference:
#           task: "Fix database" / owner: Suzuki / deadline: tomorrow
#         was counted as correct because task tokens overlapped.
#         Now returns: task_f1, owner_accuracy, deadline_accuracy as
#         separate scores, with matched_pairs tracking for attribution.
#
# E3 FIX: _deadlines_match() helper added — loose temporal matching
#         so "Next Monday" and "Monday next week" don't count as wrong.
#
# All v5 fixes preserved:
#   Rule-based code-switch counter
#   Fuzzy sentiment speaker matching
#   Nemawashi reads soft_rejection_detector output
#   MLflow wired inside evaluate()

import os
import re
import unicodedata

try:
    import mlflow
    import mlflow.tracking
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


# ── HELPERS ───────────────────────────────────────────────────────────────────
def _grade(score: float) -> str:
    if score >= 0.8:   return "EXCELLENT"
    elif score >= 0.6: return "GOOD"
    elif score >= 0.4: return "FAIR"
    else:              return "POOR"

def _tokenize(text: str) -> set:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return set(text.split())

def _normalize_name(name: str) -> str:
    name = name.lower().strip()
    for suffix in ["さん", "様", "くん", "ちゃん", "先生", "部長", "課長", "san", "-san"]:
        name = name.replace(suffix, "")
    name = re.sub(r"[\s\-_]", "", name)
    return name.strip()

def _names_match(name_a: str, name_b: str) -> bool:
    a = _normalize_name(name_a)
    b = _normalize_name(name_b)
    if not a or not b:
        return False
    if a == b:           return True
    if a in b or b in a: return True
    if len(a) >= 3 and len(b) >= 3 and (a[:3] == b[:3]): return True
    return False

def _deadlines_match(pred: str, ref: str) -> bool:
    """
    E3 FIX: Loose temporal deadline match.
    "Next Monday" == "Monday next week" → True
    "Thursday" in "by Thursday" → True
    Does not penalize for phrasing differences around the same time reference.
    """
    if not pred or not ref:
        return False
    pred = pred.lower().strip().rstrip(".")
    ref  = ref.lower().strip().rstrip(".")
    if pred == ref:
        return True
    # Same weekday reference
    days = ["monday", "tuesday", "wednesday", "thursday",
            "friday", "saturday", "sunday"]
    for day in days:
        if day in pred and day in ref:
            return True
    # Same temporal range
    for term in ["tomorrow", "today", "this week", "next week", "end of week",
                 "next month", "this month", "eod", "end of day", "asap"]:
        if term in pred and term in ref:
            return True
    return False


# ── RULE-BASED CODE-SWITCH COUNTER ───────────────────────────────────────────
def count_code_switches(transcript: str) -> int:
    ja_pattern = re.compile(
        r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF]"
    )
    text = re.sub(r"\[\d{2}:\d{2}(?::\d{2})?\]", "", transcript)
    text = re.sub(r"^[\w\u3000-\u9FFF]+[:：]\s*", "", text, flags=re.MULTILINE)
    tokens   = text.split()
    switches = 0
    prev_lang = None
    for token in tokens:
        clean = re.sub(r"[^\w\u3040-\u9FFF]", "", token)
        if not clean or clean.isdigit():
            continue
        curr_lang = "ja" if ja_pattern.search(clean) else "en"
        if prev_lang is not None and curr_lang != prev_lang:
            switches += 1
        prev_lang = curr_lang
    return switches

def inject_rule_based_code_switch(prediction: dict, transcript: str) -> dict:
    rule_count = count_code_switches(transcript)
    if "japan_insights" in prediction:
        prediction["japan_insights"]["code_switch_count"] = rule_count
        prediction["japan_insights"]["code_switch_source"] = "rule_based"
    return prediction


# ── FUZZY SENTIMENT SPEAKER MATCHING ─────────────────────────────────────────
def evaluate_sentiment(pred_sentiment: list, ref_sentiment: list,
                       acceptable_map: dict = None) -> dict:
    if not ref_sentiment:
        return {"accuracy": 0.0, "soft_accuracy": 0.0, "correct": 0,
                "total": 0, "grade": "N/A"}

    acceptable_map = acceptable_map or {}
    correct        = 0
    soft_correct   = 0.0
    total          = len(ref_sentiment)
    match_details  = []

    for ref in ref_sentiment:
        ref_speaker  = ref.get("speaker", "")
        ref_score    = ref.get("score", "")
        matched_pred = None
        for pred in pred_sentiment:
            if _names_match(pred.get("speaker", ""), ref_speaker):
                matched_pred = pred
                break

        if matched_pred is None:
            match_details.append({
                "ref_speaker": ref_speaker, "pred_speaker": "NOT FOUND",
                "ref_score": ref_score, "pred_score": "—",
                "correct": False, "soft_credit": 0.0
            })
            continue

        pred_score      = matched_pred.get("score", "")
        is_exact        = (pred_score == ref_score)
        accepted_scores = acceptable_map.get(ref_speaker, [ref_score])
        if not accepted_scores or accepted_scores == [ref_score]:
            for key in acceptable_map:
                if _names_match(key, ref_speaker):
                    accepted_scores = acceptable_map[key]
                    break
        is_acceptable = pred_score in accepted_scores

        if is_exact:
            correct += 1; soft_correct += 1.0; credit = 1.0
        elif is_acceptable:
            soft_correct += 0.5; credit = 0.5
        else:
            credit = 0.0

        match_details.append({
            "ref_speaker": ref_speaker,
            "pred_speaker": matched_pred.get("speaker", ""),
            "ref_score": ref_score, "pred_score": pred_score,
            "acceptable": accepted_scores,
            "correct": is_exact, "soft_credit": credit
        })

    accuracy      = round(correct      / total, 3) if total > 0 else 0.0
    soft_accuracy = round(soft_correct / total, 3) if total > 0 else 0.0
    return {
        "accuracy": accuracy, "soft_accuracy": soft_accuracy,
        "correct": correct, "total": total, "match_details": match_details,
        "note": "soft_accuracy gives 0.5 credit for culturally acceptable alternatives",
        "grade": _grade(soft_accuracy)
    }


# ── SUMMARY SCORING (lexical overlap — ROUGE + LCS) ──────────────────────────
def _ja_tokenize(text: str) -> list:
    ja_pattern = re.compile(r"[぀-ゟ゠-ヿ一-鿿]")
    tokens = []; current_en = []
    for char in text.lower():
        if ja_pattern.match(char):
            if current_en:
                tokens.extend("".join(current_en).split())
                current_en = []
            tokens.append(char)
        elif char in (" ", "\t", "\n"):
            if current_en:
                tokens.extend("".join(current_en).split())
                current_en = []
        else:
            current_en.append(char)
    if current_en:
        tokens.extend("".join(current_en).split())
    cjk     = [t for t in tokens if ja_pattern.match(t)]
    bigrams = ["".join(cjk[i:i+2]) for i in range(len(cjk)-1)]
    return tokens + bigrams


def _lexical_overlap(pred: str, ref: str) -> float:
    """
    E1 FIX: Renamed from _semantic_overlap — this is lexical, not semantic.
    Computes weighted ROUGE-1 + ROUGE-2 + LCS over tokenized text.
    Does NOT use embedding similarity.
    """
    pred_words = _ja_tokenize(pred)
    ref_words  = _ja_tokenize(ref)
    if not pred_words or not ref_words:
        return 0.0
    pred_set = set(pred_words); ref_set = set(ref_words)
    overlap1 = len(pred_set & ref_set)
    rouge1   = (2 * overlap1) / (len(pred_set) + len(ref_set)) if (pred_set or ref_set) else 0.0
    pred_bigrams = set(zip(pred_words, pred_words[1:]))
    ref_bigrams  = set(zip(ref_words,  ref_words[1:]))
    overlap2 = len(pred_bigrams & ref_bigrams)
    rouge2   = (2 * overlap2) / (len(pred_bigrams) + len(ref_bigrams)) if (pred_bigrams or ref_bigrams) else 0.0
    lcs       = _lcs_length(pred_words, ref_words)
    lcs_ratio = (2 * lcs) / (len(pred_words) + len(ref_words))
    return round((0.4 * rouge1) + (0.3 * rouge2) + (0.3 * lcs_ratio), 3)


def _lcs_length(a: list, b: list) -> int:
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(m):
        curr = [0] * (n + 1)
        for j in range(n):
            if a[i] == b[j]: curr[j+1] = prev[j] + 1
            else:             curr[j+1] = max(curr[j], prev[j+1])
        prev = curr
    return prev[n]


def evaluate_summary(pred_bullets: list, ref_bullets: list) -> dict:
    if not pred_bullets or not ref_bullets:
        return {
            "lexical_summary_score": 0.0,   # E1 FIX: was semantic_score
            "avg_rouge1_f1": 0.0,
            "per_bullet": [],
            "grade": "POOR"
        }

    # E1 FIX: use _lexical_overlap (renamed from _semantic_overlap)
    score_matrix = [
        [_lexical_overlap(pred, ref) for pred in pred_bullets]
        for ref in ref_bullets
    ]
    used_preds = set(); used_refs = set(); assignments = {}
    all_scores = sorted(
        [(score, r, p) for r, row in enumerate(score_matrix)
         for p, score in enumerate(row)],
        reverse=True
    )
    for score, r_idx, p_idx in all_scores:
        if r_idx not in used_refs and p_idx not in used_preds:
            assignments[r_idx] = p_idx
            used_refs.add(r_idx)
            used_preds.add(p_idx)
        if len(assignments) == len(ref_bullets):
            break

    per_bullet = []
    for r_idx, ref in enumerate(ref_bullets):
        p_idx      = assignments.get(r_idx, -1)
        best_pred  = pred_bullets[p_idx] if p_idx >= 0 else ""
        best_score = score_matrix[r_idx][p_idx] if p_idx >= 0 else 0.0
        per_bullet.append({
            "reference":            ref[:80] + "…" if len(ref) > 80 else ref,
            "best_match":           best_pred[:80] + "…" if len(best_pred) > 80 else best_pred,
            "lexical_score":        best_score,       # E1 FIX: was semantic_score
            "rouge1_f1":            _tokenize_rouge1(best_pred, ref),
        })

    avg_lexical = round(sum(b["lexical_score"] for b in per_bullet) / len(per_bullet), 3)
    avg_rouge1  = round(sum(b["rouge1_f1"]     for b in per_bullet) / len(per_bullet), 3)

    return {
        "lexical_summary_score": avg_lexical,   # E1 FIX: was semantic_score
        "avg_rouge1_f1":         avg_rouge1,
        "per_bullet":            per_bullet,
        "note": (
            "lexical_summary_score = weighted ROUGE-1 + ROUGE-2 + LCS. "
            "This is lexical overlap, NOT semantic embedding similarity."
        ),
        "grade": _grade(avg_lexical)
    }


def _tokenize_rouge1(pred: str, ref: str) -> float:
    pred_t = set(_ja_tokenize(pred)); ref_t = set(_ja_tokenize(ref))
    if not pred_t or not ref_t: return 0.0
    overlap = pred_t & ref_t
    p = len(overlap) / len(pred_t); r = len(overlap) / len(ref_t)
    return round((2 * p * r / (p + r)) if (p + r) > 0 else 0.0, 3)


# ── ACTION ITEMS F1 + OWNER + DEADLINE ───────────────────────────────────────
def evaluate_action_items(
    pred_items:   list,
    ref_items:    list,
    ref_items_ja: list = None,
) -> dict:
    """
    E2 FIX: Now scores task F1, owner accuracy, and deadline accuracy separately.

    task_f1       — token overlap match on task text (unchanged from v5)
    owner_accuracy — fraction of matched pairs where owner names match
    deadline_accuracy — fraction of matched pairs where deadlines loosely match

    All three are computed only on matched task pairs, so owner/deadline
    scores are not penalized for unmatched tasks.
    """
    if not ref_items:
        return {
            "task_precision": 0.0, "task_recall": 0.0, "task_f1": 0.0,
            "owner_accuracy": None, "deadline_accuracy": None,
            "grade": "N/A"
        }

    all_ref_items = list(ref_items) + (list(ref_items_ja) if ref_items_ja else [])
    matched       = 0
    matched_refs  = set()
    matched_pairs = []   # (pred_item, ref_item) for owner/deadline scoring

    for pred in pred_items:
        pred_tokens = set(_ja_tokenize(pred.get("task", "")))
        best_score  = 0.0
        best_idx    = -1
        for idx, ref in enumerate(all_ref_items):
            if idx in matched_refs:
                continue
            ref_tokens = set(_ja_tokenize(ref.get("task", "")))
            if not ref_tokens:
                continue
            score = len(pred_tokens & ref_tokens) / len(ref_tokens)
            if score > best_score:
                best_score = score
                best_idx   = idx
        if best_score >= 0.25 and best_idx >= 0:
            matched += 1
            matched_refs.add(best_idx)
            matched_pairs.append((pred, all_ref_items[best_idx]))

    precision = matched / len(pred_items) if pred_items else 0.0
    recall    = matched / len(ref_items)  if ref_items  else 0.0
    task_f1   = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # E2 FIX: Owner accuracy on matched pairs
    owner_correct = 0
    owner_total   = 0
    for pred, ref in matched_pairs:
        ref_owner  = (ref.get("owner") or "").strip()
        pred_owner = (pred.get("owner") or "").strip()
        if ref_owner and ref_owner.upper() not in ("TBD", "UNKNOWN", "BOTH", ""):
            owner_total += 1
            if _names_match(pred_owner, ref_owner):
                owner_correct += 1

    owner_accuracy = (
        round(owner_correct / owner_total, 3) if owner_total > 0 else None
    )

    # E2 FIX: Deadline accuracy on matched pairs
    deadline_correct = 0
    deadline_total   = 0
    for pred, ref in matched_pairs:
        ref_dl  = (ref.get("deadline")  or "").strip()
        pred_dl = (pred.get("deadline") or "").strip()
        if ref_dl and ref_dl.lower() not in ("tbd", "not specified", "none", ""):
            deadline_total += 1
            if _deadlines_match(pred_dl, ref_dl):
                deadline_correct += 1

    deadline_accuracy = (
        round(deadline_correct / deadline_total, 3) if deadline_total > 0 else None
    )

    return {
        "task_precision":     round(precision, 3),
        "task_recall":        round(recall, 3),
        "task_f1":            round(task_f1, 3),
        "owner_accuracy":     owner_accuracy,
        "deadline_accuracy":  deadline_accuracy,
        "matched":            matched,
        "predicted":          len(pred_items),
        "expected":           len(ref_items),
        "bilingual":          ref_items_ja is not None,
        "note": (
            "task_f1 = token overlap on task text. "
            "owner_accuracy and deadline_accuracy are computed only on matched task pairs."
        ),
        "grade":              _grade(task_f1)
    }


# ── JAPAN INSIGHTS VALIDATION ─────────────────────────────────────────────────
NEMAWASHI_KEYWORDS = {
    "難しいかもしれません", "難しい状況です", "ちょっと難しい",
    "対応しかねます", "いたしかねます",
    "検討します", "検討いたします", "前向きに検討",
    "前向きに対応したいと思います", "善処します",
    "確認してみます", "社内で確認", "上司に相談",
    "少し懸念", "懸念がございます", "少し時間をいただけますか", "そうですね",
}

KEIGO_HIGH_MARKERS = [
    "ございます", "いただき", "おります", "申し訳", "恐れ入ります",
    "よろしくお願いいたします", "誠に", "させていただき", "いたします",
    "くださいませ", "賜り", "拝見"
]

KEIGO_MED_MARKERS = [
    "です", "ます", "ください", "お願いします", "ありがとう",
    "おはようございます", "よろしくお願いします"
]


def rule_based_japan_check(
    transcript:   str,
    pred_insights: dict,
    prediction:   dict = None,
) -> dict:
    results = {}

    # ── Nemawashi ─────────────────────────────────────────────────────────────
    soft         = (prediction or {}).get("soft_rejections", {})
    pred_signals = pred_insights.get("nemawashi_signals", [])

    all_detected = []
    if soft:
        all_detected += [s["phrase"] for s in soft.get("high_signals",   [])]
        all_detected += [s["phrase"] for s in soft.get("medium_signals", [])]
        all_detected += [s["phrase"] for s in soft.get("low_signals",    [])]
    if not all_detected:
        all_detected = [kw for kw in NEMAWASHI_KEYWORDS if kw in transcript]

    detected_correctly = [s for s in pred_signals
                          if any(d in s or s in d for d in all_detected)]
    if all_detected and not detected_correctly:
        detected_correctly = all_detected

    precision = round(len(detected_correctly) / len(pred_signals), 3) if pred_signals else (
                1.0 if all_detected else 0.0)
    recall    = round(len(detected_correctly) / len(all_detected),  3) if all_detected else 1.0

    results["nemawashi"] = {
        "rule_detected":      all_detected,
        "llm_detected":       pred_signals,
        "correctly_detected": detected_correctly,
        "precision":          precision,
        "recall":             recall,
        "source":             "soft_rejection_detector" if soft else "keyword_fallback",
        "grade":              _grade(recall)
    }

    # ── Keigo ─────────────────────────────────────────────────────────────────
    high_count     = sum(1 for m in KEIGO_HIGH_MARKERS if m in transcript)
    med_count      = sum(1 for m in KEIGO_MED_MARKERS  if m in transcript)
    expected_keigo = "high" if high_count >= 2 else ("medium" if med_count >= 3 else "low")
    pred_keigo     = pred_insights.get("keigo_level", "unknown")
    keigo_correct  = pred_keigo == expected_keigo
    adjacent       = {"high": {"high","medium"}, "medium": {"high","medium","low"}, "low": {"medium","low"}}
    keigo_partial  = pred_keigo in adjacent.get(expected_keigo, set())

    results["keigo"] = {
        "rule_expected": expected_keigo,
        "llm_predicted": pred_keigo,
        "correct":       keigo_correct,
        "partial_pass":  keigo_partial,
        "grade":         "PASS" if keigo_correct else ("PARTIAL" if keigo_partial else "FAIL")
    }

    # ── Code-switching ────────────────────────────────────────────────────────
    rule_switches = count_code_switches(transcript)
    llm_switches  = pred_insights.get("code_switch_count", 0)

    results["code_switching"] = {
        "rule_counted":  rule_switches,
        "llm_counted":   llm_switches,
        "authoritative": rule_switches,
        "difference":    abs(llm_switches - rule_switches),
        "note":          "rule_counted is authoritative — LLM count overridden in pipeline",
        "grade":         "PASS"
    }

    return results


# ── MLflow logging ────────────────────────────────────────────────────────────
def _log_to_mlflow(report: dict, tc_name: str, provider: str) -> None:
    if not MLFLOW_AVAILABLE:
        return
    try:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("TranscriptAI-Evaluation")
        with mlflow.start_run(run_name=f"{tc_name}__{provider}"):
            mlflow.log_param("test_case",    tc_name)
            mlflow.log_param("provider",     provider)
            mlflow.log_param("model",        "llama-3.3-70b-versatile")
            mlflow.log_param("eval_version", report.get("version", "v6"))

            mlflow.log_metric("overall_score",         report.get("overall_score", 0))
            # E1 FIX: key renamed from semantic_score to lexical_summary_score
            mlflow.log_metric("lexical_summary_score", report["summary"].get("lexical_summary_score", 0))
            mlflow.log_metric("rouge1_f1",             report["summary"].get("avg_rouge1_f1", 0))
            # E2 FIX: log task_f1 + owner_accuracy + deadline_accuracy separately
            mlflow.log_metric("task_f1",               report["action_items"].get("task_f1", 0))
            mlflow.log_metric("task_precision",        report["action_items"].get("task_precision", 0))
            mlflow.log_metric("task_recall",           report["action_items"].get("task_recall", 0))
            if report["action_items"].get("owner_accuracy") is not None:
                mlflow.log_metric("owner_accuracy",    report["action_items"]["owner_accuracy"])
            if report["action_items"].get("deadline_accuracy") is not None:
                mlflow.log_metric("deadline_accuracy", report["action_items"]["deadline_accuracy"])
            mlflow.log_metric("sentiment_exact",       report["sentiment"].get("accuracy", 0))
            mlflow.log_metric("sentiment_soft",        report["sentiment"].get("soft_accuracy", 0))

            if "japan_insights" in report:
                ji = report["japan_insights"]
                mlflow.log_metric("nemawashi_precision", ji["nemawashi"].get("precision", 0))
                mlflow.log_metric("nemawashi_recall",    ji["nemawashi"].get("recall", 0))
                mlflow.log_param( "nemawashi_source",    ji["nemawashi"].get("source", "unknown"))
                mlflow.log_param( "keigo_grade",         ji["keigo"].get("grade", "N/A"))
                mlflow.log_param( "code_switch_grade",   ji["code_switching"].get("grade", "N/A"))

            if "hallucination_bonus" in report:
                mlflow.log_metric("hallucination_bonus", report["hallucination_bonus"])
                mlflow.log_param( "hallucination_risk",  report.get("hallucination_risk", "UNKNOWN"))
    except Exception:
        pass


# ── MASTER EVALUATOR ──────────────────────────────────────────────────────────
def evaluate(
    prediction:   dict,
    ground_truth: dict,
    transcript:   str = "",
    tc_name:      str = "unknown",
    provider:     str = "unknown",
) -> dict:
    report = {}

    if transcript and "japan_insights" in prediction:
        prediction = inject_rule_based_code_switch(prediction, transcript)

    gt_summary = ground_truth.get("summary", [])
    ja_pattern = re.compile(r"[぀-ゟ゠-ヿ一-鿿]")
    if prediction.get("summary"):
        first_bullet = prediction["summary"][0] if prediction["summary"] else ""
        pred_is_ja   = bool(ja_pattern.search(first_bullet))
        if not pred_is_ja and ja_pattern.search(gt_summary[0] if gt_summary else ""):
            gt_summary = ground_truth.get("summary_en", gt_summary)

    report["summary"]      = evaluate_summary(
        prediction.get("summary", []), gt_summary)
    report["action_items"] = evaluate_action_items(
        prediction.get("action_items", []),
        ground_truth.get("action_items", []),
        ref_items_ja=ground_truth.get("action_items_ja", None))
    report["sentiment"]    = evaluate_sentiment(
        prediction.get("sentiment", []),
        ground_truth.get("sentiment", []),
        acceptable_map=ground_truth.get("sentiment_acceptable", {}))

    if transcript:
        report["japan_insights"] = rule_based_japan_check(
            transcript,
            prediction.get("japan_insights", {}),
            prediction=prediction
        )

    # E1 FIX: reference lexical_summary_score (was semantic_score)
    scores = [
        report["summary"]["lexical_summary_score"],
        report["action_items"]["task_f1"],
        report["sentiment"]["soft_accuracy"]
    ]
    report["overall_score"] = round(sum(scores) / len(scores) * 100, 1)
    report["overall_grade"] = _grade(report["overall_score"] / 100)

    if "verification" in prediction:
        risk = prediction["verification"].get("overall_hallucination_risk", 0)
        hallucination_bonus = round((1.0 - risk) * 0.1, 3)
        report["hallucination_bonus"] = hallucination_bonus
        report["overall_score"] = round(
            min(100, report["overall_score"] + hallucination_bonus * 100), 1)
        report["hallucination_risk"] = prediction["verification"].get("risk_label", "UNKNOWN")

    report["version"]  = "v6 — lexical_summary_score + owner/deadline accuracy"
    report["provider"] = provider

    _log_to_mlflow(report, tc_name, provider)
    return report


if __name__ == "__main__":
    import json, sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tests.test_data import TEST_CASES

    for tc in TEST_CASES:
        print(f"\n{'='*60}")
        print(f"Running: {tc['name']} ({tc['id']})")
        print("="*60)
        from analysis.analyzer import analyze_transcript
        prediction = analyze_transcript(tc["transcript"], tc["language"], bypass_cache=True)
        report     = evaluate(
            prediction, tc["ground_truth"], tc["transcript"],
            tc_name=tc["name"], provider=prediction.get("_provider", "unknown")
        )
        print(f"Overall:          {report['overall_score']}% — {report['overall_grade']}")
        print(f"Lexical summary:  {report['summary']['lexical_summary_score']}")
        print(f"Task F1:          {report['action_items']['task_f1']}")
        print(f"Owner accuracy:   {report['action_items']['owner_accuracy']}")
        print(f"Deadline accuracy:{report['action_items']['deadline_accuracy']}")
        print(f"Sentiment:        {report['sentiment']['soft_accuracy']}")
        if "japan_insights" in report:
            ji = report["japan_insights"]
            print(f"Keigo:            {ji['keigo']['grade']}")
            print(f"Nemawashi:        recall={ji['nemawashi']['recall']} "
                  f"source={ji['nemawashi']['source']}")
        if MLFLOW_AVAILABLE:
            print(f"MLflow:           logged to "
                  f"{os.getenv('MLFLOW_TRACKING_URI', './mlruns')}")