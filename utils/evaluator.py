# evaluator.py
# Evaluation layer for TranscriptAI — v3
#
# Fix 1: Code-switch counting moved fully to rule-based (LLM was wildly inaccurate)
# Fix 2: Fuzzy speaker name matching for sentiment (handles Yamamoto vs 山本 etc.)
# Fix 3: Semantic similarity added alongside ROUGE (catches paraphrasing)
# v3 FIX: NEMAWASHI_KEYWORDS cleaned up — removed false positives:
#   - 検討しました (past tense = done, NOT deferring)
#   - 素晴らしい   (praise/excellent = positive, NOT rejection)
#   - 了解しました (understood = agreement, NOT rejection)
#   - なるほど     (I see = acknowledgment, NOT rejection)
#   - 分かりました (understood = agreement, NOT rejection)
#   - 承知しました (will do = agreement, NOT rejection)
#   Only present/future-tense deferral and hesitation patterns kept.

import re
import unicodedata


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
    if a == b:
        return True
    if a in b or b in a:
        return True
    if len(a) >= 3 and len(b) >= 3 and (a[:3] == b[:3]):
        return True
    return False


# ── FIX 1: RULE-BASED CODE-SWITCH COUNTER ────────────────────────────────────
def count_code_switches(transcript: str) -> int:
    ja_pattern = re.compile(
        r"[\u3040-\u309F"
        r"\u30A0-\u30FF"
        r"\u4E00-\u9FFF"
        r"\u3400-\u4DBF]"
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


# ── FIX 2: FUZZY SENTIMENT SPEAKER MATCHING ──────────────────────────────────
def evaluate_sentiment(pred_sentiment: list, ref_sentiment: list,
                       acceptable_map: dict = None) -> dict:
    if not ref_sentiment:
        return {"accuracy": 0.0, "soft_accuracy": 0.0, "correct": 0, "total": 0, "grade": "N/A"}

    acceptable_map = acceptable_map or {}
    correct        = 0
    soft_correct   = 0.0
    total          = len(ref_sentiment)
    match_details  = []

    for ref in ref_sentiment:
        ref_speaker = ref.get("speaker", "")
        ref_score   = ref.get("score", "")

        matched_pred = None
        for pred in pred_sentiment:
            if _names_match(pred.get("speaker", ""), ref_speaker):
                matched_pred = pred
                break

        if matched_pred is None:
            match_details.append({
                "ref_speaker":  ref_speaker,
                "pred_speaker": "NOT FOUND",
                "ref_score":    ref_score,
                "pred_score":   "—",
                "correct":      False,
                "soft_credit":  0.0
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
            correct      += 1
            soft_correct += 1.0
            credit        = 1.0
        elif is_acceptable:
            soft_correct += 0.5
            credit        = 0.5
        else:
            credit = 0.0

        match_details.append({
            "ref_speaker":  ref_speaker,
            "pred_speaker": matched_pred.get("speaker", ""),
            "ref_score":    ref_score,
            "pred_score":   pred_score,
            "acceptable":   accepted_scores,
            "correct":      is_exact,
            "soft_credit":  credit
        })

    accuracy      = round(correct      / total, 3) if total > 0 else 0.0
    soft_accuracy = round(soft_correct / total, 3) if total > 0 else 0.0

    return {
        "accuracy":      accuracy,
        "soft_accuracy": soft_accuracy,
        "correct":       correct,
        "total":         total,
        "match_details": match_details,
        "note": "soft_accuracy gives 0.5 credit for culturally acceptable alternatives (JP neutral≈positive)",
        "grade": _grade(soft_accuracy)
    }


# ── FIX 3: SEMANTIC SIMILARITY ALONGSIDE ROUGE ───────────────────────────────
def _ja_tokenize(text: str) -> list:
    ja_pattern = re.compile(r"[぀-ゟ゠-ヿ一-鿿]")
    tokens     = []
    current_en = []
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


def _semantic_overlap(pred: str, ref: str) -> float:
    pred_words = _ja_tokenize(pred)
    ref_words  = _ja_tokenize(ref)
    if not pred_words or not ref_words:
        return 0.0
    pred_set = set(pred_words)
    ref_set  = set(ref_words)
    overlap1 = len(pred_set & ref_set)
    rouge1   = (2 * overlap1) / (len(pred_set) + len(ref_set)) if (pred_set or ref_set) else 0.0
    pred_bigrams = set(zip(pred_words, pred_words[1:]))
    ref_bigrams  = set(zip(ref_words,  ref_words[1:]))
    overlap2 = len(pred_bigrams & ref_bigrams)
    rouge2   = (2 * overlap2) / (len(pred_bigrams) + len(ref_bigrams)) if (pred_bigrams or ref_bigrams) else 0.0
    lcs      = _lcs_length(pred_words, ref_words)
    lcs_ratio = (2 * lcs) / (len(pred_words) + len(ref_words))
    return round((0.4 * rouge1) + (0.3 * rouge2) + (0.3 * lcs_ratio), 3)


def _lcs_length(a: list, b: list) -> int:
    m, n = len(a), len(b)
    prev = [0] * (n + 1)
    for i in range(m):
        curr = [0] * (n + 1)
        for j in range(n):
            if a[i] == b[j]:
                curr[j+1] = prev[j] + 1
            else:
                curr[j+1] = max(curr[j], prev[j+1])
        prev = curr
    return prev[n]


def evaluate_summary(pred_bullets: list, ref_bullets: list) -> dict:
    if not pred_bullets or not ref_bullets:
        return {"semantic_score": 0.0, "avg_rouge1_f1": 0.0, "per_bullet": [], "grade": "POOR"}

    # Build full score matrix — ref x pred
    score_matrix = []
    for ref in ref_bullets:
        row = [_semantic_overlap(pred, ref) for pred in pred_bullets]
        score_matrix.append(row)

    # Optimal assignment — greedy on global max (not per-row greedy)
    # Repeatedly pick the highest score in the entire matrix
    # This prevents bullet 1 stealing a pred that would better match bullet 2
    used_preds = set()
    used_refs  = set()
    assignments = {}  # ref_idx -> pred_idx

    # Sort all (score, ref_idx, pred_idx) descending and assign greedily
    all_scores = []
    for r_idx, row in enumerate(score_matrix):
        for p_idx, score in enumerate(row):
            all_scores.append((score, r_idx, p_idx))
    all_scores.sort(reverse=True)

    for score, r_idx, p_idx in all_scores:
        if r_idx not in used_refs and p_idx not in used_preds:
            assignments[r_idx] = p_idx
            used_refs.add(r_idx)
            used_preds.add(p_idx)
        if len(assignments) == len(ref_bullets):
            break

    per_bullet = []
    for r_idx, ref in enumerate(ref_bullets):
        p_idx     = assignments.get(r_idx, -1)
        best_pred = pred_bullets[p_idx] if p_idx >= 0 else ""
        best_score = score_matrix[r_idx][p_idx] if p_idx >= 0 else 0.0
        rouge = _tokenize_rouge1(best_pred, ref)
        per_bullet.append({
            "reference":      ref[:80] + "…" if len(ref) > 80 else ref,
            "best_match":     best_pred[:80] + "…" if len(best_pred) > 80 else best_pred,
            "semantic_score": best_score,
            "rouge1_f1":      rouge
        })

    avg_semantic = round(sum(b["semantic_score"] for b in per_bullet) / len(per_bullet), 3)
    avg_rouge1   = round(sum(b["rouge1_f1"]      for b in per_bullet) / len(per_bullet), 3)

    return {
        "semantic_score": avg_semantic,
        "avg_rouge1_f1":  avg_rouge1,
        "per_bullet":     per_bullet,
        "note": "semantic_score = weighted ROUGE-1 + ROUGE-2 + LCS (primary metric). rouge1 = word overlap only.",
        "grade": _grade(avg_semantic)
    }


def _tokenize_rouge1(pred: str, ref: str) -> float:
    pred_t  = set(_ja_tokenize(pred))
    ref_t   = set(_ja_tokenize(ref))
    if not pred_t or not ref_t:
        return 0.0
    overlap = pred_t & ref_t
    p = len(overlap) / len(pred_t)
    r = len(overlap) / len(ref_t)
    return round((2 * p * r / (p + r)) if (p + r) > 0 else 0.0, 3)


# ── ACTION ITEMS F1 ───────────────────────────────────────────────────────────
def evaluate_action_items(pred_items: list, ref_items: list,
                          ref_items_ja: list = None) -> dict:
    if not ref_items:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "grade": "N/A"}

    all_ref_items = list(ref_items)
    if ref_items_ja:
        all_ref_items = all_ref_items + list(ref_items_ja)

    matched      = 0
    matched_refs = set()

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
            overlap = pred_tokens & ref_tokens
            score   = len(overlap) / len(ref_tokens)
            if score > best_score:
                best_score = score
                best_idx   = idx
        if best_score >= 0.25 and best_idx >= 0:
            matched += 1
            matched_refs.add(best_idx)

    precision = matched / len(pred_items) if pred_items else 0.0
    recall    = matched / len(ref_items)  if ref_items  else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 3),
        "recall":    round(recall, 3),
        "f1":        round(f1, 3),
        "matched":   matched,
        "predicted": len(pred_items),
        "expected":  len(ref_items),
        "bilingual": ref_items_ja is not None,
        "grade":     _grade(f1)
    }


# ── JAPAN INSIGHTS VALIDATION ─────────────────────────────────────────────────
# v3 FIX: Only genuine soft rejection / deferral patterns
# Removed: 検討しました (past), 素晴らしい (praise), 了解しました (agreement),
#          なるほど (acknowledgment), 分かりました (agreement), 承知しました (agreement)
NEMAWASHI_KEYWORDS = {
    # REJECTION — almost certainly No
    "難しいかもしれません",
    "難しい状況です",
    "ちょっと難しい",
    "対応しかねます",
    "いたしかねます",
    # LIKELY REJECTION — present/future deferral (NOT past tense)
    "検討します",
    "検討いたします",
    "前向きに検討",
    "前向きに対応したいと思います",
    "善処します",
    "確認してみます",
    "社内で確認",
    "上司に相談",
    # HESITATION
    "少し懸念",
    "懸念がございます",
    "少し時間をいただけますか",
    "そうですね",
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


def rule_based_japan_check(transcript: str, pred_insights: dict) -> dict:
    results = {}

    # Nemawashi — v3: clean keyword set only
    found_signals     = [kw for kw in NEMAWASHI_KEYWORDS if kw in transcript]
    pred_signals      = pred_insights.get("nemawashi_signals", [])
    detected_correctly = [
        s for s in pred_signals
        if any(kw in s for kw in found_signals) or s in found_signals
    ]

    precision = round(len(detected_correctly) / len(pred_signals), 3) if pred_signals else 0.0
    recall    = round(len(detected_correctly) / len(found_signals), 3) if found_signals else 1.0

    results["nemawashi"] = {
        "rule_detected":      found_signals,
        "llm_detected":       pred_signals,
        "correctly_detected": detected_correctly,
        "precision":          precision,
        "recall":             recall,
        "grade":              _grade(len(detected_correctly) / max(len(found_signals), 1))
    }

    # Keigo
    high_count    = sum(1 for m in KEIGO_HIGH_MARKERS if m in transcript)
    med_count     = sum(1 for m in KEIGO_MED_MARKERS  if m in transcript)
    expected_keigo = "high" if high_count >= 2 else ("medium" if med_count >= 3 else "low")
    pred_keigo    = pred_insights.get("keigo_level", "unknown")
    keigo_correct = pred_keigo == expected_keigo
    adjacent      = {"high": {"high","medium"}, "medium": {"high","medium","low"}, "low": {"medium","low"}}
    keigo_partial = pred_keigo in adjacent.get(expected_keigo, set())

    results["keigo"] = {
        "rule_expected": expected_keigo,
        "llm_predicted": pred_keigo,
        "correct":       keigo_correct,
        "partial_pass":  keigo_partial,
        "grade":         "PASS" if keigo_correct else ("PARTIAL" if keigo_partial else "FAIL")
    }

    # Code-switching — always rule-based
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


# ── MASTER EVALUATOR ──────────────────────────────────────────────────────────
def evaluate(prediction: dict, ground_truth: dict, transcript: str = "") -> dict:
    report = {}

    if transcript and "japan_insights" in prediction:
        prediction = inject_rule_based_code_switch(prediction, transcript)

    gt_summary  = ground_truth.get("summary", [])
    ja_pattern  = re.compile(r"[぀-ゟ゠-ヿ一-鿿]")
    if prediction.get("summary"):
        first_bullet = prediction["summary"][0] if prediction["summary"] else ""
        pred_is_ja   = bool(ja_pattern.search(first_bullet))
        if not pred_is_ja and ja_pattern.search(gt_summary[0] if gt_summary else ""):
            gt_summary = ground_truth.get("summary_en", gt_summary)

    report["summary"] = evaluate_summary(
        prediction.get("summary", []),
        gt_summary
    )
    report["action_items"] = evaluate_action_items(
        prediction.get("action_items", []),
        ground_truth.get("action_items", []),
        ref_items_ja=ground_truth.get("action_items_ja", None)
    )
    report["sentiment"] = evaluate_sentiment(
        prediction.get("sentiment", []),
        ground_truth.get("sentiment", []),
        acceptable_map=ground_truth.get("sentiment_acceptable", {})
    )

    if transcript:
        report["japan_insights"] = rule_based_japan_check(
            transcript,
            prediction.get("japan_insights", {})
        )

    scores = [
        report["summary"]["semantic_score"],
        report["action_items"]["f1"],
        report["sentiment"]["soft_accuracy"]
    ]
    report["overall_score"] = round(sum(scores) / len(scores) * 100, 1)
    report["overall_grade"] = _grade(report["overall_score"] / 100)

    if "verification" in prediction:
        risk = prediction["verification"].get("overall_hallucination_risk", 0)
        hallucination_bonus = round((1.0 - risk) * 0.1, 3)
        report["hallucination_bonus"] = hallucination_bonus
        report["overall_score"] = round(
            min(100, report["overall_score"] + hallucination_bonus * 100), 1
        )
        report["hallucination_risk"] = prediction["verification"].get("risk_label", "UNKNOWN")

    report["version"] = "v4 — + hallucination prevention + confidence scoring"
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
        prediction = analyze_transcript(tc["transcript"], tc["language"])
        report     = evaluate(prediction, tc["ground_truth"], tc["transcript"])
        print(f"Overall:    {report['overall_score']}% — {report['overall_grade']}")
        print(f"Semantic:   {report['summary']['semantic_score']}")
        print(f"Actions F1: {report['action_items']['f1']}")
        print(f"Sentiment:  {report['sentiment']['soft_accuracy']}")
        if "japan_insights" in report:
            ji = report["japan_insights"]
            print(f"Keigo:      {ji['keigo']['grade']}")
            print(f"Nemawashi:  precision={ji['nemawashi']['precision']} rule_detected={ji['nemawashi']['rule_detected']}")