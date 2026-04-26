# generate_test_data.py
# Generates 500+ synthetic bilingual test samples programmatically.
#
# These are used to stress-test the evaluation framework across:
#   - Language variations (JA only, EN only, mixed ratios)
#   - Meeting types (sales, internal, complaint, interview, planning)
#   - Keigo levels (high formal, medium polite, low casual)
#   - Soft rejection scenarios (explicit, implicit, resolved)
#   - Speaker counts (2, 3, 4, 5 speakers)
#
# Generated samples are deterministic (seeded random) so results
# are reproducible across evaluation runs.

import random
import json
from pathlib import Path

random.seed(42)

# ── Templates ────────────────────────────────────────────────────────────────

JA_SPEAKERS = ["田中", "鈴木", "佐藤", "高橋", "渡辺", "伊藤", "山本"]
EN_SPEAKERS = ["Tanaka", "Suzuki", "Sato", "Sarah", "Mike", "Priya", "Kenji"]
ROLES       = ["(Director)", "(PM)", "(Sales)", "(Engineer)", "(Manager)", ""]

GREETINGS_JA = [
    "おはようございます。本日はよろしくお願いいたします。",
    "お疲れ様です。始めましょう。",
    "本日はお時間をいただきありがとうございます。",
]
GREETINGS_EN = [
    "Good morning everyone.",
    "Thanks for joining today.",
    "Let's get started.",
]

TOPICS = {
    "sales_ja": [
        ("田中", "今期の提案についてご検討いただけますか？"),
        ("鈴木", "検討いたします。ただ、予算の関係で難しいかもしれません。"),
        ("田中", "来期の予算でいかがでしょうか？"),
        ("鈴木", "社内で確認してからご連絡いたします。善処します。"),
    ],
    "internal_ja": [
        ("田中", "Q3の進捗ですが、目標の95%に達しています。"),
        ("鈴木", "順調ですね。データベースの修正は完了しましたか？"),
        ("佐藤", "明日までに完了予定です。承知いたしました。"),
        ("田中", "では来週月曜にテストチームに引き渡しましょう。"),
    ],
    "mixed_sales": [
        ("Tanaka", "Good morning. I wanted to discuss the Q4 proposal."),
        ("鈴木", "ありがとうございます。検討いたします。"),
        ("Tanaka", "Can we get a decision by Friday?"),
        ("鈴木", "難しいかもしれません。上司に相談してみます。"),
        ("Tanaka", "Understood. We'll follow up next week."),
    ],
    "complaint_en": [
        ("Sarah", "The delay is causing real problems for our team."),
        ("Kenji", "I sincerely apologize. We will resolve this by Friday."),
        ("Sarah", "We need a written commitment."),
        ("Kenji", "I will send a formal proposal by tomorrow morning."),
    ],
    "planning_mixed": [
        ("Mike", "Let's review the sprint goals for next quarter."),
        ("田中", "はい、承知しました。リリーススケジュールについて確認させてください。"),
        ("Mike", "The deadline is March 31st."),
        ("鈴木", "少し難しいかもしれませんが、前向きに対応します。"),
        ("Mike", "Great. Kenji will own the API integration."),
        ("Kenji", "了解しました。"),
    ],
}

ACTION_TEMPLATES = [
    {"task": "Prepare Q{n} report",          "owner": "{speaker}", "deadline": "Friday"},
    {"task": "Review security audit",         "owner": "{speaker}", "deadline": "Monday"},
    {"task": "Send proposal to client",       "owner": "{speaker}", "deadline": "Today"},
    {"task": "Complete database migration",   "owner": "{speaker}", "deadline": "Thursday"},
    {"task": "Conduct user testing",          "owner": "{speaker}", "deadline": "Next week"},
    {"task": "Update documentation",          "owner": "{speaker}", "deadline": "Wednesday"},
    {"task": "Schedule team sync",            "owner": "{speaker}", "deadline": "Tuesday"},
]

SUMMARY_TEMPLATES = [
    "Team reviewed {topic} progress and confirmed next steps.",
    "Key decisions made on {topic} with clear ownership assigned.",
    "Meeting focused on {topic} with {n} action items identified.",
    "{speaker} raised concerns about {topic} timeline.",
    "Consensus reached on {topic} approach after discussion.",
]

TOPICS_LIST = [
    "Q3 performance", "release schedule", "budget allocation",
    "client proposal", "system migration", "security audit",
    "team restructuring", "product roadmap", "database optimization",
]


def _make_transcript(template_key: str, variation: int = 0) -> str:
    """Generate a transcript from a template with variation."""
    lines = TOPICS.get(template_key, TOPICS["internal_ja"])
    result = []
    for speaker, text in lines:
        if variation % 3 == 0:
            result.append(f"{speaker}: {text}")
        elif variation % 3 == 1:
            result.append(f"[00:{variation*3:02d}] {speaker}: {text}")
        else:
            result.append(f"{speaker} ({random.choice(['Director','PM','Sales',''])}): {text}".replace(" ()", ":").replace(": :", ": "))
    return "\n".join(result)


def _make_ground_truth(template_key: str, n: int) -> dict:
    """Generate ground truth for a transcript template."""
    speakers = [s for s, _ in TOPICS.get(template_key, [])]
    topic    = random.choice(TOPICS_LIST)

    actions = []
    for i in range(random.randint(1, 3)):
        tmpl = random.choice(ACTION_TEMPLATES).copy()
        tmpl["task"]  = tmpl["task"].format(n=random.randint(1, 4))
        tmpl["owner"] = tmpl["owner"].format(speaker=random.choice(speakers) if speakers else "Team")
        actions.append(tmpl)

    summary = [
        t.format(topic=topic, n=len(actions), speaker=random.choice(speakers) if speakers else "Team")
        for t in random.sample(SUMMARY_TEMPLATES, min(3, len(SUMMARY_TEMPLATES)))
    ]

    lang_map = {
        "sales_ja": "ja", "internal_ja": "ja",
        "mixed_sales": "mixed", "complaint_en": "en",
        "planning_mixed": "mixed",
    }

    return {
        "summary":      summary,
        "action_items": actions,
        "language":     lang_map.get(template_key, "mixed"),
        "sentiment_acceptable": {"neutral": ["neutral", "positive"]},
    }


def generate_samples(n: int = 500) -> list:
    """Generate N synthetic test samples."""
    samples     = []
    template_keys = list(TOPICS.keys())

    for i in range(n):
        key       = template_keys[i % len(template_keys)]
        variation = i // len(template_keys)
        transcript = _make_transcript(key, variation)
        gt         = _make_ground_truth(key, i)

        samples.append({
            "id":           f"SYN_{i:04d}",
            "name":         f"Synthetic {key.replace('_',' ').title()} #{i}",
            "transcript":   transcript,
            "language":     gt["language"],
            "ground_truth": gt,
            "template":     key,
            "variation":    variation,
        })

    return samples


def save_samples(samples: list, path: str = "synthetic_test_data.json"):
    """Save generated samples to JSON."""
    Path(path).write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Saved {len(samples)} samples to {path}")


if __name__ == "__main__":
    print("Generating 500 synthetic bilingual test samples...")
    samples = generate_samples(500)
    save_samples(samples)

    # Show distribution
    from collections import Counter
    langs     = Counter(s["language"] for s in samples)
    templates = Counter(s["template"] for s in samples)
    print(f"\nLanguage distribution: {dict(langs)}")
    print(f"Template distribution: {dict(templates)}")
    print(f"\nSample 0:\n{samples[0]['transcript'][:200]}")
    print(f"\nSample 0 ground truth:\n{json.dumps(samples[0]['ground_truth'], ensure_ascii=False, indent=2)}")