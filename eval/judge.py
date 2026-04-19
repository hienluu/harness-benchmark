"""LLM-as-judge for open-ended research tasks.

Judge sees the task prompt, the rubric, and the final answer only —
not the trajectory. This keeps outcome evaluation independent of process
(the harness shouldn't be judged on how it got there, only on what it produced).
"""

from __future__ import annotations

import json
import re

import anthropic

from harnesses.common import MODEL


JUDGE_SYSTEM = (
    "You are an impartial evaluator. Given a task, a scoring rubric, and a "
    "candidate answer, return a JSON object with fields `score` (integer "
    "1-5) and `rationale` (1-3 sentences explaining the score). "
    "Respond with ONLY the JSON object, no prose."
)


def judge(task_prompt: str, rubric: str, answer: str) -> tuple[int, str]:
    client = anthropic.Anthropic()
    user = (
        f"TASK:\n{task_prompt}\n\n"
        f"RUBRIC:\n{rubric}\n\n"
        f"CANDIDATE ANSWER:\n{answer or '<empty>'}"
    )
    resp = client.messages.create(
        model=MODEL,
        max_tokens=512,
        temperature=0.0,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return 1, f"unparseable judge output: {text[:200]}"
    try:
        obj = json.loads(m.group(0))
        score = int(obj.get("score", 1))
        rationale = str(obj.get("rationale", ""))
        return max(1, min(5, score)), rationale
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        return 1, f"judge parse error: {e}"


def score_to_normalized(judge_score: int) -> float:
    """Map 1-5 judge score onto 0.0-1.0 for aggregation."""
    return (judge_score - 1) / 4.0
