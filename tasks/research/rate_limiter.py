"""Design a rate limiter for an API spec; produce pseudocode + tradeoffs."""

from __future__ import annotations

from pathlib import Path

from tasks.registry import Score, Task, Trajectory


API_SPEC = """# Payments API — Rate Limit Requirements

- Each API key has a limit of 100 requests per minute.
- Burst allowed: up to 20 requests in any 1-second window.
- Exceeding either limit returns HTTP 429.
- The API is deployed across 8 stateless edge nodes behind a round-robin load
  balancer. Any request can land on any node. Redis is available as shared
  state.
- Latency budget per request: p99 < 20ms including the rate-limit check.
- Must be fair under contention (no single tenant can starve others).
"""

PROMPT = (
    "Read `api_spec.md` in the scratch directory. Design a rate limiter that "
    "satisfies every requirement. Produce: (1) pseudocode for the check "
    "function running on an edge node, (2) a bulleted list of at least 3 "
    "design tradeoffs you considered (e.g., token bucket vs sliding window, "
    "Redis RTT vs local cache, drift under clock skew), and (3) one concrete "
    "failure mode and its mitigation. Save to `design.md` AND pass the full "
    "text as the `answer` argument to `finish`."
)


RUBRIC = """Score on a 1-5 scale:
- 5: Picks a sensible algorithm (e.g., sliding window log or token bucket with atomic Redis ops / Lua) that handles BOTH the per-minute and per-second limits; pseudocode is concrete and correct; discusses Redis RTT vs latency budget; mentions fairness (e.g., per-key key-space); identifies a real failure mode with a mitigation.
- 4: Correct algorithm and most tradeoffs; one minor gap (e.g., doesn't address both limits, or weak on fairness).
- 3: Plausible design but at least one requirement unaddressed or pseudocode is vague.
- 2: Serious gap (e.g., ignores distributed constraint, or picks algorithm that can't enforce both limits).
- 1: Wrong or missing."""


def _evaluator(traj: Trajectory, scratch: Path) -> Score:
    answer = (traj.final_answer or "").strip() if traj else ""
    return Score(
        passed=bool(answer),
        score=0.0,
        details=f"answer_len={len(answer)}",
    )


TASK = Task(
    id="research_rate_limiter",
    category="research",
    prompt=PROMPT,
    scratch_files={"api_spec.md": API_SPEC},
    evaluator=_evaluator,
)
TASK.rubric = RUBRIC  # type: ignore[attr-defined]
