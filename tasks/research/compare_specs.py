"""Compare three small tech specs and produce a 200-word summary."""

from __future__ import annotations

from pathlib import Path

from tasks.registry import Score, Task, Trajectory


SPEC_A = """# Spec A: SimpleCache
- In-memory key-value store
- Single-node only, no replication
- TTL per key, max 1M entries
- Synchronous API, written in Go
- Pricing: free, self-hosted
"""

SPEC_B = """# Spec B: DistCache
- Distributed key-value store, 3-node minimum
- Strong consistency via Raft consensus
- TTL per key, unlimited entries (disk-backed)
- Synchronous + async APIs, written in Rust
- Pricing: self-hosted, enterprise support available
"""

SPEC_C = """# Spec C: CloudCache
- Managed serverless cache
- Multi-region replication, eventual consistency
- TTL per key, pay-per-request
- Async API only, multiple SDKs
- Pricing: $0.15 per 1M requests, no minimum
"""

PROMPT = (
    "Read the three specs in `spec_a.md`, `spec_b.md`, and `spec_c.md`. "
    "Write a 150-200 word comparison that helps a reader pick between them. "
    "Address: consistency model, operational burden, cost profile, and typical "
    "use case. Save your comparison to `comparison.md` AND pass it as the "
    "`answer` argument to `finish`."
)


RUBRIC = """Score the answer on a 1-5 scale using these criteria:
- 5: Covers all four dimensions (consistency, operational burden, cost profile, use case) accurately for all 3 specs; actionable recommendation.
- 4: Covers most dimensions accurately; minor omissions.
- 3: Covers the main differences but misses a dimension or misstates a detail.
- 2: Surface-level; significant errors or omissions.
- 1: Wrong or largely missing.
Also penalize if length is far outside 150-200 words."""


def _evaluator(traj: Trajectory, scratch: Path) -> Score:
    # Judged externally by eval/judge.py; this just reports the answer length sanity.
    answer = (traj.final_answer or "").strip() if traj else ""
    return Score(
        passed=bool(answer),
        score=0.0,  # overridden by LLM judge
        details=f"answer_len={len(answer)}",
    )


TASK = Task(
    id="research_compare_specs",
    category="research",
    prompt=PROMPT,
    scratch_files={
        "spec_a.md": SPEC_A,
        "spec_b.md": SPEC_B,
        "spec_c.md": SPEC_C,
    },
    evaluator=_evaluator,
)
TASK.rubric = RUBRIC  # type: ignore[attr-defined]
