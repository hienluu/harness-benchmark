"""Root-cause hypothesis from a bug report + stack trace."""

from __future__ import annotations

from pathlib import Path

from tasks.registry import Score, Task, Trajectory


BUG_REPORT = """# Bug Report #842

**Title:** Orders occasionally saved with negative totals after checkout

**Severity:** High
**Environment:** production, EU region

## Description
Starting last Tuesday, ~0.1% of orders in the EU region are saved with a
negative total_amount_cents. US region is unaffected. The affected orders
all went through the "promo_code" code path. Non-promo orders are fine.

## Recent changes
Deployed last Monday:
- Promo code engine rewrite (migrated from Python service to Rust service)
- Added support for stackable promo codes (multiple promos per order)
- EU region rolled out first; US rollout scheduled for next week
"""

STACK_TRACE = """[ERROR] order_service::checkout::apply_promos
  at order_service/src/checkout.rs:184
  Panic: attempt to subtract with overflow (i32)
  caller: apply_promo_stack(order, promos=[PROMO_A, PROMO_B, PROMO_WELCOME20])

  context:
    order.subtotal_cents = 4500
    promos = [
      { code: "PROMO_A", kind: "percent", value: 20 },
      { code: "PROMO_B", kind: "flat",    value: 2000 },
      { code: "PROMO_WELCOME20", kind: "percent", value: 20 },
    ]
  final_discount_cents = 6100
  computed: total_amount_cents = subtotal_cents - final_discount_cents
"""

PROMPT = (
    "Read `bug_report.md` and `stack_trace.txt` in the scratch directory. "
    "Write a root-cause hypothesis (2-3 paragraphs) and a minimal fix plan "
    "(bullet list of 3-5 concrete steps an engineer should take). Save it to "
    "`rca.md` AND pass the full text as the `answer` argument to `finish`."
)


RUBRIC = """Score on a 1-5 scale:
- 5: Correctly identifies stacking promos can exceed subtotal (discount > subtotal causes negative total), ties it to the stackable-promo rewrite, flags lack of floor-at-zero clamp, and proposes specific fixes (clamp to 0, cap discount to subtotal, add invariant check).
- 4: Correct root cause and most of the fix plan; minor omissions.
- 3: Gets the general category right (promo stacking / discount overflow) but fix plan is vague or partial.
- 2: Plausible but misses the core mechanism; fix is generic.
- 1: Wrong root cause."""


def _evaluator(traj: Trajectory, scratch: Path) -> Score:
    answer = (traj.final_answer or "").strip() if traj else ""
    return Score(
        passed=bool(answer),
        score=0.0,
        details=f"answer_len={len(answer)}",
    )


TASK = Task(
    id="research_bug_rca",
    category="research",
    prompt=PROMPT,
    scratch_files={
        "bug_report.md": BUG_REPORT,
        "stack_trace.txt": STACK_TRACE,
    },
    evaluator=_evaluator,
)
TASK.rubric = RUBRIC  # type: ignore[attr-defined]
