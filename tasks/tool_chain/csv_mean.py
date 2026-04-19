"""Compute mean of a column in a CSV and write result to JSON."""

from __future__ import annotations

import json
from pathlib import Path

from tasks.registry import Score, Task, Trajectory


CSV = """id,value,label
1,10,a
2,25,b
3,30,a
4,45,c
5,50,b
6,60,a
7,75,c
8,90,b
"""

EXPECTED_MEAN = 48.125  # sum(10,25,30,45,50,60,75,90) / 8

PROMPT = (
    "There is a `data.csv` in the scratch directory. Compute the arithmetic "
    "mean of the `value` column and write the result to `out.json` as a JSON "
    'object of the form `{"mean": <float>}`. Round to 3 decimal places. '
    "Then call `finish`."
)


def _evaluator(traj: Trajectory, scratch: Path) -> Score:
    out = scratch / "out.json"
    if not out.exists():
        return Score(passed=False, score=0.0, details="out.json not created")
    try:
        data = json.loads(out.read_text())
    except json.JSONDecodeError as e:
        return Score(passed=False, score=0.0, details=f"invalid JSON: {e}")
    got = data.get("mean")
    if not isinstance(got, (int, float)):
        return Score(passed=False, score=0.0, details=f"mean missing or wrong type: {got!r}")
    if abs(got - EXPECTED_MEAN) < 0.01:
        return Score(passed=True, score=1.0, details=f"mean={got}")
    return Score(passed=False, score=0.0, details=f"expected {EXPECTED_MEAN}, got {got}")


TASK = Task(
    id="tool_csv_mean",
    category="tool_chain",
    prompt=PROMPT,
    scratch_files={"data.csv": CSV},
    evaluator=_evaluator,
)
