"""Debug a broken multi-file import chain."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tasks.registry import Score, Task, Trajectory


PKG_INIT = ""  # missing __init__ content is fine

MATH_UTILS = '''\
from .helpers import double  # helpers.py exports `triple`, not `double`

def quadruple(x):
    return double(x) + double(x)
'''

HELPERS = '''\
def triple(x):
    return x * 3
'''

MAIN = '''\
from pkg.math_utils import quadruple

if __name__ == "__main__":
    print(quadruple(5))
'''

TESTS = '''\
import subprocess, sys, os

def test_runs():
    here = os.path.dirname(os.path.abspath(__file__))
    result = subprocess.run(
        [sys.executable, "main.py"],
        cwd=here,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "20"
'''

PROMPT = (
    "Running `python main.py` fails with an ImportError. Find and fix the "
    "broken import chain in `pkg/` so that `main.py` prints `20` "
    "(quadruple of 5). Do not modify `main.py` or the test. When tests pass, call `finish`."
)


def _evaluator(traj: Trajectory, scratch: Path) -> Score:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_main.py"],
        cwd=str(scratch),
        capture_output=True,
        text=True,
        timeout=60,
    )
    passed = result.returncode == 0
    return Score(
        passed=passed,
        score=1.0 if passed else 0.0,
        details=(result.stdout + result.stderr)[-2000:],
    )


TASK = Task(
    id="code_import_chain",
    category="code",
    prompt=PROMPT,
    scratch_files={
        "pkg/__init__.py": PKG_INIT,
        "pkg/math_utils.py": MATH_UTILS,
        "pkg/helpers.py": HELPERS,
        "main.py": MAIN,
        "test_main.py": TESTS,
    },
    evaluator=_evaluator,
)
