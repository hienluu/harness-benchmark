"""Rename a function across a small package, preserving imports."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tasks.registry import Score, Task, Trajectory


A = '''\
def foo(x):
    return x + 1

def wrapper(x):
    return foo(x) * 2
'''

B = '''\
from a import foo

def double_foo(x):
    return foo(x) + foo(x)
'''

MAIN = '''\
from a import wrapper
from b import double_foo

if __name__ == "__main__":
    assert wrapper(3) == 8, wrapper(3)
    assert double_foo(3) == 8, double_foo(3)
    print("ok")
'''

TESTS = '''\
import subprocess, sys, os, re

HERE = os.path.dirname(os.path.abspath(__file__))

def test_renamed():
    # foo must no longer appear as a defined or imported symbol
    for name in ("a.py", "b.py"):
        src = open(os.path.join(HERE, name)).read()
        assert "def foo(" not in src, f"{name} still defines foo"
        assert re.search(r"\\bfoo\\b", src) is None, f"{name} still references foo"
    # bar must be defined somewhere
    a_src = open(os.path.join(HERE, "a.py")).read()
    assert "def bar(" in a_src, "a.py must define bar"

def test_still_runs():
    r = subprocess.run([sys.executable, "main.py"], cwd=HERE, capture_output=True, text=True, timeout=10)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"
'''

PROMPT = (
    "Rename the function `foo` to `bar` across the scratch directory. "
    "Update `a.py` where it is defined, and update `b.py` (and anywhere else) "
    "that imports or calls it, so that no reference to `foo` remains in any "
    ".py file. `main.py` must still run successfully. Do not modify `main.py` "
    "or `test_rename.py`. Call `finish` when tests pass."
)


def _evaluator(traj: Trajectory, scratch: Path) -> Score:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_rename.py"],
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
    id="tool_rename_symbol",
    category="tool_chain",
    prompt=PROMPT,
    scratch_files={"a.py": A, "b.py": B, "main.py": MAIN, "test_rename.py": TESTS},
    evaluator=_evaluator,
)
