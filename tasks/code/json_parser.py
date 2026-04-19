"""Add missing error handling to a JSON parser wrapper."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tasks.registry import Score, Task, Trajectory


BUGGY = '''\
import json

def safe_parse(text, default=None):
    """Parse JSON and return the value, or `default` on any parse failure.

    - If `text` is None or empty string, return `default`.
    - If `text` is not valid JSON, return `default`.
    - Otherwise return the parsed value.
    """
    return json.loads(text)  # TODO: implement the spec
'''

TESTS = '''\
from parser import safe_parse

def test_valid_object():
    assert safe_parse('{"a": 1}') == {"a": 1}

def test_valid_array():
    assert safe_parse('[1, 2, 3]') == [1, 2, 3]

def test_empty_string():
    assert safe_parse('') is None
    assert safe_parse('', default="x") == "x"

def test_none():
    assert safe_parse(None) is None

def test_invalid():
    assert safe_parse('{not json}') is None
    assert safe_parse('{"a":', default=[]) == []

def test_number():
    assert safe_parse('42') == 42
'''

PROMPT = (
    "`parser.py` contains a `safe_parse` function whose implementation does "
    "not match its docstring. Fix the implementation so every test in "
    "`test_parser.py` passes. Do not modify the tests. Call `finish` when done."
)


def _evaluator(traj: Trajectory, scratch: Path) -> Score:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "test_parser.py"],
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
    id="code_json_parser",
    category="code",
    prompt=PROMPT,
    scratch_files={"parser.py": BUGGY, "test_parser.py": TESTS},
    evaluator=_evaluator,
)
