"""Find .py files with TODO comments and write sorted list to todos.txt."""

from __future__ import annotations

from pathlib import Path

from tasks.registry import Score, Task, Trajectory


SRC_FILES = {
    "src/a.py": "def foo():\n    # TODO: handle None\n    pass\n",
    "src/b.py": "def bar():\n    return 42\n",
    "src/nested/c.py": "# TODO: write docs\n\ndef baz(): pass\n",
    "src/nested/d.py": "# nothing to do here\n",
    "src/e.py": "def qux():\n    pass\n    # TODO refactor\n",
    "README.md": "# TODO: add docs  (not a .py file)\n",
}

EXPECTED = "src/a.py\nsrc/e.py\nsrc/nested/c.py\n"

PROMPT = (
    "Under the `src/` directory in the current scratch dir there are several "
    ".py files. Find ALL .py files under `src/` that contain the substring "
    "`TODO` anywhere in their content. Write their paths (relative to the "
    "scratch dir, so they start with `src/`), one per line, sorted "
    "alphabetically, into a file named `todos.txt` at the top of the scratch "
    "dir. The file must end with a trailing newline. Then call `finish`."
)


def _evaluator(traj: Trajectory, scratch: Path) -> Score:
    out = scratch / "todos.txt"
    if not out.exists():
        return Score(passed=False, score=0.0, details="todos.txt not created")
    got = out.read_text()
    if got == EXPECTED:
        return Score(passed=True, score=1.0, details="exact match")
    return Score(
        passed=False,
        score=0.0,
        details=f"expected {EXPECTED!r}, got {got!r}",
    )


TASK = Task(
    id="tool_find_todos",
    category="tool_chain",
    prompt=PROMPT,
    scratch_files=SRC_FILES,
    evaluator=_evaluator,
)
