"""Task schema + loader.

A task bundles: a natural-language prompt, seed files materialized into a
scratch directory, and an evaluator that grades a completed trajectory.
"""

from __future__ import annotations

import importlib
import pkgutil
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal


Category = Literal["code", "tool_chain", "research"]


@dataclass
class Score:
    passed: bool
    score: float  # 0.0-1.0 for deterministic, 0.2-1.0 for 1-5 judge scale
    details: str = ""


@dataclass
class Trajectory:
    messages: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)
    final_answer: str | None = None
    num_turns: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cache_read: int = 0
    cache_write: int = 0
    latency_s: float = 0.0
    stopped_reason: str = "unknown"  # success, iter_cap, error, token_cap


@dataclass
class Task:
    id: str
    category: Category
    prompt: str
    scratch_files: dict[str, str]
    evaluator: Callable[[Trajectory, Path], Score]


def materialize(task: Task) -> Path:
    """Create a fresh scratch directory seeded with the task's files."""
    d = Path(tempfile.mkdtemp(prefix=f"hb_{task.id}_"))
    for rel, content in task.scratch_files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return d


def cleanup(scratch_dir: Path) -> None:
    shutil.rmtree(scratch_dir, ignore_errors=True)


def load_all() -> list[Task]:
    """Import every task module under tasks/{code,tool_chain,research} and collect TASK objects."""
    import tasks  # noqa: F401 - parent package

    collected: list[Task] = []
    for cat in ("code", "tool_chain", "research"):
        pkg = importlib.import_module(f"tasks.{cat}")
        for modinfo in pkgutil.iter_modules(pkg.__path__):
            mod = importlib.import_module(f"tasks.{cat}.{modinfo.name}")
            task = getattr(mod, "TASK", None)
            if isinstance(task, Task):
                collected.append(task)
    return collected
