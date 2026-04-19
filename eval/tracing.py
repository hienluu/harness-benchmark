"""Opt-in LangSmith tracing.

Enabled only when the `LANGSMITH_TRACING` env var is truthy (and
`LANGSMITH_API_KEY` is set). When disabled, every helper is a no-op so
the benchmark runs exactly as before with zero overhead and zero new
runtime dependency on langsmith at call time.

Usage:
    from eval.tracing import wrap_anthropic_client, traced, run_context

    client = wrap_anthropic_client(anthropic.Anthropic())   # auto-wraps when enabled
    with run_context(harness="thin", task_id=task.id, trial=0):
        ...                                                  # nested calls inherit metadata
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, Callable


def is_enabled() -> bool:
    return os.environ.get("LANGSMITH_TRACING", "").lower() in ("1", "true", "yes", "on")


def wrap_anthropic_client(client: Any) -> Any:
    """Wrap an Anthropic SDK client so every `messages.create` call shows up
    as a trace with token usage and latency. No-op when tracing is disabled."""
    if not is_enabled():
        return client
    try:
        from langsmith.wrappers import wrap_anthropic

        return wrap_anthropic(client)
    except Exception:
        return client


def traced(name: str | None = None, tags: list[str] | None = None) -> Callable:
    """Decorator: trace a function as a LangSmith run when enabled."""
    def deco(fn: Callable) -> Callable:
        if not is_enabled():
            return fn
        from langsmith import traceable

        return traceable(name=name or fn.__name__, tags=tags or [])(fn)

    return deco


@contextlib.contextmanager
def run_context(**metadata: Any):
    """Open a LangSmith trace scope tagged with the given metadata.

    Any wrapped-Anthropic calls or @traced functions called inside this
    block nest under a single top-level run, making it easy to filter the
    LangSmith UI by harness / task / trial.
    """
    if not is_enabled():
        yield None
        return
    try:
        from langsmith import trace

        meta = {k: v for k, v in metadata.items() if v is not None}
        name = f"{meta.get('harness', 'run')}/{meta.get('task_id', '?')}#{meta.get('trial', 0)}"
        with trace(name=name, metadata=meta, tags=[f"harness:{meta.get('harness', '?')}"]) as run:
            yield run
    except Exception:
        yield None
