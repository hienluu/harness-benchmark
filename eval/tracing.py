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


def wrap_openai_client(client: Any) -> Any:
    """Wrap an OpenAI SDK client so every `chat.completions.create` call shows
    up as a trace with token usage and latency. No-op when tracing is disabled."""
    if not is_enabled():
        return client
    try:
        from langsmith.wrappers import wrap_openai

        return wrap_openai(client)
    except Exception:
        return client


def wrap_gemini_client(client: Any) -> Any:
    """Wrap a google-genai Client so every `models.generate_content` call shows
    up as a trace with token usage, tool calls, and latency. No-op when tracing
    is disabled. Requires langsmith >= 0.4.33 (the wrapper is in beta)."""
    if not is_enabled():
        return client
    try:
        from langsmith.wrappers import wrap_gemini

        return wrap_gemini(client)
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


def configure_claude_sdk_tracing() -> bool:
    """Enable LangSmith's native Claude Agent SDK integration.

    Safe to call unconditionally: no-op when tracing is disabled, and swallows
    any import/runtime errors so a missing optional dep never breaks a run.
    Should be called once, at process startup, before the first `query()` call.
    """
    if not is_enabled():
        return False
    try:
        from langsmith.integrations.claude_agent_sdk import configure_claude_agent_sdk

        return configure_claude_agent_sdk()
    except Exception:
        pass


def configure_openai_agents_tracing() -> bool:
    """Enable LangSmith's native OpenAI Agents SDK integration.

    Installs `OpenAIAgentsTracingProcessor` as the Agents SDK's trace processor
    so every Runner.run produces a nested LangSmith trace covering model calls,
    tool calls, and handoffs.

    Safe to call unconditionally: no-op when tracing is disabled, and swallows
    any import/runtime errors so a missing optional dep never breaks a run.
    Should be called once, at process startup, before the first `Runner.run`.
    """
    if not is_enabled():
        return False
    try:
        from agents import set_trace_processors
        from langsmith.wrappers import OpenAIAgentsTracingProcessor

        set_trace_processors([OpenAIAgentsTracingProcessor()])
        return True
    except Exception:
        return False
