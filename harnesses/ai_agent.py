"""Structured-thin "agent SDK" harness, provider-agnostic dispatcher.

Each major provider ships an "agent SDK" that wraps the bare model API with
batteries-included scaffolding — tool loop, session management, permission
model. They are the closest peers across providers:

  - Anthropic: claude-agent-sdk (in-process MCP server + ClaudeSDKClient)
  - OpenAI:    openai-agents    (Agent + Runner + @function_tool)
  - Google:    google-adk       (TODO)

Unlike `thin`, these SDKs are structurally very different (different tool
registration, different event/result shapes, different async patterns), so
the consolidation here is at the dispatcher level — each provider's
implementation lives in its own module (`claude_sdk.py`, `openai_agents.py`)
and this file just routes to the right one based on `--provider`.

When the Google ADK harness exists, add a `gemini` branch below.
"""

from __future__ import annotations

from pathlib import Path

from tasks.registry import Task, Trajectory


def run(task: Task, scratch_dir: Path, provider: str = "anthropic") -> Trajectory:
    if provider == "anthropic":
        # Imported lazily so a missing optional SDK doesn't break runs that
        # don't use it.
        from harnesses import claude_sdk

        return claude_sdk.run(task, scratch_dir)
    if provider == "openai":
        from harnesses import openai_agents

        return openai_agents.run(task, scratch_dir)
    if provider == "gemini":
        raise NotImplementedError(
            "ai_agent --provider gemini is not yet implemented. "
            "Add a Google ADK harness (see README TODOs) and dispatch here. "
            "Note: `thin --provider gemini` works today via google-genai."
        )
    raise ValueError(
        f"unknown provider {provider!r}; expected one of: anthropic, openai, gemini"
    )
