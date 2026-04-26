"""Shared helpers for the OpenAI-side harnesses.

Keeps the schema/usage conversion in one place so `openai_thin.py` and
`openai_langgraph_h.py` (which both call the OpenAI Chat Completions API
directly) stay parallel and consistent.

The OpenAI SDK reads `OPENAI_API_KEY` and `OPENAI_BASE_URL` from the
environment automatically, which is what makes these harnesses work
unchanged against any OpenAI-compatible server (vLLM, Ollama, OpenRouter,
Together, LM Studio, llama.cpp, ...). The model name is the only thing the
runner explicitly threads in via `OPENAI_MODEL` (set from the
`--openai-model` CLI flag).
"""

from __future__ import annotations

import os

from tools import TOOL_SCHEMAS

# Default if the user set neither --openai-model nor $OPENAI_MODEL. The
# runner warns when this fallback fires.
#DEFAULT_OPENAI_MODEL = "gpt-5-nano-2025-08-07"
DEFAULT_OPENAI_MODEL = "gpt-5-mini-2025-08-07"
#DEFAULT_OPENAI_MODEL = "gpt-5"


def get_openai_model() -> str:
    return os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)


def is_reasoning_model(model: str) -> bool:
    """OpenAI reasoning models (o1/o3/o4 and the gpt-5 family) reject the
    `temperature` and `top_p` sampling params, and prefer `max_completion_tokens`
    over `max_tokens`. This helper is used by every OpenAI harness to gate
    those params.
    """
    name = (model or "").lower()
    return name.startswith(("o1", "o3", "o4", "gpt-5"))


def chat_completions_kwargs(
    model: str, *, max_tokens: int, temperature: float
) -> dict:
    """Build kwargs for `client.chat.completions.create(...)` that respect a
    model's parameter constraints. Reasoning models reject `temperature` and
    use `max_completion_tokens` instead of `max_tokens`.
    """
    if is_reasoning_model(model):
        return {"model": model, "max_completion_tokens": max_tokens}
    return {"model": model, "max_tokens": max_tokens, "temperature": temperature}


def to_openai_tools(schemas: list[dict] = TOOL_SCHEMAS) -> list[dict]:
    """Convert our Anthropic-format TOOL_SCHEMAS to OpenAI's `tools` array.

    Anthropic: {"name", "description", "input_schema": {...}}
    OpenAI:    {"type": "function", "function": {"name", "description", "parameters": {...}}}
    """
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["input_schema"],
            },
        }
        for s in schemas
    ]


def extract_usage(resp) -> tuple[int, int, int, int]:
    """Return (tokens_in, tokens_out, cache_read, cache_write) from an OpenAI response.

    OpenAI exposes cached input tokens via `usage.prompt_tokens_details.cached_tokens`
    on the official API; OpenAI-compatible servers usually omit it. There is no
    cache_creation/cache_write equivalent in OpenAI's API today, so cache_write
    is always 0.
    """
    u = getattr(resp, "usage", None)
    if u is None:
        return 0, 0, 0, 0
    tokens_in = getattr(u, "prompt_tokens", 0) or 0
    tokens_out = getattr(u, "completion_tokens", 0) or 0
    details = getattr(u, "prompt_tokens_details", None)
    cache_read = getattr(details, "cached_tokens", 0) if details else 0
    return tokens_in, tokens_out, cache_read or 0, 0
