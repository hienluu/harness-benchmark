"""Constants shared across harnesses.

The fairness of this benchmark depends on keeping every knob identical
except the harness structure itself. Both harnesses import from here.
"""

from __future__ import annotations

import os

MODEL = "claude-sonnet-4-6"
MAX_ITERS = 25
MAX_TOKENS_PER_CALL = 4096
TEMPERATURE = 0.0

# The system prompt is deliberately minimal and identical for both harnesses.
# The "thick" harness adds its own node-specific role prompts *in addition to*
# this shared system message, which is the whole point of the comparison.
SYSTEM_PROMPT = (
    "You are an AI assistant working inside a sandboxed scratch directory. "
    "You have access to a small set of tools: bash, read_file, write_file, "
    "edit_file, list_dir, and finish. "
    "Complete the user's task using these tools. "
    "When done, call the `finish` tool with your final answer or a brief "
    "success note. Do not ask clarifying questions — proceed with reasonable "
    "assumptions. Keep tool calls focused and minimal."
)

# Sonnet 4.6 pricing, USD per 1M tokens. Used for cost accounting only.
PRICE_IN_PER_MTOK = 3.0
PRICE_OUT_PER_MTOK = 15.0
PRICE_CACHE_READ_PER_MTOK = 0.30
PRICE_CACHE_WRITE_PER_MTOK = 3.75

# OpenAI per-model pricing, USD per 1M tokens, as (in, out, cache_read).
# OpenAI has no cache-write pricing line item — input tokens that get cached
# are billed at the full input rate the first time and then at the cached
# rate on subsequent reads.
#
# This table exists so cost numbers in summary.jsonl are sensible for the
# common official-OpenAI cases. Self-hosted OpenAI-compatible endpoints
# (vLLM/Ollama) and unknown models default to (0, 0, 0) — there's no sane
# universal price, and printing $0 makes that obvious in the report.
OPENAI_PRICING: dict[str, tuple[float, float, float]] = {
    # Frontier
    "gpt-5":              (1.25, 10.00, 0.125),
    "gpt-5-mini":         (0.25,  2.00, 0.025),
    "gpt-5-nano":         (0.05,  0.40, 0.005),
    # GPT-4 family (legacy but still common)
    "gpt-4o":             (2.50, 10.00, 1.25),
    "gpt-4o-mini":        (0.15,  0.60, 0.075),
    "gpt-4.1":            (2.00,  8.00, 0.50),
    "gpt-4.1-mini":       (0.40,  1.60, 0.10),
    "gpt-4.1-nano":       (0.10,  0.40, 0.025),
}

# Gemini per-model pricing, USD per 1M tokens, as (in, out, cache_read).
# Gemini's context-cache "creation" is metered separately on a per-hour basis
# in real billing — we ignore that here and only price the input/output/cache-read
# rates that map directly onto our Trajectory token counters.
GEMINI_PRICING: dict[str, tuple[float, float, float]] = {
    "gemini-2.5-pro":     (1.25, 10.00, 0.31),    # ≤200k context tier
    "gemini-2.5-flash":   (0.30,  2.50, 0.075),
    "gemini-2.5-flash-lite": (0.10, 0.40, 0.025),
    "gemini-2.0-flash":   (0.10,  0.40, 0.025),
    "gemini-2.0-flash-lite": (0.075, 0.30, 0.0),
}


def _lookup_pricing(table: dict, model: str) -> tuple[float, float, float]:
    """Match the longest prefix in `table` against `model` so e.g.
    'gemini-2.5-pro-preview-05-06' picks up gemini-2.5-pro pricing.
    Returns (0, 0, 0) when no prefix matches — the right behavior for
    unknown / self-hosted models."""
    for key, prices in sorted(table.items(), key=lambda kv: -len(kv[0])):
        if model.startswith(key):
            return prices
    return (0.0, 0.0, 0.0)


def usd_cost(
    tokens_in: int,
    tokens_out: int,
    cache_read: int,
    cache_write: int,
    provider: str = "anthropic",
) -> float:
    if provider == "openai":
        model = os.environ.get("OPENAI_MODEL", "")
        price_in, price_out, price_cache_read = _lookup_pricing(OPENAI_PRICING, model)
        # Cached input tokens are billed *instead of* full input tokens, so
        # subtract the cached portion from the input total.
        billed_in = max(0, tokens_in - cache_read)
        return (
            billed_in * price_in
            + cache_read * price_cache_read
            + tokens_out * price_out
        ) / 1_000_000

    if provider == "gemini":
        model = os.environ.get("GEMINI_MODEL", "")
        price_in, price_out, price_cache_read = _lookup_pricing(GEMINI_PRICING, model)
        billed_in = max(0, tokens_in - cache_read)
        return (
            billed_in * price_in
            + cache_read * price_cache_read
            + tokens_out * price_out
        ) / 1_000_000

    # Anthropic (default)
    return (
        tokens_in * PRICE_IN_PER_MTOK
        + tokens_out * PRICE_OUT_PER_MTOK
        + cache_read * PRICE_CACHE_READ_PER_MTOK
        + cache_write * PRICE_CACHE_WRITE_PER_MTOK
    ) / 1_000_000
