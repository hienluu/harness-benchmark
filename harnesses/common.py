"""Constants shared across harnesses.

The fairness of this benchmark depends on keeping every knob identical
except the harness structure itself. Both harnesses import from here.
"""

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


def usd_cost(tokens_in: int, tokens_out: int, cache_read: int, cache_write: int) -> float:
    return (
        tokens_in * PRICE_IN_PER_MTOK
        + tokens_out * PRICE_OUT_PER_MTOK
        + cache_read * PRICE_CACHE_READ_PER_MTOK
        + cache_write * PRICE_CACHE_WRITE_PER_MTOK
    ) / 1_000_000
