"""Provider abstraction for the unified `thin` harness.

The thin harness's loop is the same regardless of LLM provider:
  send messages → get tool calls → execute → loop. The only things that
differ are the SDK call shape, the tool/message format, and the usage fields.

Each provider implements the small `Provider` protocol below, returning a
normalized `ModelStep` from `call(...)`. `thin.py` operates on these
normalized types and never touches any provider SDK directly.

To add a new provider (e.g. Gemini): create `harnesses/providers/<name>.py`
implementing `Provider`, register it in `get_provider()` below, and you're
done — no changes to thin.py or the runner's HARNESSES dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass
class ToolCall:
    """Provider-neutral tool call surfaced from a model response."""

    id: str          # the call id the provider will expect on tool result messages
    name: str        # the tool name (e.g. "bash", "finish")
    args: dict       # parsed arguments dict (provider-specific JSON-string parsing
                     # happens inside the provider so thin.py sees a real dict)


@dataclass
class ModelStep:
    """Provider-neutral result of one chat-completion call."""

    tool_calls: list[ToolCall]
    text: str                          # assistant's text content; may be empty
    usage: tuple[int, int, int, int]   # (tokens_in, tokens_out, cache_read, cache_write)
    assistant_message: dict            # provider-format dict to append to messages
                                       # so the next turn sees the right history


class Provider(Protocol):
    name: str

    def make_client(self) -> Any:
        """Return an SDK client (already wrapped for LangSmith if enabled)."""

    def get_model(self) -> str:
        """Return the model name to use (resolved from env / defaults)."""

    def encode_tools(self, schemas: list[dict]) -> list[dict]:
        """Convert our shared TOOL_SCHEMAS (Anthropic-format) into the
        provider's tool array format."""

    def initial_messages(self, system: str, user: str) -> list[dict]:
        """Seed the messages list. Some providers (Anthropic) keep the system
        prompt out-of-band; others (OpenAI) put it in the messages list."""

    def call(
        self,
        client: Any,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> ModelStep:
        """Make one model call and return the normalized step."""

    def encode_tool_results(
        self, results: list[tuple[ToolCall, str]]
    ) -> list[dict]:
        """Convert (tool_call, result_string) pairs into the provider-format
        message(s) the next turn will see. May return one message containing
        all results (Anthropic) or one message per result (OpenAI)."""


def get_provider(name: str) -> Provider:
    """Factory: resolve a provider name string to a Provider instance.

    Imports are deferred so a missing optional SDK doesn't break harnesses
    that don't use it.
    """
    if name == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider()
    if name == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider()
    if name == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider()
    raise ValueError(
        f"unknown provider {name!r}; expected one of: anthropic, openai, gemini"
    )


SUPPORTED_PROVIDERS = ("anthropic", "openai", "gemini")
