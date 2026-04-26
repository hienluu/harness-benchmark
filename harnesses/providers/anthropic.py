"""Anthropic implementation of the unified `thin` Provider.

Mirrors the original `harnesses/thin.py` exactly: same client wrapping,
same cache_control on system blocks, same usage extraction. The provider
abstraction just relocates the Anthropic-specific bits here so the loop
in `thin.py` is provider-agnostic.
"""

from __future__ import annotations

from typing import Any

import anthropic

from eval.tracing import wrap_anthropic_client
from harnesses.common import MODEL

from . import ModelStep, ToolCall


class AnthropicProvider:
    name = "anthropic"

    def make_client(self) -> Any:
        return wrap_anthropic_client(anthropic.Anthropic())

    def get_model(self) -> str:
        return MODEL

    def encode_tools(self, schemas: list[dict]) -> list[dict]:
        # Our shared TOOL_SCHEMAS already use Anthropic format.
        return schemas

    def initial_messages(self, system: str, user: str) -> list[dict]:
        # Anthropic takes the system prompt as a separate API arg, not in messages.
        # `system` is unused here — see `call(...)` where it's passed as `system=`.
        return [{"role": "user", "content": user}]

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
        system_blocks = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "cache_control": {"type": "ephemeral"},
            "temperature": temperature,
            "system": system_blocks,
            "messages": messages,
        }
        # Omit `tools` when empty — Anthropic accepts the kwarg but it's
        # cleaner not to declare zero tools (and lets nodes that don't need
        # tools, e.g. langgraph_h's planner/reflector, pass tools=[]).
        if tools:
            kwargs["tools"] = tools
        resp = client.messages.create(**kwargs)
        u = resp.usage
        usage = (
            u.input_tokens,
            u.output_tokens,
            getattr(u, "cache_read_input_tokens", 0) or 0,
            getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        assistant_content = [b.model_dump() for b in resp.content]
        tool_calls = [
            ToolCall(id=b.id, name=b.name, args=b.input)
            for b in resp.content
            if b.type == "tool_use"
        ]
        text = "".join(b.text for b in resp.content if b.type == "text")
        return ModelStep(
            tool_calls=tool_calls,
            text=text,
            usage=usage,
            assistant_message={"role": "assistant", "content": assistant_content},
        )

    def encode_tool_results(
        self, results: list[tuple[ToolCall, str]]
    ) -> list[dict]:
        # Anthropic packs all tool results into one user message as content blocks.
        blocks = [
            {"type": "tool_result", "tool_use_id": tc.id, "content": result}
            for tc, result in results
        ]
        return [{"role": "user", "content": blocks}]
