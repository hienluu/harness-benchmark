"""OpenAI implementation of the unified `thin` Provider.

Mirrors the original `harnesses/openai_thin.py`: same client wrapping, same
reasoning-model gate (`chat_completions_kwargs`), same usage extraction.
Uses Chat Completions (not the Responses API) so this works against any
OpenAI-compatible server (vLLM, Ollama, OpenRouter, …) with no changes.
"""

from __future__ import annotations

import json
from typing import Any

import openai

from eval.tracing import wrap_openai_client
from harnesses.openai_common import (
    chat_completions_kwargs,
    extract_usage,
    get_openai_model,
    to_openai_tools,
)

from . import ModelStep, ToolCall


class OpenAIProvider:
    name = "openai"

    def make_client(self) -> Any:
        return wrap_openai_client(openai.OpenAI())

    def get_model(self) -> str:
        return get_openai_model()

    def encode_tools(self, schemas: list[dict]) -> list[dict]:
        return to_openai_tools(schemas)

    def initial_messages(self, system: str, user: str) -> list[dict]:
        # Chat Completions has no separate system arg; the system prompt lives
        # in the messages list as the first message.
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

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
        # `system` is unused here — already injected via initial_messages and
        # carried in the messages list across turns.
        resp = client.chat.completions.create(
            **chat_completions_kwargs(
                model, max_tokens=max_tokens, temperature=temperature
            ),
            messages=messages,
            tools=tools,
        )
        usage = extract_usage(resp)
        msg = resp.choices[0].message
        raw_calls = msg.tool_calls or []
        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                args=_parse_args(tc.function.arguments),
            )
            for tc in raw_calls
        ]
        assistant_message = {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.model_dump() for tc in raw_calls] if raw_calls else None,
        }
        return ModelStep(
            tool_calls=tool_calls,
            text=msg.content or "",
            usage=usage,
            assistant_message=assistant_message,
        )

    def encode_tool_results(
        self, results: list[tuple[ToolCall, str]]
    ) -> list[dict]:
        # OpenAI: one separate "tool" role message per tool call.
        return [
            {"role": "tool", "tool_call_id": tc.id, "content": result}
            for tc, result in results
        ]


def _parse_args(raw: str | None) -> dict:
    """OpenAI returns tool args as a JSON string; some OpenAI-compatible
    servers occasionally return malformed JSON. Be tolerant — return {} on
    parse failure so the harness records the call and the loop continues."""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
