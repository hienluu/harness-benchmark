"""Gemini implementation of the unified `thin` Provider.

Uses the `google-genai` Python SDK (`google.genai.Client().models.generate_content`).
Auto function-calling is disabled — we run our own tool loop, just like the
Anthropic and OpenAI providers, so the harness's behavior stays identical
across providers.

Environment:
    GOOGLE_API_KEY   (preferred) or GEMINI_API_KEY — picked up by `genai.Client()`.
    GEMINI_MODEL     (optional) overrides the default model.
"""

from __future__ import annotations

import os
from typing import Any

from google import genai
from google.genai import types

from eval.tracing import wrap_gemini_client

from . import ModelStep, ToolCall


# Default model for the Gemini provider when neither --gemini-model (TBD) nor
# $GEMINI_MODEL is set. 2.5 Pro is the closest peer to Sonnet 4.6.
DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"


# Map JSON-schema lowercase types → Gemini Schema enum strings (uppercase).
_TYPE_MAP = {
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
    "object": "OBJECT",
    "array": "ARRAY",
}


def _convert_schema(schema: Any) -> Any:
    """Recursively translate JSON-schema types to Gemini Schema's uppercase enum.

    Gemini's `parameters` schema accepts the same nested structure as JSON
    Schema (properties / items / required) but rejects lowercase type strings.
    """
    if not isinstance(schema, dict):
        return schema
    out = dict(schema)
    if isinstance(out.get("type"), str):
        out["type"] = _TYPE_MAP.get(out["type"].lower(), out["type"].upper())
    if isinstance(out.get("properties"), dict):
        out["properties"] = {k: _convert_schema(v) for k, v in out["properties"].items()}
    if "items" in out:
        out["items"] = _convert_schema(out["items"])
    return out


class GeminiProvider:
    name = "gemini"

    def make_client(self) -> Any:
        # The SDK reads GOOGLE_API_KEY (preferred) or GEMINI_API_KEY from env
        # automatically. The runner gates on GOOGLE_API_KEY before we get here.
        # wrap_gemini_client patches generate_content for LangSmith tracing
        # when LANGSMITH_TRACING=true; no-op otherwise.
        return wrap_gemini_client(genai.Client())

    def get_model(self) -> str:
        return os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    def encode_tools(self, schemas: list[dict]) -> list[dict]:
        # Gemini groups all function declarations under a single Tool.
        return [
            {
                "function_declarations": [
                    {
                        "name": s["name"],
                        "description": s["description"],
                        "parameters": _convert_schema(s["input_schema"]),
                    }
                    for s in schemas
                ]
            }
        ]

    def initial_messages(self, system: str, user: str) -> list[dict]:
        # Gemini takes the system prompt via config.system_instruction, not in
        # messages. The first message is just the user's prompt.
        return [{"role": "user", "parts": [{"text": user}]}]

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
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=tools,
            temperature=temperature,
            max_output_tokens=max_tokens,
            # We run our own tool loop — disable the SDK's auto function calling
            # so we get raw FunctionCall parts back instead of the SDK silently
            # invoking Python callables.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        resp = client.models.generate_content(
            model=model,
            contents=messages,
            config=config,
        )

        u = getattr(resp, "usage_metadata", None)
        tokens_in = (getattr(u, "prompt_token_count", 0) or 0) if u else 0
        tokens_out = (getattr(u, "candidates_token_count", 0) or 0) if u else 0
        cache_read = (getattr(u, "cached_content_token_count", 0) or 0) if u else 0
        usage = (tokens_in, tokens_out, cache_read, 0)

        candidate = resp.candidates[0] if resp.candidates else None
        parts = candidate.content.parts if (candidate and candidate.content) else []

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        # Gemini doesn't issue tool-call ids; synthesize stable ones from part
        # index so tool results in the next turn can be matched back if needed.
        # (encode_tool_results matches by tool name, not id, since Gemini
        # function_response is name-keyed.)
        for i, part in enumerate(parts):
            fc = getattr(part, "function_call", None)
            if fc and getattr(fc, "name", None):
                args = dict(fc.args) if getattr(fc, "args", None) else {}
                tool_calls.append(ToolCall(id=f"call_{i}", name=fc.name, args=args))
            elif getattr(part, "text", None):
                text_parts.append(part.text)

        # Serialize the model's content so the next turn sees the same history
        # the model just produced. SDK content objects are Pydantic models.
        if candidate and candidate.content:
            assistant_message = candidate.content.model_dump(exclude_none=True)
            assistant_message.setdefault("role", "model")
        else:
            assistant_message = {"role": "model", "parts": []}

        return ModelStep(
            tool_calls=tool_calls,
            text="".join(text_parts),
            usage=usage,
            assistant_message=assistant_message,
        )

    def encode_tool_results(
        self, results: list[tuple[ToolCall, str]]
    ) -> list[dict]:
        # Gemini packs all tool results into one user-role message containing
        # function_response parts (one per tool call), matched by name.
        return [
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "name": tc.name,
                            "response": {"result": result},
                        }
                    }
                    for tc, result in results
                ],
            }
        ]
