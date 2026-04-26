"""Structured-thin harness, OpenAI flavor: OpenAI Agents SDK with our shared tools.

Mirror of `harnesses/claude_sdk.py` for the OpenAI Agents SDK
(`openai-agents` package, importable as `agents`). Intentional structural
parallel: the SDK runs the loop, we just register our six shared tools as
function tools, drain events, and translate them into our `Trajectory`.

Like the other OpenAI harnesses, this picks up `OPENAI_API_KEY` and
`OPENAI_BASE_URL` from the environment (via `set_default_openai_client`) so
it works against any OpenAI-compatible server.

Fairness notes:
- Same 6 custom tools, wrapped as `@function_tool` instead of MCP tools;
  the underlying `tools.execute()` dispatcher is identical.
- Same `MAX_ITERS` (passed as `max_turns`).
- The Agents SDK does not expose `temperature` / `max_tokens` uniformly
  across model settings — we set what the SDK supports and let the rest
  fall back to defaults. This is the same kind of structural difference
  the Claude Agent SDK harness already documents.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from agents import Agent, ModelSettings, Runner, function_tool, set_default_openai_client
from openai import AsyncOpenAI

from harnesses.common import MAX_ITERS, SYSTEM_PROMPT, TEMPERATURE
from harnesses.openai_common import get_openai_model, is_reasoning_model
from tasks.registry import Task, Trajectory
from tools import execute as tools_execute


def _build_function_tools(scratch_dir: Path) -> list:
    """Wrap the shared tool dispatcher as Agents SDK function tools."""

    @function_tool
    def bash(command: str, timeout_s: int = 30) -> str:
        """Run a shell command in the task scratch directory. Returns combined stdout+stderr and the exit code. 30s default timeout."""
        return tools_execute(
            "bash", {"command": command, "timeout_s": timeout_s}, scratch_dir
        )

    @function_tool
    def read_file(path: str) -> str:
        """Read a text file relative to the scratch directory."""
        return tools_execute("read_file", {"path": path}, scratch_dir)

    @function_tool
    def write_file(path: str, content: str) -> str:
        """Overwrite a text file relative to the scratch directory."""
        return tools_execute(
            "write_file", {"path": path, "content": content}, scratch_dir
        )

    @function_tool
    def edit_file(path: str, old: str, new: str) -> str:
        """Replace the first exact occurrence of `old` with `new` in the file. Fails if `old` is not present or appears more than once."""
        return tools_execute(
            "edit_file", {"path": path, "old": old, "new": new}, scratch_dir
        )

    @function_tool
    def list_dir(path: str = ".") -> str:
        """List files and directories under the given path (recursive, max 200 entries)."""
        return tools_execute("list_dir", {"path": path}, scratch_dir)

    @function_tool
    def finish(answer: str) -> str:
        """Terminal tool. Call exactly once when the task is complete. Pass the final answer or a brief success note."""
        return tools_execute("finish", {"answer": answer}, scratch_dir)

    return [bash, read_file, write_file, edit_file, list_dir, finish]


async def _run_async(task: Task, scratch_dir: Path) -> Trajectory:
    traj = Trajectory()

    # AsyncOpenAI picks up OPENAI_API_KEY and OPENAI_BASE_URL from env.
    # set_default_openai_client makes the Agents SDK route all calls through it,
    # which is how we point the harness at any OpenAI-compatible server.
    set_default_openai_client(AsyncOpenAI())

    model_name = get_openai_model()
    # Reasoning models (gpt-5/o1/o3/o4) reject the temperature param entirely.
    # Pass the default ModelSettings in that case so the SDK doesn't forward it.
    settings = (
        ModelSettings()
        if is_reasoning_model(model_name)
        else ModelSettings(temperature=TEMPERATURE)
    )
    agent = Agent(
        name="hb-agent",
        instructions=SYSTEM_PROMPT,
        model=model_name,
        tools=_build_function_tools(scratch_dir),
        model_settings=settings,
    )

    t0 = time.time()
    try:
        result = await Runner.run(agent, task.prompt, max_turns=MAX_ITERS)
    except Exception as e:
        traj.latency_s = time.time() - t0
        traj.stopped_reason = f"crashed:{type(e).__name__}"
        return traj
    traj.latency_s = time.time() - t0

    # Walk the new items the agent produced. The Agents SDK exposes typed items
    # (MessageOutputItem, ToolCallItem, ToolCallOutputItem, ...) on result.
    for item in getattr(result, "new_items", []) or []:
        traj.messages.append({"type": type(item).__name__, "repr": repr(item)[:2000]})
        item_type = type(item).__name__

        if item_type == "ToolCallItem":
            raw = getattr(item, "raw_item", None)
            name = _get_attr(raw, "name") or _get_attr(getattr(raw, "function", None), "name")
            args = _parse_args(_get_attr(raw, "arguments")) if raw is not None else {}
            call_id = _get_attr(raw, "call_id") or _get_attr(raw, "id")
            traj.tool_calls.append(
                {"name": name, "input": args, "result": None, "id": call_id}
            )
            if name == "finish":
                traj.final_answer = (args or {}).get("answer", "")

        elif item_type == "ToolCallOutputItem":
            raw = getattr(item, "raw_item", None) or {}
            call_id = _get_attr(raw, "call_id") or _get_attr(raw, "tool_call_id")
            output = getattr(item, "output", None)
            if output is None:
                output = _get_attr(raw, "output")
            for call in reversed(traj.tool_calls):
                if call["id"] == call_id and call["result"] is None:
                    call["result"] = (
                        output if isinstance(output, str) else str(output)
                    )
                    break

        elif item_type == "MessageOutputItem":
            text = _extract_message_text(item)
            if text and not traj.final_answer:
                traj.final_answer = text

    # Token usage: the Agents SDK aggregates per-turn usage on
    # `result.context_wrapper.usage` in recent versions. Fall back gracefully
    # if the field has moved.
    usage = _aggregate_usage(result)
    traj.tokens_in = usage[0]
    traj.tokens_out = usage[1]
    traj.cache_read = usage[2]
    traj.cache_write = usage[3]

    traj.num_turns = _count_turns(result)

    if not traj.final_answer:
        final_output = getattr(result, "final_output", None)
        if final_output:
            traj.final_answer = str(final_output)

    if traj.num_turns >= MAX_ITERS and not traj.final_answer:
        traj.stopped_reason = "iter_cap"
    else:
        traj.stopped_reason = "success" if traj.final_answer else "no_answer"
    return traj


def _get_attr(obj, name):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _parse_args(raw):
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        import json
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _extract_message_text(item) -> str:
    raw = getattr(item, "raw_item", None)
    if raw is None:
        return ""
    content = _get_attr(raw, "content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            text = _get_attr(block, "text") or _get_attr(block, "content")
            if isinstance(text, str):
                parts.append(text)
        return "\n".join(parts)
    return ""


def _aggregate_usage(result) -> tuple[int, int, int, int]:
    """Best-effort extraction of (in, out, cache_read, cache_write) tokens.

    The Agents SDK's usage surface has shifted across versions; try a couple
    of known locations and fall back to walking raw_responses.
    """
    cw = getattr(result, "context_wrapper", None)
    u = getattr(cw, "usage", None) if cw is not None else None
    if u is not None:
        ti = getattr(u, "input_tokens", 0) or getattr(u, "prompt_tokens", 0) or 0
        to = getattr(u, "output_tokens", 0) or getattr(u, "completion_tokens", 0) or 0
        details = getattr(u, "input_tokens_details", None) or getattr(
            u, "prompt_tokens_details", None
        )
        cr = getattr(details, "cached_tokens", 0) if details else 0
        return ti, to, cr or 0, 0

    ti = to = cr = 0
    for resp in getattr(result, "raw_responses", []) or []:
        ru = getattr(resp, "usage", None)
        if ru is None:
            continue
        ti += getattr(ru, "input_tokens", 0) or getattr(ru, "prompt_tokens", 0) or 0
        to += getattr(ru, "output_tokens", 0) or getattr(ru, "completion_tokens", 0) or 0
        details = getattr(ru, "input_tokens_details", None) or getattr(
            ru, "prompt_tokens_details", None
        )
        cr += getattr(details, "cached_tokens", 0) if details else 0
    return ti, to, cr, 0


def _count_turns(result) -> int:
    """One turn = one assistant model response."""
    n = 0
    for item in getattr(result, "new_items", []) or []:
        if type(item).__name__ == "MessageOutputItem":
            n += 1
    if n == 0:
        # Fall back to counting raw responses if the SDK didn't emit message items.
        n = len(getattr(result, "raw_responses", []) or [])
    return n


def run(task: Task, scratch_dir: Path) -> Trajectory:
    return asyncio.run(_run_async(task, scratch_dir))
