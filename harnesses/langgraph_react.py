"""Baseline LangGraph/LangChain harness: the prebuilt `create_agent`.

This is "LangChain/LangGraph as they want to be used" — the canonical prebuilt
agent from `langchain.agents.create_agent` (successor to the deprecated
`langgraph.prebuilt.create_react_agent`). It's a simple tool-use loop: LLM
node decides → ToolNode executes → LLM node decides → ... → stop when the
LLM stops calling tools.

Point of this harness in the benchmark: it isolates *our topology's* overhead
(planner / router / executor / reflector / finalizer in `langgraph_h.py`)
from *LangGraph's own* overhead. If this harness matches `thin` on cost and
quality, the cost of `langgraph_h` is attributable to our graph design, not
to LangGraph the framework. If this harness also costs substantially more
than `thin`, there's a real framework tax.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.messages import SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool as lc_tool

from harnesses.common import (
    MAX_ITERS,
    MAX_TOKENS_PER_CALL,
    MODEL,
    SYSTEM_PROMPT,
    TEMPERATURE,
)
from tasks.registry import Task, Trajectory
from tools import execute as tools_execute


def _build_tools(scratch_dir: Path) -> list:
    """Wrap our shared `tools.execute()` dispatcher as LangChain tools.

    Descriptions intentionally mirror the Anthropic-format schemas in
    `tools/__init__.py::TOOL_SCHEMAS` so the tool surface the model sees is
    as close to the other harnesses as LangChain's tool schema allows.
    """

    @lc_tool
    def bash(command: str, timeout_s: int = 30) -> str:
        """Run a shell command in the task scratch directory. Returns combined stdout+stderr and the exit code. 30s default timeout."""
        return tools_execute(
            "bash", {"command": command, "timeout_s": timeout_s}, scratch_dir
        )

    @lc_tool
    def read_file(path: str) -> str:
        """Read a text file relative to the scratch directory."""
        return tools_execute("read_file", {"path": path}, scratch_dir)

    @lc_tool
    def write_file(path: str, content: str) -> str:
        """Overwrite a text file relative to the scratch directory."""
        return tools_execute(
            "write_file", {"path": path, "content": content}, scratch_dir
        )

    @lc_tool
    def edit_file(path: str, old: str, new: str) -> str:
        """Replace the first exact occurrence of `old` with `new` in the file. Fails if `old` is not present or appears more than once."""
        return tools_execute(
            "edit_file", {"path": path, "old": old, "new": new}, scratch_dir
        )

    @lc_tool
    def list_dir(path: str = ".") -> str:
        """List files and directories under the given path (recursive, max 200 entries)."""
        return tools_execute("list_dir", {"path": path}, scratch_dir)

    @lc_tool
    def finish(answer: str) -> str:
        """Terminal tool. Call exactly once when the task is complete. Pass the final answer or a brief success note."""
        return tools_execute("finish", {"answer": answer}, scratch_dir)

    return [bash, read_file, write_file, edit_file, list_dir, finish]


def run(task: Task, scratch_dir: Path) -> Trajectory:
    traj = Trajectory()

    model = ChatAnthropic(
        model=MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS_PER_CALL,
    )
    # Pass the system prompt as a SystemMessage (not a plain string) so we can
    # attach Anthropic `cache_control` to the content block. This mirrors the
    # thin and langgraph harnesses, which both cache the system prompt via
    # `cache_control: {"type": "ephemeral"}` — without it, this harness would
    # re-send the full system prompt uncached on every turn, inflating tokens_in
    # relative to the others and muddying the fairness comparison.
    agent = create_agent(
        model,
        _build_tools(scratch_dir),
        system_prompt=SystemMessage(
            content=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        ),
    )

    # LangGraph's recursion_limit counts total graph steps (agent node + tool
    # node pairs). Each "turn" in our other harnesses corresponds to one agent
    # node step + one tool node step, so give it ~2x MAX_ITERS plus a small
    # margin for the final no-tool-call exit step.
    config = {"recursion_limit": MAX_ITERS * 2 + 2}

    t0 = time.time()
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=task.prompt)]},
            config,
        )
    except Exception as e:
        traj.latency_s = time.time() - t0
        traj.stopped_reason = f"crashed:{type(e).__name__}"
        return traj
    traj.latency_s = time.time() - t0

    msgs = result.get("messages", [])
    for msg in msgs:
        if isinstance(msg, AIMessage):
            traj.num_turns += 1

            usage = getattr(msg, "usage_metadata", None) or {}
            traj.tokens_in += int(usage.get("input_tokens", 0) or 0)
            traj.tokens_out += int(usage.get("output_tokens", 0) or 0)
            details = usage.get("input_token_details") or {}
            traj.cache_read += int(details.get("cache_read", 0) or 0)
            traj.cache_write += int(details.get("cache_creation", 0) or 0)

            for tc in (msg.tool_calls or []):
                traj.tool_calls.append(
                    {
                        "name": tc["name"],
                        "input": tc.get("args", {}),
                        "result": None,
                        "id": tc.get("id"),
                    }
                )
                if tc["name"] == "finish":
                    traj.final_answer = (tc.get("args") or {}).get("answer", "")

            # Fall-through final answer: if the model produced text without
            # calling `finish`, use the last non-empty text as the answer.
            text = _message_text(msg)
            if text and not traj.final_answer:
                traj.final_answer = text

        elif isinstance(msg, ToolMessage):
            for call in reversed(traj.tool_calls):
                if call["id"] == msg.tool_call_id and call["result"] is None:
                    call["result"] = (
                        msg.content
                        if isinstance(msg.content, str)
                        else str(msg.content)
                    )
                    break

    traj.messages = [
        {"type": type(m).__name__, "content": str(getattr(m, "content", m))[:2000]}
        for m in msgs
    ]

    if traj.num_turns >= MAX_ITERS and not traj.final_answer:
        traj.stopped_reason = "iter_cap"
    else:
        traj.stopped_reason = "success" if traj.final_answer else "no_answer"
    return traj


def _message_text(msg: AIMessage) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        return "\n".join(parts)
    return ""
