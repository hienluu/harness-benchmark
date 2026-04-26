"""OpenAI flavor of langgraph_react.py: prebuilt LangGraph `create_agent`
backed by `ChatOpenAI`.

Mirror of `harnesses/langgraph_react.py`. The only meaningful change is the
provider wrapper (`ChatOpenAI` instead of `ChatAnthropic`) and the system
prompt construction (no Anthropic `cache_control` block — that field is
provider-specific and silently dropped by other providers, but the OpenAI
SDK rejects unknown keys in some configurations, so we use a plain string
system prompt).

Like `openai_thin`, this harness picks up `OPENAI_API_KEY` and
`OPENAI_BASE_URL` from the environment so it works against any
OpenAI-compatible endpoint.
"""

from __future__ import annotations

import time
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool as lc_tool
from langchain_openai import ChatOpenAI

from harnesses.common import (
    MAX_ITERS,
    MAX_TOKENS_PER_CALL,
    SYSTEM_PROMPT,
    TEMPERATURE,
)
from harnesses.openai_common import get_openai_model, is_reasoning_model
from tasks.registry import Task, Trajectory
from tools import execute as tools_execute


def _build_tools(scratch_dir: Path) -> list:
    """Wrap the shared `tools.execute()` dispatcher as LangChain tools.

    Descriptions intentionally mirror the schemas in `tools/__init__.py` so
    the surface the model sees matches the other harnesses as closely as
    LangChain's tool format allows.
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

    # ChatOpenAI picks up base_url and api_key from OPENAI_BASE_URL /
    # OPENAI_API_KEY env vars automatically, which is how this harness works
    # transparently against any OpenAI-compatible server.
    model_name = get_openai_model()
    chat_kwargs: dict = {"model": model_name}
    if is_reasoning_model(model_name):
        # Reasoning models reject temperature and use max_completion_tokens
        # in place of max_tokens.
        chat_kwargs["max_completion_tokens"] = MAX_TOKENS_PER_CALL
    else:
        chat_kwargs["temperature"] = TEMPERATURE
        chat_kwargs["max_tokens"] = MAX_TOKENS_PER_CALL
    model = ChatOpenAI(**chat_kwargs)
    agent = create_agent(
        model,
        _build_tools(scratch_dir),
        system_prompt=SYSTEM_PROMPT,
    )

    # Match the LangGraph recursion-limit budget used by the Anthropic
    # langgraph_react harness so the harnesses cap at the same number of turns.
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
            # OpenAI exposes cached tokens as `cache_read`; cache_creation is
            # not a thing in the OpenAI API, so it'll just be 0 here.
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
