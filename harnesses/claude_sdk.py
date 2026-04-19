"""Structured-thin harness: Anthropic's Claude Agent SDK with our shared tools.

The SDK provides batteries-included scaffolding (tool loop, MCP integration,
permission model, session mgmt) around the same underlying API. This is the
"thin with official defaults" baseline — useful for distinguishing "raw loop"
from "scaffolded loop" from "framework-heavy graph".

Fairness notes:
- Same model (`MODEL`), same `MAX_ITERS` (via `max_turns`).
- Same 6 custom tools, wrapped as in-process MCP tools; built-in SDK tools
  are blocked via `disallowed_tools` and not listed in `allowed_tools`.
- CAVEAT: `ClaudeAgentOptions` does not expose `temperature` or `max_tokens`
  at this SDK version, so sampling params fall back to the SDK's defaults.
  This is an honest structural difference and should be flagged in the report.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    query,
    tool,
)

from harnesses.common import MAX_ITERS, MODEL, SYSTEM_PROMPT
from tasks.registry import Task, Trajectory
from tools import execute as tools_execute


# Tool schemas mirror the Anthropic-format schemas in `tools/__init__.py`,
# but restated here because `@tool(...)` takes an MCP-style input schema.
_SERVER_NAME = "hb"
_TOOL_PREFIX = f"mcp__{_SERVER_NAME}__"


def _build_mcp_server(scratch_dir: Path):
    """Build an MCP server whose tools dispatch into our shared tools.execute()."""

    async def _run(name: str, args: dict[str, Any]) -> dict[str, Any]:
        # Offload the blocking tool call (subprocess / file I/O) to a worker
        # thread so we don't stall the SDK's asyncio loop.
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, tools_execute, name, args, scratch_dir)
        return {"content": [{"type": "text", "text": result}]}

    @tool("bash", "Run a shell command in the task scratch directory.",
          {"command": str, "timeout_s": int})
    async def bash_t(args):
        return await _run("bash", args)

    @tool("read_file", "Read a text file relative to the scratch directory.",
          {"path": str})
    async def read_file_t(args):
        return await _run("read_file", args)

    @tool("write_file", "Overwrite a text file relative to the scratch directory.",
          {"path": str, "content": str})
    async def write_file_t(args):
        return await _run("write_file", args)

    @tool("edit_file", "Replace the first exact occurrence of old with new in a file.",
          {"path": str, "old": str, "new": str})
    async def edit_file_t(args):
        return await _run("edit_file", args)

    @tool("list_dir", "Recursively list files and directories under path.",
          {"path": str})
    async def list_dir_t(args):
        return await _run("list_dir", args)

    @tool("finish", "Terminal tool. Call once when the task is complete.",
          {"answer": str})
    async def finish_t(args):
        return await _run("finish", args)

    return create_sdk_mcp_server(
        name=_SERVER_NAME,
        version="1.0",
        tools=[bash_t, read_file_t, write_file_t, edit_file_t, list_dir_t, finish_t],
    )


_ALLOWED_TOOLS = [
    f"{_TOOL_PREFIX}{n}"
    for n in ("bash", "read_file", "write_file", "edit_file", "list_dir", "finish")
]

# Explicitly deny the SDK's built-in tools so the only tools available match
# the thin/langgraph harnesses.
_DISALLOWED_BUILTINS = [
    "Bash", "BashOutput", "Read", "Write", "Edit", "NotebookEdit",
    "Glob", "Grep", "WebSearch", "WebFetch", "Task", "TodoWrite",
    "SlashCommand",
]


async def _run_async(task: Task, scratch_dir: Path) -> Trajectory:
    traj = Trajectory()
    server = _build_mcp_server(scratch_dir)

    options = ClaudeAgentOptions(
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        max_turns=MAX_ITERS,
        mcp_servers={_SERVER_NAME: server},
        allowed_tools=_ALLOWED_TOOLS,
        disallowed_tools=_DISALLOWED_BUILTINS,
        cwd=str(scratch_dir),
        setting_sources=[],  # don't load user/project settings (reproducibility)
        permission_mode="bypassPermissions",
    )

    t0 = time.time()
    async for message in query(prompt=task.prompt, options=options):
        traj.messages.append({"type": type(message).__name__, "repr": repr(message)[:2000]})

        if isinstance(message, AssistantMessage):
            traj.num_turns += 1
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    # We only see the tool call here; result comes in UserMessage.
                    traj.tool_calls.append(
                        {
                            "name": _strip_prefix(block.name),
                            "input": block.input,
                            "result": None,
                            "id": block.id,
                        }
                    )
                    if _strip_prefix(block.name) == "finish":
                        traj.final_answer = (block.input or {}).get("answer", "")
                elif isinstance(block, TextBlock):
                    # Keep latest text as a fallback final answer.
                    if not traj.final_answer:
                        traj.final_answer = block.text

        elif isinstance(message, UserMessage):
            for block in getattr(message, "content", []) or []:
                if isinstance(block, ToolResultBlock):
                    for call in reversed(traj.tool_calls):
                        if call["id"] == block.tool_use_id and call["result"] is None:
                            call["result"] = _extract_text(block.content)
                            break

        elif isinstance(message, ResultMessage):
            u = getattr(message, "usage", None) or {}
            if isinstance(u, dict):
                traj.tokens_in += int(u.get("input_tokens", 0) or 0)
                traj.tokens_out += int(u.get("output_tokens", 0) or 0)
                traj.cache_read += int(u.get("cache_read_input_tokens", 0) or 0)
                traj.cache_write += int(u.get("cache_creation_input_tokens", 0) or 0)
            if getattr(message, "result", None) and not traj.final_answer:
                traj.final_answer = str(message.result)
            traj.stopped_reason = "success" if not message.is_error else "sdk_error"

    traj.latency_s = time.time() - t0
    if traj.num_turns >= MAX_ITERS and not traj.final_answer:
        traj.stopped_reason = "iter_cap"
    elif not traj.stopped_reason or traj.stopped_reason == "unknown":
        traj.stopped_reason = "success" if traj.final_answer else "no_answer"
    return traj


def _strip_prefix(name: str) -> str:
    return name[len(_TOOL_PREFIX):] if name.startswith(_TOOL_PREFIX) else name


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", item)))
        return "\n".join(parts)
    return str(content)


def run(task: Task, scratch_dir: Path) -> Trajectory:
    return asyncio.run(_run_async(task, scratch_dir))
