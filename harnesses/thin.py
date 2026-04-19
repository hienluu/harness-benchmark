"""Thin harness: Anthropic SDK + tool loop, no framework.

This is the "minimal scaffolding" baseline for the benchmark. It does only
what is strictly necessary to run a tool-using agent:
  1. Send user prompt + tool list to the model.
  2. If the model calls tools, execute them and append results.
  3. Repeat until the model stops or `finish` is called.
"""

from __future__ import annotations

import time
from pathlib import Path

import anthropic

from eval.tracing import wrap_anthropic_client
from harnesses.common import (
    MAX_ITERS,
    MAX_TOKENS_PER_CALL,
    MODEL,
    SYSTEM_PROMPT,
    TEMPERATURE,
)
from tasks.registry import Task, Trajectory
from tools import TOOL_SCHEMAS, execute


def run(task: Task, scratch_dir: Path) -> Trajectory:
    client = wrap_anthropic_client(anthropic.Anthropic())
    traj = Trajectory()

    system_blocks = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]
    messages: list[dict] = [{"role": "user", "content": task.prompt}]

    t0 = time.time()
    for _ in range(MAX_ITERS):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS_PER_CALL,
            temperature=TEMPERATURE,
            system=system_blocks,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )
        traj.num_turns += 1
        u = resp.usage
        traj.tokens_in += u.input_tokens
        traj.tokens_out += u.output_tokens
        traj.cache_read += getattr(u, "cache_read_input_tokens", 0) or 0
        traj.cache_write += getattr(u, "cache_creation_input_tokens", 0) or 0

        assistant_content = [b.model_dump() for b in resp.content]
        messages.append({"role": "assistant", "content": assistant_content})
        traj.messages.append({"role": "assistant", "content": assistant_content})

        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        finish_call = next((b for b in tool_uses if b.name == "finish"), None)

        if finish_call:
            traj.final_answer = finish_call.input.get("answer", "")
            traj.tool_calls.append(
                {"name": "finish", "input": finish_call.input, "result": "FINISHED"}
            )
            traj.stopped_reason = "success"
            break

        if not tool_uses:
            # Model ended turn without calling finish — treat as soft stop.
            traj.stopped_reason = "no_tool_call"
            break

        tool_results = []
        for tu in tool_uses:
            result = execute(tu.name, tu.input, scratch_dir)
            tool_results.append(
                {"type": "tool_result", "tool_use_id": tu.id, "content": result}
            )
            traj.tool_calls.append(
                {"name": tu.name, "input": tu.input, "result": result}
            )
        messages.append({"role": "user", "content": tool_results})
        traj.messages.append({"role": "user", "content": tool_results})
    else:
        traj.stopped_reason = "iter_cap"

    traj.latency_s = time.time() - t0
    return traj
