"""Thin harness: provider-agnostic SDK + tool loop, no framework.

The original "minimal scaffolding" baseline. Same loop shape as before — send
prompt + tools, execute any tool calls, repeat until the model stops or
calls `finish` — but the LLM provider is now selected via the runner's
`--provider {anthropic,openai,gemini}` flag and dispatched through the small
abstraction in `harnesses/providers/`.

To add a new provider, drop a file in `harnesses/providers/<name>.py` that
implements the `Provider` protocol and register it in
`harnesses/providers/__init__.py::get_provider()`. No change here is needed.
"""

from __future__ import annotations

import time
from pathlib import Path

from harnesses.common import (
    MAX_ITERS,
    MAX_TOKENS_PER_CALL,
    SYSTEM_PROMPT,
    TEMPERATURE,
)
from harnesses.providers import get_provider
from tasks.registry import Task, Trajectory
from tools import TOOL_SCHEMAS, execute


def run(task: Task, scratch_dir: Path, provider: str = "anthropic") -> Trajectory:
    p = get_provider(provider)
    client = p.make_client()
    model = p.get_model()
    tools = p.encode_tools(TOOL_SCHEMAS)
    messages = p.initial_messages(SYSTEM_PROMPT, task.prompt)

    traj = Trajectory()
    t0 = time.time()
    for _ in range(MAX_ITERS):
        step = p.call(
            client,
            model=model,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=tools,
            max_tokens=MAX_TOKENS_PER_CALL,
            temperature=TEMPERATURE,
        )
        traj.num_turns += 1
        ti, to, cr, cw = step.usage
        traj.tokens_in += ti
        traj.tokens_out += to
        traj.cache_read += cr
        traj.cache_write += cw

        messages.append(step.assistant_message)
        traj.messages.append(step.assistant_message)

        finish_call = next(
            (tc for tc in step.tool_calls if tc.name == "finish"), None
        )
        if finish_call:
            traj.final_answer = finish_call.args.get("answer", "")
            traj.tool_calls.append(
                {"name": "finish", "input": finish_call.args, "result": "FINISHED"}
            )
            traj.stopped_reason = "success"
            break

        if not step.tool_calls:
            # Model ended turn without calling finish — treat as soft stop.
            traj.stopped_reason = "no_tool_call"
            break

        results: list = []
        for tc in step.tool_calls:
            result = execute(tc.name, tc.args, scratch_dir)
            results.append((tc, result))
            traj.tool_calls.append(
                {"name": tc.name, "input": tc.args, "result": result}
            )

        result_msgs = p.encode_tool_results(results)
        messages.extend(result_msgs)
        traj.messages.extend(result_msgs)
    else:
        traj.stopped_reason = "iter_cap"

    traj.latency_s = time.time() - t0
    return traj
