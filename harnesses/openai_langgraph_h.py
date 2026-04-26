"""Thick LangGraph harness, OpenAI flavor: planner / router / executor / reflector / finalizer.

Mirror of `harnesses/langgraph_h.py` with the Anthropic API calls swapped
for OpenAI Chat Completions. The graph topology, node responsibilities,
budgets, and accounting are intentionally identical so a thin-vs-thick
comparison can be done on either provider's side independently — and so a
cross-provider comparison stays apples-to-apples.

Picks up `OPENAI_API_KEY` and `OPENAI_BASE_URL` from the environment, so it
works against any OpenAI-compatible server.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, TypedDict

import openai
from langgraph.graph import END, StateGraph

from eval.tracing import wrap_openai_client
from harnesses.common import (
    MAX_ITERS,
    MAX_TOKENS_PER_CALL,
    SYSTEM_PROMPT,
    TEMPERATURE,
)
from harnesses.openai_common import (
    chat_completions_kwargs,
    extract_usage,
    get_openai_model,
    to_openai_tools,
)
from tasks.registry import Task, Trajectory
from tools import execute


EXECUTOR_STEP_BUDGET = 8
MAX_SUBGOALS = 6


class GState(TypedDict, total=False):
    task_prompt: str
    scratch_dir: str
    plan: list[str]
    current_idx: int
    final_answer: Optional[str]
    done: bool
    # accounting
    turns: int
    tokens_in: int
    tokens_out: int
    cache_read: int
    cache_write: int
    tool_calls: list[dict]
    messages_log: list[dict]


def _client() -> openai.OpenAI:
    return wrap_openai_client(openai.OpenAI())


def _system_msg(extra: str = "") -> dict:
    text = SYSTEM_PROMPT + ("\n\n" + extra if extra else "")
    return {"role": "system", "content": text}


def _bump(state: GState, resp) -> None:
    state["turns"] = state.get("turns", 0) + 1
    ti, to, cr, cw = extract_usage(resp)
    state["tokens_in"] = state.get("tokens_in", 0) + ti
    state["tokens_out"] = state.get("tokens_out", 0) + to
    state["cache_read"] = state.get("cache_read", 0) + cr
    state["cache_write"] = state.get("cache_write", 0) + cw


def _parse_args(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def planner_node(state: GState) -> GState:
    prompt = (
        "Break the following task into a short ordered list of 2 to "
        f"{MAX_SUBGOALS} concrete subgoals that can be executed with the "
        "available tools. Respond with ONLY a JSON array of strings, nothing else.\n\n"
        f"TASK:\n{state['task_prompt']}"
    )
    resp = _client().chat.completions.create(
        **chat_completions_kwargs(
            get_openai_model(),
            max_tokens=MAX_TOKENS_PER_CALL,
            temperature=TEMPERATURE,
        ),
        messages=[
            _system_msg("You are the PLANNER. Decompose tasks into subgoals."),
            {"role": "user", "content": prompt},
        ],
    )
    _bump(state, resp)

    text = (resp.choices[0].message.content or "").strip()
    plan = _parse_plan(text)
    state["plan"] = plan
    state["current_idx"] = 0
    state["done"] = False
    state.setdefault("tool_calls", [])
    state.setdefault("messages_log", []).append(
        {"node": "planner", "plan": plan, "raw": text}
    )
    return state


def _parse_plan(text: str) -> list[str]:
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, list):
                return [str(x) for x in parsed][:MAX_SUBGOALS]
        except json.JSONDecodeError:
            pass
    lines = [ln.strip(" -*0123456789.") for ln in text.splitlines() if ln.strip()]
    return lines[:MAX_SUBGOALS] or ["Complete the task."]


def router_node(state: GState) -> GState:
    if state.get("done") or state.get("current_idx", 0) >= len(state.get("plan", [])):
        state["done"] = True
    return state


def _route(state: GState) -> str:
    if state.get("done") or state.get("turns", 0) >= MAX_ITERS:
        return "finalizer"
    return "executor"


def executor_node(state: GState) -> GState:
    """Run a bounded inner tool-loop to accomplish the current subgoal."""
    scratch = Path(state["scratch_dir"])
    plan = state.get("plan", [])
    idx = state.get("current_idx", 0)
    subgoal = plan[idx] if idx < len(plan) else "Complete the task."

    user_msg = (
        f"OVERALL TASK:\n{state['task_prompt']}\n\n"
        f"PLAN:\n"
        + "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))
        + f"\n\nCURRENT SUBGOAL ({idx+1}/{len(plan)}): {subgoal}\n\n"
        "Execute this subgoal with tools. When the subgoal is done, stop "
        "calling tools. Only call `finish` if the entire task is complete."
    )
    messages: list[dict] = [
        _system_msg("You are the EXECUTOR. Complete the CURRENT SUBGOAL using tools."),
        {"role": "user", "content": user_msg},
    ]
    client = _client()
    tools = to_openai_tools()
    model = get_openai_model()

    for _ in range(EXECUTOR_STEP_BUDGET):
        if state.get("turns", 0) >= MAX_ITERS:
            break
        resp = client.chat.completions.create(
            **chat_completions_kwargs(
                model, max_tokens=MAX_TOKENS_PER_CALL, temperature=TEMPERATURE
            ),
            messages=messages,
            tools=tools,
        )
        _bump(state, resp)

        msg = resp.choices[0].message
        tool_calls = msg.tool_calls or []
        assistant_record = {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.model_dump() for tc in tool_calls] if tool_calls else None,
        }
        messages.append(assistant_record)
        state["messages_log"].append({"node": "executor", "content": assistant_record})

        finish_call = next(
            (tc for tc in tool_calls if tc.function.name == "finish"), None
        )
        if finish_call:
            args = _parse_args(finish_call.function.arguments)
            state["final_answer"] = args.get("answer", "")
            state["done"] = True
            state["tool_calls"].append(
                {"name": "finish", "input": args, "result": "FINISHED"}
            )
            return state
        if not tool_calls:
            break

        tool_results = []
        for tc in tool_calls:
            args = _parse_args(tc.function.arguments)
            result = execute(tc.function.name, args, scratch)
            tool_results.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )
            state["tool_calls"].append(
                {"name": tc.function.name, "input": args, "result": result}
            )
        messages.extend(tool_results)
        state["messages_log"].append(
            {"node": "executor", "content": tool_results}
        )
    return state


def reflector_node(state: GState) -> GState:
    """Ask the model if progress is acceptable and possibly revise the plan."""
    if state.get("done"):
        return state
    plan = state.get("plan", [])
    idx = state.get("current_idx", 0)
    recent_calls = state["tool_calls"][-6:]
    prompt = (
        f"PLAN: {plan}\n"
        f"JUST FINISHED SUBGOAL INDEX: {idx}\n"
        f"RECENT TOOL CALLS: {json.dumps(recent_calls, default=str)[:2000]}\n\n"
        "Respond with a JSON object: "
        '{"progress": "ok"|"revise"|"done", "new_plan": [optional list of subgoals if revise]}. '
        "Use \"done\" only if the entire task is complete, \"revise\" if the plan is wrong, "
        "\"ok\" otherwise."
    )
    resp = _client().chat.completions.create(
        **chat_completions_kwargs(
            get_openai_model(), max_tokens=1024, temperature=TEMPERATURE
        ),
        messages=[
            _system_msg("You are the REFLECTOR. Decide if the plan needs revision."),
            {"role": "user", "content": prompt},
        ],
    )
    _bump(state, resp)

    text = (resp.choices[0].message.content or "").strip()
    decision = _parse_reflection(text)
    state["messages_log"].append(
        {"node": "reflector", "decision": decision, "raw": text}
    )
    if decision.get("progress") == "done":
        state["done"] = True
    elif decision.get("progress") == "revise" and isinstance(
        decision.get("new_plan"), list
    ):
        state["plan"] = [str(x) for x in decision["new_plan"]][:MAX_SUBGOALS]
        state["current_idx"] = 0
    else:
        state["current_idx"] = idx + 1
    return state


def _parse_reflection(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    return {"progress": "ok"}


def finalizer_node(state: GState) -> GState:
    """If we exited without the executor calling `finish`, synthesize a final answer."""
    if state.get("final_answer") is not None:
        return state
    recent = state["tool_calls"][-8:]
    prompt = (
        f"TASK:\n{state['task_prompt']}\n\n"
        f"PLAN WAS: {state.get('plan')}\n"
        f"RECENT TOOL CALLS: {json.dumps(recent, default=str)[:2000]}\n\n"
        "Write the final answer to the task in plain text."
    )
    resp = _client().chat.completions.create(
        **chat_completions_kwargs(
            get_openai_model(),
            max_tokens=MAX_TOKENS_PER_CALL,
            temperature=TEMPERATURE,
        ),
        messages=[
            _system_msg("You are the FINALIZER. Produce the final answer."),
            {"role": "user", "content": prompt},
        ],
    )
    _bump(state, resp)
    text = (resp.choices[0].message.content or "").strip()
    state["final_answer"] = text
    state["done"] = True
    state["messages_log"].append({"node": "finalizer", "answer": text})
    return state


def _build_graph():
    g = StateGraph(GState)
    g.add_node("planner", planner_node)
    g.add_node("router", router_node)
    g.add_node("executor", executor_node)
    g.add_node("reflector", reflector_node)
    g.add_node("finalizer", finalizer_node)

    g.set_entry_point("planner")
    g.add_edge("planner", "router")
    g.add_conditional_edges(
        "router", _route, {"executor": "executor", "finalizer": "finalizer"}
    )
    g.add_edge("executor", "reflector")
    g.add_edge("reflector", "router")
    g.add_edge("finalizer", END)
    return g.compile()


_GRAPH = None


def run(task: Task, scratch_dir: Path) -> Trajectory:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()

    initial: GState = {
        "task_prompt": task.prompt,
        "scratch_dir": str(scratch_dir),
        "plan": [],
        "current_idx": 0,
        "done": False,
        "turns": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "cache_read": 0,
        "cache_write": 0,
        "tool_calls": [],
        "messages_log": [],
    }

    t0 = time.time()
    final = _GRAPH.invoke(initial, config={"recursion_limit": 60})
    elapsed = time.time() - t0

    traj = Trajectory()
    traj.final_answer = final.get("final_answer")
    traj.num_turns = final.get("turns", 0)
    traj.tokens_in = final.get("tokens_in", 0)
    traj.tokens_out = final.get("tokens_out", 0)
    traj.cache_read = final.get("cache_read", 0)
    traj.cache_write = final.get("cache_write", 0)
    traj.tool_calls = final.get("tool_calls", [])
    traj.messages = final.get("messages_log", [])
    traj.latency_s = elapsed
    if traj.num_turns >= MAX_ITERS and not final.get("final_answer"):
        traj.stopped_reason = "iter_cap"
    else:
        traj.stopped_reason = "success" if traj.final_answer else "no_answer"
    return traj
