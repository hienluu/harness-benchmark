"""Thick LangGraph harness, provider-agnostic.

Deliberately structured. Each task goes through:
  planner -> router -> executor -> reflector -> router -> ... -> finalizer -> END

Every node calls the LLM via the same `Provider` abstraction `thin.py` uses
(`harnesses/providers/`), so the only thing that changes between providers
is which SDK call gets made under the hood. The graph topology, node
budgets, prompts, and accounting are identical regardless of provider —
which is the whole point of being able to compare thick-vs-thin on either
backbone independently.

Run with `--provider {anthropic,openai,gemini}` (gemini works too, since
the GeminiProvider already exists and the planner/reflector/finalizer
nodes don't need tools).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from harnesses.common import (
    MAX_ITERS,
    MAX_TOKENS_PER_CALL,
    SYSTEM_PROMPT,
    TEMPERATURE,
)
from harnesses.providers import ModelStep, Provider, get_provider
from tasks.registry import Task, Trajectory
from tools import TOOL_SCHEMAS, execute


EXECUTOR_STEP_BUDGET = 8
MAX_SUBGOALS = 6


class GState(TypedDict, total=False):
    task_prompt: str
    scratch_dir: str
    provider: str
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


def _system(extra: str = "") -> str:
    return SYSTEM_PROMPT + ("\n\n" + extra if extra else "")


def _provider_obj(state: GState) -> Provider:
    return get_provider(state["provider"])


def _bump(state: GState, step: ModelStep) -> None:
    state["turns"] = state.get("turns", 0) + 1
    ti, to, cr, cw = step.usage
    state["tokens_in"] = state.get("tokens_in", 0) + ti
    state["tokens_out"] = state.get("tokens_out", 0) + to
    state["cache_read"] = state.get("cache_read", 0) + cr
    state["cache_write"] = state.get("cache_write", 0) + cw


def planner_node(state: GState) -> GState:
    p = _provider_obj(state)
    sys_text = _system("You are the PLANNER. Decompose tasks into subgoals.")
    user_msg = (
        "Break the following task into a short ordered list of 2 to "
        f"{MAX_SUBGOALS} concrete subgoals that can be executed with the "
        "available tools. Respond with ONLY a JSON array of strings, nothing else.\n\n"
        f"TASK:\n{state['task_prompt']}"
    )
    step = p.call(
        p.make_client(),
        model=p.get_model(),
        system=sys_text,
        messages=p.initial_messages(sys_text, user_msg),
        tools=[],
        max_tokens=MAX_TOKENS_PER_CALL,
        temperature=TEMPERATURE,
    )
    _bump(state, step)

    plan = _parse_plan(step.text.strip())
    state["plan"] = plan
    state["current_idx"] = 0
    state["done"] = False
    state.setdefault("tool_calls", [])
    state.setdefault("messages_log", []).append(
        {"node": "planner", "plan": plan, "raw": step.text}
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
    # Fallback: treat each non-empty line as a subgoal.
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
    p = _provider_obj(state)
    client = p.make_client()
    scratch = Path(state["scratch_dir"])
    plan = state.get("plan", [])
    idx = state.get("current_idx", 0)
    subgoal = plan[idx] if idx < len(plan) else "Complete the task."

    sys_text = _system(
        "You are the EXECUTOR. Complete the CURRENT SUBGOAL using tools."
    )
    user_msg = (
        f"OVERALL TASK:\n{state['task_prompt']}\n\n"
        f"PLAN:\n"
        + "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan))
        + f"\n\nCURRENT SUBGOAL ({idx+1}/{len(plan)}): {subgoal}\n\n"
        "Execute this subgoal with tools. When the subgoal is done, stop "
        "calling tools. Only call `finish` if the entire task is complete."
    )
    messages = p.initial_messages(sys_text, user_msg)
    tools = p.encode_tools(TOOL_SCHEMAS)
    model = p.get_model()

    for _ in range(EXECUTOR_STEP_BUDGET):
        if state.get("turns", 0) >= MAX_ITERS:
            break
        step = p.call(
            client,
            model=model,
            system=sys_text,
            messages=messages,
            tools=tools,
            max_tokens=MAX_TOKENS_PER_CALL,
            temperature=TEMPERATURE,
        )
        _bump(state, step)

        messages.append(step.assistant_message)
        state["messages_log"].append(
            {"node": "executor", "content": step.assistant_message}
        )

        finish_call = next(
            (tc for tc in step.tool_calls if tc.name == "finish"), None
        )
        if finish_call:
            state["final_answer"] = finish_call.args.get("answer", "")
            state["done"] = True
            state["tool_calls"].append(
                {"name": "finish", "input": finish_call.args, "result": "FINISHED"}
            )
            return state
        if not step.tool_calls:
            break

        results: list = []
        for tc in step.tool_calls:
            result = execute(tc.name, tc.args, scratch)
            results.append((tc, result))
            state["tool_calls"].append(
                {"name": tc.name, "input": tc.args, "result": result}
            )
        result_msgs = p.encode_tool_results(results)
        messages.extend(result_msgs)
        state["messages_log"].append({"node": "executor", "content": result_msgs})
    return state


def reflector_node(state: GState) -> GState:
    """Ask the model if progress is acceptable and possibly revise the plan."""
    if state.get("done"):
        return state
    p = _provider_obj(state)
    plan = state.get("plan", [])
    idx = state.get("current_idx", 0)
    recent_calls = state["tool_calls"][-6:]
    sys_text = _system("You are the REFLECTOR. Decide if the plan needs revision.")
    user_msg = (
        f"PLAN: {plan}\n"
        f"JUST FINISHED SUBGOAL INDEX: {idx}\n"
        f"RECENT TOOL CALLS: {json.dumps(recent_calls, default=str)[:2000]}\n\n"
        "Respond with a JSON object: "
        '{"progress": "ok"|"revise"|"done", "new_plan": [optional list of subgoals if revise]}. '
        "Use \"done\" only if the entire task is complete, \"revise\" if the plan is wrong, "
        "\"ok\" otherwise."
    )
    step = p.call(
        p.make_client(),
        model=p.get_model(),
        system=sys_text,
        messages=p.initial_messages(sys_text, user_msg),
        tools=[],
        max_tokens=1024,
        temperature=TEMPERATURE,
    )
    _bump(state, step)

    decision = _parse_reflection(step.text.strip())
    state["messages_log"].append(
        {"node": "reflector", "decision": decision, "raw": step.text}
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
    p = _provider_obj(state)
    recent = state["tool_calls"][-8:]
    sys_text = _system("You are the FINALIZER. Produce the final answer.")
    user_msg = (
        f"TASK:\n{state['task_prompt']}\n\n"
        f"PLAN WAS: {state.get('plan')}\n"
        f"RECENT TOOL CALLS: {json.dumps(recent, default=str)[:2000]}\n\n"
        "Write the final answer to the task in plain text."
    )
    step = p.call(
        p.make_client(),
        model=p.get_model(),
        system=sys_text,
        messages=p.initial_messages(sys_text, user_msg),
        tools=[],
        max_tokens=MAX_TOKENS_PER_CALL,
        temperature=TEMPERATURE,
    )
    _bump(state, step)
    state["final_answer"] = step.text.strip()
    state["done"] = True
    state["messages_log"].append({"node": "finalizer", "answer": step.text})
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


def run(task: Task, scratch_dir: Path, provider: str = "anthropic") -> Trajectory:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()

    initial: GState = {
        "task_prompt": task.prompt,
        "scratch_dir": str(scratch_dir),
        "provider": provider,
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
