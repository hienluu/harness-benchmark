"""Run the (harness × task × trial) matrix and log per-run artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import sys
import threading
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env from the repo root if present. Optional dep; users can also export
# vars directly. We do this before importing harnesses so OPENAI_BASE_URL etc.
# are visible when the OpenAI SDK clients are constructed.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from functools import partial

from harnesses import (
    ai_agent,
    langgraph_h,
    langgraph_react,
    openai_langgraph_h,
    openai_langgraph_react,
    thin,
)
from harnesses.common import MODEL as ANTHROPIC_MODEL, usd_cost
from harnesses.openai_common import DEFAULT_OPENAI_MODEL
from harnesses.providers import SUPPORTED_PROVIDERS, get_provider
from harnesses.providers.gemini import DEFAULT_GEMINI_MODEL
from tasks.registry import Task, Trajectory, cleanup, load_all, materialize
from eval.judge import judge, score_to_normalized
from eval.tracing import (
    configure_claude_sdk_tracing,
    configure_openai_agents_tracing,
    is_enabled as tracing_enabled,
    run_context,
)


# `thin` and `ai_agent` are provider-agnostic — both honor --provider at run
# time (default: anthropic). The other harnesses are provider-pinned because
# they wrap a provider-specific framework (LangChain ChatX classes).
HARNESSES = {
    "thin": thin.run,
    "ai_agent": ai_agent.run,
    "langgraph": langgraph_h.run,
    "langgraph_react": langgraph_react.run,
    "openai_langgraph": openai_langgraph_h.run,
    "openai_langgraph_react": openai_langgraph_react.run,
}

# Harnesses whose provider is determined by --provider rather than the harness id.
PROVIDER_AGNOSTIC = {"thin", "ai_agent"}

ANTHROPIC_HARNESSES = {"langgraph", "langgraph_react"}
OPENAI_HARNESSES = {
    "openai_langgraph",
    "openai_langgraph_react",
}


def _provider_for(harness_name: str, provider: str) -> str:
    """Which provider should be used for cost lookup, API-key gating, etc.

    Provider-agnostic harnesses follow --provider; everything else is pinned
    to its module's provider by virtue of wrapping a provider-specific framework.
    """
    if harness_name in PROVIDER_AGNOSTIC:
        return provider
    return "openai" if harness_name in OPENAI_HARNESSES else "anthropic"


def print_tasks() -> None:
    """Print the task registry as an aligned table: category, id, first line of prompt."""
    tasks = sorted(load_all(), key=lambda t: (t.category, t.id))
    if not tasks:
        print("no tasks registered")
        return
    cat_w = max(len(t.category) for t in tasks)
    id_w = max(len(t.id) for t in tasks)
    header = f"{'CATEGORY':<{cat_w}}  {'ID':<{id_w}}  DESCRIPTION"
    print(header)
    print("-" * len(header))
    by_cat: dict[str, int] = {}
    for t in tasks:
        first = t.prompt.strip().split("\n", 1)[0]
        if len(first) > 80:
            first = first[:77] + "..."
        print(f"{t.category:<{cat_w}}  {t.id:<{id_w}}  {first}")
        by_cat[t.category] = by_cat.get(t.category, 0) + 1
    breakdown = ", ".join(f"{v} {k}" for k, v in sorted(by_cat.items()))
    print(f"\n{len(tasks)} tasks total ({breakdown})")


def classify_failure(traj: Trajectory, evaluator_passed: bool) -> str:
    """Post-hoc failure-mode classification from a trajectory."""
    if evaluator_passed:
        return "success"
    if traj.stopped_reason == "iter_cap":
        return "hit_iter_cap"
    if traj.stopped_reason in ("no_tool_call", "no_answer"):
        return "gave_up"
    errors = sum(
        1 for c in traj.tool_calls if isinstance(c.get("result"), str) and c["result"].startswith("ERROR:")
    )
    if errors >= 3:
        return "tool_error_loop"
    if traj.final_answer:
        return "wrong_answer"
    return "unclassified"


def run_one(
    task: Task,
    harness_name: str,
    trial_idx: int,
    out_dir: Path,
    keep_scratch: bool = False,
    provider: str = "anthropic",
) -> dict:
    scratch = materialize(task)
    start = time.time()
    # Provider-agnostic harnesses take a provider arg; everything else has a
    # fixed (task, scratch_dir) signature.
    fn = HARNESSES[harness_name]
    if harness_name in PROVIDER_AGNOSTIC:
        fn = partial(fn, provider=provider)
    try:
        with run_context(
            harness=harness_name,
            task_id=task.id,
            trial=trial_idx,
            category=task.category,
            provider=_provider_for(harness_name, provider),
        ):
            traj = fn(task, scratch)
        crashed = False
        err = None
    except Exception as e:
        traj = Trajectory()
        traj.latency_s = time.time() - start
        traj.stopped_reason = "crashed"
        crashed = True
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

    try:
        score = task.evaluator(traj, scratch)
    except Exception as e:
        from tasks.registry import Score

        score = Score(passed=False, score=0.0, details=f"evaluator crashed: {e}")

    judge_score = None
    judge_rationale = None
    final_score = score.score
    if task.category == "research" and traj.final_answer:
        rubric = getattr(task, "rubric", "") or ""
        try:
            judge_score, judge_rationale = judge(task.prompt, rubric, traj.final_answer)
            final_score = score_to_normalized(judge_score)
        except Exception as e:
            judge_rationale = f"judge error: {e}"

    passed = (
        final_score >= 0.75
        if task.category == "research"
        else bool(score.passed)
    )
    failure_mode = classify_failure(traj, passed)

    tool_counts = Counter(c["name"] for c in traj.tool_calls)

    record = {
        "task_id": task.id,
        "category": task.category,
        "harness": harness_name,
        "provider": _provider_for(harness_name, provider),
        "trial": trial_idx,
        "passed": passed,
        "score_normalized": final_score,
        "deterministic_score": score.score,
        "deterministic_details": score.details,
        "judge_score": judge_score,
        "judge_rationale": judge_rationale,
        "latency_s": traj.latency_s,
        "num_turns": traj.num_turns,
        "num_tool_calls": len(traj.tool_calls),
        "tool_counts": dict(tool_counts),
        "tokens_in": traj.tokens_in,
        "tokens_out": traj.tokens_out,
        "cache_read": traj.cache_read,
        "cache_write": traj.cache_write,
        "cost_usd": usd_cost(
            traj.tokens_in,
            traj.tokens_out,
            traj.cache_read,
            traj.cache_write,
            provider=_provider_for(harness_name, provider),
        ),
        "stopped_reason": traj.stopped_reason,
        "failure_mode": failure_mode,
        "final_answer": (traj.final_answer or "")[:4000],
        "crashed": crashed,
        "error": err,
    }

    task_dir = out_dir / harness_name / task.id
    task_dir.mkdir(parents=True, exist_ok=True)

    if keep_scratch:
        preserved = task_dir / f"trial_{trial_idx}_scratch"
        # Replace any stale preserved copy from an earlier rerun with the same out_dir.
        if preserved.exists():
            shutil.rmtree(preserved, ignore_errors=True)
        try:
            shutil.move(str(scratch), str(preserved))
            record["scratch_dir"] = str(preserved)
        except Exception as e:
            record["scratch_dir"] = None
            record["scratch_preserve_error"] = f"{type(e).__name__}: {e}"
            cleanup(scratch)
    else:
        record["scratch_dir"] = None

    (task_dir / f"trial_{trial_idx}.json").write_text(json.dumps(record, indent=2, default=str))
    (task_dir / f"trial_{trial_idx}.trajectory.json").write_text(
        json.dumps(
            {
                "tool_calls": traj.tool_calls,
                "messages": traj.messages,
            },
            indent=2,
            default=str,
        )
    )

    if not keep_scratch:
        cleanup(scratch)
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harnesses", default="thin,langgraph,langgraph_react,ai_agent")
    ap.add_argument("--tasks", default="all", help="'all' or comma-separated task ids")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument(
        "--provider",
        default="anthropic",
        choices=list(SUPPORTED_PROVIDERS),
        help=(
            "Which LLM provider the `thin` and `ai_agent` harnesses use "
            "(default: anthropic). `ai_agent` dispatches to the matching "
            "agent SDK: claude-agent-sdk for anthropic, openai-agents for "
            "openai (gemini not yet implemented for ai_agent). Other "
            "harnesses are provider-pinned and ignore this flag."
        ),
    )
    ap.add_argument(
        "--openai-model",
        default=None,
        help=(
            "Model name to use for any OpenAI-backed harness (openai_* and "
            "`thin --provider openai`). Overrides $OPENAI_MODEL. If neither is "
            f"set, defaults to {DEFAULT_OPENAI_MODEL!r}. OPENAI_API_KEY and "
            "OPENAI_BASE_URL come from the .env file (or the process "
            "environment) so this flag is the only thing you typically change "
            "to swap models or point at an OpenAI-compatible server."
        ),
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Run this many (task, harness, trial) triples concurrently via threads.",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the task registry and exit (no runs).",
    )
    ap.add_argument(
        "--keep-scratch",
        action="store_true",
        help=(
            "After each trial, move the scratch directory into "
            "results/run_<ts>/<harness>/<task>/trial_N_scratch/ instead of "
            "deleting it. Useful for inspecting the code the agent produced."
        ),
    )
    args = ap.parse_args()

    if args.list_tasks:
        print_tasks()
        return

    all_tasks = load_all()
    if args.tasks != "all":
        wanted = set(args.tasks.split(","))
        all_tasks = [t for t in all_tasks if t.id in wanted]
    harnesses = args.harnesses.split(",")

    unknown = [h for h in harnesses if h not in HARNESSES]
    if unknown:
        sys.exit(
            f"ERROR: unknown harness(es): {unknown}. "
            f"Known: {sorted(HARNESSES.keys())}"
        )

    needs_anthropic = any(h in ANTHROPIC_HARNESSES for h in harnesses)
    needs_openai = any(h in OPENAI_HARNESSES for h in harnesses)
    needs_gemini = False
    # Provider-agnostic harnesses (thin, ai_agent) follow --provider; their
    # provider also drives key gating and tracing setup.
    uses_provider_agnostic = any(h in PROVIDER_AGNOSTIC for h in harnesses)
    if uses_provider_agnostic:
        # Validate the provider is implemented for thin (the broader Provider
        # protocol). ai_agent is gated separately below since the agent SDKs
        # are independently optional.
        if "thin" in harnesses:
            try:
                get_provider(args.provider)
            except (NotImplementedError, ValueError) as e:
                sys.exit(f"ERROR: --provider {args.provider!r}: {e}")
        # ai_agent doesn't yet have a Gemini implementation (no Google ADK
        # harness). Fail fast with a clear message instead of crashing mid-run.
        if "ai_agent" in harnesses and args.provider == "gemini":
            sys.exit(
                "ERROR: ai_agent --provider gemini is not yet implemented "
                "(no Google ADK harness). Use `thin --provider gemini` for the "
                "Gemini thin harness, or pick a different --provider for ai_agent."
            )
        if args.provider == "anthropic":
            needs_anthropic = True
        elif args.provider == "openai":
            needs_openai = True
        elif args.provider == "gemini":
            needs_gemini = True

    if needs_anthropic and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "ERROR: ANTHROPIC_API_KEY is not set, but the selected harnesses "
            "include Anthropic-backed ones.\n"
            "Fix:  export ANTHROPIC_API_KEY=sk-ant-...   (or remove those "
            "harnesses with --harnesses)."
        )
    if needs_openai and not os.environ.get("OPENAI_API_KEY"):
        sys.exit(
            "ERROR: OPENAI_API_KEY is not set, but the selected harnesses "
            "include OpenAI-backed ones.\n"
            "Fix:  add OPENAI_API_KEY=... (and optionally OPENAI_BASE_URL=...) "
            "to a .env file at the repo root, or export them directly."
        )
    if needs_gemini and not (
        os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    ):
        sys.exit(
            "ERROR: GOOGLE_API_KEY (or GEMINI_API_KEY) is not set, but the "
            "selected run uses --provider gemini.\n"
            "Fix:  add GOOGLE_API_KEY=... to .env (the google-genai SDK reads "
            "it automatically — GOOGLE_API_KEY takes precedence if both are set)."
        )

    # Thread the model selection through env so harnesses (and usd_cost
    # pricing lookup) read the same value via os.environ["OPENAI_MODEL"].
    # If we don't materialize the default into the env here, usd_cost sees an
    # empty string, fails the OPENAI_PRICING prefix match, and reports $0 for
    # every OpenAI run — even though the harness itself ran fine on the default.
    if needs_openai:
        if args.openai_model:
            os.environ["OPENAI_MODEL"] = args.openai_model
        elif not os.environ.get("OPENAI_MODEL"):
            os.environ["OPENAI_MODEL"] = DEFAULT_OPENAI_MODEL
            print(
                f"WARNING: --openai-model not set and OPENAI_MODEL not in env; "
                f"defaulting to {DEFAULT_OPENAI_MODEL!r}."
            )

    # Same defaulting for Gemini — if GEMINI_MODEL isn't set in .env, materialize
    # the default into env so usd_cost and the manifest both see the same value.
    if needs_gemini and not os.environ.get("GEMINI_MODEL"):
        os.environ["GEMINI_MODEL"] = DEFAULT_GEMINI_MODEL
        print(
            f"NOTE: GEMINI_MODEL not in env; defaulting to {DEFAULT_GEMINI_MODEL!r}."
        )

    started_at = dt.datetime.now().astimezone()
    ts = started_at.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out or ROOT / "results" / f"run_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing results to {out_dir}")
    print(f"started at {started_at.isoformat(timespec='seconds')}")
    if tracing_enabled():
        project = os.environ.get("LANGSMITH_PROJECT", "(default)")
        print(f"langsmith tracing: ON (project={project})")
        # ai_agent dispatches to either claude-agent-sdk (anthropic) or
        # openai-agents (openai); install only the matching trace processor.
        if "ai_agent" in harnesses and args.provider == "anthropic":
            if not configure_claude_sdk_tracing():
                print("WARNING: failed to configure Claude SDK tracing")
        if "ai_agent" in harnesses and args.provider == "openai":
            if not configure_openai_agents_tracing():
                print("WARNING: failed to configure OpenAI Agents SDK tracing")
    else:
        print("langsmith tracing: OFF")

    manifest = {
        "timestamp": ts,
        "started_at": started_at.isoformat(timespec="seconds"),
        "started_at_human": started_at.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "harnesses": harnesses,
        "tasks": [t.id for t in all_tasks],
        "trials": args.trials,
        "workers": args.workers,
        "provider": args.provider if uses_provider_agnostic else None,
        "anthropic_model": ANTHROPIC_MODEL if needs_anthropic else None,
        "openai_model": os.environ.get("OPENAI_MODEL") if needs_openai else None,
        "openai_base_url": os.environ.get("OPENAI_BASE_URL") if needs_openai else None,
        "gemini_model": os.environ.get("GEMINI_MODEL") if needs_gemini else None,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    jobs = [
        (task, h, trial)
        for task in all_tasks
        for h in harnesses
        for trial in range(args.trials)
    ]
    total = len(jobs)
    summary: list[dict] = []
    print_lock = threading.Lock()
    t_wall = time.time()

    def _format_line(rec: dict, idx: int) -> str:
        tin = rec.get("tokens_in", 0) or 0
        tout = rec.get("tokens_out", 0) or 0
        prov = rec.get("provider", "?")
        return (
            f"[{idx}/{total}] {rec['harness']:10} [{prov:9}] {rec['task_id']:28} "
            f"trial={rec['trial']} "
            f"pass={rec['passed']} "
            f"score={rec['score_normalized']:.2f} "
            f"t={rec['latency_s']:.1f}s "
            f"turns={rec['num_turns']} "
            f"tok={tin + tout} (in={tin} out={tout}) "
            f"cost=${rec['cost_usd']:.4f} "
            f"mode={rec['failure_mode']}"
        )

    if args.keep_scratch:
        print(f"keep-scratch: ON (preserved under {out_dir}/<harness>/<task>/trial_N_scratch/)")

    if args.workers <= 1:
        for i, (task, h, trial) in enumerate(jobs, 1):
            rec = run_one(
                task, h, trial, out_dir,
                keep_scratch=args.keep_scratch,
                provider=args.provider,
            )
            summary.append(rec)
            print(_format_line(rec, i))
    else:
        print(f"parallel mode: {args.workers} workers over {total} jobs")
        completed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(
                    run_one, task, h, trial, out_dir,
                    args.keep_scratch, args.provider,
                ): (task.id, h, trial)
                for (task, h, trial) in jobs
            }
            for fut in as_completed(futures):
                try:
                    rec = fut.result()
                except Exception as e:
                    tid, h, trial = futures[fut]
                    rec = {
                        "task_id": tid, "harness": h, "trial": trial,
                        "passed": False, "score_normalized": 0.0,
                        "latency_s": 0.0, "num_turns": 0, "cost_usd": 0.0,
                        "failure_mode": f"runner_crash:{type(e).__name__}",
                    }
                summary.append(rec)
                completed += 1
                with print_lock:
                    print(_format_line(rec, completed))

    (out_dir / "summary.jsonl").write_text(
        "\n".join(json.dumps(r, default=str) for r in summary)
    )

    ended_at = dt.datetime.now().astimezone()
    duration_s = (ended_at - started_at).total_seconds()
    manifest["ended_at"] = ended_at.isoformat(timespec="seconds")
    manifest["ended_at_human"] = ended_at.strftime("%Y-%m-%d %H:%M:%S %Z")
    manifest["duration_s"] = duration_s
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(
        f"\ndone. {total} runs in {duration_s:.1f}s wall "
        f"(workers={args.workers}) -> {out_dir}"
    )


if __name__ == "__main__":
    main()
