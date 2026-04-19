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

from harnesses import claude_sdk, langgraph_h, thin
from harnesses.common import usd_cost
from tasks.registry import Task, Trajectory, cleanup, load_all, materialize
from eval.judge import judge, score_to_normalized
from eval.tracing import (
    configure_claude_sdk_tracing,
    is_enabled as tracing_enabled,
    run_context,
)


HARNESSES = {
    "thin": thin.run,
    "langgraph": langgraph_h.run,
    "claude_sdk": claude_sdk.run,
}


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
) -> dict:
    scratch = materialize(task)
    start = time.time()
    try:
        with run_context(
            harness=harness_name,
            task_id=task.id,
            trial=trial_idx,
            category=task.category,
        ):
            traj = HARNESSES[harness_name](task, scratch)
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
            traj.tokens_in, traj.tokens_out, traj.cache_read, traj.cache_write
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
    ap.add_argument("--harnesses", default="thin,langgraph,claude_sdk")
    ap.add_argument("--tasks", default="all", help="'all' or comma-separated task ids")
    ap.add_argument("--trials", type=int, default=3)
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

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit(
            "ERROR: ANTHROPIC_API_KEY is not set.\n"
            "All three harnesses call the Anthropic API; exiting before "
            "wasting time materializing scratch dirs.\n"
            "Fix:  export ANTHROPIC_API_KEY=sk-ant-..."
        )

    all_tasks = load_all()
    if args.tasks != "all":
        wanted = set(args.tasks.split(","))
        all_tasks = [t for t in all_tasks if t.id in wanted]
    harnesses = args.harnesses.split(",")

    started_at = dt.datetime.now().astimezone()
    ts = started_at.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out or ROOT / "results" / f"run_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing results to {out_dir}")
    print(f"started at {started_at.isoformat(timespec='seconds')}")
    if tracing_enabled():
        project = os.environ.get("LANGSMITH_PROJECT", "(default)")
        print(f"langsmith tracing: ON (project={project})")
        if not configure_claude_sdk_tracing():
            print("WARNING: failed to configure Claude SDK tracing")
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
        return (
            f"[{idx}/{total}] {rec['harness']:10} {rec['task_id']:28} "
            f"trial={rec['trial']} "
            f"pass={rec['passed']} "
            f"score={rec['score_normalized']:.2f} "
            f"t={rec['latency_s']:.1f}s "
            f"turns={rec['num_turns']} "
            f"cost=${rec['cost_usd']:.4f} "
            f"mode={rec['failure_mode']}"
        )

    if args.keep_scratch:
        print(f"keep-scratch: ON (preserved under {out_dir}/<harness>/<task>/trial_N_scratch/)")

    if args.workers <= 1:
        for i, (task, h, trial) in enumerate(jobs, 1):
            rec = run_one(task, h, trial, out_dir, keep_scratch=args.keep_scratch)
            summary.append(rec)
            print(_format_line(rec, i))
    else:
        print(f"parallel mode: {args.workers} workers over {total} jobs")
        completed = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(run_one, task, h, trial, out_dir, args.keep_scratch): (task.id, h, trial)
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
