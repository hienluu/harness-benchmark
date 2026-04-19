"""Aggregate runner outputs into a markdown report.

Usage:
    python analyze.py results/run_20260418_144300
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import webbrowser
from collections import Counter, defaultdict
from pathlib import Path


def load_summary(run_dir: Path) -> list[dict]:
    sp = run_dir / "summary.jsonl"
    if not sp.exists():
        raise SystemExit(f"no summary.jsonl in {run_dir}")
    return [json.loads(line) for line in sp.read_text().splitlines() if line.strip()]


def fmt_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _load_manifest(run_dir: Path) -> dict:
    mp = run_dir / "manifest.json"
    if mp.exists():
        try:
            return json.loads(mp.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def report(run_dir: Path) -> str:
    rows = load_summary(run_dir)
    manifest = _load_manifest(run_dir)
    harnesses = sorted({r["harness"] for r in rows})
    categories = sorted({r["category"] for r in rows})
    tasks = sorted({r["task_id"] for r in rows})

    out: list[str] = []
    out.append(f"# Benchmark Report — {run_dir.name}\n")
    started = manifest.get("started_at_human") or manifest.get("started_at") or "unknown"
    ended = manifest.get("ended_at_human") or manifest.get("ended_at")
    duration = manifest.get("duration_s")
    out.append(f"- **Started:** {started}")
    if ended:
        out.append(f"- **Ended:** {ended}")
    if duration is not None:
        out.append(f"- **Duration:** {duration:.1f}s wall-clock")
    if "workers" in manifest:
        out.append(f"- **Workers:** {manifest['workers']}")
    out.append(f"- **Trials per (harness, task):** {manifest.get('trials', '?')}")
    out.append(f"- **Total runs:** {len(rows)}\n")

    # ---- Overall per-harness table ----
    out.append("## Overall per-harness metrics\n")
    out.append(
        fmt_row(
            [
                "Harness",
                "Pass rate",
                "Mean score",
                "p50 latency (s)",
                "p95 latency (s)",
                "Mean turns",
                "Mean cost ($)",
                "Total cost ($)",
            ]
        )
    )
    out.append(fmt_row(["---"] * 8))
    for h in harnesses:
        hr = [r for r in rows if r["harness"] == h]
        passes = sum(1 for r in hr if r["passed"])
        scores = [r["score_normalized"] for r in hr]
        lats = sorted(r["latency_s"] for r in hr)
        costs = [r["cost_usd"] for r in hr]
        turns = [r["num_turns"] for r in hr]
        out.append(
            fmt_row(
                [
                    h,
                    f"{passes}/{len(hr)} ({100*passes/len(hr):.0f}%)",
                    f"{statistics.mean(scores):.2f}",
                    f"{lats[len(lats)//2]:.1f}",
                    f"{lats[int(len(lats)*0.95)] if lats else 0:.1f}",
                    f"{statistics.mean(turns):.1f}",
                    f"{statistics.mean(costs):.4f}",
                    f"{sum(costs):.3f}",
                ]
            )
        )
    out.append("")

    # ---- Per category pass rate ----
    out.append("## Pass rate by category\n")
    out.append(fmt_row(["Category"] + harnesses))
    out.append(fmt_row(["---"] * (len(harnesses) + 1)))
    for c in categories:
        cells = [c]
        for h in harnesses:
            hr = [r for r in rows if r["harness"] == h and r["category"] == c]
            if not hr:
                cells.append("-")
                continue
            p = sum(1 for r in hr if r["passed"])
            cells.append(f"{p}/{len(hr)} ({100*p/len(hr):.0f}%)")
        out.append(fmt_row(cells))
    out.append("")

    # ---- Per task detail ----
    out.append("## Per-task results\n")
    out.append(fmt_row(["Task"] + [f"{h} pass" for h in harnesses] + [f"{h} turns" for h in harnesses] + [f"{h} cost$" for h in harnesses]))
    out.append(fmt_row(["---"] * (1 + 3 * len(harnesses))))
    for t in tasks:
        cells = [t]
        for h in harnesses:
            tr = [r for r in rows if r["task_id"] == t and r["harness"] == h]
            passes = sum(1 for r in tr if r["passed"])
            cells.append(f"{passes}/{len(tr)}")
        for h in harnesses:
            tr = [r for r in rows if r["task_id"] == t and r["harness"] == h]
            turns = [r["num_turns"] for r in tr] or [0]
            cells.append(f"{statistics.mean(turns):.1f}")
        for h in harnesses:
            tr = [r for r in rows if r["task_id"] == t and r["harness"] == h]
            costs = [r["cost_usd"] for r in tr] or [0.0]
            cells.append(f"{statistics.mean(costs):.4f}")
        out.append(fmt_row(cells))
    out.append("")

    # ---- Failure mode breakdown ----
    out.append("## Failure mode breakdown\n")
    modes = sorted({r["failure_mode"] for r in rows})
    out.append(fmt_row(["Failure mode"] + harnesses))
    out.append(fmt_row(["---"] * (len(harnesses) + 1)))
    for m in modes:
        cells = [m]
        for h in harnesses:
            hr = [r for r in rows if r["harness"] == h]
            c = sum(1 for r in hr if r["failure_mode"] == m)
            cells.append(f"{c}/{len(hr)}")
        out.append(fmt_row(cells))
    out.append("")

    # ---- Tool call mix ----
    out.append("## Tool call mix (mean per run)\n")
    tool_names = sorted(
        {name for r in rows for name in (r.get("tool_counts") or {})}
    )
    out.append(fmt_row(["Harness"] + tool_names))
    out.append(fmt_row(["---"] * (1 + len(tool_names))))
    for h in harnesses:
        hr = [r for r in rows if r["harness"] == h]
        cells = [h]
        for tn in tool_names:
            vals = [(r.get("tool_counts") or {}).get(tn, 0) for r in hr]
            cells.append(f"{statistics.mean(vals):.1f}")
        out.append(fmt_row(cells))
    out.append("")

    # ---- Win/loss summary ----
    out.append("## Head-to-head (per task, aggregated over trials)\n")
    out.append(
        "For each task, compare the two harnesses on pass-rate first, "
        "then on mean cost as a tiebreaker."
    )
    out.append("")
    if len(harnesses) == 2:
        h1, h2 = harnesses
        wins = Counter()
        for t in tasks:
            def stat(h):
                tr = [r for r in rows if r["task_id"] == t and r["harness"] == h]
                p = sum(1 for r in tr if r["passed"]) / max(1, len(tr))
                cost = statistics.mean([r["cost_usd"] for r in tr] or [0])
                return p, cost
            p1, c1 = stat(h1)
            p2, c2 = stat(h2)
            if p1 > p2:
                wins[h1] += 1
            elif p2 > p1:
                wins[h2] += 1
            else:
                if c1 < c2:
                    wins[f"{h1} (tiebreak)"] += 1
                elif c2 < c1:
                    wins[f"{h2} (tiebreak)"] += 1
                else:
                    wins["tie"] += 1
        for k, v in wins.most_common():
            out.append(f"- {k}: {v}")
    out.append("")

    return "\n".join(out)


_HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 980px; margin: 2rem auto; padding: 0 1rem; color: #1f2328; line-height: 1.5; }}
  h1, h2 {{ border-bottom: 1px solid #d1d9e0; padding-bottom: 0.3em; }}
  code {{ background: #f6f8fa; padding: 0.1em 0.3em; border-radius: 3px; font-size: 0.9em; }}
  table {{ border-collapse: collapse; margin: 1em 0; }}
  th, td {{ border: 1px solid #d1d9e0; padding: 6px 10px; text-align: left; }}
  th {{ background: #f6f8fa; }}
  tr:nth-child(even) td {{ background: #fafbfc; }}
</style></head><body>{body}</body></html>
"""


def _render_html(md_text: str, title: str) -> str:
    import markdown as md_lib

    html = md_lib.markdown(md_text, extensions=["tables", "fenced_code"])
    return _HTML_TEMPLATE.format(title=title, body=html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", nargs="?", help="path to results/run_<ts> (default: latest)")
    ap.add_argument(
        "--open",
        action="store_true",
        help="Render report.html and open it in the default browser.",
    )
    args = ap.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        runs = sorted(Path(__file__).parent.glob("results/run_*"))
        if not runs:
            raise SystemExit("no results/run_* dirs found; pass a path")
        run_dir = runs[-1]
        print(f"using latest run: {run_dir}")

    text = report(run_dir)
    md_path = run_dir / "report.md"
    md_path.write_text(text)
    print(text)
    print(f"\nwrote {md_path}")

    if args.open:
        html_path = run_dir / "report.html"
        html_path.write_text(_render_html(text, run_dir.name))
        url = html_path.resolve().as_uri()
        print(f"wrote {html_path}")
        webbrowser.open(url)


if __name__ == "__main__":
    main()
