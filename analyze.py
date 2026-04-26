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


def _human_tokens(n: float) -> str:
    """Compact human-readable token counts: 1234 -> 1.2k, 1_500_000 -> 1.5M."""
    n = float(n or 0)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return f"{n:.0f}"


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
    # Group by (harness, provider) so re-running `thin` against multiple
    # providers (and concatenating summary.jsonl files) shows up as separate
    # rows. Records without a `provider` field (older runs) fall back to "?".
    pairs = sorted({(r["harness"], r.get("provider", "?")) for r in rows})
    categories = sorted({r["category"] for r in rows})
    tasks = sorted({r["task_id"] for r in rows})

    # The common case is a single-provider run — labels and columns stay clean.
    # Only when the same harness appears with multiple providers (e.g. someone
    # concatenated summary.jsonl from runs with different --provider flags) do
    # we add a "(provider)" suffix to disambiguate.
    providers_per_harness: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        providers_per_harness[r["harness"]].add(r.get("provider", "?"))

    def _label(harness: str, provider: str) -> str:
        if len(providers_per_harness[harness]) > 1:
            return f"{harness} ({provider})"
        return harness

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
    out.append(f"- **Total runs:** {len(rows)}")

    # Provider summary: single line for the common case; grouped breakdown
    # when a single run mixes providers (or summary.jsonl was concatenated).
    # Suppress entirely for legacy runs that don't have a provider field.
    known_providers = sorted(
        {r.get("provider", "?") for r in rows} - {"?"}
    )
    if len(known_providers) == 1:
        line = f"- **Provider:** {known_providers[0]}"
        # Surface the model name when we can — comes from the run's manifest
        # (set by the runner when --provider is the matching one).
        if known_providers[0] == "openai" and manifest.get("openai_model"):
            line += f" ({manifest['openai_model']})"
        elif known_providers[0] == "gemini" and manifest.get("gemini_model"):
            line += f" ({manifest['gemini_model']})"
        out.append(line)
    elif known_providers:
        # Mixed providers — group harnesses by provider so it's easy to see
        # who used what.
        by_prov: dict[str, set[str]] = defaultdict(set)
        for r in rows:
            p = r.get("provider")
            if p:
                by_prov[p].add(r["harness"])
        breakdown = "; ".join(
            f"**{p}** ({', '.join(sorted(by_prov[p]))})" for p in known_providers
        )
        out.append(f"- **Providers:** {breakdown}")
    out.append("")

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
                "Mean tokens (in/out)",
                "Total tokens",
                "Mean cost ($)",
                "Total cost ($)",
            ]
        )
    )
    out.append(fmt_row(["---"] * 10))
    for h, prov in pairs:
        hr = [r for r in rows if r["harness"] == h and r.get("provider", "?") == prov]
        passes = sum(1 for r in hr if r["passed"])
        scores = [r["score_normalized"] for r in hr]
        lats = sorted(r["latency_s"] for r in hr)
        costs = [r["cost_usd"] for r in hr]
        turns = [r["num_turns"] for r in hr]
        tin = [r.get("tokens_in", 0) or 0 for r in hr]
        tout = [r.get("tokens_out", 0) or 0 for r in hr]
        mean_in = statistics.mean(tin) if tin else 0
        mean_out = statistics.mean(tout) if tout else 0
        total_tokens = sum(tin) + sum(tout)
        out.append(
            fmt_row(
                [
                    _label(h, prov),
                    f"{passes}/{len(hr)} ({100*passes/len(hr):.0f}%)",
                    f"{statistics.mean(scores):.2f}",
                    f"{lats[len(lats)//2]:.1f}",
                    f"{lats[int(len(lats)*0.95)] if lats else 0:.1f}",
                    f"{statistics.mean(turns):.1f}",
                    f"{_human_tokens(mean_in)} / {_human_tokens(mean_out)}",
                    _human_tokens(total_tokens),
                    f"{statistics.mean(costs):.4f}",
                    f"{sum(costs):.3f}",
                ]
            )
        )
    out.append("")

    # ---- Per category pass rate ----
    labels = [_label(h, p) for h, p in pairs]
    out.append("## Pass rate by category\n")
    out.append(fmt_row(["Category"] + labels))
    out.append(fmt_row(["---"] * (len(pairs) + 1)))
    for c in categories:
        cells = [c]
        for h, prov in pairs:
            hr = [
                r for r in rows
                if r["harness"] == h and r.get("provider", "?") == prov and r["category"] == c
            ]
            if not hr:
                cells.append("-")
                continue
            p = sum(1 for r in hr if r["passed"])
            cells.append(f"{p}/{len(hr)} ({100*p/len(hr):.0f}%)")
        out.append(fmt_row(cells))
    out.append("")

    # ---- Per task detail ----
    out.append("## Per-task results\n")
    out.append(
        fmt_row(
            ["Task"]
            + [f"{lbl} pass" for lbl in labels]
            + [f"{lbl} turns" for lbl in labels]
            + [f"{lbl} tokens" for lbl in labels]
            + [f"{lbl} cost$" for lbl in labels]
        )
    )
    out.append(fmt_row(["---"] * (1 + 4 * len(pairs))))

    def _select(t: str, h: str, prov: str) -> list[dict]:
        return [
            r for r in rows
            if r["task_id"] == t and r["harness"] == h and r.get("provider", "?") == prov
        ]

    for t in tasks:
        cells = [t]
        for h, prov in pairs:
            tr = _select(t, h, prov)
            passes = sum(1 for r in tr if r["passed"])
            cells.append(f"{passes}/{len(tr)}" if tr else "-")
        for h, prov in pairs:
            tr = _select(t, h, prov)
            turns = [r["num_turns"] for r in tr] or [0]
            cells.append(f"{statistics.mean(turns):.1f}" if tr else "-")
        for h, prov in pairs:
            tr = _select(t, h, prov)
            toks = [
                (r.get("tokens_in", 0) or 0) + (r.get("tokens_out", 0) or 0)
                for r in tr
            ] or [0]
            cells.append(_human_tokens(statistics.mean(toks)) if tr else "-")
        for h, prov in pairs:
            tr = _select(t, h, prov)
            costs = [r["cost_usd"] for r in tr] or [0.0]
            cells.append(f"{statistics.mean(costs):.4f}" if tr else "-")
        out.append(fmt_row(cells))
    out.append("")

    # ---- Failure mode breakdown ----
    out.append("## Failure mode breakdown\n")
    modes = sorted({r["failure_mode"] for r in rows})
    out.append(fmt_row(["Failure mode"] + labels))
    out.append(fmt_row(["---"] * (len(pairs) + 1)))
    for m in modes:
        cells = [m]
        for h, prov in pairs:
            hr = [
                r for r in rows
                if r["harness"] == h and r.get("provider", "?") == prov
            ]
            c = sum(1 for r in hr if r["failure_mode"] == m)
            cells.append(f"{c}/{len(hr)}" if hr else "-")
        out.append(fmt_row(cells))
    out.append("")

    # ---- Tool call mix ----
    out.append("## Tool call mix (mean per run)\n")
    tool_names = sorted(
        {name for r in rows for name in (r.get("tool_counts") or {})}
    )
    out.append(fmt_row(["Harness"] + tool_names))
    out.append(fmt_row(["---"] * (1 + len(tool_names))))
    for (h, prov), lbl in zip(pairs, labels):
        hr = [
            r for r in rows
            if r["harness"] == h and r.get("provider", "?") == prov
        ]
        cells = [lbl]
        for tn in tool_names:
            vals = [(r.get("tool_counts") or {}).get(tn, 0) for r in hr]
            cells.append(f"{statistics.mean(vals):.1f}" if vals else "-")
        out.append(fmt_row(cells))
    out.append("")

    # ---- Win/loss summary ----
    out.append("## Head-to-head (per task, aggregated over trials)\n")
    out.append(
        "For each task, compare the two harnesses on pass-rate first, "
        "then on mean cost as a tiebreaker."
    )
    out.append("")
    if len(pairs) == 2:
        (h1, p1_prov), (h2, p2_prov) = pairs
        l1, l2 = labels
        wins = Counter()
        for t in tasks:
            def stat(h, prov):
                tr = [
                    r for r in rows
                    if r["task_id"] == t
                    and r["harness"] == h
                    and r.get("provider", "?") == prov
                ]
                p = sum(1 for r in tr if r["passed"]) / max(1, len(tr))
                cost = statistics.mean([r["cost_usd"] for r in tr] or [0])
                return p, cost
            pa, ca = stat(h1, p1_prov)
            pb, cb = stat(h2, p2_prov)
            if pa > pb:
                wins[l1] += 1
            elif pb > pa:
                wins[l2] += 1
            else:
                if ca < cb:
                    wins[f"{l1} (tiebreak)"] += 1
                elif cb < ca:
                    wins[f"{l2} (tiebreak)"] += 1
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
