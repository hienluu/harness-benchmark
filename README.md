# Thin vs. Thick Harness Benchmark

A controlled head-to-head comparison of three agent harnesses — raw thin loop, Claude Agent SDK (structured thin), and LangGraph (thick) — on the same tasks, same tools, same model. Answers: does a thick framework-based harness buy you anything over a simple tool loop, or does it just cost tokens and latency? And where does "structured thin" (official SDK defaults) land between those extremes?

## Design

- **Thin harness** (`harnesses/thin.py`): Anthropic SDK + while-loop. No framework.
- **Claude Agent SDK harness** (`harnesses/claude_sdk.py`): structured-thin — Anthropic's official agent SDK with our six custom tools exposed as an in-process MCP server. Built-in SDK tools are explicitly denied.
- **Thick harness** (`harnesses/langgraph_h.py`): LangGraph StateGraph with planner → router → executor → reflector nodes.
- **Shared:** model (`claude-sonnet-4-6`), system prompt, max iterations, tool implementations (`tools/`), task suite, evaluators. The only variable is harness structure. Note: the Claude Agent SDK does not expose `temperature` / `max_tokens` via its options surface, so those fall back to SDK defaults — flagged in the report as an honest structural difference.
- **Tasks** (10): 4 code fixes (pytest-scored), 3 tool-chain tasks (file-check scored), 3 research tasks (LLM-judge scored).
- **Metrics:** pass rate, latency, token cost (with prompt caching), num turns, tool-call mix, failure-mode breakdown.

## Run

This project uses [uv](https://docs.astral.sh/uv/) for dependency and env management. `pyproject.toml` is the source of truth.

```bash
# Install deps (creates .venv, writes uv.lock)
uv sync

# Run the full matrix (3 harnesses × 10 tasks × 3 trials = 90 runs)
export ANTHROPIC_API_KEY=sk-ant-...
uv run python eval/runner.py --trials 3

# Speed it up: run N (task, harness, trial) triples concurrently
uv run python eval/runner.py --trials 3 --workers 8

# Subset harnesses or tasks
uv run python eval/runner.py --harnesses thin,claude_sdk --tasks code_merge_intervals,tool_csv_mean --trials 1

# Keep each trial's scratch dir around to inspect the code the agent wrote
uv run python eval/runner.py --tasks code_lru_cache --trials 1 --keep-scratch
# → results/run_<ts>/<harness>/code_lru_cache/trial_0_scratch/cache.py

# Aggregate the latest run into a markdown report
uv run python analyze.py

# Also render report.html and open it in the default browser
uv run python analyze.py --open
```

Harness ids: `thin`, `claude_sdk`, `langgraph`.

List all tasks (category, id, first line of prompt):
```bash
uv run python eval/runner.py --list-tasks
```

Results land in `results/run_<timestamp>/` with per-run `trial_N.json` (metrics) and `trial_N.trajectory.json` (full message history).

## Layout

```
harnesses/   thin.py + claude_sdk.py + langgraph_h.py + common.py (shared constants)
tools/       bash.py, fs.py, __init__.py (shared schemas + dispatcher)
tasks/       10 Task modules under code/, tool_chain/, research/
eval/        runner.py (matrix execution), judge.py (LLM-as-judge for research)
analyze.py   aggregation -> markdown report
```

## Tasks

Ten tasks across three categories. All tasks share the same six-tool surface (`bash`, `read_file`, `write_file`, `edit_file`, `list_dir`, `finish`) and run in an isolated scratch directory. Run `uv run python eval/runner.py --list-tasks` to see them at a glance.

### Code (4, deterministic pytest scoring)
Each task ships a buggy or stub source file plus hidden tests; the evaluator runs pytest and passes/fails the trial.

| ID | What the agent must do |
| --- | --- |
| `code_merge_intervals` | Fix an off-by-one bug in `merge_intervals` — touching intervals like `[1,4]` + `[4,5]` must merge. |
| `code_lru_cache` | Implement an `LRUCache` class from its docstring spec; eviction + update-refresh must be correct. |
| `code_import_chain` | Fix a broken multi-file import chain so `python main.py` prints `20`. |
| `code_json_parser` | Make `safe_parse` honor its docstring: handle `None`, empty strings, and invalid JSON by returning the default. |

### Tool chain (3, deterministic file-check scoring)
Multi-step tool sequences; the evaluator inspects files the agent wrote.

| ID | What the agent must do |
| --- | --- |
| `tool_find_todos` | Find all `.py` files under `src/` containing `TODO`, write paths (sorted, one per line) to `todos.txt`. |
| `tool_csv_mean` | Read `data.csv`, compute the mean of the `value` column, write `{"mean": <float>}` to `out.json`. |
| `tool_rename_symbol` | Rename function `foo` → `bar` across a small package with no references left and `main.py` still passing. |

### Research (3, LLM-judge scoring on a 1-5 rubric)
Open-ended writing; the judge sees task + final answer only (not the trajectory) to keep outcome evaluation independent of process.

| ID | What the agent must do |
| --- | --- |
| `research_compare_specs` | Read three cache specs, write a 150–200 word comparison across consistency, ops burden, cost, use case. |
| `research_bug_rca` | Given a bug report + stack trace, write a root-cause hypothesis and a 3-5 step fix plan. |
| `research_rate_limiter` | Design a per-key distributed rate limiter: pseudocode, tradeoffs discussed, one failure mode + mitigation. |

## Observability (optional)

Opt-in LangSmith tracing. When enabled, every model call is traced with token counts, latency, and a nested view of planner → router → executor calls for the LangGraph harness. Each run is tagged with `harness`, `task_id`, `trial`, and `category` so you can filter in the LangSmith UI.

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=lsv2_...
export LANGSMITH_PROJECT=harness-benchmark   # optional
uv run python eval/runner.py --trials 3 --workers 8
```

When the env vars are unset, tracing is a complete no-op — no network calls, no import cost at hot path, no behavior change.

**Coverage:**
- `thin` + `langgraph` — Anthropic client wrapped via `langsmith.wrappers.wrap_anthropic`; full per-call traces with token usage attached to each span.
- `claude_sdk` — uses LangSmith's first-class `configure_claude_agent_sdk()` integration (from the `langsmith[claude-agent-sdk]` extra), called once at runner startup. Traces agent queries, tool invocations, model calls, and MCP server operations natively — no hand-rolled spans. **Caveat:** subagent tool calls aren't captured today (see [langchain-ai/langsmith-sdk#2091](https://github.com/langchain-ai/langsmith-sdk/issues/2091)); the `claude_sdk` harness here doesn't use subagents, so that limitation doesn't apply.

## Fairness audit

All three harnesses share:
- `harnesses/common.py::MODEL` — same model (`claude-sonnet-4-6`).
- `harnesses/common.py::SYSTEM_PROMPT` — same base system prompt. The LangGraph harness prepends node-specific roles (planner, executor, reflector, finalizer) on top; that extra scaffolding *is* the point of the comparison.
- `tools/` — identical tool implementations and schemas. The Claude Agent SDK harness exposes them as an in-process MCP server (`mcp__hb__*`); the other two use native Anthropic tool-use format. The underlying `tools.execute()` dispatcher is the same for all three.
- `harnesses/common.py::MAX_ITERS` — same model-call budget (mapped to `max_turns` in the Claude SDK harness).
- Prompt caching on the system prompt for thin and langgraph; the Claude SDK handles caching on its own.

**Known structural differences (not confounds to hide — call them out in the writeup):**
- `temperature` / `max_tokens`: thin and langgraph are pinned to `temperature=0.0`, `max_tokens=4096`. `ClaudeAgentOptions` in `claude-agent-sdk` 0.1.63 does not expose these, so that harness uses SDK defaults.
- Built-in tools: the Claude Agent SDK ships with its own `Bash`/`Read`/`Edit`/`WebSearch`/etc. — we deny them via `disallowed_tools` and omit them from `allowed_tools` so the agent can only call our six custom tools. This is enforced by an allowlist, not hard-disabled.
- `setting_sources=[]` on the Claude SDK harness so user/project `CLAUDE.md` and other settings don't leak into reproducibility.
