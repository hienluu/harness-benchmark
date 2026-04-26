# Thin vs. Thick Harness Benchmark

Licensed under the [Apache License 2.0](LICENSE).


A controlled head-to-head comparison of agent harnesses — raw thin loop, official agent SDK (structured thin), LangGraph `create_agent` (idiomatic thin-in-LangGraph baseline), and a custom thick LangGraph topology (planner/router/executor/reflector) — on the same tasks, same tools, same model within a single run. Answers: does a thick framework-based harness buy you anything over a simple tool loop, or does it just cost tokens and latency? And if a thick topology *does* cost more, is the cost attributable to the framework (LangGraph) or to the topology on top of it?

**Every harness is provider-agnostic.** Pick the LLM backend with `--provider {anthropic,openai,gemini}`. The OpenAI provider uses the Chat Completions API, so it works against any OpenAI-compatible server (vLLM, Ollama, OpenRouter, Together, LM Studio, …) — letting the same matrix run against open-source models with no code changes. The thin/langgraph/langgraph_react/ai_agent files share a `Provider` abstraction (`harnesses/providers/`) plus tiny per-harness dispatchers, so adding a fourth provider is one new file. To compare *across* providers, do separate runs and concatenate the resulting `summary.jsonl`s — `analyze.py` will surface them side-by-side.

## Design

- **Thin harness** (`harnesses/thin.py`): provider-agnostic SDK + while-loop. No framework. Provider is selected with `--provider {anthropic,openai,gemini}`; the format-specific bits live in `harnesses/providers/`. To add a new provider, drop in `harnesses/providers/<name>.py`.
- **Structured-thin harness** (`harnesses/ai_agent.py`): provider-agnostic dispatcher that routes to the matching official agent SDK based on `--provider` — `claude-agent-sdk` (anthropic) or `openai-agents` (openai). The actual SDK-specific implementations live in `claude_sdk.py` and `openai_agents.py`; `ai_agent.py` just picks one. Built-in SDK tools are explicitly denied where applicable.
- **LangGraph prebuilt harness** (`harnesses/langgraph_react.py`): the prebuilt idiomatic LangGraph agent (`create_agent`) — a simple LLM → ToolNode loop. Provider-agnostic via `--provider` (anthropic / openai today; gemini requires the optional `langchain-google-genai` dep); LangChain handles message format and tool schemas, the harness just picks the right `ChatX` class. Serves as the baseline for "LangGraph as LangGraph wants to be used" so the thick harness's cost can be split into "framework tax" vs. "our custom topology tax."
- **Thick harness** (`harnesses/langgraph_h.py`): LangGraph StateGraph with planner → router → executor → reflector nodes. Provider-agnostic via `--provider`; every node calls the LLM through the same `Provider` abstraction `thin.py` uses, so the only thing that changes between providers is which SDK call gets made under the hood.
- **Shared within a run:** model (per-provider default — `claude-sonnet-4-6` for anthropic, `--openai-model` for openai, `GEMINI_MODEL` env for gemini), system prompt, max iterations, tool implementations (`tools/`), task suite, evaluators. The only variable is harness structure. Note: the Claude Agent SDK does not expose `temperature` / `max_tokens` via its options surface, so the `ai_agent --provider anthropic` path falls back to SDK defaults — flagged in the report as an honest structural difference.
- **Tasks** (10): 4 code fixes (pytest-scored), 3 tool-chain tasks (file-check scored), 3 research tasks (LLM-judge scored).
- **Metrics:** pass rate, latency, token cost (with prompt caching), num turns, tool-call mix, failure-mode breakdown.

## Run

This project uses [uv](https://docs.astral.sh/uv/) for dependency and env management. `pyproject.toml` is the source of truth.

```bash
# Install deps (creates .venv, writes uv.lock)
uv sync

# Configure secrets in .env (see .env.example for the full template).
# At minimum: ANTHROPIC_API_KEY for --provider anthropic, OPENAI_API_KEY for
# --provider openai, GOOGLE_API_KEY for --provider gemini, and optionally
# LANGSMITH_* for tracing. Research tasks always need ANTHROPIC_API_KEY (the
# LLM-as-judge is pinned to Anthropic for cross-provider scoring consistency).
cp .env.example .env && $EDITOR .env

# Run the default Anthropic matrix (4 harnesses × 10 tasks × 3 trials = 120 runs)
uv run python eval/runner.py --trials 3

# Speed it up: run N (task, harness, trial) triples concurrently
uv run python eval/runner.py --trials 3 --workers 8

# Subset harnesses or tasks
uv run python eval/runner.py --harnesses thin,ai_agent --tasks code_merge_intervals,tool_csv_mean --trials 1

# Keep each trial's scratch dir around to inspect the code the agent wrote
uv run python eval/runner.py --tasks code_lru_cache --trials 1 --keep-scratch
# → results/run_<ts>/<harness>/code_lru_cache/trial_0_scratch/cache.py

# Aggregate the latest run into a markdown report
uv run python analyze.py

# Also render report.html and open it in the default browser
uv run python analyze.py --open
```

Harness ids: `thin`, `ai_agent`, `langgraph_react`, `langgraph`. **All four are provider-agnostic** — pair any of them with `--provider anthropic` (default), `--provider openai`, or `--provider gemini`. Caveats: `ai_agent --provider gemini` is not yet implemented (no Google ADK harness exists), and `langgraph_react --provider gemini` requires the optional `langchain-google-genai` dep. `thin` and `langgraph` support all three providers via the shared `harnesses/providers/` abstraction.

### OpenAI / OpenAI-compatible models

Configuration lives in a `.env` file at the repo root (read automatically via `python-dotenv`). The `--openai-model` flag picks the model name; everything else (key, endpoint) comes from `.env`, which means **swapping between official OpenAI, vLLM, Ollama, OpenRouter, Together, LM Studio, etc. is just changing two env vars** — no code changes.

```bash
# .env at repo root
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1     # default — omit for OpenAI proper

# Run all four harnesses against OpenAI
uv run python eval/runner.py \
  --harnesses thin,ai_agent,langgraph_react,langgraph \
  --provider openai \
  --openai-model gpt-5 \
  --tasks code_merge_intervals --trials 1
```

Point at any OpenAI-compatible server by changing `OPENAI_BASE_URL`:

```bash
# Ollama serving qwen2.5-coder
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
uv run python eval/runner.py --harnesses thin --provider openai --openai-model qwen2.5-coder --tasks tool_csv_mean --trials 1

# vLLM, OpenRouter, Together, LM Studio, ... — same shape
```

Same code in every case — only `--provider` (and for OpenAI, `OPENAI_BASE_URL` + `--openai-model`) swaps the backend. Each trial reports tokens, latency, and cost; pricing is provider-aware (Anthropic prices for anthropic runs, OpenAI prices for openai runs, Gemini prices for gemini runs; self-hosted / unknown models report cost = 0). The provider used for the run shows up in the report header (e.g. `**Provider:** openai (gpt-5)`).

### Gemini

Same shape as the OpenAI flow — drop the key in `.env` and pick the model with `GEMINI_MODEL` (default `gemini-2.5-pro`):

```bash
# .env
GOOGLE_API_KEY=AIza...
# GEMINI_MODEL=gemini-2.5-flash    # optional override

uv run python eval/runner.py --harnesses thin --provider gemini --tasks tool_csv_mean --trials 1
```

Three-way comparison on a single task:

```bash
uv run python eval/runner.py --harnesses thin --provider anthropic --out results/run_anth --tasks tool_csv_mean --trials 1
uv run python eval/runner.py --harnesses thin --provider openai    --out results/run_oai  --tasks tool_csv_mean --trials 1
uv run python eval/runner.py --harnesses thin --provider gemini    --out results/run_gem  --tasks tool_csv_mean --trials 1
```

List all tasks (category, id, first line of prompt):
```bash
uv run python eval/runner.py --list-tasks
```

Results land in `results/run_<timestamp>/` with `manifest.json` (run config — harnesses, provider, models, base URL), `summary.jsonl` (one row per trial), and per-trial `trial_N.json` + `trial_N.trajectory.json` (full message history). `analyze.py` reads these and emits `report.md` with:
- A header showing the provider and model in use (e.g. `**Provider:** openai (gpt-5)`).
- Per-harness aggregates: pass rate, p50/p95 latency, mean turns, mean tokens (in/out), total tokens, mean / total cost.
- Per-task and per-failure-mode breakdowns.
- The provider name is also included on every per-trial JSON record so concatenated runs (cross-provider comparisons) render correctly.

## Layout

```
harnesses/             thin.py, ai_agent.py, langgraph_react.py, langgraph_h.py
                       (all four provider-agnostic),
                       claude_sdk.py + openai_agents.py (agent-SDK impls dispatched by ai_agent),
                       common.py + openai_common.py (shared constants)
harnesses/providers/   __init__.py (Provider protocol + factory), anthropic.py, openai.py, gemini.py
tools/                 bash.py, fs.py, __init__.py (shared schemas + dispatcher)
tasks/                 10 Task modules under code/, tool_chain/, research/
eval/                  runner.py (matrix execution), judge.py (LLM-as-judge for research)
analyze.py             aggregation -> markdown report
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

Add the LangSmith vars to `.env` (uncomment them in `.env.example`):

```bash
# .env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=harness-benchmark   # optional
```

Then run normally:

```bash
uv run python eval/runner.py --trials 3 --workers 8
```

When `LANGSMITH_TRACING` is unset (or falsy), tracing is a complete no-op — no network calls, no import cost at hot path, no behavior change.

**Coverage:**
- `thin` and `langgraph` (any `--provider`) — Anthropic / OpenAI / Gemini clients wrapped via `langsmith.wrappers.wrap_anthropic` / `wrap_openai` / `wrap_gemini` (all transparent to the harness via `harnesses/providers/`); full per-call traces with token usage attached to each span.
- `thin` (`--provider gemini`) — google-genai client wrapped via `langsmith.wrappers.wrap_gemini` (beta in LangSmith ≥ 0.4.33); patches `models.generate_content` so each tool-loop step shows up as a traced span with token usage and tool calls.
- `langgraph_react` (any `--provider`) — LangChain / LangGraph emit LangSmith spans natively when `LANGSMITH_TRACING=true`; every agent and tool node step is traced without any extra wiring, regardless of which `ChatX` provider class backs the model.
- `ai_agent` (`--provider anthropic`) — dispatches to the Claude Agent SDK; LangSmith's first-class `configure_claude_agent_sdk()` integration (from the `langsmith[claude-agent-sdk]` extra) is installed at runner startup when this combination is selected. Traces agent queries, tool invocations, model calls, and MCP server operations natively. **Caveat:** subagent tool calls aren't captured today (see [langchain-ai/langsmith-sdk#2091](https://github.com/langchain-ai/langsmith-sdk/issues/2091)); the harness here doesn't use subagents, so that limitation doesn't apply.
- `ai_agent` (`--provider openai`) — dispatches to the OpenAI Agents SDK; the `OpenAIAgentsTracingProcessor` (from the `langsmith[openai-agents]` extra) is installed at runner startup when this combination is selected. Parallel to the Claude SDK integration above.

## Fairness audit

All four harnesses in a single run share:
- **Model.** All harnesses in a given run use the same model — `claude-sonnet-4-6` for `--provider anthropic` (from `harnesses/common.py::MODEL`), the value of `--openai-model` / `OPENAI_MODEL` for `--provider openai`, and `GEMINI_MODEL` for `--provider gemini`. **Cross-provider comparisons are not "controlled" in the way same-provider comparisons are** — they're a different kind of measurement (capability comparison rather than harness-overhead comparison).
- **System prompt.** Same `harnesses/common.py::SYSTEM_PROMPT` across all four. The thick LangGraph harness prepends node-specific roles (planner, executor, reflector, finalizer) on top; that extra scaffolding *is* the point of the comparison.
- **Tools.** Identical implementations and schemas in `tools/`. The Claude Agent SDK exposes them as an in-process MCP server (`mcp__hb__*`); the OpenAI Agents SDK exposes them as `@function_tool` functions; the other harnesses go through native tool-use format via `harnesses/providers/`. The underlying `tools.execute()` dispatcher is the same for all of them.
- **Iteration budget.** Same `harnesses/common.py::MAX_ITERS` (mapped to `max_turns` in the agent SDKs).
- **Prompt caching.** `cache_control: {"type": "ephemeral"}` is attached to the system message for thin and langgraph when `--provider anthropic` (Anthropic-only feature). The Claude Agent SDK handles caching internally. The OpenAI provider reports `cache_read` only when the official OpenAI API auto-caches inputs (most OpenAI-compatible servers omit this field). Gemini reports `cache_read` only when the input matched a manually-created context cache. `cache_write` is Anthropic-only.
- **LLM-as-judge.** `eval/judge.py` is intentionally pinned to Anthropic / `MODEL` regardless of which harness produced the answer — a single judge keeps cross-provider rubric scoring consistent. Requires `ANTHROPIC_API_KEY` even in OpenAI-only or Gemini-only runs that include research tasks.

**Known structural differences (not confounds to hide — call them out in the writeup):**
- `temperature` / `max_tokens`: thin, langgraph, and langgraph_react are pinned to `temperature=0.0`, `max_tokens=4096`. `ClaudeAgentOptions` in `claude-agent-sdk` 0.1.63 does not expose these, so `ai_agent --provider anthropic` uses SDK defaults. Reasoning models on the OpenAI side (gpt-5 / o1 / o3 / o4) reject `temperature` entirely and use `max_completion_tokens` — handled inside `openai_common.chat_completions_kwargs`.
- Built-in tools: the Claude Agent SDK ships with its own `Bash`/`Read`/`Edit`/`WebSearch`/etc. — we deny them via `disallowed_tools` and omit them from `allowed_tools` so the agent can only call our six custom tools. Enforced by an allowlist, not hard-disabled.
- `setting_sources=[]` on `ai_agent --provider anthropic` so user/project `CLAUDE.md` and other settings don't leak into reproducibility.

## TODOs

### Multi-provider harnesses

Done for Anthropic and OpenAI; mostly done for Gemini. The matrix is now `(harness × task × trial × provider)` where `harness ∈ {thin, ai_agent, langgraph_react, langgraph}` (4) and `provider ∈ {anthropic, openai, gemini}` (3 with caveats — see below). Adding a new provider is one new file in `harnesses/providers/` plus a branch in `harnesses/providers/__init__.py::get_provider()`.

- [x] **Google: `google-genai` thin harness** — folded into the unified `harnesses/thin.py` via `harnesses/providers/gemini.py`. Run with `--harnesses thin --provider gemini`. Uses `google.genai.Client().models.generate_content(...)` with auto function-calling disabled so the harness loop matches the other providers exactly. Requires `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) in `.env`. Model selection via `GEMINI_MODEL` env var (default `gemini-2.5-pro`).
- [ ] **Google: Gemini ADK harness** — `harnesses/gemini_adk.py` (or similar), then add a `gemini` branch to `harnesses/ai_agent.py::run()` and `--harnesses ai_agent --provider gemini` Just Works. Uses Google's [Agent Development Kit](https://google.github.io/adk-docs/) (`google-adk` package). Register our tools as ADK `Tool`s, run via the ADK's session/runner API, drain events equivalent to our current `AssistantMessage`/`ToolUseBlock`/`ResultMessage` parsing. LangSmith has a native integration (`langsmith[google-adk]`), use that for observability.
- [x] **OpenAI: `openai` client thin harness** — folded into the unified `harnesses/thin.py` via `harnesses/providers/openai.py`. Run with `--harnesses thin --provider openai`. Implemented against Chat Completions (not Responses API) so it also works against any OpenAI-compatible server (vLLM, Ollama, OpenRouter, Together, …). Model selection via `--openai-model`; key + endpoint via `.env`.
- [x] **OpenAI: Agents SDK harness** — `harnesses/openai_agents.py`, dispatched by `harnesses/ai_agent.py` when `--provider openai`. Uses the [`openai-agents`](https://openai.github.io/openai-agents-python/) Python package with our six shared tools registered via `@function_tool`. LangSmith integration via `langsmith[openai-agents]`.
- [x] **OpenAI: thick LangGraph harness** — folded into the unified `harnesses/langgraph_h.py` via the shared `Provider` abstraction. Run with `--harnesses langgraph --provider openai`. Same for Gemini: `--harnesses langgraph --provider gemini` Just Works (the `GeminiProvider` is reused).

With three providers and four harnesses already in place (modulo the two Gemini caveats), the central question shifts from "does scaffolding help within Anthropic?" to "does the *same* scaffolding pattern buy more on weaker backbones than on stronger ones, and do different providers' official agent frameworks converge on similar quality/cost tradeoffs?"

### Other follow-ons

- [ ] **Model-size ablation** — rerun the full matrix on a weaker model (e.g., Haiku 4.5 on the Anthropic side) to test the blog's central hypothesis: thick harnesses compensate for weaker models.
- [ ] **Multi-agent orchestrator harness** — a third "thick" style: orchestrator + specialist subagents (planner, coder, reviewer) delegating via tool-call handoff. Complements LangGraph's single-agent graph.
- [ ] **Statistical significance** — 3 trials is thin for small deltas. Scale to 10+ once a full matrix costs < $1.
