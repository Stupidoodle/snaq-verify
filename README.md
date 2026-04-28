# snaq-verify

A take-home for SNAQ. Verifies the nutrition values in `food_items.json` against authoritative external sources, flags discrepancies, proposes corrections, and (bonus) judges its own output against a hand-curated golden set.

> **TL;DR:** *the LLM agent orchestrates tool calls; tools produce all facts.* Numeric values, candidate selection, verdicts, and human-readable summaries all trace back to a tool call — never to LLM prose. The LLM is a router; the tools are the truth. Output guardrails re-check the agent's claims with pure math (Atwater equation), so hallucinations cannot survive into the report.

---

## The original task (from `home_task.md`)

> SNAQ is an app that helps people understand what they eat. A core challenge is reliability: food databases are incomplete, product data varies across sources, and LLM knowledge alone is not a trustworthy source for calorie counts and macros.
>
> Your task is to build an agentic system that takes a set of food items as structured input and verifies their nutritional content.
>
> You are given a file called `food_items.json`. Each entry in the file represents a food item with an existing nutritional profile. Your agent should verify whether the provided nutrition data is correct, flag discrepancies, and where possible propose corrections.

The full brief (input schema, evaluation criteria, bonus, deliverables) is in [`home_task.md`](./home_task.md). What the brief asks for, mapped to this repo:

| Brief asks for | Where it lives |
| --- | --- |
| **Working code, runnable locally with minimal setup, taking `food_items.json` as input** | `make verify` or `docker compose up` — see [Quickstart](#quickstart-single-command) below |
| **Output for the provided `food_items.json`** | `verification_report.json` (generated; per-item verdict + evidence + proposed correction) |
| **Agent design** ("not just a single LLM call dressed up as an agent") | [Decisions matrix](#decisions-matrix) — the tools-as-truth, agent-as-router framing |
| **Handling uncertainty** (sources disagree, generic vs branded, farmed vs wild) | `select_best_candidate` with deterministic tie-break, `derive_confidence` rule, `low_confidence` verdict path, web-search fallback when USDA + OFF both miss |
| **Tool use** (interact with at least one external data source) | Three sources behind ports: USDA FoodData Central, Open Food Facts, Tavily. Six computation tools layered on top |
| **Code quality** ("write code you'd be comfortable shipping") | Hexagonal layout, ABC ports, 355 unit tests, ruff + mypy strict, atomic commits |
| **Bonus: a layer that verifies the agent itself** | `SelfVerifyStep` (judge inline, retry verifier with judge feedback as a hint if score is low) + `make eval` produces `eval_report.json` against a hand-curated 11-item golden set |
| **AI conversation log** | [`docs/conversation_log.txt`](./docs/conversation_log.txt) — the full session transcript, including peer DMs between the eight teammates |

---

## Quickstart (single command)

```bash
cp .env.example .env       # paste 3 keys: USDA_API_KEY, OPENAI_API_KEY, TAVILY_API_KEY
docker compose up          # builds image, runs verify + eval, writes output/*.json
```

That's it. The reviewer's first command after cloning. Outputs land in `./output/`:

- `verification_report.json` — the per-item verdicts (the deliverable the brief asks for)
- `eval_report.json` — the bonus: LLM-as-judge scores against `tests/data/ground_truth.json`

Without Docker:

```bash
uv sync
make verify        # → verification_report.json
make eval          # → eval_report.json (after make verify)
```

## Output shape (one item)

```json
{
  "item_id": "fage-total-0-greek-yogurt",
  "item_name": "Total 0% Greek Yogurt, Plain",
  "reported_nutrition": { "calories_kcal": 57, "protein_g": 5.5, ... },
  "verdict": "major_discrepancy",
  "confidence": "medium",
  "evidence": [
    { "source": "usda", "candidate": { ... }, "bundle": { ... } },
    { "source": "web",  "candidate": { ... }, "bundle": { ... } }
  ],
  "proposed_correction": { "calories_kcal": 61, "protein_g": 10.3, ... },
  "atwater_check_input": { "expected_kcal": 49.4, "reported_kcal": 57, "is_consistent": true },
  "summary": "Total 0% Greek Yogurt, Plain: cross-checked against 2 source(s)...",
  "reasoning": "<the model's chain-of-thought, native or self-reported>",
  "notes": [
    "off.lookup_by_barcode.not_found barcode='5200435000027'",
    "off.search_by_name.no_results name='Total 0% Greek Yogurt, Plain' brand='Fage'",
    "tavily.search query='Fage Total 0% Greek Yogurt nutrition per 100g'"
  ]
}
```

Every numeric field traces to a tool call. Every line in `notes` is a real event the tools observed — never LLM prose.

---

## Decisions matrix

This is the section the brief tells you to read first. Each row is a real call we (Bryan as PO + the agent team) made during the build, with the trade-off and rationale.

| Decision | Picked | Rejected | Why |
|---|---|---|---|
| **Architecture style** | Hexagonal: `domain/`, `ports/`, `infrastructure/`, `application/` | A flat script | Bryan's first move after reading the brief: *"I would say a quick hexagonal architecture... check ../brella-mass-outbound for more info on the project structure."* The reference repo set the conventions; we mirrored them so the structure is familiar to the reviewer if they've seen Bryan's prior code. |
| **Caching** | `CachePort` ABC + `FileCache` (`.cache/{sha256(key)}.json`, atomic tmp+rename writes) + `InMemoryCache` for tests | Valkey/Redis with docker-compose | Bryan asked mid-design: *"hold up is adding valkey overengineering?"* — yes for an 11-item batch that runs in 30s. We kept the port (the architectural statement) so the swap to a distributed cache is one line in `bootstrap.py` if it's ever wanted. |
| **The architecture's center of gravity** | The agent is a **tool-orchestrating router**; tools produce all facts. Output schema (`output_type=ItemVerification`) is the contract. | An LLM-heavy "agent that thinks" approach | Bryan's `ULTRATHINK`-stamped framing: *"the .md asks for an agentic system, so we should focus on the tool calls and functions that the agents use thats where the undeniable truth and determinism will come from."* This is the project's north star — see [Determinism guarantees](#determinism-guarantees) below. |
| **Compute as tools** | 6 deterministic, pure-function tools wrapped via `function_tool(...)`: `score_candidate_match`, `select_best_candidate`, `compute_per_nutrient_delta`, `verdict_from_deltas`, `check_atwater_consistency`, `format_human_summary` | Letting the LLM "compute" deltas / verdicts in prose | Same north star. The LLM cannot ship a verdict without calling `verdict_from_deltas`. It cannot pick a candidate without calling `select_best_candidate`. The summary is a string template, never LLM prose. |
| **`notes` field source** | Mechanically derived from a `tool_events: list[str]` accumulator in the `VerifierContext` | LLM-written free-text notes | First live run caught the agent **fabricating a "Foundation tool validation error" that never happened** (HTTP logs showed two clean 200s). That broke the "tools as truth" promise. Fix: every note line is appended by the tool itself when it observes a notable outcome (a 404, a cache miss, a fallback). The fabrication channel is closed. |
| **Three output guardrails** | `atwater_output_guardrail`, `schema_output_guardrail`, `confidence_output_guardrail` — all from the SDK `@output_guardrail` pattern | Trusting the LLM | Atwater re-runs the math on the agent's claimed nutrition and tripwires when the verdict label contradicts physics (4P + 4C + 9F vs reported kcal). Schema guardrail rejects empty `evidence` for non-NO_DATA verdicts. Confidence guardrail enforces the deterministic `derive_confidence()` rule (≥2 sources at score ≥0.85 = HIGH). All three exist because the LLM can't be trusted on any of these. |
| **Confidence calibration** | A pure function `derive_confidence(evidence, max_score, verdict) -> ConfidenceLevel`, called in the adapter to **override** whatever the LLM put | LLM-judged confidence | First run produced 100% low/medium and zero HIGH. The deterministic rule (≥2 sources confirming + max score ≥0.85 + verdict ≤ minor → HIGH) is calibrated against the agent's actual evidence, not vibes. |
| **Reasoning surfacing** | Optional `ItemVerification.reasoning: str \| None`. The LLM populates it via `instructions`. If the run produces native `ReasoningItem`s (gpt-5/o-series), the adapter overrides with the real reasoning summary. Else None. | Either schema-only (LLM paraphrase) or extraction-only (silent on non-reasoning models) | Bryan's call: *"force a reasoning flag or if the model has reasoning tokens then show reasoning in agent output model... gpt 5-4 mini likely has no native reasoning but we can still force that field on the agent output else if has reasoning_tokens use reasoning field of native."* The fallback chain is: native > LLM-self-report > None. We never fabricate. |
| **Source clients** | Three **specific** ports: `USDAClientPort`, `OpenFoodFactsClientPort`, `TavilyClientPort` — each with the operations its tools actually need | One generic `NutritionSourcePort` | Source-specific typing means the agent's tools are typed end-to-end. `USDACandidate` carries `fdc_id` and `data_type` because those matter for downstream scoring; `OFFProduct` carries `completeness` and `popularity_key` for the same reason. A generic port flattens this into a lossy lowest-common-denominator. |
| **OFF client** | `httpx` direct against `https://world.openfoodfacts.org/api/v2/...` | The official `openfoodfacts` PyPI lib | The official lib uses sync `requests`. Our stack is async. Wrapping in `asyncio.to_thread` would be awkward and would tie us to the lib's call shape. (off-domain teammate's call after surveying both options.) |
| **Foundation FDC ID filter** | Drop USDA `Foundation` candidates with `fdc_id < 2_000_000` at the source layer | LLM-side retry on 404 | First live run revealed *50%* of FDC IDs returned by `/foods/search` 404 on `/food/{id}`. docs-oracle found the cause: USDA migrated Foundation Foods to the 2M+ ID range and never pruned the search index. A one-line filter at the adapter turns runtime LLM-recovery into deterministic O(1) behavior. |
| **OpenAI Agents SDK vs hand-rolled tool loop** | OpenAI Agents SDK 0.14.6 — `Agent`, `Runner.run`, `@function_tool`, `@output_guardrail`, `model_settings=ModelSettings(reasoning=Reasoning(effort="low", summary="auto"))` | A custom tool-calling loop on raw `openai` | The SDK is stateless, clean to wrap behind ports, and gives us guardrails + tracing for free. Wrapping it cost ~30 lines of adapter code. |
| **Determinism contract** | Run `make verify` twice with a warm cache → byte-identical `verification_report.json` modulo `RunMetadata.timestamp`. `make verify-determinism` enforces this with the `diff-runs` CLI subcommand. | "Just trust the agent" | Re-running a verification system on the same input must produce the same answer. Because (a) all numeric outputs come from cached deterministic tool calls and (b) tie-breaks are explicit (lowest `source_id` wins), the output values are stable across runs even though the agent's tool-call order may vary. |
| **Pipeline as composable steps** | `PipelineRunner` runs an ordered list of `PipelineStep`s. Each step reads/writes specific slices of `PipelineState`. Adding the bonus eval was just adding 4 more steps. | A monolithic `verify_all_items` function | Bryan asked specifically for *"easily extendable so we can plug and play more steps to go for the full eval rig."* When the bonus needed `LoadGroundTruthStep` + `JudgeStep` + `SelfVerifyStep`, they slotted in without touching the verification path. |
| **TeamCreate-driven build** | Eight teammates owning disjoint domains end-to-end (adapter + tools + fakes + tests). All peer-to-peer DMs, not via PO. | Solo build / sequential dispatch | Bryan: *"agents should communicate directly between each other rather than through you the PO to avoid overhead."* Two specialists ran the whole time: **docs-oracle** (research-only, queried before any teammate wrote API code) and **credentials-runner** (provisioned `.env`). Six implementers ran in parallel after the contracts were frozen. The conversation log shows the peer DMs that resolved the FunctionTool wrapping pattern, the sodium=1093 correction, and the FDC 2M filter. |
| **TDD discipline** | Every teammate writes the test file first, runs RED, implements GREEN, refactors. Atomic commits per change. | "Just write code" | Bryan: *"I want you the PO to manually test and verify the work out... read the tests beforehand to make sure they're not fake lmfao."* The 355-test suite is the cumulative output. We caught real bugs this way (`FakeCache` was falsy when empty because `__len__` returned 0; off-domain found it, search-cache-domain fixed it at the source). |
| **Audit pass on the golden set** | docs-oracle ran an explicit audit against the live USDA / OFF endpoints | Trust the first hand-curated values | The audit caught 7 wrong FDC IDs in `tests/data/ground_truth.json` — including **`169426` which is pistachios** (not almonds) and **`172686` which is wheat bread** (not white bread). A reviewer who clicked the citation URLs would have caught this. Now they don't have to. |
| **Concurrency in `VerifyStep`** | `_DEFAULT_CONCURRENCY = 1` (sequential per-item) | `concurrency=3` | First live run at 3 burst over OpenAI's 200k TPM cap on `gpt-5.4-mini` after 7 items. Sequential is the safest default; the comment in the file documents the trade-off. |
| **Concurrency in `JudgeStep`** | `asyncio.gather` with `Semaphore(3)` | Sequential | Judge calls are smaller (no tool use, just structured output), so 3-way parallelism is safe. Saves ~20s on the 11-item judge phase. |
| **`max_turns=30` per agent run** | Explicit in the adapter | The SDK default of 10 | First live run with the soft-retry-on-low-confidence loop hit `MaxTurnsExceeded` on three items. The current flow can use ~15 turns happy-path: USDA search → 2-3 candidate fetches → OFF lookup/search → web search fallback → 6 compute tool calls → synthesis. 30 leaves headroom for the retry. |
| **Self-verify (the bonus)** | A `SelfVerifyStep` that runs the judge inline against ground truth; if score below threshold, re-runs the verifier with the judge's reasoning as a `hint` | A separate retry-eval pass at the end | The brief's bonus asks for *"a layer that verifies the agent itself, not just the food data."* The retry loop directly maps to that — the verifier audits its own work using the judge's structured feedback, then writes the corrected verdict. The hint flows in via `VerifierAgentPort.verify(item, hint=...)`. |

---

## Architecture

```
food_items.json
   │
   ▼
┌────────────────────────────────────────────────────────────┐
│ Pipeline (PipelineRunner, wrapped in trace())              │
│                                                            │
│   LoadInputStep                                            │
│   LoadGroundTruthStep   (run-and-eval only, idempotent)    │
│                                                            │
│   VerifyStep ──► VerifierAgent (one Runner.run per item)   │
│                  ├─► IO tools (deterministic adapters)     │
│                  │     search_usda                         │
│                  │     get_usda_food                        │
│                  │     lookup_off_by_barcode  (gated by    │
│                  │                            is_enabled   │
│                  │                            on barcode)  │
│                  │     search_off_by_name                  │
│                  │     search_tavily                       │
│                  ├─► Compute tools (pure functions)        │
│                  │     score_candidate_match               │
│                  │     select_best_candidate               │
│                  │     compute_per_nutrient_delta          │
│                  │     verdict_from_deltas                 │
│                  │     check_atwater_consistency           │
│                  │     format_human_summary                │
│                  └─► output_type=ItemVerification          │
│                        + output_guardrails:                │
│                          atwater_output_guardrail          │
│                          schema_output_guardrail           │
│                          confidence_output_guardrail       │
│                        + soft retry on LOW confidence      │
│                                                            │
│   SelfVerifyStep ──► judge inline, retry verifier with     │
│                      judge feedback as hint if score low   │
│                      (run-and-eval only — the bonus)       │
│                                                            │
│   AggregateStep ──► VerificationReport                     │
│   WriteReportStep ──► verification_report.json             │
│                                                            │
│   JudgeStep (parallel, gather + Semaphore(3))              │
│   WriteEvalReportStep ──► eval_report.json                 │
└────────────────────────────────────────────────────────────┘
```

### File layout

```
src/snaq_verify/
├── core/config.py              # pydantic-settings + thresholds
├── bootstrap.py                # composition root — only place real adapters are instantiated
├── domain/
│   ├── models/                 # 9 Pydantic models, one per file
│   └── ports/                  # 8 ABC ports, one per file
├── application/
│   ├── pipeline/
│   │   ├── runner.py           # trace() wrap, sequential step loop
│   │   └── steps/              # 9 steps (load/verify/self_verify/aggregate/...)
│   └── tools/                  # 11 tools — 5 IO, 6 deterministic compute
├── infrastructure/
│   ├── observability/          # structlog adapter
│   ├── cache/                  # FileCache + InMemoryCache
│   ├── sources/                # USDA / OFF / Tavily clients (httpx-async)
│   └── agents/                 # verifier + judge + 3 guardrails
└── cli/
    ├── main.py                 # Typer: verify / eval / run-and-eval / diff-runs
    └── diff_runs.py            # determinism check helper
```

## Determinism guarantees

A reviewer running `make verify` twice on the same input expects the same JSON. The system makes this possible by enforcing:

1. **All tool functions return structured Pydantic data** — never prose that gets propagated to the report.
2. **All arithmetic lives in tools.** The LLM never computes a number itself.
3. **`select_best_candidate` uses a fixed scoring formula** with deterministic tie-break (lowest `source_id` wins lexicographically).
4. **`format_human_summary` is a string template**, not an LLM call.
5. **The output schema (`ItemVerification`)** requires fields that only specific tools produce — the LLM cannot ship a verdict without calling `verdict_from_deltas`.
6. **`FileCache` pins API responses** across runs so even the underlying source data is stable.
7. **Three output guardrails re-check** the agent's claims against pure math and reject contradictions.
8. **Confidence is overridden** by `derive_confidence()` after the run.
9. **`notes` are mechanically derived** from a `tool_events` accumulator — the LLM has no write access to that field.

Stochasticity that remains is confined to *which tools the LLM happened to call in which order* — not what numbers it produced.

`make verify-determinism` runs `verify` twice and diffs the outputs; any value-field mismatch fails CI.

## Local development

```bash
uv sync                  # install deps (uv only — never edit pyproject.toml deps by hand)
make all                 # setup + lint + typecheck + test + verify
make test                # 355 unit tests, fakes only, no network
make test-int            # integration tests against real APIs
make verify              # run the verifier; writes verification_report.json
make eval                # run the judge against ground_truth.json; writes eval_report.json
make verify-determinism  # run verify twice, diff value-fields, fail on mismatch
make lint                # ruff
make typecheck           # mypy strict
make fmt                 # ruff format
make clean               # nuke caches
```

Tests use **fakes**, not mocks. Each port has an in-memory fake under `tests/fakes/`. Adapters are tested with `respx`-mocked HTTP. The agent layer is tested with `set_tracing_disabled(True)` and stubbed `Runner.run`.

## What's in the box

| Metric | Count |
| --- | --- |
| Atomic commits | 41 |
| Source files (`.py`, excluding `__init__`) | 55 |
| Test files | 31 |
| Unit tests | 355 (all green) |
| Pydantic domain models | 9 |
| ABC ports | 8 |
| Tools (5 IO + 6 compute) | 11 |
| Pipeline steps | 9 |
| Output guardrails | 3 |
| Lines of code (`src` + `tests`) | ~12k |

## What I'd do differently with more time

These are real gaps the build surfaced. Each one was deliberately left as-is to ship; the rationale is in the conversation log.

- **Inline-hydrate `nutrition_per_100g` from USDA search responses.** The fix is implemented (`_parse_search_hit` returns `None` if any of 8 nutrients are missing), but most of the time the search response *does* carry enough, and the agent could skip `get_food()` entirely. Currently the agent always round-trips through `get_food`. Saving that round-trip would cut latency ~30%.
- **Smarter confidence rule.** The current `derive_confidence()` is sound but conservative; a real ML calibration against the golden set would be a Phase 4 win.
- **Judge feedback loop bounds.** `SelfVerifyStep` retries once on low score. A real eval rig would tune the threshold and possibly retry with a different model on persistent failure.
- **OFF rate-limit handling.** When OFF returns 503 (their server was flapping during the live runs), we just log and move on. A proper backoff-and-retry with jitter would hide this from the verdict.
- **Trace correlation.** `runner.py` wraps the pipeline in a single `trace()` context, but the per-item traces aren't tagged with `item_id`. Easy improvement: pass `metadata={"item_id": item.id}` to a nested trace per item.
- **Schema-only `proposed_correction`.** Right now if the agent can't recover all 8 fields, `proposed_correction` is set to `None`. A more useful product would surface a partial correction *with explicit per-field nullability* — but that requires changing the domain model and writing migration logic, which felt out of scope.
- **Determinism CI gate.** `make verify-determinism` exists, but isn't wired to a real CI. With more time it would run on every PR.

## AI conversation log

Per the brief: *"Your AI conversation log. We expect you to use an AI coding tool... Please export and include the conversation or session transcript."*

The full session transcript is at [`docs/conversation_log.txt`](./docs/conversation_log.txt) (~170KB). What it shows:

- The arc from "valkey + agent swarm" through "wait, is valkey overengineering?" to the final "tools as truth, agent as router" framing
- Every peer DM between the eight teammates (docs-oracle's API research, the FDC 2M filter discovery, the FunctionTool wrapping pattern that converged across three teams)
- Every PO audit (docs-oracle catching wrong FDC IDs in the golden set; me catching the LLM fabricating a "Foundation tool validation error" that never happened)
- Every wrong turn we corrected (3 attempts at the right tool-wrapping pattern; OpenAI 429 forcing concurrency=1; MaxTurnsExceeded forcing max_turns=30; guardrail tripwires aborting the pipeline before they were caught and converted to NO_DATA fallbacks)

If you want to know *why* a specific design choice exists, the conversation log will tell you — including the cases where the AI got it wrong and we backed it out.

## License & attribution

Licensed under the **GNU Affero General Public License v3.0** (AGPLv3) — see [`LICENSE`](./LICENSE). Network use is distribution: if you run a modified version as a service, you must offer the corresponding source under the same license.

Built by Bryan Tran for the SNAQ take-home. Implementation work delegated to a Claude Code (Opus 4.7 orchestrator + Sonnet teammates) team via TeamCreate; orchestration, all design calls, and final review by Bryan. The full transcript is in [`docs/conversation_log.txt`](./docs/conversation_log.txt) — including the cases where the AI got it wrong and we backed it out.
