# Agent starter template (practice mock)

A **self-made imitation** of the hackathon's `agent-starter-template`, built from the
briefing so we can practise before the event. The real template arrives on the day,
so treat the structure here as a best guess. **All data is synthetic.**

## Quick start

```bash
cd agent-starter-template
.venv/bin/python main.py --dates 2025-10-31          # one date with Groq gpt-oss-120b (~2-3 min)
LLM_PROVIDER=mock .venv/bin/python main.py           # all 6 dates, no LLM, ~15 s
.venv/bin/python -m pytest                           # unit tests + output validation
.venv/bin/python scripts/generate_synthetic_data.py  # regenerate data (seeded)
```

### Setup after cloning

```bash
cd agent-starter-template
python3 -m venv .venv
.venv/bin/pip install langgraph langchain-core langchain-groq langchain-openai langchain-ollama \
    langfuse pandas numpy scipy matplotlib reportlab pytest python-dotenv ipykernel
cp .env.example .env        # add your GROQ_API_KEY (and Langfuse keys if you want tracing)
```

Mock mode (`LLM_PROVIDER=mock`) needs no keys at all. Open `multi_agents_market.ipynb` for the interactive flow.

## Layout

```
main.py                      CLI: runs dates (retries a failed date in the same session),
                             writes weights.csv only when every date has a result
multi_agents_market.ipynb    interactive version (same name as the official notebook)
src/config.py                dates, constraints, extra limits, strategic benchmark, model id
src/core/                    = hackathon "DO NOT CHANGE" folder (mocked here)
  llm.py                     get_llm(model_id): groq | ollama | openai | mock
  session.py                 start_session() -> outputs/<session id>/
  run.py                     run_date(workflow, date, model_name, session): one Langfuse
                             trace per date, fingerprints of PDF + weights on the trace
  results.py                 checks approved_weights has all 23 assets, calls build_report(),
                             write_weights(results, session) -> weights.csv (10 decimals)
src/utils/
  data.py                    read_assets / read_financial / read_macro / read_news (raw)
  pit.py                     point-in-time views, the ONLY data access decisions should use
  analytics.py               quant core: stats, FX conversion, VaR/ES, risk contributions,
                             constraint check, least-squares projection, safe rounding
  report.py                  build_report(result, path), keep the signature!
src/agents/
  state.py                   shared state (everything inspectable)
  tools.py                   analyst tools; the as-of date is injected, not chosen by the LLM
  prompts.py                 system prompts (time discipline in every one)
  nodes.py                   analyst, analyst_signals, portfolio_manager, risk_review, risk_manager
  graph.py                   LangGraph wiring
scripts/generate_synthetic_data.py
tests/                       analytics + PIT tests, mock end-to-end run, output validation
```

## The graph

```
START -> analyst <-> tools (asset stats, market indicators, macro snapshot, news)
           |
     analyst_signals          per-ticker views -2..+2 with confidence  (LLM, structured)
           |
     portfolio_manager  <---- feedback once if constraints violated   (LLM, structured)
           |                 |
     risk_review  -----------+    plain Python: clip/normalise, check limits, project if
           |                      still violating, safe rounding, risk metrics in EUR
     risk_manager -> END          risk commentary (LLM)
```

## Output (as in deliverables.pdf)

```
outputs/<session id>/          exactly 7 files
├── 2025-10-31.pdf … 2026-03-31.pdf   one report per date (build_report)
└── weights.csv                date + 23 asset columns, 10 decimals, rows sum to 1
```
The final graph state must hold `approved_weights` with all 23 assets (zeros included).
Submit the folder plus the Langfuse session URL (`session.url`).

## Synthetic data (guessing the real formats)

> The organisers' `data-detailed.pdf` shows the real formats differ (long files, day-first
> dates, code identifiers, macro dated at the start of the month, versioned news with topic
> lists). Aligning the synthetic data with them is the next step.

| file | shape |
|---|---|
| `assets.csv` | wide: `date` + 23 total-return index levels (local currency) |
| `assets-metadata.csv` | ticker, name, asset_class, jurisdiction, currency, duration |
| `financial.csv` | long: date, series_id, value (31 series: FX, rates, vol, CDS, swaps, commodities) |
| `macro.csv` | long: date (= **reference period end**), series_id, value (monthly/quarterly) |
| `news.csv` | published_at, headline, body, topic, source (~5k synthetic headlines) |

The paths roughly follow the shape of 2018-2026 (COVID, 2022 hikes, 2024-25 cuts).
**Oct 2025 - Mar 2026 is invented**, e.g. an ECB cut on 2026-03-18 that did not
necessarily happen. That's useful: if the LLM "knows" something that isn't in the
tools' output, it's leaking memory.

## Practice ideas

1. Replace "PM outputs 23 weights" with "PM outputs views → Black-Litterman → optimiser".
2. Add a macro agent and a news/sentiment agent, and let them debate before the PM.
3. Add a "challenger" agent (the investment committee) that must approve or send back.
4. Backtest the deterministic part (mock mode) over 2019-2025 month-ends against the benchmark.
5. Compare models (gpt-oss-120b vs 20b vs Qwen) on the same date: stability of weights.
