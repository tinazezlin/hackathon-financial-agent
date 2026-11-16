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
  data.py                    official-style readers: read_assets/read_financial/read_macro(end_date=)
  news.py                    official-style news: read_news(start, end, topics=, title_contains=), find_topics
  universe.py                ADAPTER: official codes/categories -> our fields (name, asset_class,
                             region, currency, estimated duration), rule-based benchmark, macro y/y
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

## Synthetic data (in the official formats of data-detailed.pdf)

| file | shape |
|---|---|
| `assets.csv` | long: `date` (dd/mm/yyyy), `index` (code, e.g. `IBBEU008E`), `price`, `price_description` |
| `assets_metadata.csv` | `jurisdiction` (EUR/US/DE/IT/FR), `category` (Gov Bonds, Corp Bonds, Equities, Cash), `indicator`, `index`, `field`, `description`. Gov bonds are labelled "Total Return Yield" but hold index levels, as in the real file |
| `financial.csv` + metadata | same long format, Bloomberg-like codes (`GDBR10 Index`, `EURUSD Curncy`, ...) |
| `macro.csv` + metadata | `date` = FIRST day of the reference month (PMIs: last day; quarterly: first day of the quarter), `identifier`, `value`; HICP/CPI as price INDICES |
| `news.csv` | stand-in for the parquet files: versioned stories (`story_id`, `version`, `version_role`), UTC `published_at`, `topic_name` list (exact Reuters names) |
| `news_topics.csv` | topic, count |

The universe (16 government bonds, 2 corporate, 3 equity, 2 cash) is our guess. The paths
roughly follow 2018-2026 (COVID, 2022 hikes, 2024-25 cuts); **Oct 2025 - Mar 2026 is
invented**, e.g. an ECB cut on 2026-03-18. If the LLM "knows" something that isn't in the
tools' output, it's leaking memory.

**On the day:** point `src/utils/data.py`/`news.py` at the official modules (or delete
ours), then fix `src/utils/universe.py` (category names, `MARKET_SERIES` codes, duration
parsing) until `pytest` passes. Nothing else should need to change.

## Practice ideas

1. Replace "PM outputs 23 weights" with "PM outputs views → Black-Litterman → optimiser".
2. Add a macro agent and a news/sentiment agent, and let them debate before the PM.
3. Add a "challenger" agent (the investment committee) that must approve or send back.
4. Backtest the deterministic part (mock mode) over 2019-2025 month-ends against the benchmark.
5. Compare models (gpt-oss-120b vs 20b vs Qwen) on the same date: stability of weights.
