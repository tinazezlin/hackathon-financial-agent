# Things to be careful about (input for the hackathon plan)

Collected while reading the briefing and building the practice template.
Items marked **[seen]** actually happened in our practice runs. Items marked **[docs]**
come from the organisers' detailed documents of 16 Nov (`deliverables.pdf`,
`data-detailed.pdf`, `polymarket_detailed_3557720.pdf`), which override the briefing.

---

## A. Look-ahead bias & time discipline (most important)

1. **LLM knowledge cutoff leaks the future.** Qwen3.8 (cutoff Aug 2026) and Qwen3.6
   (Apr 2026) *know* what happened Oct 2025 - Mar 2026. Any decision they make can use
   that, consciously or not.
   - Prefer gpt-oss-120b / 20b (cutoff Jun 2024) for decision-making agents.
   - Put the as-of date and "do not use knowledge after this date" in every prompt.
   - Make agents ground every claim in tool output and cite numbers.
   - Write the mitigation in the documentation. The jury will probably ask.
2. **Tools must not be able to see the future.** Don't let the LLM pass the date as a
   tool argument. Inject it from state (`InjectedState`) and filter all data `<= as_of`.
   One central point-in-time module, with tests.
3. **Macro data is dated by reference period, not by release date.** March CPI is
   published in April, Q4 GDP in late Jan/Feb. Using "date <= as_of" leaks 1-2 months.
   Apply release lags (we used 35 days monthly, 60 days quarterly) or find real
   release dates in the metadata.
   - **[docs] Most macro series are dated on the FIRST day of the reference month**
     (01/03/2025 = March 2025), but the 7 PMI series on the LAST day (31/03/2025). Count
     the lag from the END of the reference month: "first-of-month date + 35 days" would
     leak about a month. Align the series on the month before combining them.
   - **[docs] HICP comes as an INDEX** (e.g. `P023ZH`), not as y/y inflation. Compute
     y/y ourselves, using only index values that were published by the as-of date.
   - **[docs] Quarterly and annual series end at 01/01/2026**, monthly ones at March 2026.
   - **[seen] But not every series has a lag.** Policy rates are known the moment they
     are announced. Our blanket 35-day lag showed the Fed at 4.50% while the news said
     it had cut to 4.00%, and the analyst wrote both numbers in one sentence. Use
     per-series lags (`SERIES_RELEASE_LAG_DAYS` in `pit.py`).
   - **[seen] A monthly series is the wrong source for something that changes on a given
     day.** Even with zero lag, the December value (dated 31 Dec) is invisible on 30 Dec,
     so the 11 Dec Fed cut was missing. On day one, look for a daily policy-rate series
     (€STR, deposit rate, fed funds) in `financial.csv`, or build it from decision events.
4. **Data revisions.** Macro values in the file are probably the latest vintage, not
   what was known at the time. We can't fix it; state it as a limitation.
5. **News timestamps and time zones.** Decide whether the decision is at the close of the
   rebalancing date (use news up to 23:59 that day) or before the open. Be consistent.
   Watch out for UTC vs CET and US news after the European close.
   - **[docs] News `published_at` is UTC.** Filter on `published_at` (when the version was
     recorded), not on `version_created`.
6. **Full-sample statistics are a leak too.** Z-scores, normalisation, PCA, model
   training, "best parameters" must all be fitted on data up to each as-of date only.
7. **[docs] The readers return the future unless told otherwise.** `read_assets`,
   `read_financial` and `read_macro` return everything when called without
   `end_date=`. Always pass `end_date=T` (inclusive), in one central PIT module, and keep
   our own `<= T` filter as a second layer. The organisers check for look-ahead and
   award points for handling it well.
8. **Our own knowledge leaks too.** We humans also know how late 2025 / early 2026 went.
   Don't hand-tune prompts or constraints until the March 2026 answer "looks good".

## B. Dates & data handling

9. **[docs] The reporting dates changed.** `deliverables.pdf` lists 2025-10-31, 2025-11-28,
   **2025-12-29, 2026-01-30, 2026-02-27**, 2026-03-31, all business days. The briefing
   said 30 Dec, 31 Jan and 28 Feb. Use the template's list on the day, and keep
   "last observation on or before the date" logic for holidays.
10. **Different calendars.** US and EU holidays differ. Forward-fill carefully. Don't
   compute correlations on pairwise-misaligned days (stale prices = fake low correlation).
11. **Coverage varies across series** (the briefing says so). Some start late or have
    gaps. Check before using one in a tool, or the agent gets NaNs and invents things.
12. **Units.** Yields in %, spreads in bp, rates in % p.a., index levels. Label units in
    every tool output, or the LLM mixes them up.
13. **Know what "assets.csv" levels are:** total return vs price index, local currency or
    EUR, hedged or not. It changes the return and risk numbers.
    - **[docs] The 16 government bonds say "Total Return Yield"** (`TOT_RETURN`) in the
      metadata, but the values are total return index LEVELS (144 to 2,020). Compute
      returns from them as from prices; do not treat them as yields.
14. **[docs] The real files are long and coded.** `assets.csv` and `financial.csv` have
    `date, index, price, price_description`; dates are day-first (`02/01/2018`);
    identifiers are codes (`IBBEU008E`, `P023ZH`). Parse dates with `dayfirst=True`,
    read the metadata with `encoding="utf-8-sig"`, and map codes to readable names in
    one place before anything reaches an LLM.
15. **[docs] The metadata has no duration and no currency.** Asset metadata has
    jurisdiction (EUR, US, DE, IT, FR: "EUR", not "EA", and no ES), category (Gov Bonds,
    Corp Bonds, Equities, Cash), indicator, index, field, description. Derive currency
    from jurisdiction, and estimate duration (maturity bucket in the name, or a
    regression of returns on yield changes) and say how in the report.
16. **[docs] News comes in versions and with many topics.** Each story has versions
    (`story_id`, `version`, `version_role` first/last), so deduplicate by story. There
    are 636 topics, each item has a list of them, and `read_news` accepts only exact
    names (look them up with `find_topics`). Don't query by geography alone. `body` is
    returned only on request (it is 95% of the data). Recommended topics: Central Banks /
    Central Bank Events, Central Bank Interest Rates / Policy, European Central Bank,
    Inflation Data, Employment / Unemployment Data / Policy, Consumer Sentiment Survey
    Data, Fixed Income, Equities, Gold, Crude Oil (no Brent topic: North Sea Crudes).
    - **[docs] There is no "Federal Reserve" topic** (only the regional Fed banks). Fed news
      comes through "Central Bank Interest Rates / Policy" plus `title_contains="Fed"`.
    - **[seen] One unknown topic name makes the whole `read_news` call fail.** Filter the
      topic list against `find_topics()` first, and log the names that were dropped.
    - **[seen] Merge repeated headlines and keep the latest version of each story by T.**
      After that, the noise headline the analyst treated as a "key risk" disappeared.

## C. Currency & investor perspective

17. **Base currency.** A euro-area central bank thinks in EUR. USD assets carry FX risk
    unless hedged. Decide hedged vs unhedged, compute risk in EUR, and say so in the report.
18. **"US jurisdiction <= 20%" includes everything US:** treasuries, US corporates, S&P,
    USD cash. Map it from metadata, not by hand.

## D. LLM engineering

19. **[seen] Structured output fails on small type mismatches.** The model answered
    `view: 0.5` where the schema said `int`, and the whole call was rejected. Keep schemas
    tolerant (floats, optional fields), validate in Python, retry with the error message.
20. **[seen] gpt-oss can spend its whole token budget on reasoning and return EMPTY
    content** (`finish_reason="length"`). It happened when the PM had to do arithmetic on
    23 weights. Set `max_tokens` high, tune `reasoning_effort`, and don't make the LLM do
    arithmetic: let it give views/tilts and let Python compute the weights.
21. **[seen] Structured output via tool calling is less reliable than native JSON
    schema** on gpt-oss/Groq ("model did not call a tool"). Check what the STACKIT
    proxy supports on day one.
22. **[seen] The LLM misreads statistics.** It called a 12-month return "YTD" (in two runs out of two). Name
    columns explicitly (`ret_12m_trailing`) and describe them in the tool docstring.
23. **[seen] "Moderate tilts" in the prompt is not enough.** The PM went to 33%
    equities against a 14% benchmark, inside the hard limits, so nothing stopped it. Add
    quantitative guards: max active weight per asset, tracking-error budget, or a
    deterministic optimiser.
24. **Non-determinism.** Same input, different weights. Use temperature 0, log
    everything, maybe run N times and average or take the median. Measure the stability.
    - **[seen] One run proves nothing about a fix.** In our second analyst run an
      invented-causality error disappeared without any fix: it was just a different
      sample. Judge a prompt or tool change on 3-5 runs and count errors.
    - **[seen] Python's `hash()` of a string changes between runs** (hash randomisation),
      so seeding random numbers with it makes "deterministic" code non-reproducible. Use
      `zlib.crc32` or a fixed seed.
25. **Latency and rate limits.** Our practice run took ~150 s per date (Groq free tier).
    At the event the proxy is shared by 8 teams. Keep prompts compact, avoid
    unnecessary tool loops, start the final run early.
26. **Context size.** ~5k news items is far too many for a prompt. Filter by date window
    and topic, rank, summarise (map-reduce), or build a sentiment index in code.
27. **[seen] Unicode in LLM text breaks the PDF** (non-breaking hyphens, thin spaces show
    as black boxes in ReportLab's built-in fonts). Normalise text before rendering. LLMs
    also write markdown tables, so render them properly.
28. **Hallucinated numbers.** Cross-check the numbers in the narrative against the tool
    output (a simple regex check, or a "fact-checker" node).
    - **[seen] Numbers copied correctly, interpretation wrong.** The analyst wrote "HICP
      1.89%, down from 1.85%" and, in another run, "1.89% remains above the 2% target".
      A generic prompt line ("state figures correctly") does not fix this: the model
      already thinks it is right.
    - **[seen] Pre-computing comparisons in the tool works**: after we added
      `change` / `direction` columns to the macro snapshot, the "down" error disappeared.
      **But the errors move**: the model then made a comparison we hadn't pre-computed
      (against the 2% target) and got it wrong. Let Python compute what it can, and keep
      a fact-checker as the safety net.
    - **[seen] Prompt rules work better when specific**: "copy directions from the tool,
      don't compute changes", "use the horizon named in the column, never YTD", "don't
      state causes unless a tool states them".
29. **[seen] Units get lost.** The header said "changes are absolute", yet the model wrote
    EUR/USD "down 0.016 %" (the move was -0.016 in level, about -1.4%) and HICP "up 0.04 %"
    (percentage points). Put units in column names (`chg_1m_bp`, `change_pp`) or in the values.
30. **[seen] Invented causality and noise as signal.** The analyst attributed an IT CDS
    move to an unrelated headline, and treated a randomly repeated headline as a key risk
    (twice in two runs). Deduplicate news in the tool, and tell the model that repetition
    is not evidence and that it should describe moves without explaining them unless a
    source does.
31. **[seen] The analyst's errors become investment decisions.** Signals and the PM only see
    the brief. The analyst also added allocation advice ("favour short-duration
    sovereigns…") it was not asked for, which already steers the PM.

## E. Portfolio construction & risk

32. **[seen] Rounding can breach a limit.** Rounding weights to 6 decimals turned an
    exact 20.000% US exposure into 20.0001%. Round down (floor) and put the residual
    into EUR cash. Validate the final CSV, not the in-memory weights.
    - **[docs] weights.csv uses 10 decimals.** The same problem exists at 10 decimals, and
      floating point adds its own: `floor(0.13 * 1e10)` gives 0.1299999999.
33. **Weights must sum to 1, be >= 0, and cover all tickers.** Validate before writing.
    **[docs]** The final graph state must be called `approved_weights` and hold all 23
    assets, zeros included; the template fails the date otherwise.
    Always keep a deterministic last line of defence (projection onto the constraints).
34. **The LLM should not be the optimiser.** Agents give views and confidence. A transparent
    method (Black-Litterman, risk budgeting, constrained mean-variance) turns them into
    weights. That is the "transparent, inspectable" story the jury wants.
35. **Covariance estimation.** Short windows are noisy, long ones are stale. Use shrinkage
    (Ledoit-Wolf) or EWMA. State the window. VaR from 3y history won't include a crash
    that isn't in the window.
36. **Turnover & transaction costs.** Monthly rebalancing with big swings is unrealistic for
    a reserve manager. Consider a turnover penalty or cap, and report turnover per date.
37. **Duration.** For a bond-heavy portfolio, duration/DV01 is the main risk number, not
    equity beta. Report it, maybe cap it.
38. **Extra constraints must be explained** (briefing). E.g. min liquidity/cash tranche,
    max single asset, max issuer-country concentration (IT/ES), duration band.
39. **A benchmark to compare against.** Without a neutral portfolio (strategic benchmark,
    equal weight, minimum variance) "good performance" means nothing.

## F. Hackathon rules & logistics

40. **Don't touch `src/core/`**, don't change `build_report(result, path)`'s signature,
    don't hand-edit outputs (SHA-256 is checked), don't rebuild the shared venv, and don't
    read or print `~/.env.secrets`.
    - **[docs] Don't edit, regenerate or copy files into the session folder.** Each date's
      trace records a fingerprint of its PDF and weights, and the organisers recompute
      them. A row from another session or a date that never ran won't match, and the
      submission is rejected. To change something, run it again.
41. **[docs] Submit one complete session: its folder and its Langfuse URL.** The folder
    holds exactly 7 files (`2025-10-31.pdf` … `2026-03-31.pdf` and `weights.csv`). Email
    both to Hackathon@ecb.europa.eu **by 17:00 on 18 November**. Plan a "final run" slot
    well before that and freeze the code before it.
    - **[docs] Langfuse is mandatory.** The session must have 6 traces, one per date.
      An untraced session is penalised. Check that tracing is on before the final run.
42. **One failing date blocks the submission.** Make nodes robust (retries, fallbacks)
    and test all 6 dates early. **[docs]** A failed date can be rerun in the SAME session
    (`results[d] = run_date(...)`, then `write_weights(results, session)` again); its last
    successful trace counts. Don't start a new session for one date.
43. **8 people, one repo, one shared environment.** Agree on ownership per folder/file,
    use git branches, and set one integration owner. Hermes must "respect teammates' files".
44. **Install nothing new** (shared venv). Check on day one what is available (scipy?
    cvxpy? scikit-learn?) before planning methods.
45. **Deliverables are more than code:** 6 PDF reports with charts, documentation of the
    architecture and assumptions, and a presentation. Assign owners early. Don't leave
    the narrative for the last hour.
46. **Polymarket library** is offered. Prediction-market odds (e.g. for Fed/ECB decisions)
    could be a neat extra signal, but check its point-in-time availability.
    - **[docs] Pass `end=T` as END-OF-DAY UTC** (`datetime(…, 23, 59, 59, tzinfo=utc)`).
      A plain date means midnight, i.e. the START of T, and drops that day.
    - **[docs] Some fields are today's values, not values at T:** `volume`, `liquidity`,
      `closed`, order books, midpoints and spreads. Don't filter or rank events by them.
    - **[docs] A market that closed before T raises an error** with `end` after its last
      price: use its last date.
    - **[docs] YES prices of exclusive answers only roughly sum to 1** (97.95% in their
      example). Normalise them, and don't do that for non-exclusive events (nested
      deadlines). Thin markets can be stale: check liquidity before trusting a price.
    - **[docs] `discover_tags("ECB")` doesn't find `european-central-bank`**: read tags
      from a known event, or search with `query`.
    - **There is no "next meeting" field.** Select the events open at T by schedule
      (`start_date_max=T`, `end_date_min=T`) and take the one with the earliest
      `schedule.end_date`. Never filter on `closed`, which describes today.
    - **We could not test the library before the event** (it isn't on pip). Our wrapper is
      written from the PDF; the column names returned by `get_prices_by_event` are a guess.
      Run the check cell in the notebook first thing on day one.
