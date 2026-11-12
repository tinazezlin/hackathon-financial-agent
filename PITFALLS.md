# Things to be careful about (input for the hackathon plan)

Collected while reading the briefing and building the practice template.
Items marked **[seen]** actually happened in our practice runs.

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
6. **Full-sample statistics are a leak too.** Z-scores, normalisation, PCA, model
   training, "best parameters" must all be fitted on data up to each as-of date only.
7. **Our own knowledge leaks too.** We humans also know how late 2025 / early 2026 went.
   Don't hand-tune prompts or constraints until the March 2026 answer "looks good".

## B. Dates & data handling

8. **Two reporting dates are Saturdays: 31 Jan 2026 and 28 Feb 2026.** And 30 Dec 2025
   rather than 31 Dec. Use the last observation on or before the date, and keep the exact
   date strings the organisers expect in weights.csv (check the template's validation tests).
9. **Different calendars.** US and EU holidays differ. Forward-fill carefully. Don't
   compute correlations on pairwise-misaligned days (stale prices = fake low correlation).
10. **Coverage varies across series** (the briefing says so). Some start late or have
    gaps. Check before using one in a tool, or the agent gets NaNs and invents things.
11. **Units.** Yields in %, spreads in bp, rates in % p.a., index levels. Label units in
    every tool output, or the LLM mixes them up.
12. **Know what "assets.csv" levels are:** total return vs price index, local currency or
    EUR, hedged or not. It changes the return and risk numbers.

## C. Currency & investor perspective

13. **Base currency.** A euro-area central bank thinks in EUR. USD assets carry FX risk
    unless hedged. Decide hedged vs unhedged, compute risk in EUR, and say so in the report.
14. **"US jurisdiction <= 20%" includes everything US:** treasuries, US corporates, S&P,
    USD cash. Map it from metadata, not by hand.

## D. LLM engineering

15. **[seen] Structured output fails on small type mismatches.** The model answered
    `view: 0.5` where the schema said `int`, and the whole call was rejected. Keep schemas
    tolerant (floats, optional fields), validate in Python, retry with the error message.
16. **[seen] gpt-oss can spend its whole token budget on reasoning and return EMPTY
    content** (`finish_reason="length"`). It happened when the PM had to do arithmetic on
    23 weights. Set `max_tokens` high, tune `reasoning_effort`, and don't make the LLM do
    arithmetic: let it give views/tilts and let Python compute the weights.
17. **[seen] Structured output via tool calling is less reliable than native JSON
    schema** on gpt-oss/Groq ("model did not call a tool"). Check what the STACKIT
    proxy supports on day one.
18. **[seen] The LLM misreads statistics.** It called a 12-month return "YTD" (in two runs out of two). Name
    columns explicitly (`ret_12m_trailing`) and describe them in the tool docstring.
19. **[seen] "Moderate tilts" in the prompt is not enough.** The PM went to 33%
    equities against a 14% benchmark, inside the hard limits, so nothing stopped it. Add
    quantitative guards: max active weight per asset, tracking-error budget, or a
    deterministic optimiser.
20. **Non-determinism.** Same input, different weights. Use temperature 0, log
    everything, maybe run N times and average or take the median. Measure the stability.
    - **[seen] One run proves nothing about a fix.** In our second analyst run an
      invented-causality error disappeared without any fix: it was just a different
      sample. Judge a prompt or tool change on 3-5 runs and count errors.
21. **Latency and rate limits.** Our practice run took ~150 s per date (Groq free tier).
    At the event the proxy is shared by 8 teams. Keep prompts compact, avoid
    unnecessary tool loops, start the final run early.
22. **Context size.** ~5k news items is far too many for a prompt. Filter by date window
    and topic, rank, summarise (map-reduce), or build a sentiment index in code.
23. **[seen] Unicode in LLM text breaks the PDF** (non-breaking hyphens, thin spaces show
    as black boxes in ReportLab's built-in fonts). Normalise text before rendering. LLMs
    also write markdown tables, so render them properly.
24. **Hallucinated numbers.** Cross-check the numbers in the narrative against the tool
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
25. **[seen] Units get lost.** The header said "changes are absolute", yet the model wrote
    EUR/USD "down 0.016 %" (the move was -0.016 in level, about -1.4%) and HICP "up 0.04 %"
    (percentage points). Put units in column names (`chg_1m_bp`, `change_pp`) or in the values.
26. **[seen] Invented causality and noise as signal.** The analyst attributed an IT CDS
    move to an unrelated headline, and treated a randomly repeated headline as a key risk
    (twice in two runs). Deduplicate news in the tool, and tell the model that repetition
    is not evidence and that it should describe moves without explaining them unless a
    source does.
27. **[seen] The analyst's errors become investment decisions.** Signals and the PM only see
    the brief. The analyst also added allocation advice ("favour short-duration
    sovereigns…") it was not asked for, which already steers the PM.

## E. Portfolio construction & risk

28. **[seen] Rounding can breach a limit.** Rounding weights to 6 decimals turned an
    exact 20.000% US exposure into 20.0001%. Round down (floor) and put the residual
    into EUR cash. Validate the final CSV, not the in-memory weights.
29. **Weights must sum to 1, be >= 0, and cover all tickers.** Validate before writing.
    Always keep a deterministic last line of defence (projection onto the constraints).
30. **The LLM should not be the optimiser.** Agents give views and confidence. A transparent
    method (Black-Litterman, risk budgeting, constrained mean-variance) turns them into
    weights. That is the "transparent, inspectable" story the jury wants.
31. **Covariance estimation.** Short windows are noisy, long ones are stale. Use shrinkage
    (Ledoit-Wolf) or EWMA. State the window. VaR from 3y history won't include a crash
    that isn't in the window.
32. **Turnover & transaction costs.** Monthly rebalancing with big swings is unrealistic for
    a reserve manager. Consider a turnover penalty or cap, and report turnover per date.
33. **Duration.** For a bond-heavy portfolio, duration/DV01 is the main risk number, not
    equity beta. Report it, maybe cap it.
34. **Extra constraints must be explained** (briefing). E.g. min liquidity/cash tranche,
    max single asset, max issuer-country concentration (IT/ES), duration band.
35. **A benchmark to compare against.** Without a neutral portfolio (strategic benchmark,
    equal weight, minimum variance) "good performance" means nothing.

## F. Hackathon rules & logistics

36. **Don't touch `src/core/`**, don't change `build_report(result, path)`'s signature,
    don't hand-edit outputs (SHA-256 is checked), don't rebuild the shared venv, and don't
    read or print `~/.env.secrets`.
37. **The submission is the LAST session.** A half-finished test run after the good one
    overwrites what is submitted. Plan a "final run" slot and freeze the code before it.
38. **weights.csv is only written if ALL dates succeed.** One failing date = no weights.
    Make nodes robust (retries, fallbacks) and test all 6 dates early.
39. **8 people, one repo, one shared environment.** Agree on ownership per folder/file,
    use git branches, and set one integration owner. Hermes must "respect teammates' files".
40. **Install nothing new** (shared venv). Check on day one what is available (scipy?
    cvxpy? scikit-learn?) before planning methods.
41. **Deliverables are more than code:** 6 PDF reports with charts, documentation of the
    architecture and assumptions, and a presentation. Assign owners early. Don't leave
    the narrative for the last hour.
42. **Polymarket library** is offered. Prediction-market odds (e.g. for Fed/ECB decisions)
    could be a neat extra signal, but check its point-in-time availability.
