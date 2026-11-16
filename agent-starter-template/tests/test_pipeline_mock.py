"""End-to-end run of the graph in mock mode (no LLM, no Langfuse) for one date."""

import pandas as pd


def test_pipeline_mock(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("LANGFUSE_DISABLED", "1")
    from src.agents.graph import build_workflow
    from src.core.results import write_weights
    from src.core.run import run_date
    from src.core.session import start_session
    from src.utils import universe
    from src.utils.analytics import check_constraints

    s = start_session(tmp_path)
    res = run_date(build_workflow(), "2026-02-27", "mock", s)
    assert (s.dir / "2026-02-27.pdf").stat().st_size > 10_000
    w = pd.Series(res["approved_weights"])
    assert set(w.index) == set(universe.tickers())  # all 23, zeros included
    assert check_constraints(w, universe.assets_meta()) == []
    df = pd.read_csv(write_weights({"2026-02-27": res}, s), index_col=0)
    assert list(df.index) == ["2026-02-27"]
    assert {p.name for p in s.dir.iterdir()} == {"2026-02-27.pdf", "weights.csv"}
