"""End-to-end run of the graph in mock mode (no LLM, no Langfuse) for one date."""

import pandas as pd


def test_pipeline_mock(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("LANGFUSE_DISABLED", "1")
    from src.agents.graph import build_workflow
    from src.core.results import write_weights
    from src.core.run import run_date
    from src.core.session import start_session
    from src.utils import data
    from src.utils.analytics import check_constraints

    s = start_session(tmp_path)
    res = run_date(build_workflow(), "2026-02-28", s)
    assert (s.dir / "report_2026-02-28.pdf").stat().st_size > 10_000
    w = pd.Series(res["weights"])
    assert check_constraints(w, data.read_assets_metadata()) == []
    df = pd.read_csv(write_weights(s), index_col=0)
    assert list(df.index) == ["2026-02-28"]
