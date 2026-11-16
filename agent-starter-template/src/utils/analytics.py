"""Quant core: returns, statistics, risk metrics, constraints, projection.

Pure functions (no LLM, no I/O) so they are easy to unit-test and to reuse in
tools, the risk review node and the report.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.config import LIMITS, MAX_SINGLE_ASSET, Limit

TD = 252


# --------------------------------------------------------------------------- returns
def daily_returns(levels: pd.DataFrame) -> pd.DataFrame:
    return levels.sort_index().pct_change(fill_method=None).iloc[1:]


def to_base_currency(returns: pd.DataFrame, meta: pd.DataFrame, eurusd: pd.Series,
                     hedged: bool = False) -> pd.DataFrame:
    """Convert USD-asset returns to EUR (unhedged). eurusd = USD per 1 EUR.

    If EURUSD rises the USD weakens, so a EUR investor loses on USD assets:
        r_eur = (1 + r_usd) * fx_{t-1} / fx_t - 1
    hedged=True keeps local returns (ignores the hedge carry - a simplification).
    """
    if hedged:
        return returns.copy()
    out = returns.copy()
    usd = [t for t in meta.loc[meta["currency"] == "USD", "ticker"] if t in out.columns]
    fx = eurusd.reindex(out.index.union(eurusd.index)).ffill().reindex(out.index)
    fx_ret = fx.pct_change(fill_method=None).fillna(0.0)
    out[usd] = (1 + out[usd]).div(1 + fx_ret, axis=0) - 1
    return out


def max_drawdown(r: pd.Series) -> float:
    wealth = (1 + r.fillna(0)).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


# --------------------------------------------------------------------------- asset statistics
def asset_statistics(levels: pd.DataFrame, lookback_days: int = 252) -> pd.DataFrame:
    """Per-asset trailing statistics from a PIT levels frame."""
    lv = levels.dropna(how="all").iloc[-(lookback_days + 1):]
    r = daily_returns(lv)

    def trailing(n):
        return levels.iloc[-1] / levels.iloc[-(n + 1)] - 1 if len(levels) > n else np.nan

    stats = pd.DataFrame({
        "ret_1m": trailing(21),
        "ret_3m": trailing(63),
        "ret_12m": trailing(252),
        "ann_vol": r.std() * np.sqrt(TD),
        "max_dd": r.apply(max_drawdown),
    })
    stats["ret_ann"] = (lv.iloc[-1] / lv.iloc[0]) ** (TD / max(len(r), 1)) - 1
    stats["sharpe_like"] = stats["ret_ann"] / stats["ann_vol"].replace(0, np.nan)
    return stats


# --------------------------------------------------------------------------- portfolio risk
def exposures(weights: pd.Series, meta: pd.DataFrame, field: str) -> pd.Series:
    m = meta.set_index("ticker")[field]
    return weights.groupby(m.reindex(weights.index)).sum().sort_values(ascending=False)


def portfolio_metrics(weights: pd.Series, returns: pd.DataFrame, meta: pd.DataFrame,
                      benchmark: pd.Series | None = None) -> dict:
    """Historical-simulation risk metrics for fixed weights over the given return window.

    Assumes constant weights (daily rebalanced) - a simplification worth stating.
    """
    w = weights.reindex(returns.columns).fillna(0.0)
    r = returns.fillna(0.0)
    port = r @ w
    vol = port.std() * np.sqrt(TD)
    q05 = port.quantile(0.05)
    m21 = (1 + port).rolling(21).apply(np.prod, raw=True).dropna() - 1

    cov = r.cov() * TD
    sigma_w = cov @ w
    rc = w * sigma_w / vol if vol > 0 else w * 0
    rc_pct = rc / rc.sum() if rc.sum() != 0 else rc

    dur = meta.set_index("ticker")["duration"].reindex(w.index).fillna(0.0)
    out = {
        "ann_return_hist": float((1 + port).prod() ** (TD / len(port)) - 1),
        "ann_vol": float(vol),
        "var95_1d": float(-q05),
        "es95_1d": float(-port[port <= q05].mean()),
        "var95_1m": float(-m21.quantile(0.05)) if len(m21) else float("nan"),
        "max_drawdown": max_drawdown(port),
        "duration": float((w * dur).sum()),
        "risk_contrib_by_class": exposures(rc_pct, meta, "asset_class").round(4).to_dict(),
        "risk_contrib_by_asset": rc_pct.sort_values(ascending=False).head(8).round(4).to_dict(),
        "window": f"{returns.index[0]:%Y-%m-%d} to {returns.index[-1]:%Y-%m-%d}",
    }
    if benchmark is not None:
        b = benchmark.reindex(returns.columns).fillna(0.0)
        active = port - r @ b
        out["tracking_error"] = float(active.std() * np.sqrt(TD))
        out["benchmark_ann_vol"] = float((r @ b).std() * np.sqrt(TD))
    return out


# --------------------------------------------------------------------------- constraints
def _mask(meta: pd.DataFrame, lim: Limit, tickers) -> np.ndarray:
    m = meta.set_index("ticker").reindex(tickers)
    col = m.index.to_series() if lim.field == "ticker" else m[lim.field]
    return (col == lim.value).to_numpy()


def check_constraints(weights: pd.Series, meta: pd.DataFrame, limits: list[Limit] = LIMITS,
                      max_single: float | None = MAX_SINGLE_ASSET, tol: float = 1e-6) -> list[str]:
    """Return a list of human-readable violations (empty = compliant)."""
    v = []
    w = weights
    if (w < -tol).any():
        v.append(f"Short positions not allowed: {w[w < -tol].round(4).to_dict()}")
    if abs(w.sum() - 1) > 1e-4:
        v.append(f"Weights sum to {w.sum():.4f}, not 1")
    for lim in limits:
        x = float(w[_mask(meta, lim, w.index)].sum())
        if (lim.sense == "<=" and x > lim.bound + tol) or (lim.sense == ">=" and x < lim.bound - tol):
            v.append(f"{lim.label or lim.value}: is {x:.2%}, limit {lim.sense} {lim.bound:.0%}")
    if max_single is not None and (w > max_single + tol).any():
        v.append(f"Single-asset cap {max_single:.0%} breached: {w[w > max_single + tol].round(4).to_dict()}")
    return v


def round_weights(w: pd.Series, decimals: int = 10, sink: str | None = None) -> pd.Series:
    """Round for output WITHOUT breaking limits: floor every weight, then put the
    (tiny, positive) residual into a domestic cash asset. Flooring can never push a
    capped group above its cap; plain rounding can (a 20.00004% US weight).
    The official weights.csv uses 10 decimals (deliverables.pdf)."""
    if sink is None:
        from src.utils.universe import cash_sink
        sink = cash_sink()
    f = 10.0**decimals
    # np.round(.., 6) first: 0.13 * 1e10 is 1299999999.9999998 in floating point, and a
    # plain floor would turn it into 0.1299999999.
    out = np.floor(np.round(w.clip(lower=0) * f, 6)) / f
    out[sink] = out.get(sink, 0.0) + round(1.0 - out.sum(), decimals)
    return out


def project_to_constraints(w0: pd.Series, meta: pd.DataFrame, limits: list[Limit] = LIMITS,
                           max_single: float | None = MAX_SINGLE_ASSET) -> pd.Series:
    """Closest (least-squares) feasible portfolio to w0. Deterministic safety net."""
    tickers = list(w0.index)
    x0 = np.clip(w0.to_numpy(dtype=float), 0, None)
    x0 = x0 / x0.sum() if x0.sum() > 0 else np.full(len(x0), 1 / len(x0))
    cons = [{"type": "eq", "fun": lambda x: x.sum() - 1}]
    for lim in limits:
        mk = _mask(meta, lim, tickers).astype(float)
        if lim.sense == "<=":
            cons.append({"type": "ineq", "fun": lambda x, mk=mk, b=lim.bound: b - mk @ x})
        else:
            cons.append({"type": "ineq", "fun": lambda x, mk=mk, b=lim.bound: mk @ x - b})
    ub = 1.0 if max_single is None else max_single
    res = minimize(lambda x: ((x - x0) ** 2).sum(), x0, jac=lambda x: 2 * (x - x0),
                   bounds=[(0, ub)] * len(x0), constraints=cons, method="SLSQP",
                   options={"ftol": 1e-12, "maxiter": 500})
    if not res.success:
        raise RuntimeError(f"Projection failed: {res.message}")
    x = np.clip(res.x, 0, None)
    x[x < 1e-6] = 0.0
    return pd.Series(x / x.sum(), index=tickers)
