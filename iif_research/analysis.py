"""Numerical analysis for the IIF newsletter; all report periods are bounded."""
from __future__ import annotations

from itertools import combinations
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.covariance import LedoitWolf

STOCK_HORIZONS = {5: .15, 15: .15, 21: .20, 42: .20, 63: .20, 126: .10}
INDEX_HORIZONS = {2: .10, 5: .20, 10: .25, 21: .25, 42: .15, 63: .05}
MACRO_NAMES = ("Brent", "USD/INR", "Gold", "Nifty 50")


def _empty(columns):
    return pd.DataFrame(columns=columns)


def _dates(frame, col="date"):
    if col not in frame:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    return pd.to_datetime(frame[col], errors="coerce").dt.normalize()


def _bounded(frame, start, end, col="date"):
    if frame.empty or col not in frame:
        return frame.iloc[:0].copy()
    dates = _dates(frame, col)
    return frame.loc[dates.between(start, end)].copy()


def _number(s):
    return pd.to_numeric(s, errors="coerce").astype(float)


def _prepare_prices(prices):
    if prices.empty or not {"date", "symbol", "close"}.issubset(prices.columns):
        return prices.iloc[:0].copy()
    data = prices.copy()
    data["date"] = _dates(data)
    data["symbol"] = data.symbol.astype(str)
    data["close"] = _number(data.close)
    data["raw_close"] = _number(data.raw_close) if "raw_close" in data else data.close
    data["volume"] = _number(data.volume) if "volume" in data else np.nan
    return (data.dropna(subset=["date"]).sort_values(["symbol", "date"])
            .drop_duplicates(["symbol", "date"], keep="last"))


def _metadata(members, markets):
    stocks = members.copy() if not members.empty else pd.DataFrame(columns=["symbol", "name", "sector"])
    indexes = markets.copy() if not markets.empty else pd.DataFrame(columns=["symbol", "name", "group", "unit"])
    for frame, cols in ((stocks, ("symbol", "name", "sector")), (indexes, ("symbol", "name", "group", "unit"))):
        for col in cols:
            if col not in frame:
                frame[col] = np.nan
        frame.drop_duplicates("symbol", keep="last", inplace=True)
    return stocks, indexes


def _period_measure(history, start, end):
    cols = ["symbol", "period_return_pct", "anchor_date", "period_end_date", "period_start_value", "period_end_value"]
    if history.empty:
        return _empty(cols)
    base = history.loc[(history.date < start) & history.close.notna()].groupby("symbol", as_index=False).tail(1)
    last = history.loc[history.date.between(start, end) & history.close.notna()].groupby("symbol", as_index=False).tail(1)
    if base.empty or last.empty:
        return _empty(cols)
    base = base[["symbol", "date", "close"]].rename(columns={"date": "anchor_date", "close": "period_start_value"})
    last = last[["symbol", "date", "close"]].rename(columns={"date": "period_end_date", "close": "period_end_value"})
    out = base.merge(last, on="symbol", how="inner")
    out["period_return_pct"] = (out.period_end_value / out.period_start_value - 1) * 100
    return out[cols]


def _momentum(history, metadata, horizons, start, end, yield_groups=None):
    if "group" in metadata.columns:
        meta_cols = ["symbol", "name", "group", "unit"]
        out_cols = meta_cols
    else:
        meta_cols = ["symbol", "name", "sector"]
        out_cols = meta_cols
    horizon_cols = [f"return_{n}s_pct" for n in horizons]
    cols = out_cols + horizon_cols + ["composite_score_pct", "period_return_pct", "period_return_unit", "anchor_date", "period_end_date", "eligible", "momentum_history_complete", "eligibility_reason"]
    if history.empty:
        return _empty(cols)
    data = history.loc[history.date <= end].sort_values(["symbol", "date"])
    rows = []
    for symbol, group in data.groupby("symbol", sort=False):
        group = group.dropna(subset=["close"]).reset_index(drop=True)
        if group.empty:
            continue
        meta = metadata.loc[metadata.symbol.astype(str) == str(symbol)]
        row = {"symbol": symbol}
        if not meta.empty:
            row.update(meta.iloc[0].to_dict())
        is_yield = bool(yield_groups and str(row.get("group", "")) in yield_groups)
        for n in horizons:
            row[f"return_{n}s_pct"] = ((group.close.iloc[-1] - group.close.iloc[-n-1]) if is_yield else
                                        (group.close.iloc[-1] / group.close.iloc[-n-1] - 1) * 100) if len(group) > n else np.nan
        vals = [row[f"return_{n}s_pct"] for n in horizons]
        row["composite_score_pct"] = sum(v * w for v, w in zip(vals, horizons.values())) if all(pd.notna(v) for v in vals) else np.nan
        row["eligible"] = all(pd.notna(row[f"return_{n}s_pct"]) and row[f"return_{n}s_pct"] > 0 for n in list(horizons)[:3])
        row["momentum_history_complete"] = all(pd.notna(v) for v in vals)
        row["eligibility_reason"] = ("eligible" if row["eligible"] else "fails positive 5/15/21-session filter")
        if not row["momentum_history_complete"]:
            row["eligibility_reason"] = "insufficient history for all weighted horizons"
        rows.append(row)
    out = pd.DataFrame(rows)
    for c in out_cols:
        if c not in out:
            out[c] = np.nan
    period = _period_measure(history, start, end)
    out = out.merge(period, on="symbol", how="left")
    out["period_return_unit"] = "%"
    if yield_groups and "group" in out:
        is_yield = out.group.astype(str).isin(yield_groups)
        out.loc[is_yield, "period_return_pct"] = (out.loc[is_yield, "period_end_value"] - out.loc[is_yield, "period_start_value"])
        out.loc[is_yield, "period_return_unit"] = "percentage points"
    for c in cols:
        if c not in out:
            out[c] = np.nan
    return out[cols].sort_values(["eligible", "composite_score_pct"], ascending=[False, False], na_position="last").reset_index(drop=True)


def _period_prices(history, metadata, start, end):
    cols = ["date", "symbol", "name", "group", "unit", "close", "normalized", "anchor_date", "period_return_pct"]
    if history.empty:
        return _empty(cols)
    history = history.drop(columns=[c for c in ("name", "group", "unit") if c in history])
    anchor = history.loc[(history.date < start) & history.close.notna()].groupby("symbol", as_index=False).tail(1)[["symbol", "date", "close"]]
    anchor = anchor.rename(columns={"date": "anchor_date", "close": "base_close"})
    period = history.loc[history.date.between(start, end) & history.close.notna()].merge(anchor, on="symbol", how="inner")
    if period.empty:
        return _empty(cols)
    period["normalized"] = period.close / period.base_close * 100
    period["period_return_pct"] = (period.close / period.base_close - 1) * 100
    period = period.merge(metadata[[c for c in ("symbol", "name", "group", "unit") if c in metadata]], on="symbol", how="left")
    for c in ("name", "group", "unit"):
        if c not in period:
            period[c] = np.nan
    return period[cols].sort_values(["symbol", "date"]).reset_index(drop=True)


def _daily_returns(history, start, end):
    cols = ["date", "symbol", "close", "daily_return_pct"]
    if history.empty:
        return _empty(cols)
    dates = sorted(history.date.dropna().unique())
    levels = history.pivot(index="date", columns="symbol", values="close").reindex(dates)
    returns = levels.pct_change(fill_method=None) * 100
    returns = returns.rename_axis("date").stack(future_stack=True).rename("daily_return_pct").reset_index()
    data = history[["date", "symbol", "close"]].merge(returns, on=["date", "symbol"], how="left")
    return data.loc[data.date.between(start, end), cols].reset_index(drop=True)


def _breadth(history, members, start, end):
    cols = ["date", "eligible_adv_dec", "advancers", "decliners", "unchanged", "advance_decline_ratio", "above_50dma_pct", "above_200dma_pct", "eligible_50dma", "eligible_200dma"]
    if history.empty or members.empty or "symbol" not in members:
        return _empty(cols)
    stock_history = history.loc[history.symbol.astype(str).isin(set(members.symbol.astype(str)))].copy()
    dates = sorted(history.date.dropna().unique())
    levels = stock_history.pivot(index="date", columns="symbol", values="close").reindex(index=dates)
    daily = levels.pct_change(fill_method=None) * 100
    ma50 = levels.shift(1).rolling(50, min_periods=50).mean()
    ma200 = levels.shift(1).rolling(200, min_periods=200).mean()
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    rows = []
    for date in (d for d in dates if start <= d <= end):
        eligible_returns = daily.loc[date].dropna()
        adv = int((eligible_returns > 0).sum())
        dec = int((eligible_returns < 0).sum())
        unchanged = int((eligible_returns == 0).sum())
        eligible50 = levels.loc[date].notna() & ma50.loc[date].notna()
        eligible200 = levels.loc[date].notna() & ma200.loc[date].notna()
        rows.append({"date": date, "eligible_adv_dec": int(len(eligible_returns)), "advancers": adv,
                     "decliners": dec, "unchanged": unchanged,
                     "advance_decline_ratio": adv / dec if dec else np.nan,
                     "above_50dma_pct": (levels.loc[date, eligible50] > ma50.loc[date, eligible50]).mean() * 100 if eligible50.any() else np.nan,
                     "above_200dma_pct": (levels.loc[date, eligible200] > ma200.loc[date, eligible200]).mean() * 100 if eligible200.any() else np.nan,
                     "eligible_50dma": int(eligible50.sum()), "eligible_200dma": int(eligible200.sum())})
    return pd.DataFrame(rows, columns=cols).sort_values("date").reset_index(drop=True) if rows else _empty(cols)


def _highs_lows(history, start, end):
    cols = ["date", "symbol", "close", "prior_252_high", "prior_252_low", "new_252_session_high", "new_252_session_low"]
    if history.empty:
        return _empty(cols)
    d = history.sort_values(["symbol", "date"]).copy()
    g = d.groupby("symbol").close
    d["prior_252_high"] = g.transform(lambda s: s.shift(1).rolling(252, min_periods=252).max())
    d["prior_252_low"] = g.transform(lambda s: s.shift(1).rolling(252, min_periods=252).min())
    d["new_252_session_high"] = d.close > d.prior_252_high
    d["new_252_session_low"] = d.close < d.prior_252_low
    return d.loc[d.date.between(start, end), cols].reset_index(drop=True)


def _volume_anomalies(history, start, end):
    cols = ["date", "symbol", "volume", "prior_20_session_avg_volume", "volume_vs_prior_avg"]
    if history.empty or "volume" not in history:
        return _empty(cols)
    d = history.sort_values(["symbol", "date"]).copy()
    d["prior_20_session_avg_volume"] = d.groupby("symbol").volume.transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    d["volume_vs_prior_avg"] = d.volume / d.prior_20_session_avg_volume
    return d.loc[d.date.between(start, end), cols].reset_index(drop=True)


def _period_date(value):
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.notna(parsed):
        return parsed
    label = str(value).strip().upper()
    # Common Indian quarter labels: Q1 FY27 / FY27 Q1 (fiscal year ends in March).
    match = re.search(r"Q([1-4]).*?FY\s?(\d{2,4})|FY\s?(\d{2,4}).*?Q([1-4])", label)
    if match:
        q = int(match.group(1) or match.group(4))
        fy = int(match.group(2) or match.group(3))
        fy = fy + 2000 if fy < 100 else fy
        month = ((q + 1) * 3) % 12 or 12
        if q < 4:
            year = fy - 1
        else:
            year = fy
        return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    match = re.search(r"(\d{4})\s*[-/]?\s*Q([1-4])|Q([1-4])\s*[-/]?\s*(\d{4})", label)
    if match:
        year = int(match.group(1) or match.group(4))
        q = int(match.group(2) or match.group(3))
        month = q * 3
        return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
    return pd.NaT


def _holdings_changes(holdings, end):
    cols = ["symbol", "previous_period", "latest_period", "previous_fii_pct", "latest_fii_pct", "fii_change_pp", "previous_dii_pct", "latest_dii_pct", "dii_change_pp", "basis_note"]
    if holdings.empty or not {"symbol", "period"}.issubset(holdings):
        return _empty(cols)
    d = holdings.copy()
    d["_pdate"] = d.period.map(_period_date)
    d = d.loc[d._pdate.le(end)].sort_values("_pdate").drop_duplicates(["symbol", "period"], keep="last")
    periods = d[["period", "_pdate"]].drop_duplicates().tail(2)
    if len(periods) < 2:
        return _empty(cols)
    p0, p1 = periods.period.tolist()
    a, b = d.loc[d.period == p0].set_index("symbol"), d.loc[d.period == p1].set_index("symbol")
    rows = []
    for sym in a.index.union(b.index):
        x = a.loc[sym] if sym in a.index else pd.Series(dtype=object)
        y = b.loc[sym] if sym in b.index else pd.Series(dtype=object)
        get = lambda s, k: pd.to_numeric(s.get(k, np.nan), errors="coerce")
        af, bf, ad, bd = get(x, "fii_pct"), get(y, "fii_pct"), get(x, "dii_pct"), get(y, "dii_pct")
        rows.append({"symbol": sym, "previous_period": p0, "latest_period": p1, "previous_fii_pct": af, "latest_fii_pct": bf,
                     "fii_change_pp": bf-af if pd.notna(af) and pd.notna(bf) else np.nan,
                     "previous_dii_pct": ad, "latest_dii_pct": bd, "dii_change_pp": bd-ad if pd.notna(ad) and pd.notna(bd) else np.nan,
                     "basis_note": "Latest two quarter labels on or before report end; disclosure/as-of dates are not inferred."})
    return pd.DataFrame(rows, columns=cols)


def _quarterly(quarterly, end):
    d = quarterly.copy()
    if d.empty or not {"symbol", "period"}.issubset(d):
        d["snapshot_note"] = pd.Series(dtype=str)
        return d, d.copy()
    d["_pdate"] = d.period.map(_period_date)
    eligible = d.loc[d._pdate.le(end)].copy()
    if eligible.empty:
        latest = d.iloc[:0].copy()
    else:
        sort_cols = ["_pdate"] + (["fetched_at"] if "fetched_at" in eligible else [])
        latest = eligible.sort_values(sort_cols).groupby("symbol", as_index=False).tail(1).copy()
    latest["snapshot_note"] = "Selected by report period label on or before report end; filing/disclosure date is not guaranteed."
    return latest.drop(columns="_pdate", errors="ignore").reset_index(drop=True), eligible.drop(columns="_pdate", errors="ignore")


def _flows(flows, start, end):
    scols = ["category", "buy_cr", "sell_cr", "net_cr", "first_date", "last_date"]
    ccols = ["date", "category", "daily_net_cr", "cumulative_net_cr"]
    if flows.empty or "date" not in flows:
        return _empty(scols), _empty(ccols)
    d = _bounded(flows, start, end)
    if d.empty:
        return _empty(scols), _empty(ccols)
    d["date"] = _dates(d)
    for c in ("buy_cr", "sell_cr", "net_cr"):
        d[c] = _number(d[c]) if c in d else np.nan
    if "category" not in d:
        d["category"] = "All"
    summary = d.groupby("category", dropna=False).agg(buy_cr=("buy_cr", "sum"), sell_cr=("sell_cr", "sum"), net_cr=("net_cr", "sum"), first_date=("date", "min"), last_date=("date", "max")).reset_index()
    cumulative = d.groupby(["date", "category"], dropna=False, as_index=False).net_cr.sum().rename(columns={"net_cr": "daily_net_cr"}).sort_values(["category", "date"])
    cumulative["cumulative_net_cr"] = cumulative.groupby("category").daily_net_cr.cumsum()
    return summary[scols], cumulative[ccols].reset_index(drop=True)


def _bh(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if not len(p):
        return []
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(len(p))
    out[order] = np.clip(q, 0, 1)
    return out.tolist()


def _sector_round_robin(d, count):
    if d.empty:
        return []
    d = d.sort_values(["sector", "avg_traded_value_cr"], ascending=[True, False], na_position="last")
    sectors = sorted(d.sector.fillna("Unknown").astype(str).unique())
    queues = {s: d.loc[d.sector.fillna("Unknown").astype(str) == s, "symbol"].astype(str).tolist() for s in sectors}
    selected = []
    while len(selected) < count and any(queues.values()):
        for s in sectors:
            if queues[s] and len(selected) < count:
                selected.append(queues[s].pop(0))
    return selected


def _correlations(history, members, markets, config, end):
    empty = {
        "correlation_universe": _empty(["symbol", "sector", "avg_traded_value_cr", "coverage", "selected", "exclusion_reason"]),
        "correlation_pearson": pd.DataFrame(), "correlation_shrinkage": pd.DataFrame(),
        "correlation_shrinkage_method": _empty(["method", "complete_joint_observations", "minimum_observations"]),
        "correlation_source_method": _empty(["return_window_sessions", "price_dates_used", "minimum_price_coverage", "minimum_turnover_cr", "coverage_basis", "turnover_basis", "first_date", "last_date"]),
        "correlation_sample_sizes": pd.DataFrame(),
        "correlation_pair_summary": _empty(["symbol_a", "symbol_b", "n_observations", "pearson_r", "p_value", "fdr_q_value", "shrinkage_r"]),
        "rolling_pair_correlation": _empty(["date", "symbol_a", "symbol_b", "rolling_60d_correlation"]),
        "macro_correlations": _empty(["symbol", "stock_name", "macro_name", "macro_symbol", "n_observations", "pearson_r", "p_value", "basis_note"])}
    if history.empty:
        return empty
    sessions = int(getattr(config, "correlation_sessions", 126))
    minimum = int(getattr(config, "min_correlation_observations", 60))
    count = int(getattr(config, "correlation_count", 24))
    # N daily returns require N+1 price observations for the initial close.
    dates = sorted(history.loc[history.date <= end, "date"].dropna().unique())[-(sessions + 1):]
    if not dates:
        return empty
    window = history.loc[history.date.isin(dates)].copy()
    window["traded_value"] = window.raw_close * window.volume / 10_000_000
    m = members.copy() if not members.empty else pd.DataFrame(columns=["symbol", "sector", "name"])
    for c in ("symbol", "sector", "name"):
        if c not in m: m[c] = np.nan
    m = m.drop_duplicates("symbol", keep="last")
    liq = window.groupby("symbol").agg(avg_traded_value_cr=("traded_value", "mean"), observations=("close", "count")).reset_index()
    liq = liq.merge(m[["symbol", "sector"]], on="symbol", how="left")
    liq["sector"] = liq.sector.fillna("Unknown")
    liq["coverage"] = liq.observations / len(dates)
    liq["selected"] = False
    min_coverage = .90
    min_turnover_cr = 1.0
    liq["exclusion_reason"] = "outside sector-balanced liquidity selection"
    liq.loc[liq.sector.eq("Unknown"), "exclusion_reason"] = "missing sector metadata"
    liq.loc[liq.coverage < min_coverage, "exclusion_reason"] = "price coverage below 90%"
    liq.loc[liq.avg_traded_value_cr < min_turnover_cr, "exclusion_reason"] = "average traded value below INR 1 crore"
    eligible = liq.loc[(liq.coverage >= min_coverage) & (liq.avg_traded_value_cr >= min_turnover_cr) & liq.sector.ne("Unknown")].copy()
    selected = _sector_round_robin(eligible, count)
    liq.loc[liq.symbol.astype(str).isin(selected), ["selected", "exclusion_reason"]] = [True, ""]
    empty["correlation_universe"] = liq[["symbol", "sector", "avg_traded_value_cr", "coverage", "selected", "exclusion_reason"]].sort_values(["selected", "sector", "avg_traded_value_cr"], ascending=[False, True, False]).reset_index(drop=True)
    empty["correlation_source_method"] = pd.DataFrame([{"return_window_sessions": sessions, "price_dates_used": len(dates),
                                                         "minimum_price_coverage": min_coverage, "minimum_turnover_cr": min_turnover_cr,
                                                         "coverage_basis": "observed non-null closes / shared market price dates",
                                                         "turnover_basis": "mean(raw_close * volume) converted to INR crore",
                                                         "first_date": dates[0], "last_date": dates[-1]}])
    if not selected:
        return empty
    # Pivot levels first, so an entirely absent bar stays NaN on the shared
    # calendar and cannot turn a two-session move into a one-session return.
    levels = window.loc[window.symbol.astype(str).isin(selected)].pivot(index="date", columns="symbol", values="close")
    levels = levels.reindex(index=dates, columns=selected)
    returns = levels.pct_change(fill_method=None)
    corr = pd.DataFrame(np.nan, index=selected, columns=selected, dtype=float)
    ns = pd.DataFrame(0, index=selected, columns=selected, dtype=int)
    records = []
    for a, b in combinations(selected, 2):
        pair = returns[[a, b]].dropna()
        n = len(pair); r = p = np.nan
        if n >= minimum and pair[a].nunique() > 1 and pair[b].nunique() > 1:
            r, p = pearsonr(pair[a], pair[b]); corr.loc[a, b] = corr.loc[b, a] = r
        ns.loc[a, b] = ns.loc[b, a] = n
        records.append({"symbol_a": a, "symbol_b": b, "n_observations": n, "pearson_r": r, "p_value": p})
    valid_positions = [i for i, rec in enumerate(records) if pd.notna(rec["p_value"])]
    qs = _bh([float(records[i]["p_value"]) for i in valid_positions])
    for i, q in zip(valid_positions, qs):
        records[i]["fdr_q_value"] = q
    for i, rec in enumerate(records):
        if i not in valid_positions:
            rec["fdr_q_value"] = np.nan
    np.fill_diagonal(corr.values, 1.)
    np.fill_diagonal(ns.values, returns.reindex(columns=selected).notna().sum(axis=0).to_numpy())
    # Ledoit-Wolf uses only fully observed joint rows. If the joint panel is
    # too sparse, report no shrinkage estimate instead of imputing returns.
    shrink = pd.DataFrame(np.nan, index=selected, columns=selected, dtype=float)
    returns = returns.reindex(columns=selected)
    complete = returns.dropna(how="any")
    if len(complete) >= minimum and len(selected) >= 2:
        lw = LedoitWolf().fit(complete.to_numpy())
        cov = lw.covariance_; scale = np.sqrt(np.diag(cov)); denom = np.outer(scale, scale)
        shrink.loc[:, :] = np.divide(cov, denom, out=np.full_like(cov, np.nan), where=denom != 0)
        shrink_method = "LedoitWolf"
    else:
        shrink_method = "unavailable: insufficient complete joint observations"
    for rec in records: rec["shrinkage_r"] = shrink.loc[rec["symbol_a"], rec["symbol_b"]]
    empty["correlation_pearson"] = corr.rename_axis("symbol").reset_index()
    empty["correlation_shrinkage"] = shrink.rename_axis("symbol").reset_index()
    empty["correlation_shrinkage_method"] = pd.DataFrame([{"method": shrink_method, "complete_joint_observations": len(complete), "minimum_observations": minimum}])
    empty["correlation_sample_sizes"] = ns.rename_axis("symbol").reset_index()
    empty["correlation_pair_summary"] = pd.DataFrame(records, columns=["symbol_a", "symbol_b", "n_observations", "pearson_r", "p_value", "fdr_q_value", "shrinkage_r"])
    rolling = []
    for a, b in combinations(selected, 2):
        pair = returns[[a, b]].dropna()
        for i in range(59, len(pair)):
            sample = pair.iloc[i-59:i+1]
            if sample[a].nunique() > 1 and sample[b].nunique() > 1:
                rolling.append({"date": pair.index[i], "symbol_a": a, "symbol_b": b, "rolling_60d_correlation": sample[a].corr(sample[b])})
    if rolling: empty["rolling_pair_correlation"] = pd.DataFrame(rolling)

    markets = markets.copy() if not markets.empty else pd.DataFrame(columns=["symbol", "name"])
    if "name" not in markets: markets["name"] = np.nan
    macros = markets.loc[markets.name.isin(MACRO_NAMES), ["symbol", "name"]].drop_duplicates()
    if not macros.empty:
        macro_history = _prepare_prices(markets)
        md = macro_history.loc[macro_history.symbol.isin(macros.symbol) & macro_history.date.isin(dates)]
        macro_levels = md.pivot(index="date", columns="symbol", values="close").reindex(index=dates)
        mr = macro_levels.pct_change(fill_method=None)
        names = m.set_index("symbol").name.to_dict(); macro_names = macros.set_index("symbol").name.to_dict()
        macro_rows = []
        for stock in selected:
            for ms in macros.symbol:
                if ms not in mr: continue
                pair = pd.concat([returns[stock], mr[ms]], axis=1).dropna()
                if len(pair) >= minimum and pair.iloc[:, 0].nunique() > 1 and pair.iloc[:, 1].nunique() > 1:
                    r, p = pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
                    macro_rows.append({"symbol": stock, "stock_name": names.get(stock), "macro_name": macro_names.get(ms), "macro_symbol": ms, "n_observations": len(pair), "pearson_r": r, "p_value": p,
                                       "basis_note": "Descriptive same-date returns across instruments with different trading calendars; no predictive interpretation."})
        empty["macro_correlations"] = pd.DataFrame(macro_rows, columns=list(empty["macro_correlations"].columns))
    return empty


def analyze(tables: dict[str, pd.DataFrame], config: Any) -> dict[str, pd.DataFrame]:
    """Return source tables and derived report tables for an inclusive date range."""
    start, end = pd.Timestamp(config.start).normalize(), pd.Timestamp(config.end).normalize()
    result = {k: v.copy() for k, v in tables.items() if isinstance(v, pd.DataFrame)}
    prices = _prepare_prices(tables.get("prices", pd.DataFrame()))
    markets, members = tables.get("markets", pd.DataFrame()), tables.get("members", pd.DataFrame())
    stock_meta, market_meta = _metadata(members, markets)
    market_history = _prepare_prices(markets)
    result["stock_momentum"] = _momentum(prices, stock_meta, STOCK_HORIZONS, start, end)
    valid_returns = result["stock_momentum"].dropna(subset=["period_return_pct"])
    result["period_gainers"] = valid_returns.sort_values("period_return_pct", ascending=False).reset_index(drop=True)
    result["period_losers"] = valid_returns.sort_values("period_return_pct", ascending=True).reset_index(drop=True)
    result["index_momentum"] = _momentum(market_history, market_meta, INDEX_HORIZONS, start, end, {"Yield"})
    # Chart series cover both constituent shares and market indicators, each
    # rebased to its own last observed close strictly before the report start.
    chart_history = pd.concat([prices, market_history], ignore_index=True, sort=False)
    chart_meta = pd.concat([
        stock_meta.assign(group=np.nan, unit="INR/share"),
        market_meta[["symbol", "name", "group", "unit"]],
    ], ignore_index=True, sort=False).drop_duplicates("symbol", keep="last")
    result["period_prices"] = _period_prices(chart_history, chart_meta, start, end)
    result["daily_returns"] = _daily_returns(prices, start, end)
    result["breadth"] = _breadth(prices, members, start, end)
    result["highs_lows"] = _highs_lows(prices, start, end)
    result["volume_anomalies"] = _volume_anomalies(prices, start, end)
    result["holdings_changes"] = _holdings_changes(tables.get("holdings", pd.DataFrame()), end)
    result["quarterly_latest"], result["quarterly_history"] = _quarterly(tables.get("quarterly", pd.DataFrame()), end)
    result["flow_summary"], result["flow_cumulative"] = _flows(tables.get("flows", pd.DataFrame()), start, end)
    result.update(_correlations(prices, members, markets, config, end))
    return result
