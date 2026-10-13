"""Observed-data risk, breadth, rotation, and market-state measures."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


STOCK_RISK_COLUMNS = ["symbol", "name", "sector", "as_of", "return_63_pct", "vol_63_ann_pct", "downside_vol_63_ann_pct", "max_drawdown_63_pct", "report_return_pct", "avg_traded_value_cr", "liquidity_sessions", "price_observations", "data_quality"]
SECTOR_ROTATION_COLUMNS = ["sector", "date", "relative_strength_100", "relative_momentum_21_pct", "sector_close", "nifty_close", "relative_observations"]
ADVANCE_DECLINE_COLUMNS = ["date", "advancers", "decliners", "unchanged", "eligible", "advance_decline_ratio", "ad_line", "ad_line_normalized", "new_highs_52w", "new_lows_52w", "highlow_eligible", "above_50ma_pct", "above_200ma_pct", "breadth_thrust_10s_pct"]
PRICE_VOLUME_COLUMNS = ["symbol", "date", "return_pct", "volume_ratio_20d", "delivery_pct", "traded_value_cr", "delivery_observations"]
NIFTY_STATE_COLUMNS = ["date", "return_pct", "return_20_pct", "return_63_pct", "vol_20_ann_pct", "vol_63_ann_pct", "range_position_20d_pct", "overnight_gap_pct", "intraday_spread_pct"]
INSTITUTION_COLUMNS = ["date", "fii_net_cr", "dii_net_cr", "combined_net_cr", "domestic_absorption_ratio"]
SPREAD_COLUMNS = ["date", "spread_name", "value", "unit"]
REGIME_COLUMNS = ["metric", "value", "unit", "as_of", "definition"]
RETURN_DISTRIBUTION_COLUMNS = ["date", "symbol", "name", "sector", "return_pct", "traded_value_cr"]


def _empty_outputs() -> dict[str, pd.DataFrame]:
    schemas = {
        "stock_risk_return": STOCK_RISK_COLUMNS, "sector_rotation": SECTOR_ROTATION_COLUMNS,
        "advance_decline": ADVANCE_DECLINE_COLUMNS, "stock_price_volume_delivery": PRICE_VOLUME_COLUMNS,
        "nifty_market_state": NIFTY_STATE_COLUMNS, "institution_absorption": INSTITUTION_COLUMNS,
        "factor_spreads": SPREAD_COLUMNS, "regime_snapshot": REGIME_COLUMNS,
        "stock_return_distribution": RETURN_DISTRIBUTION_COLUMNS,
    }
    return {key: pd.DataFrame(columns=columns) for key, columns in schemas.items()}


def _date_col(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["date"], errors="coerce").dt.normalize() if "date" in frame else pd.Series(pd.NaT, index=frame.index)


def _numeric(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame[col], errors="coerce") if col in frame else pd.Series(np.nan, index=frame.index, dtype=float)


def _bounded(frame: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    if frame.empty or "date" not in frame:
        return frame.iloc[:0].copy()
    out = frame.copy()
    out["date"] = _date_col(out)
    return out.loc[out.date.notna() & out.date.le(end)].copy()


def _prepare_prices(frame: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    columns = ["date", "symbol", "close", "raw_close", "volume", "high", "low", "open"]
    if frame.empty or not {"date", "symbol", "close"}.issubset(frame.columns):
        return pd.DataFrame(columns=columns)
    out = _bounded(frame, end)
    out["symbol"] = out.symbol.astype(str).str.strip().str.upper()
    for col in ("close", "raw_close", "volume", "high", "low", "open"):
        out[col] = _numeric(out, col)
    if "raw_close" not in frame:
        out["raw_close"] = out.close
    return out.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")[columns]


def _prepare_members(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "symbol" not in frame:
        return pd.DataFrame(columns=["symbol", "name", "sector"])
    out = frame.copy()
    out["symbol"] = out.symbol.astype(str).str.strip().str.upper()
    for col in ("name", "sector"):
        if col not in out:
            out[col] = ""
    out["name"] = out.name.fillna("").astype(str)
    out["sector"] = out.sector.fillna("Unknown").astype(str)
    return out.drop_duplicates("symbol", keep="last")[["symbol", "name", "sector"]]


def _stock_risk(prices: pd.DataFrame, members: pd.DataFrame, calendar: pd.DatetimeIndex,
                start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not len(calendar) or prices.empty or members.empty:
        return (pd.DataFrame(columns=STOCK_RISK_COLUMNS), pd.DataFrame(columns=RETURN_DISTRIBUTION_COLUMNS), pd.DataFrame())
    symbols = members.symbol.tolist()
    p = prices.loc[prices.symbol.isin(symbols)].copy()
    if p.empty:
        return (pd.DataFrame(columns=STOCK_RISK_COLUMNS), pd.DataFrame(columns=RETURN_DISTRIBUTION_COLUMNS), pd.DataFrame())
    closes = p.pivot(index="date", columns="symbol", values="close").reindex(index=calendar, columns=symbols)
    returns = closes.pct_change(fill_method=None)
    raw = p.pivot(index="date", columns="symbol", values="raw_close").reindex(index=calendar, columns=symbols)
    volume = p.pivot(index="date", columns="symbol", values="volume").reindex(index=calendar, columns=symbols)
    traded = (raw * volume / 10_000_000).replace([np.inf, -np.inf], np.nan)
    meta = members.set_index("symbol")
    actual_by_symbol = {symbol: group for symbol, group in p.groupby("symbol", sort=False)}
    risk_rows = []
    for symbol in symbols:
        levels, r = closes[symbol], returns[symbol]
        trailing63 = r.tail(63)
        vol63 = downside63 = np.nan
        if len(trailing63) == 63 and trailing63.notna().all():
            values = trailing63.to_numpy(dtype=float)
            vol63 = float(values.std(ddof=1) * np.sqrt(252) * 100)
            downside63 = float(np.sqrt(np.mean(np.minimum(values, 0) ** 2)) * np.sqrt(252) * 100)
        trailing64 = levels.tail(64)
        ret63 = drawdown63 = np.nan
        if len(trailing64) == 64 and trailing64.notna().all():
            values = trailing64.to_numpy(dtype=float)
            ret63 = float((values[-1] / values[0] - 1) * 100)
            drawdown63 = float((values / np.maximum.accumulate(values) - 1).min() * 100)
        actual = actual_by_symbol.get(symbol, p.iloc[:0])
        as_of = actual.date.max() if not actual.empty else pd.NaT
        base = actual.loc[actual.date.lt(start)].tail(1)
        last = actual.loc[actual.date.le(end)].tail(1)
        report_return = np.nan
        if not base.empty and not last.empty and last.date.iloc[0] >= start and float(base.close.iloc[0]) != 0:
            report_return = float((last.close.iloc[0] / base.close.iloc[0] - 1) * 100)
        value_window = traded[symbol].tail(63)
        liquidity_sessions = int(value_window.notna().sum())
        avg_value = float(value_window.mean()) if len(value_window) == 63 and liquidity_sessions >= 57 else np.nan
        quality = []
        if not np.isfinite(ret63):
            quality.append("incomplete trailing 63-session close window")
        if not np.isfinite(vol63):
            quality.append("incomplete trailing 63-session return window")
        if pd.isna(report_return):
            quality.append("report return lacks a prior-start anchor or in-period close")
        risk_rows.append({"symbol": symbol, "name": meta.at[symbol, "name"], "sector": meta.at[symbol, "sector"],
                          "as_of": as_of, "return_63_pct": ret63, "vol_63_ann_pct": vol63,
                          "downside_vol_63_ann_pct": downside63, "max_drawdown_63_pct": drawdown63,
                          "report_return_pct": report_return, "avg_traded_value_cr": avg_value,
                          "liquidity_sessions": liquidity_sessions, "price_observations": int(levels.tail(252).notna().sum()),
                          "data_quality": "; ".join(quality) if quality else "complete for displayed metrics"})
    in_range = returns.loc[(returns.index >= start) & (returns.index <= end)]
    dist = in_range.stack(future_stack=True).rename("return_pct").reset_index()
    if dist.empty:
        dist = pd.DataFrame(columns=RETURN_DISTRIBUTION_COLUMNS)
    else:
        dist.columns = ["date", "symbol", "return_pct"]
        dist = dist.loc[dist.return_pct.notna()]
        dist = dist.merge(members, on="symbol", how="left")
        turn = traded.loc[(traded.index >= start) & (traded.index <= end)].stack(future_stack=True).rename("traded_value_cr").reset_index()
        turn.columns = ["date", "symbol", "traded_value_cr"]
        turn = turn.loc[turn.traded_value_cr.notna()]
        dist = dist.merge(turn, on=["date", "symbol"], how="left")[RETURN_DISTRIBUTION_COLUMNS]
    return pd.DataFrame(risk_rows, columns=STOCK_RISK_COLUMNS), dist, returns


def _nifty_series(markets: pd.DataFrame, calendar: pd.DatetimeIndex, end: pd.Timestamp) -> pd.DataFrame:
    columns = ["close", "raw_close", "high", "low", "open"]
    if markets.empty or not {"date", "name", "close"}.issubset(markets.columns):
        return pd.DataFrame(index=calendar, columns=columns)
    d = _bounded(markets, end)
    names = d.name.fillna("").astype(str).str.strip().str.lower()
    d = d.loc[names.isin({"nifty 50", "nifty50"})].copy()
    if d.empty:
        return pd.DataFrame(index=calendar, columns=columns)
    d["close"] = _numeric(d, "close")
    d["raw_close"] = _numeric(d, "raw_close")
    if "raw_close" not in markets:
        d["raw_close"] = d.close
    # Market history may carry adjusted high/low values; those are not joined
    # to raw opens/closes for range or overnight-gap calculations.
    for col in ("high", "low", "open"):
        d[col] = np.nan
    return d.sort_values("date").drop_duplicates("date", keep="last").set_index("date")[columns].reindex(calendar)


def _nifty_state(markets: pd.DataFrame, nifty_ohlc: pd.DataFrame, calendar: pd.DatetimeIndex,
                 start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not len(calendar):
        return pd.DataFrame(columns=NIFTY_STATE_COLUMNS), pd.DataFrame()
    nifty = _nifty_series(markets, calendar, end)
    # Index OHLC is an unadjusted source table. Use its close/high/low/open
    # together for gap and range measures; never mix adjusted highs with raw close.
    if not nifty_ohlc.empty and {"date", "open", "high", "low", "close"}.issubset(nifty_ohlc.columns):
        ohlc = _bounded(nifty_ohlc, end)
        for col in ("open", "high", "low", "close"):
            ohlc[col] = _numeric(ohlc, col)
        ohlc = ohlc.drop_duplicates("date", keep="last").set_index("date").reindex(calendar)
        nifty["raw_close"] = ohlc.close
        nifty[["open", "high", "low"]] = ohlc[["open", "high", "low"]]
        if nifty.close.isna().all():
            nifty["close"] = ohlc.close
    close, raw = pd.to_numeric(nifty.close, errors="coerce"), pd.to_numeric(nifty.raw_close, errors="coerce")
    returns = close.pct_change(fill_method=None)
    vol20 = returns.rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252) * 100
    vol63 = returns.rolling(63, min_periods=63).std(ddof=1) * np.sqrt(252) * 100
    high20 = pd.to_numeric(nifty.high, errors="coerce").rolling(20, min_periods=20).max()
    low20 = pd.to_numeric(nifty.low, errors="coerce").rolling(20, min_periods=20).min()
    frame = pd.DataFrame({
        "return_pct": returns * 100,
        "return_20_pct": (close / close.shift(20) - 1) * 100,
        "return_63_pct": (close / close.shift(63) - 1) * 100,
        "vol_20_ann_pct": vol20,
        "vol_63_ann_pct": vol63,
        "range_position_20d_pct": (raw - low20) / (high20 - low20) * 100,
        "overnight_gap_pct": (pd.to_numeric(nifty.open, errors="coerce") / raw.shift(1) - 1) * 100,
        "intraday_spread_pct": (pd.to_numeric(nifty.high, errors="coerce") - pd.to_numeric(nifty.low, errors="coerce")) / raw * 100,
    }, index=calendar)
    visible = frame.loc[(calendar >= start) & (calendar <= end)].rename_axis("date").reset_index()
    return visible[NIFTY_STATE_COLUMNS], nifty


def _breadth(prices: pd.DataFrame, members: pd.DataFrame, calendar: pd.DatetimeIndex,
             start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not len(calendar) or prices.empty or members.empty:
        return pd.DataFrame(columns=ADVANCE_DECLINE_COLUMNS)
    symbols = members.symbol.tolist()
    rows = prices.loc[prices.symbol.isin(symbols)]
    levels = rows.pivot(index="date", columns="symbol", values="close").reindex(index=calendar, columns=symbols)
    returns = levels.pct_change(fill_method=None)
    adv = returns.gt(0).sum(axis=1).astype(int)
    dec = returns.lt(0).sum(axis=1).astype(int)
    unchanged = returns.eq(0).sum(axis=1).astype(int)
    eligible = returns.notna().sum(axis=1).astype(int)
    ad_ratio = adv.div(dec.where(dec.gt(0)))
    net = (adv - dec).where(eligible.gt(0))
    normalized_net = ((adv - dec) / eligible.where(eligible.gt(0))) * 100
    ad_line = net.cumsum()
    ad_line_normalized = 100 + normalized_net.cumsum()

    prior_levels = levels.shift(1)
    valid_252 = prior_levels.notna().rolling(252, min_periods=252).sum().eq(252) & levels.notna()
    high252 = prior_levels.rolling(252, min_periods=252).max()
    low252 = prior_levels.rolling(252, min_periods=252).min()
    new_high = (valid_252 & levels.ge(high252)).sum(axis=1).astype(float).where(valid_252.sum(axis=1).gt(0))
    new_low = (valid_252 & levels.le(low252)).sum(axis=1).astype(float).where(valid_252.sum(axis=1).gt(0))
    highlow_eligible = valid_252.sum(axis=1).astype(int)
    ma50 = prior_levels.rolling(50, min_periods=50).mean()
    ma200 = prior_levels.rolling(200, min_periods=200).mean()
    eligible50 = (ma50.notna() & levels.notna()).sum(axis=1)
    eligible200 = (ma200.notna() & levels.notna()).sum(axis=1)
    above50 = (levels.gt(ma50) & ma50.notna()).sum(axis=1).div(eligible50.where(eligible50.gt(0))) * 100
    above200 = (levels.gt(ma200) & ma200.notna()).sum(axis=1).div(eligible200.where(eligible200.gt(0))) * 100
    advancer_share = adv.div(eligible.where(eligible.gt(0)))
    thrust = advancer_share.rolling(10, min_periods=10).mean() * 100
    frame = pd.DataFrame({"advancers": adv, "decliners": dec, "unchanged": unchanged, "eligible": eligible,
                          "advance_decline_ratio": ad_ratio, "ad_line": ad_line,
                          "ad_line_normalized": ad_line_normalized, "new_highs_52w": new_high,
                          "new_lows_52w": new_low, "highlow_eligible": highlow_eligible,
                          "above_50ma_pct": above50, "above_200ma_pct": above200,
                          "breadth_thrust_10s_pct": thrust}, index=calendar)
    visible = frame.loc[(calendar >= start) & (calendar <= end)].copy()
    # Start both cumulative lines immediately before the report window.
    visible["ad_line"] = (adv - dec).where(eligible.gt(0)).loc[visible.index].cumsum()
    visible["ad_line_normalized"] = 100 + (((adv - dec) / eligible.where(eligible.gt(0))) * 100).loc[visible.index].cumsum()
    visible = visible.rename_axis("date").reset_index()
    return visible[ADVANCE_DECLINE_COLUMNS]


def _sector_rotation(tables: dict[str, pd.DataFrame], markets: pd.DataFrame,
                     calendar: pd.DatetimeIndex, end: pd.Timestamp) -> pd.DataFrame:
    hist = tables.get("sector_history", pd.DataFrame())
    if hist.empty or not {"date", "name", "close"}.issubset(hist.columns) or not len(calendar):
        return pd.DataFrame(columns=SECTOR_ROTATION_COLUMNS)
    snapshot = tables.get("index_snapshot", pd.DataFrame())
    allowed = None
    if not snapshot.empty and {"name", "group"}.issubset(snapshot.columns):
        allowed = set(snapshot.loc[snapshot.group.fillna("").astype(str).str.lower().eq("sectoral"), "name"].astype(str))
    d = _bounded(hist, end)
    d["name"] = d.name.fillna("").astype(str).str.strip()
    d["close"] = _numeric(d, "close")
    if allowed is not None:
        d = d.loc[d.name.isin(allowed)]
    if d.empty:
        return pd.DataFrame(columns=SECTOR_ROTATION_COLUMNS)
    d = d.drop_duplicates(["date", "name"], keep="last")
    nifty = _nifty_series(markets, calendar, end).close
    if nifty.notna().sum() == 0:
        return pd.DataFrame(columns=SECTOR_ROTATION_COLUMNS)
    visible_dates = calendar[-126:]
    rows = []
    for name, group in d.groupby("name"):
        level = group.set_index("date").close.reindex(calendar)
        relative = level / nifty
        momentum = (relative / relative.shift(21) - 1) * 100
        valid_momentum = relative.notna().rolling(22, min_periods=22).sum().eq(22)
        momentum = momentum.where(valid_momentum)
        recent = relative.reindex(visible_dates)
        if recent.empty:
            continue
        anchor = recent.iloc[0]
        rebased = recent / anchor * 100 if pd.notna(anchor) and anchor != 0 else recent * np.nan
        obs = recent.notna().cumsum()
        for dt in visible_dates:
            rows.append({"sector": name, "date": dt, "relative_strength_100": rebased.at[dt],
                         "relative_momentum_21_pct": momentum.at[dt], "sector_close": level.at[dt],
                         "nifty_close": nifty.at[dt], "relative_observations": int(obs.at[dt])})
    return pd.DataFrame(rows, columns=SECTOR_ROTATION_COLUMNS)


def _delivery_snapshot(prices: pd.DataFrame, returns: pd.DataFrame, members: pd.DataFrame,
                       delivery: pd.DataFrame, calendar: pd.DatetimeIndex,
                       start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if delivery.empty or not {"date", "symbol", "delivery_pct"}.issubset(delivery.columns) or not len(calendar):
        return pd.DataFrame(columns=PRICE_VOLUME_COLUMNS)
    d = _bounded(delivery, end)
    d["symbol"] = d.symbol.astype(str).str.strip().str.upper()
    d["delivery_pct"] = _numeric(d, "delivery_pct")
    d = d.loc[d.date.ge(start) & d.symbol.isin(set(members.symbol)) & d.delivery_pct.notna()]
    if d.empty:
        return pd.DataFrame(columns=PRICE_VOLUME_COLUMNS)
    d = d.groupby(["date", "symbol"], as_index=False).delivery_pct.mean()
    p = prices.loc[prices.symbol.isin(set(members.symbol))]
    joined = d.merge(p, on=["date", "symbol"], how="inner").sort_values(["symbol", "date"])
    if joined.empty:
        return pd.DataFrame(columns=PRICE_VOLUME_COLUMNS)
    joined = joined.groupby("symbol", as_index=False).tail(1)
    volume = p.pivot(index="date", columns="symbol", values="volume").reindex(index=calendar)
    raw = p.pivot(index="date", columns="symbol", values="raw_close").reindex(index=calendar)
    traded = raw * volume / 10_000_000
    counts = d.groupby("symbol").date.nunique()
    rows = []
    for _, row in joined.iterrows():
        symbol, date = row.symbol, row.date
        pos = calendar.get_indexer([date])[0]
        ratio = np.nan
        if pos >= 20 and symbol in volume:
            baseline = volume[symbol].iloc[pos - 20:pos]
            current = volume.at[date, symbol]
            if len(baseline) == 20 and baseline.notna().all() and pd.notna(current) and baseline.mean() != 0:
                ratio = float(current / baseline.mean())
        daily = returns.at[date, symbol] if date in returns.index and symbol in returns else np.nan
        turnover = traded.at[date, symbol] if date in traded.index and symbol in traded else np.nan
        rows.append({"symbol": symbol, "date": date, "return_pct": float(daily * 100) if pd.notna(daily) else np.nan,
                     "volume_ratio_20d": ratio, "delivery_pct": float(row.delivery_pct),
                     "traded_value_cr": float(turnover) if pd.notna(turnover) else np.nan,
                     "delivery_observations": int(counts.get(symbol, 0))})
    return pd.DataFrame(rows, columns=PRICE_VOLUME_COLUMNS).sort_values("symbol").reset_index(drop=True)


def _institution_absorption(flows: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if flows.empty or not {"date", "category"}.issubset(flows.columns):
        return pd.DataFrame(columns=INSTITUTION_COLUMNS)
    d = _bounded(flows, end)
    d = d.loc[d.date.ge(start)].copy()
    d["category_text"] = d.category.fillna("").astype(str).str.lower()
    d["net"] = _numeric(d, "net_cr")
    if "buy_cr" in d and "sell_cr" in d:
        d["net"] = d.net.fillna(_numeric(d, "buy_cr") - _numeric(d, "sell_cr"))
    d["kind"] = np.select([d.category_text.str.contains(r"fii|fpi|foreign", regex=True),
                           d.category_text.str.contains(r"dii|domestic", regex=True)], ["fii", "dii"], default="")
    d = d.loc[d.kind.ne("") & d.net.notna()]
    if d.empty:
        return pd.DataFrame(columns=INSTITUTION_COLUMNS)
    pivot = d.groupby(["date", "kind"]).net.sum().unstack("kind")
    for kind in ("fii", "dii"):
        if kind not in pivot:
            pivot[kind] = np.nan
    combined = pivot.fii + pivot.dii
    absorption = (pivot.dii / pivot.fii.abs()).where(pivot.fii.lt(0))
    return pd.DataFrame({"date": pivot.index, "fii_net_cr": pivot.fii.to_numpy(), "dii_net_cr": pivot.dii.to_numpy(),
                         "combined_net_cr": combined.to_numpy(), "domestic_absorption_ratio": absorption.to_numpy()},
                        columns=INSTITUTION_COLUMNS).reset_index(drop=True)


def _factor_spreads(markets: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    if markets.empty or not {"date", "name", "close"}.issubset(markets.columns) or not len(calendar):
        return pd.DataFrame(columns=SPREAD_COLUMNS)
    d = _bounded(markets, calendar[-1])
    aliases = {"gold": "gold", "silver": "silver", "copper": "copper", "brent": "brent",
               "brent crude": "brent", "wti": "wti", "wti crude": "wti", "usd/inr": "usd_inr",
               "dollar index": "dxy", "us 10y yield": "us10y", "us 10-year treasury yield": "us10y"}
    d["factor"] = d.name.fillna("").astype(str).str.strip().str.lower().map(aliases)
    d = d.loc[d.factor.notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=SPREAD_COLUMNS)
    d["close"] = _numeric(d, "close")
    panel = d.pivot_table(index="date", columns="factor", values="close", aggfunc="last").reindex(index=calendar[-126:])
    series = []
    if {"gold", "silver"}.issubset(panel.columns):
        series.append(("Gold / silver ratio", panel.gold / panel.silver, "ratio (USD/oz divided by USD/oz)"))
    if {"copper", "gold"}.issubset(panel.columns):
        series.append(("Copper / gold price ratio", panel.copper / panel.gold, "mixed-unit ratio (USD/lb divided by USD/troy oz)"))
    if {"brent", "wti"}.issubset(panel.columns):
        series.append(("Brent - WTI spread", panel.brent - panel.wti, "USD/barrel"))
    for factor, label, unit in (("usd_inr", "USD/INR", "INR per USD"),
                                ("dxy", "Dollar index", "index points"),
                                ("us10y", "US 10-year Treasury yield", "percent")):
        if factor in panel:
            series.append((label, panel[factor], unit))
    rows = []
    for name, values, unit in series:
        rows.extend({"date": dt, "spread_name": name, "value": float(value), "unit": unit}
                    for dt, value in values.dropna().items())
    return pd.DataFrame(rows, columns=SPREAD_COLUMNS)


def _regime(nifty: pd.DataFrame, breadth: pd.DataFrame, rotation: pd.DataFrame) -> pd.DataFrame:
    if nifty.empty:
        return pd.DataFrame(columns=REGIME_COLUMNS)
    close = pd.to_numeric(nifty.close, errors="coerce")
    valid = close.dropna()
    if valid.empty:
        return pd.DataFrame(columns=REGIME_COLUMNS)
    as_of = valid.index[-1]
    ret = close.pct_change(fill_method=None)
    tail20, tail63 = ret.tail(20), ret.tail(63)
    vol20 = float(tail20.std(ddof=1) * np.sqrt(252) * 100) if len(tail20) == 20 and tail20.notna().all() else np.nan
    vol63 = float(tail63.std(ddof=1) * np.sqrt(252) * 100) if len(tail63) == 63 and tail63.notna().all() else np.nan
    trailing_close = close.tail(63)
    trend = float((valid.iloc[-1] / trailing_close.mean() - 1) * 100) if len(trailing_close) == 63 and trailing_close.notna().all() else np.nan
    breadth_row = breadth.loc[pd.to_datetime(breadth.date, errors="coerce").dt.normalize().eq(as_of)].tail(1) if not breadth.empty else pd.DataFrame()
    above200 = breadth_row.above_200ma_pct.iloc[0] if not breadth_row.empty else np.nan
    adv_share = breadth_row.breadth_thrust_10s_pct.iloc[0] if not breadth_row.empty else np.nan
    latest_sectors = rotation.loc[rotation.date.eq(rotation.date.max())] if not rotation.empty else pd.DataFrame()
    sector_median = latest_sectors.relative_momentum_21_pct.median() if not latest_sectors.empty else np.nan
    rows = [
        {"metric": "Nifty 50 volatility (20-session)", "value": vol20, "unit": "% annualized", "as_of": as_of,
         "definition": "Sample standard deviation of 20 consecutive daily adjusted-close returns multiplied by square root of 252."},
        {"metric": "Nifty 50 volatility (63-session)", "value": vol63, "unit": "% annualized", "as_of": as_of,
         "definition": "Sample standard deviation of 63 consecutive daily adjusted-close returns multiplied by square root of 252."},
        {"metric": "Nifty 50 vs 63-session mean", "value": trend, "unit": "%", "as_of": as_of,
         "definition": "Latest close relative to the mean of the latest 63 consecutive closes."},
        {"metric": "Stocks above 200-session mean", "value": above200, "unit": "% eligible stocks", "as_of": as_of,
         "definition": "Current configured universe with an observed close and a complete prior 200-session close window."},
        {"metric": "10-session mean advancer share", "value": adv_share, "unit": "% eligible stocks", "as_of": as_of,
         "definition": "Ten-session mean of daily advancers divided by eligible stocks; not the Zweig breadth-thrust rule."},
        {"metric": "Median sector relative momentum", "value": sector_median, "unit": "% over 21 sessions", "as_of": as_of,
         "definition": "Median sector index momentum relative to Nifty 50."},
    ]
    if pd.notna(trend) and pd.notna(above200):
        label = ("Positive trend with broad participation" if trend > 0 and above200 >= 60 else
                 "Positive trend with mixed participation" if trend > 0 else
                 "Negative trend with broad participation" if above200 >= 60 else "Negative trend with mixed participation")
    else:
        label = "Insufficient history for a descriptive classification"
    rows.append({"metric": "Descriptive market-state label", "value": label, "unit": "classification", "as_of": as_of,
                 "definition": "Summarizes current trend and breadth thresholds; it is not a forecast or trading recommendation."})
    return pd.DataFrame(rows, columns=REGIME_COLUMNS)


def analyze_market_state(tables: dict[str, pd.DataFrame], analysis: dict[str, pd.DataFrame], config: Any) -> dict[str, pd.DataFrame]:
    """Calculate extra market-state measurements from observed histories."""
    del analysis  # Use source observations directly so date and coverage rules remain explicit.
    output = _empty_outputs()
    start, end = pd.Timestamp(config.start).normalize(), pd.Timestamp(config.end).normalize()
    prices = _prepare_prices(tables.get("prices", pd.DataFrame()), end)
    members = _prepare_members(tables.get("members", pd.DataFrame()))
    markets = _bounded(tables.get("markets", pd.DataFrame()), end)
    if not markets.empty:
        markets["name"] = markets.name.fillna("").astype(str)
        for col in ("close", "raw_close", "high", "low", "open"):
            markets[col] = _numeric(markets, col)
    # Union of stock sessions preserves a NaN slot when a symbol or market bar
    # is absent on a session. Nifty dates are only a fallback without prices.
    dates = (sorted(prices.date.unique()) if not prices.empty else
             sorted(markets.date.unique()) if not markets.empty and "date" in markets else [])
    calendar = pd.DatetimeIndex([d for d in dates if d <= end])
    risk, distribution, returns = _stock_risk(prices, members, calendar, start, end)
    output["stock_risk_return"] = risk
    output["stock_return_distribution"] = distribution
    output["sector_rotation"] = _sector_rotation(tables, markets, calendar, end)
    output["advance_decline"] = _breadth(prices, members, calendar, start, end)
    output["stock_price_volume_delivery"] = _delivery_snapshot(prices, returns, members,
                                                                tables.get("delivery", pd.DataFrame()), calendar, start, end)
    output["nifty_market_state"], nifty = _nifty_state(markets, tables.get("nifty_ohlc", pd.DataFrame()), calendar, start, end)
    output["institution_absorption"] = _institution_absorption(tables.get("flows", pd.DataFrame()), start, end)
    output["factor_spreads"] = _factor_spreads(markets, calendar)
    output["regime_snapshot"] = _regime(nifty, output["advance_decline"], output["sector_rotation"])
    flow = output["institution_absorption"]
    flow_sessions = int(flow.date.nunique()) if not flow.empty else np.nan
    flow_as_of = flow.date.max() if not flow.empty else pd.NaT
    flow_coverage = {
        "metric": "Institutional flow source coverage", "value": flow_sessions, "unit": "observed sessions",
        "as_of": flow_as_of,
        "definition": "Short daily FII/DII cash-flow history only; not a long-term institutional trend. Missing source sessions are not filled.",
    }
    output["regime_snapshot"] = pd.DataFrame(output["regime_snapshot"].to_dict("records") + [flow_coverage], columns=REGIME_COLUMNS)
    return output
