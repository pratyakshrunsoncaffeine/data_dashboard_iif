"""Return and macro-correlation analysis for mapped niche equity themes.

Theme membership is supplied by the explicit ``config/niche_themes.csv``
mapping. This module does not infer constituents from industry labels.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


MEMBERSHIP_COLUMNS = ["theme", "symbol", "name", "sector", "source", "classification_note", "in_current_universe"]
COVERAGE_COLUMNS = ["theme", "date", "selected_members", "eligible_members", "coverage_pct", "minimum_members", "minimum_coverage_pct", "available", "reason"]
RETURNS_COLUMNS = ["theme", "date", "return_pct", "index_value", "index_segment", "eligible_members", "selected_members", "coverage_pct", "available"]
PERFORMANCE_COLUMNS = ["theme", "selected_members", "latest_date", "return_1d_pct", "return_5d_pct", "return_21d_pct", "return_63d_pct", "return_126d_pct", "report_return_pct", "report_sessions", "distance_from_252d_high_pct", "valid_252d_returns", "availability", "reason"]
ROLLING_COLUMNS = ["date", "theme", "macro_name", "macro_symbol", "window_sessions", "rho", "joint_observations", "rho_previous_nonoverlap", "rho_change"]
LATEST_COLUMNS = ["theme", "macro_name", "macro_symbol", "window_sessions", "rho_latest", "rho_previous_nonoverlap", "rho_change", "joint_observations", "last_date", "p_value", "fdr_q_value", "interpretation"]
METHOD_COLUMNS = ["method", "details"]
WEEKLY_COLUMNS = ["date", "theme", "return_pct", "eligible_sessions", "expected_sessions", "available"]
WEEKLY_MACRO_COLUMNS = ["theme", "macro_name", "macro_symbol", "window_weeks", "rho_macro_lag1", "joint_weeks", "last_week", "interpretation"]

MIN_MEMBERS = 3
MIN_COVERAGE = .80
DAILY_WINDOWS = {60: 45, 126: 90}
MACRO_NAMES = {
    "brent", "brent crude", "wti", "wti crude", "gold", "copper", "silver",
    "usd/inr", "dollar index", "us 10y yield", "us 10-year treasury yield",
    "s&p 500", "sp500", "nifty 50", "nifty50", "nasdaq", "dax", "ftse 100",
    "nikkei 225", "hang seng", "shanghai composite", "dow jones",
}


def _empty_outputs() -> dict[str, pd.DataFrame]:
    return {
        "niche_membership": pd.DataFrame(columns=MEMBERSHIP_COLUMNS),
        "theme_coverage": pd.DataFrame(columns=COVERAGE_COLUMNS),
        "niche_returns": pd.DataFrame(columns=RETURNS_COLUMNS),
        "niche_performance": pd.DataFrame(columns=PERFORMANCE_COLUMNS),
        "niche_macro_latest": pd.DataFrame(columns=LATEST_COLUMNS),
        "niche_macro_rolling": pd.DataFrame(columns=ROLLING_COLUMNS),
        "niche_macro_method": pd.DataFrame(columns=METHOD_COLUMNS),
        "niche_weekly_returns": pd.DataFrame(columns=WEEKLY_COLUMNS),
        "niche_macro_weekly": pd.DataFrame(columns=WEEKLY_MACRO_COLUMNS),
    }


def _resolve_mapping(tables: dict[str, pd.DataFrame], config: Any) -> pd.DataFrame:
    mapping = tables.get("niche_themes", tables.get("theme_membership"))
    if isinstance(mapping, pd.DataFrame) and not mapping.empty:
        frame = mapping.copy()
    else:
        configured = getattr(config, "niche_themes_path", None)
        path = Path(configured) if configured else Path(__file__).resolve().parents[1] / "config" / "niche_themes.csv"
        if not path.exists():
            return pd.DataFrame(columns=["theme", "symbol", "source", "classification_note"])
        frame = pd.read_csv(path)
    frame.columns = [str(c).strip().lower() for c in frame.columns]
    required = {"theme", "symbol", "source", "classification_note"}
    if not required.issubset(frame.columns):
        return pd.DataFrame(columns=sorted(required))
    frame = frame.copy()
    for col in required:
        frame[col] = frame[col].fillna("").astype(str).str.strip()
    frame["symbol"] = frame.symbol.str.upper()
    # Keep explicit blank-symbol rows so unsupported pasted theme labels still
    # appear as unavailable themes instead of disappearing from coverage.
    frame = frame.loc[frame.theme.ne("")]
    return frame.drop_duplicates(["theme", "symbol"], keep="last").reset_index(drop=True)


def _prepare_prices(prices: pd.DataFrame, end: pd.Timestamp) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    required = {"date", "symbol", "close"}
    if prices.empty or not required.issubset(prices.columns):
        return pd.DataFrame(columns=["date", "symbol", "close"]), []
    data = prices[["date", "symbol", "close"]].copy()
    data["date"] = pd.to_datetime(data.date, errors="coerce").dt.normalize()
    data["symbol"] = data.symbol.astype(str).str.strip().str.upper()
    data["close"] = pd.to_numeric(data.close, errors="coerce")
    data = data.dropna(subset=["date"])
    data = data.loc[data.date.le(end)].sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")
    return data, sorted(pd.to_datetime(data.date.unique()))


def _prepare_members(members: pd.DataFrame) -> pd.DataFrame:
    if members.empty or "symbol" not in members:
        return pd.DataFrame(columns=["symbol", "name", "sector"])
    out = members.copy()
    for col in ("name", "sector"):
        if col not in out:
            out[col] = ""
    out["symbol"] = out.symbol.astype(str).str.strip().str.upper()
    return out.drop_duplicates("symbol", keep="last")[["symbol", "name", "sector"]]


def _membership(mapping: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    if mapping.empty:
        return pd.DataFrame(columns=MEMBERSHIP_COLUMNS)
    universe_symbols = set(members.symbol.astype(str))
    details = mapping.merge(members, on="symbol", how="left", suffixes=("", "_member"))
    details["in_current_universe"] = details.symbol.isin(universe_symbols)
    for col in ("name", "sector"):
        member_col = f"{col}_member"
        if member_col in details:
            details[col] = details[member_col].combine_first(details[col])
    for col in MEMBERSHIP_COLUMNS:
        if col not in details:
            details[col] = np.nan
    return details[MEMBERSHIP_COLUMNS].sort_values(["theme", "symbol"]).reset_index(drop=True)


def _stock_returns(prices: pd.DataFrame, calendar: list[pd.Timestamp]) -> pd.DataFrame:
    if prices.empty or not calendar:
        return pd.DataFrame()
    levels = prices.pivot(index="date", columns="symbol", values="close").reindex(index=calendar)
    return levels.pct_change(fill_method=None)


def _theme_daily(mapping: pd.DataFrame, membership: pd.DataFrame, returns: pd.DataFrame,
                 calendar: list[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series], dict[str, int]]:
    coverage_rows, return_rows, series_by_theme, selected_counts = [], [], {}, {}
    themes = sorted(mapping.theme.unique()) if not mapping.empty else []
    for theme in themes:
        symbols = membership.loc[(membership.theme == theme) & membership.in_current_universe.astype(bool), "symbol"].astype(str).tolist()
        selected = len(symbols)
        selected_counts[theme] = selected
        values = []
        theme_coverage = []
        for date in calendar:
            if selected:
                available_symbols = [symbol for symbol in symbols if symbol in returns.columns]
                observed = returns.loc[date, available_symbols].dropna() if available_symbols else pd.Series(dtype=float)
                eligible_count = int(len(observed))
                coverage = eligible_count / selected * 100
            else:
                observed, eligible_count, coverage = pd.Series(dtype=float), 0, 0.0
            enough = selected >= MIN_MEMBERS and eligible_count >= MIN_MEMBERS and coverage >= MIN_COVERAGE * 100
            reason = "available"
            if selected < MIN_MEMBERS:
                reason = "fewer than three mapped current-universe members"
            elif eligible_count < MIN_MEMBERS:
                reason = "fewer than three members with an observed same-session return"
            elif coverage < MIN_COVERAGE * 100:
                reason = "observed member coverage below 80%"
            daily_return = float(observed.mean() * 100) if enough else np.nan
            values.append(daily_return)
            coverage_row = {"theme": theme, "date": date, "selected_members": selected,
                            "eligible_members": eligible_count, "coverage_pct": coverage,
                            "minimum_members": MIN_MEMBERS, "minimum_coverage_pct": MIN_COVERAGE * 100,
                            "available": enough, "reason": reason}
            coverage_rows.append(coverage_row)
            theme_coverage.append(coverage_row)
        series_by_theme[theme] = pd.Series(values, index=pd.DatetimeIndex(calendar), dtype=float, name=theme)
        index_value = 100.0
        initialized = False
        index_segment = 0
        had_valid_segment = False
        for position, (date, daily_return) in enumerate(series_by_theme[theme].items()):
            if pd.notna(daily_return):
                if not initialized:
                    if had_valid_segment:
                        index_segment += 1
                    index_value = 100.0
                index_value *= 1 + daily_return / 100
                initialized = True
                had_valid_segment = True
                level = index_value
            else:
                initialized = False
                level = np.nan
            row = theme_coverage[position]
            return_rows.append({"theme": theme, "date": date, "return_pct": daily_return,
                                "index_value": level if pd.notna(daily_return) else np.nan,
                                "index_segment": index_segment if pd.notna(daily_return) else np.nan,
                                "eligible_members": row["eligible_members"], "selected_members": selected,
                                "coverage_pct": row["coverage_pct"], "available": row["available"]})
    coverage = pd.DataFrame(coverage_rows, columns=COVERAGE_COLUMNS)
    returns_out = pd.DataFrame(return_rows, columns=RETURNS_COLUMNS)
    return coverage, returns_out, series_by_theme, selected_counts


def _macro_returns(markets: pd.DataFrame, end: pd.Timestamp, calendar: list[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_cols = ["symbol", "name", "group", "unit"]
    if markets.empty or not {"date", "symbol", "close", "name"}.issubset(markets.columns):
        return pd.DataFrame(), pd.DataFrame(columns=meta_cols + ["change_unit", "interpretation"])
    d = markets.copy()
    d["date"] = pd.to_datetime(d.date, errors="coerce").dt.normalize()
    d["symbol"] = d.symbol.astype(str).str.strip().str.upper()
    d["close"] = pd.to_numeric(d.close, errors="coerce")
    d = d.dropna(subset=["date"]).loc[lambda x: x.date.le(end)].sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")
    names = d[[c for c in meta_cols if c in d]].drop_duplicates("symbol", keep="last")
    for col in meta_cols:
        if col not in names:
            names[col] = ""
    names["_match"] = names.name.fillna("").str.strip().str.lower()
    names = names.loc[names._match.isin(MACRO_NAMES)].copy()
    if names.empty:
        return pd.DataFrame(), pd.DataFrame(columns=meta_cols + ["change_unit", "interpretation"])
    names["change_unit"] = np.where(names.group.fillna("").astype(str).str.lower().eq("yield") |
                                     names.name.str.contains("yield", case=False, na=False), "percentage points", "percent")
    names["interpretation"] = np.where(names.name.str.strip().str.lower().eq("usd/inr"),
                                       "Positive change indicates INR depreciation", "Descriptive price return")
    series = {}
    market_calendar = pd.DatetimeIndex(calendar)
    for symbol, group in d.loc[d.symbol.isin(names.symbol)].groupby("symbol"):
        # Reindex to the equity-session calendar before transforming so an
        # absent market bar cannot be disguised as a one-session return.
        group = group.set_index("date").close.sort_index().reindex(market_calendar)
        info = names.loc[names.symbol == symbol].iloc[0]
        if info.change_unit == "percentage points":
            change = group.diff()
        else:
            change = group.pct_change(fill_method=None) * 100
        series[symbol] = change
    values = pd.DataFrame(series).reindex(index=market_calendar) if calendar else pd.DataFrame(series)
    return values, names[meta_cols + ["change_unit", "interpretation"]].reset_index(drop=True)


def _bh(pvalues: list[float]) -> list[float]:
    if not pvalues:
        return []
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(len(p)); out[order] = np.clip(q, 0, 1)
    return out.tolist()


def _macro_correlations(theme_returns: dict[str, pd.Series], macro_returns: pd.DataFrame,
                        macro_meta: pd.DataFrame, calendar: list[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rolling_frames, latest_candidates = [], []
    if not theme_returns or macro_returns.empty or not calendar:
        return (pd.DataFrame(columns=ROLLING_COLUMNS), pd.DataFrame(columns=LATEST_COLUMNS),
                pd.DataFrame(columns=WEEKLY_MACRO_COLUMNS))
    market_calendar = pd.DatetimeIndex(calendar)
    for theme, theme_series in theme_returns.items():
        for _, meta in macro_meta.iterrows():
            symbol = str(meta.symbol)
            if symbol not in macro_returns:
                continue
            factor = macro_returns[symbol].reindex(market_calendar)
            aligned_theme = theme_series.reindex(market_calendar)
            for window, minimum in DAILY_WINDOWS.items():
                pair_mask = aligned_theme.notna() & factor.notna()
                pair_count = pair_mask.astype(int).rolling(window, min_periods=1).sum()
                rho = aligned_theme.rolling(window, min_periods=minimum).corr(factor)
                # Pandas rolling correlation uses only complete aligned pairs.
                rho = rho.where(pair_count >= minimum)
                prior_rho = rho.shift(window)
                rho_change = rho - prior_rho
                rolling_frames.append(pd.DataFrame({
                    "date": market_calendar, "theme": theme, "macro_name": meta["name"],
                    "macro_symbol": symbol, "window_sessions": window, "rho": rho.to_numpy(),
                    "joint_observations": pair_count.to_numpy(dtype=int),
                    "rho_previous_nonoverlap": prior_rho.to_numpy(), "rho_change": rho_change.to_numpy(),
                }))

                # Always report the exact final window, including when it is
                # unavailable; never silently substitute an older estimate.
                if len(market_calendar):
                    n = int(pair_count.iloc[-1])
                    p_value = np.nan
                    latest_rho = rho.iloc[-1]
                    if pd.notna(latest_rho):
                        latest_pair = pd.concat([aligned_theme.tail(window).rename("theme"),
                                                 factor.tail(window).rename("macro")], axis=1).dropna()
                        if len(latest_pair) >= minimum and latest_pair.theme.nunique() > 1 and latest_pair.macro.nunique() > 1:
                            p_value = float(pearsonr(latest_pair.theme, latest_pair.macro).pvalue)
                    latest_candidates.append({"date": market_calendar[-1], "theme": theme, "macro_name": meta["name"],
                                              "macro_symbol": symbol, "window_sessions": window,
                                              "rho": latest_rho, "joint_observations": n,
                                              "rho_previous_nonoverlap": prior_rho.iloc[-1],
                                              "rho_change": rho_change.iloc[-1], "p_value": p_value,
                                              "interpretation": "Descriptive same-date correlation; asynchronous exchange calendars do not establish causality." if pd.notna(latest_rho)
                                              else f"Final {window}-session window unavailable: fewer than {minimum} jointly observed returns or zero variance."})
    rolling = pd.concat(rolling_frames, ignore_index=True) if rolling_frames else pd.DataFrame(columns=ROLLING_COLUMNS)
    latest = pd.DataFrame(latest_candidates)
    if not latest.empty:
        latest["fdr_q_value"] = np.nan
        for window, rows in latest.groupby("window_sessions"):
            positions = rows.index[rows.p_value.notna()].tolist()
            adjusted = _bh(latest.loc[positions, "p_value"].astype(float).tolist())
            latest.loc[positions, "fdr_q_value"] = adjusted
        latest = latest.rename(columns={"rho": "rho_latest", "joint_observations": "joint_observations", "date": "last_date"})
        latest = latest.reindex(columns=LATEST_COLUMNS)
    if not rolling.empty:
        rolling = rolling.reindex(columns=ROLLING_COLUMNS)
    weekly = _weekly_lagged(theme_returns, macro_returns, macro_meta, market_calendar)
    return rolling, latest, weekly


def _weekly_lagged(theme_returns: dict[str, pd.Series], macro_returns: pd.DataFrame,
                   macro_meta: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Friday compounded returns; macro factor is lagged one paired week."""
    rows = []
    if not len(calendar):
        return pd.DataFrame(columns=WEEKLY_MACRO_COLUMNS)
    week_codes = pd.Series(calendar.to_period("W-FRI"), index=calendar)
    unique_weeks = week_codes.drop_duplicates().tolist()
    week_dates = pd.Series(calendar, index=calendar).groupby(week_codes, sort=True).last()

    # Aggregate each theme and factor once. Pair-specific work below is limited
    # to weekly completeness masking and correlations, avoiding a daily concat
    # inside a theme × factor × week loop.
    theme_weekly: dict[str, np.ndarray] = {}
    theme_week_complete: dict[str, np.ndarray] = {}
    for theme, daily in theme_returns.items():
        values = daily.reindex(calendar)
        complete = values.notna().groupby(week_codes, sort=True).all()
        compounded = (1 + values / 100).groupby(week_codes, sort=True).prod().sub(1).mul(100)
        theme_weekly[theme] = compounded.reindex(unique_weeks).to_numpy(dtype=float)
        theme_week_complete[theme] = complete.reindex(unique_weeks).to_numpy(dtype=bool)

    macro_weekly: dict[str, np.ndarray] = {}
    macro_week_complete: dict[str, np.ndarray] = {}
    macro_is_yield: dict[str, bool] = {}
    for _, meta in macro_meta.iterrows():
        symbol = str(meta.symbol)
        if symbol not in macro_returns:
            continue
        values = macro_returns[symbol].reindex(calendar)
        complete = values.notna().groupby(week_codes, sort=True).all()
        is_yield = str(meta.get("change_unit", "")).lower() == "percentage points"
        aggregated = (values.groupby(week_codes, sort=True).sum() if is_yield else
                      (1 + values / 100).groupby(week_codes, sort=True).prod().sub(1).mul(100))
        macro_weekly[symbol] = aggregated.reindex(unique_weeks).to_numpy(dtype=float)
        macro_week_complete[symbol] = complete.reindex(unique_weeks).to_numpy(dtype=bool)
        macro_is_yield[symbol] = is_yield

    for theme, theme_daily in theme_returns.items():
        for _, meta in macro_meta.iterrows():
            symbol = str(meta.symbol)
            if symbol not in macro_weekly:
                continue
            # Each series' week is complete only when all its daily observations
            # exist. Their conjunction is the same paired-week completeness
            # check, cached at series level to keep the cross-product cheap.
            joint_complete = theme_week_complete[theme] & macro_week_complete[symbol]
            theme_values = np.where(joint_complete, theme_weekly[theme], np.nan)
            macro_values = np.where(joint_complete, macro_weekly[symbol], np.nan)
            macro_lag1 = np.roll(macro_values, 1)
            macro_lag1[0] = np.nan  # full week grid: never skip an incomplete week
            last_week = week_dates.reindex(unique_weeks).iloc[-1]
            for window, minimum in ((26, 20), (52, 40)):
                x, y = theme_values[-window:], macro_lag1[-window:]
                valid = np.isfinite(x) & np.isfinite(y)
                n_valid = int(valid.sum())
                rho = float(np.corrcoef(x[valid], y[valid])[0, 1]) if n_valid >= minimum and np.ptp(x[valid]) > 0 and np.ptp(y[valid]) > 0 else np.nan
                rows.append({"theme": theme, "macro_name": meta["name"], "macro_symbol": symbol,
                             "window_weeks": window, "rho_macro_lag1": rho, "joint_weeks": n_valid,
                             "last_week": last_week,
                             "interpretation": ("Descriptive correlation of this week's theme return with the prior calendar week's macro move; yield changes are summed in percentage points. Asynchronous sessions do not establish causality." if macro_is_yield[symbol] else "Descriptive correlation of this week's theme return with the prior calendar week's macro move; asynchronous sessions do not establish causality.")})
    return pd.DataFrame(rows, columns=WEEKLY_MACRO_COLUMNS)


def _performance(theme_returns: dict[str, pd.Series], coverage: pd.DataFrame,
                 selected_counts: dict[str, int], start: pd.Timestamp, end: pd.Timestamp,
                 calendar: list[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for theme, series in theme_returns.items():
        series = series.loc[series.index <= end]
        def compound(n):
            vals = series.tail(n)
            return float(((1 + vals / 100).prod() - 1) * 100) if len(vals) == n and vals.notna().all() else np.nan
        report = series.loc[series.index.to_series().between(start, end)]
        report_return = float(((1 + report / 100).prod() - 1) * 100) if not report.empty and report.notna().all() else np.nan
        last_252 = series.tail(252)
        distance, valid_252 = np.nan, int(last_252.notna().sum())
        if len(last_252) == 252 and last_252.notna().all():
            levels = 100 * (1 + last_252 / 100).cumprod()
            distance = float((levels.iloc[-1] / levels.max() - 1) * 100)
        current = coverage.loc[(coverage.theme == theme) & (coverage.date <= end)].tail(1)
        availability = "available" if selected_counts.get(theme, 0) >= MIN_MEMBERS else "unavailable"
        reason = "" if availability == "available" else "fewer than three mapped current-universe members"
        rows.append({"theme": theme, "selected_members": selected_counts.get(theme, 0), "latest_date": series.index.max() if len(series) else pd.NaT,
                     "return_1d_pct": compound(1), "return_5d_pct": compound(5), "return_21d_pct": compound(21),
                     "return_63d_pct": compound(63), "return_126d_pct": compound(126), "report_return_pct": report_return,
                     "report_sessions": int(len(report)), "distance_from_252d_high_pct": distance,
                     "valid_252d_returns": valid_252, "availability": availability, "reason": reason})
    return pd.DataFrame(rows, columns=PERFORMANCE_COLUMNS)


def analyze_themes(tables: dict[str, pd.DataFrame], config: Any) -> dict[str, pd.DataFrame]:
    """Analyze mapped current-universe themes and their observed market factors."""
    output = _empty_outputs()
    end = pd.Timestamp(config.end).normalize()
    start = pd.Timestamp(config.start).normalize()
    mapping = _resolve_mapping(tables, config)
    members = _prepare_members(tables.get("members", pd.DataFrame()))
    prices, calendar = _prepare_prices(tables.get("prices", pd.DataFrame()), end)
    membership = _membership(mapping, members)
    output["niche_membership"] = membership
    if mapping.empty:
        output["niche_macro_method"] = pd.DataFrame([
            {"method": "theme membership", "details": "No valid theme,symbol,source,classification_note mapping was supplied or found at config/niche_themes.csv."},
            {"method": "constituent sourcing", "details": "Niche constituents are not inferred from broad industry labels."},
        ], columns=METHOD_COLUMNS)
        return output
    if prices.empty or not calendar:
        coverage, niche_returns, theme_returns, selected_counts = _theme_daily(mapping, membership, pd.DataFrame(), [])
        output["theme_coverage"] = coverage
        output["niche_returns"] = niche_returns
        output["niche_performance"] = _performance(theme_returns, coverage, selected_counts, start, end, calendar)
        output["niche_macro_method"] = pd.DataFrame([
            {"method": "price history", "details": "No usable adjusted close observations were supplied."},
            {"method": "constituent sourcing", "details": "Mapped members are intersected with current Nifty constituents only."},
        ], columns=METHOD_COLUMNS)
        return output

    stock_returns = _stock_returns(prices, calendar)
    coverage, niche_returns, theme_returns, selected_counts = _theme_daily(mapping, membership, stock_returns, calendar)
    output["theme_coverage"] = coverage
    output["niche_returns"] = niche_returns
    output["niche_performance"] = _performance(theme_returns, coverage, selected_counts, start, end, calendar)
    macro_returns, macro_meta = _macro_returns(tables.get("markets", pd.DataFrame()), end, calendar)
    rolling, latest, weekly_macro = _macro_correlations(theme_returns, macro_returns, macro_meta, calendar)
    output["niche_macro_rolling"] = rolling
    output["niche_macro_latest"] = latest
    output["niche_macro_weekly"] = weekly_macro
    output["niche_weekly_returns"] = _weekly_theme_returns(theme_returns, calendar)
    output["niche_macro_method"] = pd.DataFrame([
        {"method": "membership", "details": "Only explicit niche_themes mapping symbols currently present in members are selected; no web discovery or broad-industry inference."},
        {"method": "basket", "details": "Daily equal-weight arithmetic mean of observed constituent returns; requires at least 3 members and at least 80% coverage of selected current constituents; no price fill."},
        {"method": "factor returns", "details": "Market price factors use daily percent changes; yield factors use first differences in percentage points; positive USD/INR change means INR depreciation."},
        {"method": "daily correlations", "details": "Contemporaneous Pearson correlations on jointly observed same-date returns; 60/126 Indian sessions, minimum 45/90 joint observations; no forward fill."},
        {"method": "change and FDR", "details": "Correlation change compares the latest window with the immediately preceding non-overlapping equal-length window. Latest-pair p-values receive Benjamini-Hochberg FDR adjustment by window. Correlation is descriptive and does not establish causality."},
        {"method": "weekly", "details": "Complete same-session paired weeks are labelled by their last observed session. Price changes compound; yield changes sum in percentage points. Lagged correlation pairs this week’s theme return with the prior calendar week’s macro move."},
        {"method": "52-week distance", "details": "Reported only with 252 valid daily basket returns; missing windows remain unavailable."},
    ], columns=METHOD_COLUMNS)
    return output


def _weekly_theme_returns(theme_returns: dict[str, pd.Series], calendar: list[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    index = pd.DatetimeIndex(calendar)
    if not len(index):
        return pd.DataFrame(columns=WEEKLY_COLUMNS)
    weeks = index.to_period("W-FRI")
    for theme, series in theme_returns.items():
        for week in weeks.unique():
            dates = index[weeks == week]
            sample = series.reindex(dates)
            eligible = int(sample.notna().sum())
            available = bool(len(dates) and eligible == len(dates))
            ret = float(((1 + sample / 100).prod() - 1) * 100) if available else np.nan
            rows.append({"date": dates[-1], "theme": theme, "return_pct": ret,
                         "eligible_sessions": eligible, "expected_sessions": len(dates), "available": available})
    return pd.DataFrame(rows, columns=WEEKLY_COLUMNS)
