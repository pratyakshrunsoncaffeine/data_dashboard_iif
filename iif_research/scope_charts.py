"""Academic-style charts for global-economy, index and market-history data."""
from __future__ import annotations

import re
import textwrap
from collections.abc import Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BLUE = "#355C7D"
TEAL = "#4C9F9A"
RED = "#B75D5D"
GREY = "#667085"
PALETTE = [BLUE, TEAL, "#8296A8", RED, "#8BA59C", "#8D7A9E"]


def _frame(tables: Mapping, key: str) -> pd.DataFrame:
    value = (tables or {}).get(key)
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if value is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(value)
    except (TypeError, ValueError):
        return pd.DataFrame()


def _asof(frame: pd.DataFrame, end: str) -> pd.DataFrame:
    data = frame.copy()
    limit = pd.Timestamp(end).normalize()
    if "date" in data:
        data["date"] = pd.to_datetime(data.date, errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.normalize()
        data = data.loc[data.date.notna() & data.date.le(limit)]
    if "available_at" in data:
        available = pd.to_datetime(data.available_at, errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.normalize()
        data["available_at"] = available
        data = data.loc[available.isna() | available.le(limit)]
    return data


def _caption(ax, text: str) -> None:
    ax.text(0, -0.2, textwrap.fill(text, 125), transform=ax.transAxes, ha="left", va="top", color=GREY, fontsize=8)


def _figure(figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")
    return fig, ax


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")[:40] or "series"


def _period_label(row) -> str:
    period = row.get("period")
    if pd.notna(period) and str(period).strip():
        return str(period)
    date = row.get("date")
    return pd.Timestamp(date).strftime("%Y-%m-%d") if pd.notna(date) else "date unavailable"


def _country_bars(data: pd.DataFrame, matches: pd.Series, output_key: str, title: str, config) -> dict[str, plt.Figure]:
    subset = data.loc[matches].copy()
    if subset.empty:
        return {}
    subset["value"] = pd.to_numeric(subset.value, errors="coerce")
    subset = subset.dropna(subset=["value", "country"])
    if subset.empty:
        return {}
    sort_cols = [c for c in ("available_at", "date") if c in subset]
    if sort_cols:
        subset = subset.sort_values(sort_cols)
    latest = subset.groupby(["indicator", "unit", "country"], dropna=False, as_index=False).tail(1)
    groups = list(latest.groupby(["indicator", "unit"], dropna=False, sort=True))
    if not groups:
        return {}
    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(max(8, 5.0 * n), max(4.5, .38 * latest.country.nunique() + 2)), squeeze=False, facecolor="white")
    axes = axes[0]
    observed = []
    for ax, ((indicator, unit), group) in zip(axes, groups):
        group = group.sort_values("country")
        labels = [f"{row.country} · {_period_label(row._asdict())}" for row in group.itertuples(index=False)]
        y = np.arange(len(group))
        values = group.value.to_numpy(dtype=float)
        ax.barh(y, values, color=[TEAL if value >= 0 else RED for value in values])
        ax.axvline(0, color="#344054", lw=.7)
        ax.set_yticks(y, labels); ax.invert_yaxis()
        ax.set_xlabel(str(unit) if pd.notna(unit) else "Value")
        ax.set_title(str(indicator), loc="left", fontweight="bold", fontsize=10)
        ax.grid(axis="x", color="#E6E8EB", lw=.7)
        for yi, value in zip(y, values):
            ax.annotate(f"{value:,.2f}", (value, yi), xytext=(4 if value >= 0 else -4, 0), textcoords="offset points",
                        va="center", ha="left" if value >= 0 else "right", fontsize=7, color="#344054")
        left, right = ax.get_xlim()
        ax.set_xlim(left - (right - left) * .03, right + (right - left) * .12)
        if "date" in group and group.date.notna().any():
            observed.extend(group.date.dropna().tolist())
    axes[0].set_title(title, loc="left", fontweight="bold", fontsize=11)
    _caption(axes[0], f"Latest available country observations as of {config.end}; period labels and observation dates are shown per bar. "
             f"Observed dates: {min(observed).date() if observed else 'not supplied'} to {max(observed).date() if observed else 'not supplied'}. "
             "Source: global_economy; publication availability is bounded by report end where supplied.")
    fig.tight_layout(rect=(0, .14, 1, 1))
    return {output_key: fig}


def _macro_trend(data: pd.DataFrame, mask: pd.Series, key: str, title: str, config) -> dict[str, plt.Figure]:
    subset = data.loc[mask].copy()
    if subset.empty:
        return {}
    subset["value"] = pd.to_numeric(subset.value, errors="coerce")
    subset = subset.dropna(subset=["date", "value"])
    if subset.empty:
        return {}
    subset = subset.sort_values("date")
    fig, ax = _figure((10, 4.7))
    for i, ((indicator, unit), group) in enumerate(subset.groupby(["indicator", "unit"], dropna=False, sort=True)):
        group = group.sort_values("date").drop_duplicates("date", keep="last")
        ax.plot(group.date, group.value, color=PALETTE[i % len(PALETTE)], lw=1.6, marker="o", ms=2.8,
                label=f"{indicator} ({unit})" if pd.notna(unit) else str(indicator))
    ax.set_title(title, loc="left", fontweight="bold")
    units = subset.unit.dropna().astype(str).unique().tolist() if "unit" in subset else []
    ax.set_ylabel(units[0] if len(units) == 1 else "Value (units identified in legend)")
    ax.set_xlabel("Observation date"); ax.grid(axis="y", color="#E6E8EB", lw=.7)
    ax.legend(frameon=False, fontsize=8)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    _caption(ax, f"Requested end: {config.end}; observed: {subset.date.min().date()} to {subset.date.max().date()}. "
             "Observation frequency and units are preserved by series. Source: global_economy.")
    fig.tight_layout(rect=(0, .08, 1, 1))
    return {key: fig}


def _nse_index_heatmap(snapshot: pd.DataFrame, config) -> dict[str, plt.Figure]:
    required = {"date", "name"}
    measures = [c for c in ("change_pct", "month_return_pct", "year_return_pct") if c in snapshot]
    if snapshot.empty or not required.issubset(snapshot.columns) or not measures:
        return {}
    data = _asof(snapshot, config.end)
    for col in measures:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["date", "name"])
    if data.empty:
        return {}
    snapshot_date = data.date.max()
    data = data.loc[data.date.eq(snapshot_date)].sort_values("name").drop_duplicates("name", keep="last")
    data = data.set_index("name")[measures]
    data = data.dropna(how="all")
    if data.empty:
        return {}
    columns = {"change_pct": "1 day (%)", "month_return_pct": "1 month (%)", "year_return_pct": "1 year (%)"}
    data = data.rename(columns=columns)
    bound = float(np.nanpercentile(np.abs(data.to_numpy(dtype=float)), 95)) if data.notna().any().any() else 1
    bound = max(bound, .1)
    figures = {}
    for offset in range(0, len(data), 35):
        part = data.iloc[offset:offset + 35]
        fig, ax = _figure((10, max(5, .25 * len(part) + 2)))
        sns.heatmap(part, cmap=sns.diverging_palette(10, 175, s=72, l=48, as_cmap=True), center=0, vmin=-bound, vmax=bound,
                    annot=True, fmt=".1f", annot_kws={"size": 8}, linewidths=.3, linecolor="white",
                    cbar_kws={"label": "Index return (%)"}, ax=ax)
        ax.set_title(f"NSE all-index returns · snapshot {snapshot_date.date()} · {offset + 1} to {offset + len(part)}", loc="left", fontsize=11)
        ax.set_xlabel("Return horizon"); ax.set_ylabel("NSE index")
        ax.tick_params(axis="y", labelsize=8)
        _caption(ax, f"Snapshot dated {snapshot_date.date()}; selected end {config.end}. Returns in percent; missing entries remain blank. Source: NSE allIndices.")
        fig.subplots_adjust(left=.38, right=.95, bottom=.14, top=.94)
        key = "nse_index_return_heatmap" + (f"_{offset // 35 + 1}" if offset else "")
        figures[key] = fig
    return figures


def _nifty_history(tables: Mapping, config) -> pd.DataFrame:
    history = _frame(tables, "long_index_history")
    if history.empty or not {"date", "name", "close"}.issubset(history.columns):
        return pd.DataFrame()
    history = _asof(history, config.end)
    history["close"] = pd.to_numeric(history.close, errors="coerce")
    history = history.dropna(subset=["date", "close"])
    nifty = history.loc[history.name.astype(str).str.contains(r"Nifty\s*50", case=False, regex=True)].copy()
    return nifty.sort_values("date").drop_duplicates("date", keep="last")


def _nifty_drawdown(nifty: pd.DataFrame, config) -> dict[str, plt.Figure]:
    if nifty.empty:
        return {}
    data = nifty.copy()
    data["running_peak"] = data.close.cummax()
    data["drawdown_pct"] = (data.close / data.running_peak - 1) * 100
    fig, ax = _figure((11, 4.7))
    ax.fill_between(data.date, data.drawdown_pct, 0, color=RED, alpha=.2)
    ax.plot(data.date, data.drawdown_pct, color=RED, lw=1.35)
    ax.axhline(0, color="#344054", lw=.7)
    ax.set_title("Nifty 50 drawdown from running peak", loc="left", fontweight="bold")
    ax.set_ylabel("Drawdown (%)"); ax.set_xlabel("Date"); ax.grid(axis="y", color="#E6E8EB", lw=.7)
    _caption(ax, f"Requested through {config.end}; observed: {data.date.min().date()} to {data.date.max().date()}. Drawdown = close / prior-or-current running peak − 1. Source: long_index_history.")
    fig.tight_layout(rect=(0, .07, 1, 1))
    return {"nifty_drawdown": fig}


def _seasonality(nifty: pd.DataFrame, config) -> dict[str, plt.Figure]:
    if nifty.empty:
        return {}
    end = pd.Timestamp(config.end)
    years = list(range(end.year - 4, end.year + 1))
    data = nifty.loc[nifty.date.le(end)].copy().sort_values("date")
    fig, ax = _figure((10.5, 5))
    plotted = []
    for i, year in enumerate(years):
        start = pd.Timestamp(year=year, month=1, day=1)
        annual = data.loc[data.date.between(start, end if year == end.year else pd.Timestamp(year=year, month=12, day=31))].copy()
        anchor = data.loc[data.date.lt(start), "close"]
        if annual.empty or anchor.empty:
            continue
        base = float(anchor.iloc[-1])
        annual["indexed"] = annual.close / base * 100
        annual["month_day"] = pd.to_datetime("2000-" + annual.date.dt.strftime("%m-%d"), errors="coerce")
        annual = annual.dropna(subset=["month_day"])
        if annual.empty:
            continue
        ax.plot(annual.month_day, annual.indexed, color=PALETTE[i % len(PALETTE)], lw=1.6,
                label=f"{year} (n={len(annual)})")
        plotted.append((year, annual.date.max()))
    if not plotted:
        plt.close(fig)
        return {}
    ax.axhline(100, color="#98A2B3", lw=.8, ls="--")
    ax.set_title("Nifty 50 calendar-year seasonality", loc="left", fontweight="bold")
    ax.set_ylabel("Indexed close (prior year-end = 100)"); ax.set_xlabel("Calendar date")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(axis="y", color="#E6E8EB", lw=.7); ax.legend(frameon=False, ncol=2, fontsize=8)
    end_label = ", ".join(f"{year} through {last.date()}" for year, last in plotted)
    _caption(ax, f"Each year is rebased to its prior calendar year’s last available close. Current year stops at selected end {end.date()}. "
             f"Available series: {end_label}. Source: long_index_history.")
    fig.tight_layout(rect=(0, .08, 1, 1))
    return {"nifty_calendar_year_seasonality": fig}


def _breadth_by_size(tables: Mapping, config) -> dict[str, plt.Figure]:
    prices = _frame(tables, "prices")
    members = _frame(tables, "members_by_size")
    if prices.empty or members.empty or not {"date", "symbol", "close"}.issubset(prices.columns) or not {"symbol", "group"}.issubset(members.columns):
        return {}
    data = prices.copy()
    data["date"] = pd.to_datetime(data.date, errors="coerce")
    data["close"] = pd.to_numeric(data.close, errors="coerce")
    data["symbol"] = data.symbol.astype(str)
    data = data.dropna(subset=["date", "symbol", "close"]).loc[lambda x: x.date.le(pd.Timestamp(config.end))]
    members = members.dropna(subset=["symbol", "group"]).copy()
    members["symbol"] = members.symbol.astype(str)
    members["group"] = members.group.astype(str)
    data = data.loc[data.symbol.isin(members.symbol)].copy()
    groups = sorted(members.group.unique().tolist())
    if len(groups) < 2 or data.empty:
        return {}
    levels = data.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
    report_dates = levels.index[(levels.index >= pd.Timestamp(config.start)) & (levels.index <= pd.Timestamp(config.end))]
    if len(report_dates) == 0:
        return {}
    rows = []
    for group_name in groups:
        symbols = sorted(set(levels.columns).intersection(members.loc[members.group.eq(group_name), "symbol"]))
        if not symbols:
            continue
        cohort = levels[symbols]
        for window in (50, 200):
            prior_average = cohort.rolling(window, min_periods=window).mean().shift(1)
            eligible = cohort.notna() & prior_average.notna()
            above = cohort.gt(prior_average) & eligible
            pct = above.sum(axis=1).div(eligible.sum(axis=1).replace(0, np.nan)).mul(100)
            counts = eligible.sum(axis=1)
            for date in report_dates:
                if pd.notna(pct.get(date)):
                    rows.append({"date": date, "group": group_name, "window": window, "above_pct": pct.loc[date], "eligible_count": int(counts.loc[date])})
    result = pd.DataFrame(rows)
    if result.empty or result.group.nunique() < 2:
        return {}
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), facecolor="white", sharex=True)
    for ax, window in zip(axes, (50, 200)):
        part = result.loc[result.window.eq(window)]
        if part.empty:
            ax.text(.5, .5, f"{window}-session average unavailable in this history", transform=ax.transAxes, ha="center", color=GREY)
        else:
            for i, (name, series) in enumerate(part.groupby("group")):
                series = series.sort_values("date")
                ax.plot(series.date, series.above_pct, label=str(name), color=PALETTE[i % len(PALETTE)], lw=1.5)
            ax.legend(frameon=False, ncol=3, fontsize=8)
        ax.set_ylabel(f"Above {window}-day average (%)"); ax.set_ylim(0, 100); ax.grid(axis="y", color="#E6E8EB", lw=.7)
    axes[0].set_title("Observed breadth by size group", loc="left", fontweight="bold")
    axes[-1].set_xlabel("Date")
    _caption(axes[-1], f"Requested: {config.start} to {config.end}; breadth uses closes above the prior-session moving average. "
             "Denominators use observed members with valid price and average data and can vary by date. Source: prices and members_by_size.")
    fig.tight_layout(rect=(0, .07, 1, 1))
    return {"breadth_by_size": fig}


def build_scope_charts(tables: Mapping, analysis: Mapping | None, config) -> dict[str, plt.Figure]:
    """Create optional global economy, index and breadth charts from supplied tables."""
    figures: dict[str, plt.Figure] = {}
    economy = _asof(_frame(tables, "global_economy"), config.end)
    if not economy.empty and {"country", "indicator", "value", "unit"}.issubset(economy.columns):
        economy["country"] = economy.country.astype(str)
        economy["indicator"] = economy.indicator.astype(str)
        indicator = economy.indicator.str.lower()
        frequency = economy.get("frequency", pd.Series("", index=economy.index)).astype(str).str.lower()
        gdp_mask = indicator.str.contains(r"\bgdp\b|gross domestic", regex=True, na=False)
        cpi_mask = indicator.str.contains(r"\bcpi\b|consumer price", regex=True, na=False)
        annual_mask = frequency.str.contains(r"annual|yearly|annualized", regex=True, na=False) | indicator.str.contains("annual", na=False)
        figures.update(_country_bars(economy, gdp_mask & annual_mask, "global_gdp_latest", "Latest annual GDP growth by country", config))
        figures.update(_country_bars(economy, cpi_mask & annual_mask, "global_annual_cpi_latest", "Latest annual CPI observations by country", config))

        country = economy.get("country", pd.Series("", index=economy.index)).astype(str).str.lower().str.strip()
        is_us = country.isin({"us", "u.s.", "usa", "united states", "united states of america"})
        monthly = frequency.str.contains(r"monthly|month", regex=True, na=False)
        us_indicators = (
            (r"\bcpi\b|consumer price", "us_monthly_cpi", "United States monthly CPI"),
            (r"unemployment|jobless rate", "us_monthly_unemployment", "United States monthly unemployment"),
            (r"policy rate|fed funds|federal funds", "us_monthly_policy_rate", "United States monthly policy rate"),
        )
        for pattern, key, title in us_indicators:
            mask = is_us & monthly & indicator.str.contains(pattern, case=False, regex=True, na=False)
            figures.update(_macro_trend(economy, mask, key, title, config))
        for pattern, key, title, frequency_mask in (
            (r"cpi", "india_monthly_cpi", "India monthly CPI inflation", monthly),
            (r"iip", "india_monthly_iip", "India monthly industrial production growth", monthly),
            (r"gdp", "india_quarterly_gdp", "India quarterly real GDP growth", frequency.eq("quarterly")),
            (r"bond yield", "india_monthly_bond_yield", "India monthly 10-year government bond yield", monthly),
        ):
            mask = country.eq("india") & frequency_mask & indicator.str.contains(pattern, regex=True, na=False)
            figures.update(_macro_trend(economy, mask, key, title, config))

    figures.update(_nse_index_heatmap(_frame(tables, "index_snapshot"), config))
    nifty = _nifty_history(tables, config)
    figures.update(_nifty_drawdown(nifty, config))
    figures.update(_seasonality(nifty, config))
    figures.update(_breadth_by_size(tables, config))
    return figures
