"""Academic-style charts for official earnings comparisons and dated releases."""
from __future__ import annotations

from collections.abc import Mapping
import re
import textwrap
from urllib.parse import urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

from .state_charts import _stock_labels

BLUE = "#355C7D"
TEAL = "#4C9F9A"
RED = "#B75D5D"
GREY = "#667085"


def _frame(mapping: Mapping, key: str) -> pd.DataFrame:
    value = (mapping or {}).get(key)
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if value is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(value)
    except (TypeError, ValueError):
        return pd.DataFrame()


def _dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return parsed.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.normalize()


def _sector(value) -> str:
    if pd.isna(value) or not str(value).strip():
        return "Unclassified"
    return re.sub(r"\s+", " ", str(value)).strip()


def _raw_percent(value, _position):
    if abs(value) >= 100:
        return f"{value:,.0f}"
    return f"{value:.1f}"


def _earnings_growth_margin(tables, analysis, config) -> dict[str, plt.Figure]:
    latest = _frame(analysis, "official_quarterly_latest")
    required = {"symbol", "period", "basis", "release_date", "profit_yoy_pct", "net_profit_margin_pct"}
    if latest.empty or not required.issubset(latest.columns):
        return {}
    latest["release_date"] = _dates(latest.release_date)
    latest["period"] = latest.period.astype(str)
    latest = latest.loc[
        latest.release_date.notna()
        & latest.release_date.le(pd.Timestamp(config.end))
    ].copy()
    latest["profit_yoy_pct"] = pd.to_numeric(latest.profit_yoy_pct, errors="coerce")
    latest["net_profit_margin_pct"] = pd.to_numeric(latest.net_profit_margin_pct, errors="coerce")
    data = latest.dropna(subset=["profit_yoy_pct", "net_profit_margin_pct"]).copy()
    if data.empty:
        return {}

    liquidity = _frame(analysis, "stock_risk_return")
    if not liquidity.empty and {"symbol", "avg_traded_value_cr"}.issubset(liquidity.columns):
        liquidity["avg_traded_value_cr"] = pd.to_numeric(liquidity.avg_traded_value_cr, errors="coerce")
        liquidity = liquidity.sort_values("avg_traded_value_cr", ascending=False).drop_duplicates("symbol")
        data = data.merge(liquidity[["symbol", "avg_traded_value_cr"]], on="symbol", how="left", validate="many_to_one")
    else:
        data["avg_traded_value_cr"] = np.nan
    if "avg_traded_value_cr" not in data:
        data["avg_traded_value_cr"] = np.nan
    data["_sector"] = data.get("sector", pd.Series("Unclassified", index=data.index)).map(_sector)
    data["_label"] = data.get("name", data.symbol).fillna(data.symbol).astype(str)

    # Use an explicit legend for common groups and combine only the sparse tail.
    counts = data._sector.value_counts()
    common = set(counts.loc[counts >= 3].index)
    data["_legend_sector"] = data._sector.where(data._sector.isin(common), "Other sectors (fewer than 3)")
    categories = sorted(data._legend_sector.unique())
    cmap = plt.get_cmap("tab20")
    colors = {sector: cmap(i % cmap.N) for i, sector in enumerate(categories)}
    labels = data.nlargest(24, "avg_traded_value_cr").index.tolist()

    fig, ax = plt.subplots(figsize=(11.5, 6.3), facecolor="white")
    ax.set_facecolor("white")
    for sector in categories:
        group = data.loc[data._legend_sector.eq(sector)]
        ax.scatter(group.profit_yoy_pct, group.net_profit_margin_pct,
                   color=colors[sector], s=31, alpha=.48, edgecolors="white", linewidths=.35,
                   label=sector, zorder=2)
    ax.set_xscale("symlog", linthresh=50)
    ax.set_yscale("symlog", linthresh=25)
    for axis, limits, threshold in ((ax.xaxis, ax.get_xlim(), 50), (ax.yaxis, ax.get_ylim(), 25)):
        candidates = [-100000, -10000, -1000, -100, -threshold, 0, threshold, 100, 1000, 10000, 100000]
        if axis is ax.xaxis:
            candidates = [value for value in candidates if abs(value) != threshold]
        ticks = sorted(set(value for value in candidates if limits[0] <= value <= limits[1]))
        if len(ticks) >= 3:
            axis.set_ticks(ticks)
        else:
            axis.set_major_locator(MaxNLocator(nbins=4))
    ax.xaxis.set_major_formatter(FuncFormatter(_raw_percent))
    ax.yaxis.set_major_formatter(FuncFormatter(_raw_percent))
    ax.axhline(0, color="#98A2B3", lw=.8, ls="--")
    ax.axvline(0, color="#98A2B3", lw=.8, ls="--")
    ax.set_xlabel("Net profit growth versus same quarter a year earlier (%)")
    ax.set_ylabel("PAT / reported revenue or income (%)")
    ax.set_title("Profit growth and reported top-line ratio", loc="left", fontweight="bold", fontsize=11)
    ax.grid(color="#E6E8EB", lw=.6, zorder=0)
    ax.legend(title="Sector", frameon=False, fontsize=7, title_fontsize=8,
              loc="center left", bbox_to_anchor=(1.01, .5), borderaxespad=0)

    period_counts = data.period.value_counts().sort_index()
    period_note = "; ".join(f"{period}: {count}" for period, count in period_counts.items())
    observed = f"{data.release_date.min().date()} to {data.release_date.max().date()}"
    basis_note = "Basis: " + ", ".join(sorted(data.basis.dropna().astype(str).unique())) if "basis" in data else "Basis not supplied."
    ratio_note = (
        "An unusually high ratio can occur when PAT includes income outside the selected top-line concept; inspect the filing concept and comparison note. "
        "This reported ratio is not an operating-margin proxy or forecast."
    )
    if {"symbol", "revenue_cr", "net_profit_cr"}.issubset(latest.columns):
        mmtc = latest.loc[latest.symbol.astype(str).str.upper().eq("MMTC")].dropna(subset=["revenue_cr", "net_profit_cr"])
        if not mmtc.empty:
            row = mmtc.iloc[-1]
            revenue_concept = str(row.get("revenue_concept") or "reported revenue")
            ratio_note += f" MMTC reports {revenue_concept} of ₹{float(row.revenue_cr):,.2f} crore and PAT of ₹{float(row.net_profit_cr):,.2f} crore in this run."
    caption = (
        f"{len(data)} companies with same-basis, same-concept prior-year profit comparisons; latest fiscal periods are mixed ({period_note}). "
        f"Release dates observed {observed}; selected as of {config.end}. {basis_note}. "
        "Bank top-line concepts may be total income rather than revenue; profit growth and net margin are percentages, while margin change in percentage points is retained in the table. "
        "Axes use symmetric log scales (linear thresholds: ±50% profit growth and ±25% net margin); tick labels show raw percentages, and all actual points remain in view. "
        f"{ratio_note} "
        "Source: official company filings; see company rows for XBRL links and comparison notes."
    )
    fig.subplots_adjust(left=.12, right=.76, bottom=.24, top=.91)
    _stock_labels(ax, data.loc[labels], "profit_yoy_pct", "net_profit_margin_pct")
    fig.text(.12, .035, textwrap.fill(caption, 150), ha="left", va="bottom", color=GREY, fontsize=7.5)
    return {"official_earnings_map": fig}


def _release_timeline(tables, config) -> dict[str, plt.Figure]:
    data = _frame(tables, "economic_calendar")
    required = {"date", "country", "event"}
    if data.empty or not required.issubset(data.columns):
        return {}
    data["date"] = _dates(data.date)
    start, end = pd.Timestamp(config.start), pd.Timestamp(config.end)
    data = data.loc[data.date.between(start, end, inclusive="both")].copy()
    data = data.dropna(subset=["date"]).sort_values(["date", "country", "event"]).reset_index(drop=True)
    if data.empty:
        return {}
    total = len(data)
    if total > 16:
        selection = np.linspace(0, total - 1, 16).round().astype(int)
        data = data.iloc[selection].copy().reset_index(drop=True)

    country_short = {"united states": "US", "united kingdom": "UK"}
    agency_short = {
        "bea.gov": "BEA", "federalreserve.gov": "Fed", "bls.gov": "BLS",
        "ons.gov.uk": "ONS", "rbi.org.in": "RBI", "mospi.gov.in": "MoSPI",
        "ec.europa.eu": "Eurostat", "ecb.europa.eu": "ECB", "statcan.gc.ca": "StatCan",
    }
    agency_text = {
        "office for national statistics": "ONS", "federal reserve": "Fed",
        "bureau of economic analysis": "BEA", "bureau of labor statistics": "BLS",
        "fred": "FRED", "reserve bank of india": "RBI", "ministry of statistics": "MoSPI",
        "eurostat": "Eurostat", "european central bank": "ECB", "statistics canada": "StatCan",
    }
    data["country_label"] = data.country.map(_sector).map(lambda name: country_short.get(name.lower(), name))
    countries = sorted(data.country_label.unique())
    palette = [BLUE, TEAL, RED, "#8296A8", "#8D7A9E", "#8BA59C", "#B18C6B", "#6F8096"]
    colors = {country: palette[i % len(palette)] for i, country in enumerate(countries)}
    data["_row"] = np.arange(len(data))
    labels = []
    for row in data.itertuples(index=False):
        event = re.sub(r"\s+", " ", str(row.event)).strip()
        country = str(row.country_label)
        source = getattr(row, "source", None)
        if source and pd.notna(source):
            source = str(source)
            host = urlparse(source).netloc
            host = host.removeprefix("www.") if host else ""
            source_name = next((agency for domain, agency in agency_short.items() if host.endswith(domain)), None)
            source_name = source_name or next((agency for name, agency in agency_text.items() if name in source.lower()), None)
            source_name = source_name or (host[:14] if host else source[:14])
            source_text = f" · {source_name}"
        else:
            source_text = ""
        labels.append(textwrap.fill(f"{country}{source_text} · {event}", width=43, max_lines=2, placeholder="…"))

    height = max(6.3, .60 * len(data) + 2.3)
    fig, ax = plt.subplots(figsize=(11.5, height), facecolor="white")
    ax.set_facecolor("white")
    for country in countries:
        group = data.loc[data.country_label.eq(country)]
        ax.scatter(group.date, group._row, s=38, color=colors[country], alpha=.9,
                   edgecolors="white", linewidths=.5, label=country, zorder=3)
    ax.set_yticks(data._row, labels)
    ax.tick_params(axis="y", labelsize=8, length=0, pad=7)
    ax.invert_yaxis()
    ax.set_xlim(start - pd.Timedelta(days=1), end + pd.Timedelta(days=1))
    locator = mdates.AutoDateLocator(minticks=3, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.set_xlabel("Event date")
    ax.set_ylabel("Country · agency · release")
    ax.set_title("Dated economic releases", loc="left", fontweight="bold", fontsize=11)
    ax.grid(axis="x", color="#E6E8EB", lw=.6)
    ax.legend(title="Country", frameon=False, fontsize=7, title_fontsize=8,
              loc="center left", bbox_to_anchor=(1.01, .5), borderaxespad=0)
    shown_dates = f"{data.date.min().date()} to {data.date.max().date()}"
    sample_note = f"Showing 16 of {total} dated releases" if total > 16 else f"Showing all {total} dated releases"
    caption = (
        f"{sample_note} in the inclusive requested window {config.start} to {config.end}; shown event dates {shown_dates}. "
        "Marker positions use source-listed dates; releases may be scheduled or published, and the date does not establish that an event occurred. Status is preserved where supplied. "
        "Full event details, status, source agency and URLs remain in the economic calendar table."
    )
    fig.subplots_adjust(left=.40, right=.78, bottom=.16, top=.92)
    fig.text(.40, .035, textwrap.fill(caption, 130), ha="left", va="bottom", color=GREY, fontsize=7.5)
    return {"official_release_timeline": fig}


def build_release_charts(tables: Mapping, analysis: Mapping | None, config) -> dict[str, plt.Figure]:
    """Build dated earnings comparison and official release timeline figures."""
    analysis = analysis or {}
    figures = {}
    figures.update(_earnings_growth_margin(tables, analysis, config))
    figures.update(_release_timeline(tables, config))
    return figures
