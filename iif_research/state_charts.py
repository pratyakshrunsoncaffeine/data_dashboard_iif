"""Plain charts for observed market-state indicators and their coverage."""
from __future__ import annotations

from collections.abc import Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#355C7D"
TEAL = "#4C9F9A"
RED = "#B75D5D"
GREY = "#667085"
PALETTE = [BLUE, TEAL, "#8296A8", RED, "#8BA59C", "#8D7A9E"]


def _frame(mapping: Mapping, *names: str) -> pd.DataFrame:
    for name in names:
        value = (mapping or {}).get(name)
        if isinstance(value, pd.DataFrame):
            return value.copy()
        if value is not None:
            try:
                return pd.DataFrame(value)
            except (TypeError, ValueError):
                continue
    return pd.DataFrame()


def _date(frame: pd.DataFrame, column="date") -> pd.DataFrame:
    data = frame.copy()
    if column in data:
        data[column] = pd.to_datetime(data[column], errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.normalize()
        data = data.dropna(subset=[column])
    return data


def _fig(figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")
    return fig, ax


def _caption(ax, text):
    import textwrap
    ax.text(0, -0.19, textwrap.fill(text, 140), transform=ax.transAxes, ha="left", va="top", color=GREY, fontsize=8)


def _footer(fig, text):
    import textwrap
    fig.text(.12, .012, textwrap.fill(text, 150), ha="left", va="bottom", color=GREY, fontsize=8)


def _numeric(frame, *columns):
    data = frame.copy()
    for column in columns:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def _stock_labels(ax, data, xcol, ycol):
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    occupied = []
    candidates = [(4, 4), (4, 16), (4, -12), (4, 28), (4, -24), (-35, 16), (-35, -12), (12, 40), (12, -36)]
    for row in data.itertuples(index=False):
        point = (getattr(row, xcol), getattr(row, ycol))
        label = ax.annotate(str(row.symbol), point, xytext=candidates[0], textcoords="offset points", fontsize=6.5, color="#344054")
        chosen = candidates[0]
        for offset in candidates:
            chosen = offset
            label.set_position(offset)
            bounds = label.get_window_extent(renderer).expanded(1.04, 1.15)
            if not any(bounds.overlaps(previous) for previous in occupied):
                break
        occupied.append(label.get_window_extent(renderer).expanded(1.04, 1.15))
        if chosen != candidates[0]:
            ax.annotate("", point, xytext=chosen, textcoords="offset points", arrowprops={"arrowstyle": "-", "color": "#98A2B3", "linewidth": .4})


def _sector_rotation(analysis, config):
    data = _frame(analysis, "sector_rotation")
    required = {"sector", "date", "relative_strength_100", "relative_momentum_21_pct"}
    if data.empty or not required.issubset(data.columns):
        return {}
    data = _date(data)
    data = _numeric(data, "relative_strength_100", "relative_momentum_21_pct").dropna(subset=["relative_strength_100", "relative_momentum_21_pct"])
    data = data.loc[data.date.le(pd.Timestamp(config.end))].sort_values(["sector", "date"])
    if data.empty:
        return {}
    fig, ax = _fig((12.8, 6.8))
    sectors = list(data.sector.dropna().astype(str).unique())
    for i, sector in enumerate(sectors):
        group = data.loc[data.sector.astype(str).eq(sector)].sort_values("date").tail(10)
        if group.empty:
            continue
        color = PALETTE[i % len(PALETTE)]
        label = str(sector).removeprefix("NIFTY ")
        ax.plot(group.relative_strength_100, group.relative_momentum_21_pct, color=color, lw=1.2, alpha=.75, label=f"{i + 1}. {label}")
        ax.scatter(group.relative_strength_100, group.relative_momentum_21_pct, color=color, s=20, alpha=.75, edgecolors="white", linewidths=.4)
        latest = group.iloc[-1]
        ax.annotate(str(i + 1), (latest.relative_strength_100, latest.relative_momentum_21_pct), xytext=(4, 4), textcoords="offset points", fontsize=8, color="#344054")
    ax.axvline(100, color="#98A2B3", lw=.8, ls="--")
    ax.axhline(0, color="#98A2B3", lw=.8, ls="--")
    ax.set_xlabel("Sector/Nifty relative strength (126-session anchor = 100)")
    ax.set_ylabel("21-session relative momentum (%)")
    ax.set_title("Sector rotation trajectory", loc="left", fontweight="bold")
    ax.grid(color="#E6E8EB", lw=.6)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8, handlelength=1.4)
    _caption(ax, f"Latest point and trailing ten observed sessions through {data.date.max().date()}; report end {config.end}. Vertical reference is relative strength 100; horizontal reference is zero relative momentum. IIF descriptive framework; not a proprietary or licensed RRG. Source: observed sector and Nifty 50 index history.")
    fig.tight_layout(rect=(0, .08, .99, 1))
    return {"state_sector_rotation": fig}


def _risk_return(analysis, config):
    data = _frame(analysis, "stock_risk_return")
    if data.empty or not {"symbol", "sector", "return_63_pct", "vol_63_ann_pct"}.issubset(data.columns):
        return {}
    xcol, ycol = "vol_63_ann_pct", "return_63_pct"
    data = _numeric(data, xcol, ycol, "avg_traded_value_cr").dropna(subset=[xcol, ycol])
    if data.empty:
        return {}
    # Preserve the most informative points while avoiding a wall of overlapping labels.
    data = data.assign(_activity=data.get("avg_traded_value_cr", 0))
    if len(data) > 24:
        data = data.nlargest(24, "_activity")
    fig, ax = _fig((10, 6))
    sector = "sector"
    groups = data[sector].fillna("Unclassified").astype(str)
    for i, (group, rows) in enumerate(data.groupby(groups, sort=True)):
        ax.scatter(rows[xcol], rows[ycol], s=34, color=PALETTE[i % len(PALETTE)], alpha=.75, label=group, edgecolors="white", linewidths=.4)
    if len(data) <= 30:
        _stock_labels(ax, data, xcol, ycol)
    ax.axhline(0, color="#98A2B3", lw=.8, ls="--")
    ax.set_xlabel("63-session annualized volatility (%)")
    ax.set_ylabel("63-session return (%)")
    ax.set_title("Observed 63-session risk and return", loc="left", fontweight="bold")
    ax.grid(color="#E6E8EB", lw=.6)
    ax.legend(frameon=False, fontsize=7, title="Sector", loc="upper right", ncol=2)
    actual_date = pd.to_datetime(data.get("as_of", pd.Series(dtype=str)), errors="coerce").max()
    actual_note = actual_date.date().isoformat() if pd.notna(actual_date) else "source date not supplied"
    _caption(ax, f"Observed through {actual_note}; report end {config.end}. Plotted {len(data)} liquid symbols with both 63-session measures observed. Constituents are current; volatility is annualized historical volatility. Source: adjusted daily closes. Coverage and data-quality fields are exported.")
    fig.tight_layout(rect=(0, .08, 1, 1))
    return {"state_risk_return": fig}


def _breadth(analysis, config):
    data = _frame(analysis, "advance_decline")
    if data.empty or "date" not in data:
        return {}
    data = _date(data)
    data = data.loc[data.date.le(pd.Timestamp(config.end))]
    candidates = [("ad_line_normalized", "Normalized advance-decline line", "Index (start = 100)"),
                  ("above_50ma_pct", "Share above 50-session average", "Percent of eligible stocks"),
                  ("above_200ma_pct", "Share above 200-session average", "Percent of eligible stocks")]
    columns = [(column, label, ylabel) for column, label, ylabel in candidates if column in data and pd.to_numeric(data[column], errors="coerce").notna().any()]
    if not columns:
        return {}
    count = len(columns)
    fig, axes = plt.subplots(count, 1, figsize=(10, max(4.2, count * 2.6)), sharex=True, squeeze=False, facecolor="white")
    for i, (column, title, ylabel) in enumerate(columns):
        ax = axes[i, 0]
        values = pd.to_numeric(data[column], errors="coerce")
        ax.plot(data.date, values, color=PALETTE[i % len(PALETTE)], lw=1.5)
        if "pct" in column:
            ax.set_ylim(0, 100)
        if column == "ad_line_normalized":
            ax.axhline(100, color="#98A2B3", lw=.7, ls="--")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
        ax.grid(axis="y", color="#E6E8EB", lw=.6)
    axes[-1, 0].set_xlabel("Date")
    locator = mdates.AutoDateLocator()
    axes[-1, 0].xaxis.set_major_locator(locator)
    axes[-1, 0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    _footer(fig, f"Observed daily market breadth through {data.date.max().date()}; selected report end {config.end}. Eligible stock counts are retained in the accompanying source table. Current constituents; history is not survivorship-free.")
    fig.tight_layout(rect=(0, .14, 1, 1))
    return {"state_breadth_trajectory": fig}


def _breadth_phase(analysis, config):
    data = _frame(analysis, "advance_decline")
    if data.empty or not {"date", "breadth_thrust_10s_pct"}.issubset(data.columns):
        return {}
    data = _date(data)
    data = _numeric(data, "breadth_thrust_10s_pct", "above_50ma_pct", "above_200ma_pct")
    data = data.loc[data.date.le(pd.Timestamp(config.end))].dropna(subset=["breadth_thrust_10s_pct"])
    if data.empty:
        return {}
    data = data.dropna(subset=["above_200ma_pct"]).sort_values("date") if "above_200ma_pct" in data else pd.DataFrame()
    if data.empty:
        return {}
    fig, ax = _fig((8.5, 6))
    ax.plot(data.above_200ma_pct, data.breadth_thrust_10s_pct, color=GREY, lw=1, alpha=.6)
    points = ax.scatter(data.above_200ma_pct, data.breadth_thrust_10s_pct, c=np.arange(len(data)), cmap="viridis", s=42, edgecolors="white", linewidths=.5)
    for i, row in enumerate(data.iloc[[0, -1]].itertuples()):
        ax.annotate(row.date.strftime("%d %b"), (row.above_200ma_pct, row.breadth_thrust_10s_pct), xytext=((-45, 25) if i == 0 else (25, -25)), textcoords="offset points", fontsize=8, arrowprops={"arrowstyle": "-", "color": GREY, "linewidth": .6})
    ax.axhline(50, color="#98A2B3", lw=.7, ls="--")
    ax.axvline(50, color="#98A2B3", lw=.7, ls="--")
    ax.set_xlim(max(0, min(50, data.above_200ma_pct.min()) - 10), min(100, max(50, data.above_200ma_pct.max()) + 10))
    ax.set_ylim(max(0, min(50, data.breadth_thrust_10s_pct.min()) - 10), min(100, max(50, data.breadth_thrust_10s_pct.max()) + 10))
    ax.set_ylabel("10-session mean advancing share (%)")
    ax.set_xlabel("Stocks above prior 200-session average (%)")
    ax.set_title("Breadth phase and participation", loc="left", fontweight="bold")
    ax.grid(color="#E6E8EB", lw=.6)
    colorbar = fig.colorbar(points, ax=ax, pad=.02)
    colorbar.set_label("Observation sequence (later = lighter)")
    _caption(ax, f"{len(data)} observed sessions from {data.date.min().date()} to {data.date.max().date()}. Long-term participation is on the horizontal axis; recent advancing share is on the vertical axis. Eligible counts and the breadth-thrust definition are exported. Current constituent universe.")
    fig.tight_layout(rect=(0, .08, 1, 1))
    return {"state_breadth_phase": fig}


def _return_distribution(analysis, config):
    data = _frame(analysis, "stock_risk_return")
    if data.empty:
        return {}
    value_col = "report_return_pct" if "report_return_pct" in data else "return_63_pct" if "return_63_pct" in data else None
    if not value_col:
        return {}
    data = _numeric(data, value_col).dropna(subset=[value_col])
    if data.empty:
        return {}
    # The distribution is a transparent empirical CDF rather than a fitted density.
    values = np.sort(data[value_col].to_numpy(dtype=float))
    cdf = np.arange(1, len(values) + 1) / len(values)
    fig, ax = _fig((9, 5))
    ax.step(values, cdf, where="post", color=BLUE, lw=1.6)
    ax.axvline(0, color="#98A2B3", lw=.8, ls="--")
    ax.axhline(.5, color="#D0D5DD", lw=.7, ls=":")
    ax.set_xlabel("Report-period return (%)" if value_col == "report_return_pct" else "63-session return (%)")
    ax.set_ylabel("Empirical cumulative share")
    ax.set_ylim(0, 1)
    ax.set_title("Cross-sectional return distribution", loc="left", fontweight="bold")
    ax.grid(color="#E6E8EB", lw=.6)
    _caption(ax, f"{len(values)} current-universe symbols with an observed {ax.get_xlabel().lower()} for {config.start} to {config.end}. Each symbol contributes one observation; missing returns are omitted. Source: adjusted daily closes.")
    fig.tight_layout(rect=(0, .08, 1, 1))
    return {"state_return_distribution": fig}


def _delivery(analysis, tables, config):
    data = _frame(analysis, "stock_price_volume_delivery")
    if data.empty:
        data = _frame(tables, "delivery")
    xcol = next((c for c in ("turnover_cr", "traded_value_cr", "average_turnover_cr", "volume") if c in data), None)
    ycol = next((c for c in ("delivery_pct", "delivery_percent", "deliverable_pct") if c in data), None)
    if data.empty or not xcol or not ycol:
        return {}
    data = _date(data)
    data = _numeric(data, xcol, ycol).dropna(subset=[xcol, ycol])
    if "date" in data:
        data = data.loc[data.date.le(pd.Timestamp(config.end))]
    if data.empty:
        return {}
    if len(data) > 500:
        data = data.nlargest(500, xcol)
    fig, ax = _fig((9, 5.5))
    if {"return_pct", "volume_ratio_20d"}.issubset(data):
        sizes = 20 + pd.to_numeric(data.volume_ratio_20d, errors="coerce").fillna(0).clip(0, 5).mul(12)
        colors = pd.to_numeric(data.return_pct, errors="coerce")
        cap = max(1, float(colors.abs().quantile(.95))) if colors.notna().any() else 1
        points = ax.scatter(data[xcol], data[ycol], s=sizes, c=colors, cmap="RdBu", vmin=-cap, vmax=cap, alpha=.6, edgecolors="none")
        fig.colorbar(points, ax=ax, pad=.02).set_label("Daily return (%)")
    else:
        ax.scatter(data[xcol], data[ycol], s=24, color=TEAL, alpha=.45, edgecolors="none")
    ax.set_xscale("log") if (data[xcol] > 0).all() else None
    ax.set_xlabel("Traded value (₹ crore, log scale when positive)" if (data[xcol] > 0).all() else "Traded value (source units)")
    ax.set_ylabel("Delivery share (%)")
    ax.set_title("Delivery participation and traded value", loc="left", fontweight="bold")
    ax.grid(color="#E6E8EB", lw=.6)
    actual = f"{data.date.min().date()} to {data.date.max().date()}" if "date" in data else "dates not supplied"
    sample_note = " Top 500 observations by traded value are shown." if len(_frame(analysis, "stock_price_volume_delivery")) > 500 else ""
    _caption(ax, f"{len(data)} observed symbol-days; dates {actual}; through report end {config.end}.{sample_note} Color shows daily return; point size shows volume relative to its prior 20-session mean, capped at 5 times. Traded value is estimated as raw close times volume, not exchange VWAP turnover. Source: NSE delivery and Yahoo price observations.")
    fig.tight_layout(rect=(0, .08, 1, 1))
    return {"state_delivery_participation": fig}


def _volatility(analysis, tables, config):
    data = _frame(analysis, "nifty_market_state")
    required = {"date", "vol_20_ann_pct", "vol_63_ann_pct"}
    if data.empty or not required.issubset(data.columns):
        return {}
    data = _date(data)
    data = data.loc[data.date.le(pd.Timestamp(config.end))]
    data = _numeric(data, "vol_20_ann_pct", "vol_63_ann_pct").dropna(subset=["vol_20_ann_pct", "vol_63_ann_pct"]).sort_values("date").drop_duplicates("date", keep="last")
    if len(data) < 2:
        return {}
    fig, ax = _fig((10, 4.8))
    ax.plot(data.date, data.vol_20_ann_pct, color=TEAL, lw=1.35, label="20-session realized volatility")
    ax.plot(data.date, data.vol_63_ann_pct, color=BLUE, lw=1.45, label="63-session realized volatility")
    ax.fill_between(data.date, data.vol_20_ann_pct, data.vol_63_ann_pct, color=BLUE, alpha=.10)
    vix = _frame(tables, "markets", "long_index_history")
    if not vix.empty and {"date", "name", "close"}.issubset(vix.columns):
        vix = _date(vix)
        vix = vix.loc[vix.date.between(data.date.min(), data.date.max()) & vix.name.astype(str).str.contains("India VIX", case=False, na=False)]
        vix = _numeric(vix, "close").dropna(subset=["close"]).sort_values("date").drop_duplicates("date", keep="last")
        if len(vix) > 1:
            twin = ax.twinx()
            twin.plot(vix.date, vix.close, color=RED, lw=1.1, alpha=.8, label="India VIX")
            twin.set_ylabel("India VIX (index points)")
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = twin.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="upper left")
        else:
            ax.legend(frameon=False, fontsize=8, loc="upper left")
    else:
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_ylabel("Annualized realized volatility (%)")
    ax.set_xlabel("Date")
    ax.set_title("Nifty realized volatility and India VIX", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E6E8EB", lw=.6)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    _caption(ax, f"Observed realized volatility: {data.date.min().date()} to {data.date.max().date()}; report end {config.end}. Realized measures use 20- and 63-session windows. India VIX is options-implied and is plotted on the right axis when observations are available. Neither series forecasts future volatility. Sources: adjusted Nifty 50 closes and NSE India VIX.")
    fig.tight_layout(rect=(0, .08, 1, 1))
    return {"state_volatility_history": fig}


def _candlestick(tables, config):
    data = _frame(tables, "nifty_ohlc", "markets", "index_candles", "nifty_candles")
    required = {"date", "open", "high", "low", "close"}
    if data.empty or not required.issubset(data.columns):
        return {}
    data = _date(data)
    if "name" in data:
        data = data.loc[data.name.astype(str).str.contains("Nifty 50", case=False, na=False)]
    data = _numeric(data, "open", "high", "low", "close")
    data = data.loc[data.date.le(pd.Timestamp(config.end))].dropna(subset=["open", "high", "low", "close"]).sort_values("date").drop_duplicates("date", keep="last")
    data["ma20"] = data.close.rolling(20, min_periods=20).mean()
    data["ma50"] = data.close.rolling(50, min_periods=50).mean()
    data = data.tail(60)
    if data.empty:
        return {}
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator
    fig, ax = _fig((11, 5.4))
    x = np.arange(len(data))
    for xi, row in zip(x, data.itertuples(index=False)):
        rising = row.close >= row.open
        color = TEAL if rising else RED
        ax.vlines(xi, row.low, row.high, color=color, lw=.8)
        bottom = min(row.open, row.close)
        height = max(abs(row.close - row.open), .02)
        ax.add_patch(Rectangle((xi - .3, bottom), .6, height, facecolor=color, edgecolor=color, alpha=.8))
    close = data.close.to_numpy(dtype=float)
    if len(data) >= 20:
        ax.plot(x, data.ma20, color=BLUE, lw=1.1, label="20-session average")
    if len(data) >= 50:
        ax.plot(x, data.ma50, color="#8D7A9E", lw=1.1, label="50-session average")
    if len(data) >= 20:
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    labels = [stamp.strftime("%d %b") for stamp in data.date]
    positions = np.unique(np.linspace(0, len(labels) - 1, min(8, len(labels)), dtype=int))
    ax.set_xticks(positions, [labels[i] for i in positions])
    ax.yaxis.set_major_locator(MaxNLocator(7))
    ax.set_xlabel("Trading date")
    ax.set_ylabel("Nifty 50 index points")
    ax.set_title("Nifty 50 daily OHLC and moving averages", loc="left", fontweight="bold")
    ax.grid(axis="y", color="#E6E8EB", lw=.6)
    _caption(ax, f"Latest {len(data)} observed sessions from {data.date.min().date()} to {data.date.max().date()}; report end {config.end}. Candles use public Yahoo Finance open, high, low and close observations. Moving averages are descriptive history.")
    fig.tight_layout(rect=(0, .08, 1, 1))
    return {"state_nifty_candlestick": fig}


def _institution_absorption(analysis, config):
    data = _frame(analysis, "institution_absorption")
    required = {"date", "fii_net_cr", "dii_net_cr", "combined_net_cr", "domestic_absorption_ratio"}
    if data.empty or not required.issubset(data.columns):
        return {}
    data = _date(data)
    data = _numeric(data, "fii_net_cr", "dii_net_cr", "combined_net_cr", "domestic_absorption_ratio")
    data = data.loc[data.date.le(pd.Timestamp(config.end))].sort_values("date").dropna(subset=["fii_net_cr", "dii_net_cr"])
    if data.empty:
        return {}
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, facecolor="white")
    axes[0].plot(data.date, data.fii_net_cr, marker="o", ms=3, color=BLUE, lw=1.2, label="FII net cash flow")
    axes[0].plot(data.date, data.dii_net_cr, marker="o", ms=3, color=TEAL, lw=1.2, label="DII net cash flow")
    axes[0].plot(data.date, data.combined_net_cr, color=GREY, lw=1, ls="--", label="Combined")
    axes[0].axhline(0, color="#98A2B3", lw=.7)
    axes[0].set_ylabel("₹ crore")
    axes[0].set_title("Institutional cash flows", loc="left", fontsize=9, fontweight="bold")
    axes[0].grid(axis="y", color="#E6E8EB", lw=.6)
    axes[0].legend(frameon=False, fontsize=8, ncol=3)
    axes[1].plot(data.date, data.domestic_absorption_ratio, marker="o", ms=3, color=RED, lw=1.2)
    axes[1].set_ylabel("DII absorption ratio")
    axes[1].set_xlabel("Observation date")
    axes[1].set_title("Domestic absorption when FII net flow is negative", loc="left", fontsize=9, fontweight="bold")
    axes[1].grid(axis="y", color="#E6E8EB", lw=.6)
    locator = mdates.AutoDateLocator()
    axes[1].xaxis.set_major_locator(locator)
    axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    _footer(fig, f"{len(data)} dated observations through {data.date.max().date()}; report end {config.end}. The absorption ratio is shown only where FII net flow is negative and is unavailable when its denominator is zero. Cash flows in ₹ crore; no future path is estimated. Source: NSE cash market FII/DII data.")
    fig.tight_layout(rect=(0, .14, 1, 1))
    return {"state_institution_absorption": fig}


def _factor_spreads(analysis, config):
    data = _frame(analysis, "factor_spreads")
    if data.empty or not {"date", "spread_name", "value", "unit"}.issubset(data.columns):
        return {}
    data = _date(data)
    data = _numeric(data, "value").dropna(subset=["value"])
    data = data.loc[data.date.le(pd.Timestamp(config.end))].sort_values("date")
    if data.empty:
        return {}
    unit_groups = list(data.groupby("unit", dropna=False, sort=True))
    fig, axes = plt.subplots(len(unit_groups), 1, figsize=(10, max(4.7, 3.1 * len(unit_groups))),
                             sharex=True, squeeze=False, facecolor="white")
    for panel_i, (unit, unit_data) in enumerate(unit_groups):
        ax = axes[panel_i, 0]
        for i, (name, group) in enumerate(unit_data.groupby("spread_name", sort=True)):
            group = group.sort_values("date")
            ax.plot(group.date, group.value, color=PALETTE[i % len(PALETTE)], lw=1.25, label=str(name))
        ax.axhline(0, color="#98A2B3", lw=.7, ls="--")
        ax.set_ylabel(str(unit) if pd.notna(unit) else "Unit not supplied")
        ax.grid(axis="y", color="#E6E8EB", lw=.6)
        ax.legend(frameon=False, fontsize=8, loc="best")
    axes[0, 0].set_title("Observed factor spreads", loc="left", fontweight="bold")
    axes[-1, 0].set_xlabel("Date")
    locator = mdates.AutoDateLocator()
    axes[-1, 0].xaxis.set_major_locator(locator)
    axes[-1, 0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    _footer(fig, f"Observed {data.date.min().date()} to {data.date.max().date()}; report end {config.end}. Units are separated into panels. Spread definitions and construction are in the factor-spread table. Source: observed market series.")
    fig.tight_layout(rect=(0, .08, 1, 1), h_pad=.8)
    return {"state_factor_spreads": fig}


def build_state_charts(tables: Mapping, analysis: Mapping | None, config) -> dict[str, plt.Figure]:
    """Return available market-state figures, omitting charts without observed inputs."""
    analysis = analysis or {}
    figures = {}
    for builder, args in ((_sector_rotation, (analysis, config)),
                          (_risk_return, (analysis, config)),
                          (_breadth, (analysis, config)),
                          (_breadth_phase, (analysis, config)),
                          (_return_distribution, (analysis, config)),
                          (_delivery, (analysis, tables, config)),
                          (_volatility, (analysis, tables, config)),
                          (_candlestick, (tables, config)),
                          (_institution_absorption, (analysis, config)),
                          (_factor_spreads, (analysis, config))):
        figures.update(builder(*args))
    return figures
