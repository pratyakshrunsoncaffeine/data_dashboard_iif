"""Plain research charts for the IIF market-monitoring report."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BLUE = "#355C7D"
TEAL = "#4C9F9A"
RED = "#B75D5D"
GREY = "#667085"
PALETTE = [BLUE, TEAL, "#8296A8", RED, "#8BA59C"]


def _frame(tables: Mapping[str, Any], name: str) -> pd.DataFrame:
    value = tables.get(name)
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value or [])


def _analysis_value(analysis: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in analysis and analysis[key] is not None:
            return analysis[key]
    return None


def _matrix(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        result = value.copy()
        if "symbol" in result.columns:
            result = result.set_index("symbol")
        return result.apply(pd.to_numeric, errors="coerce")
    if isinstance(value, Mapping):
        try:
            return pd.DataFrame(value)
        except (ValueError, TypeError):
            return pd.DataFrame()
    return pd.DataFrame()


def _caption(ax, text: str) -> None:
    ax.text(0, -0.18, text, transform=ax.transAxes, ha="left", va="top", color=GREY, fontsize=8)


def _new_fig(figsize=(10, 5.2)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")
    return fig, ax


def build_charts(tables: Mapping[str, Any], analysis: Mapping[str, Any] | None, config) -> dict[str, plt.Figure]:
    """Build available charts; missing or empty source tables simply omit a chart."""
    analysis = analysis or {}
    figures: dict[str, plt.Figure] = {}
    prices = _frame(tables, "prices")
    markets = _frame(tables, "markets")
    flows = _frame(tables, "flows")
    holdings = _frame(tables, "holdings")
    delivery = _frame(tables, "delivery")
    quarterly = _frame(tables, "quarterly")

    # Keep market performance in separate, comparable panels by source group;
    # yields and VIX retain their own units and charts below.
    period = _frame(analysis, "period_prices")
    if not period.empty and {"date", "symbol", "normalized"}.issubset(period.columns):
        period["date"] = pd.to_datetime(period.date, errors="coerce")
        period = period.dropna(subset=["date"])
        if "group" in period:
            group_label = period.group.fillna("").astype(str).str.strip()
            excluded = group_label.str.lower().isin(["yield", "volatility"])
            if "name" in period:
                excluded |= period.name.astype(str).str.contains("VIX", case=False, na=False)
            performance = period.loc[~excluded & group_label.ne("")].copy()
            performance["_panel"] = group_label.loc[performance.index]
        else:
            performance = period.iloc[:0].copy()
        if not performance.empty:
            preferred = ["India", "Sector", "Global", "Commodity", "Currency"]
            available_groups = performance._panel.dropna().unique().tolist()
            panels = [group for group in preferred if group in available_groups]
            panels.extend(sorted(group for group in available_groups if group not in panels))
            fig, axes = plt.subplots(len(panels), 1, figsize=(13, max(6.5, 2.45 * len(panels))), sharex=True, sharey=True, facecolor="white")
            axes = np.atleast_1d(axes)
            for ax, group_name in zip(axes, panels):
                panel = performance.loc[performance._panel.eq(group_name)]
                series_names = panel.get("name", panel.symbol).dropna().astype(str).unique().tolist()
                for i, name in enumerate(series_names):
                    part = panel.loc[panel.get("name", panel.symbol).astype(str).eq(name)].sort_values("date")
                    if part.empty:
                        continue
                    ax.plot(part.date, pd.to_numeric(part.normalized, errors="coerce"), label=name,
                            color=PALETTE[i % len(PALETTE)], lw=1.45)
                ax.axhline(100, color="#98A2B3", lw=.75, ls="--")
                ax.set_ylabel(f"{group_name}\nIndex (100 = anchor)", fontsize=8)
                ax.grid(axis="y", color="#E6E8EB", lw=.65)
                ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.005, .5), borderaxespad=0)
            axes[0].set_title("Market performance by series group", loc="left", fontweight="bold")
            axes[-1].set_xlabel("Date")
            fig.autofmt_xdate()
            actual = f"{performance.date.min().date()} to {performance.date.max().date()}"
            import textwrap
            caption = f"Requested: {config.start} to {config.end}; observed: {actual}. Each series is rebased to its last close before report start. Panels preserve the source groups; yields and India VIX are shown separately in native units. Source: period_prices."
            fig.text(.12, .015, textwrap.fill(caption, 160), ha="left", va="bottom", fontsize=8, color=GREY)
            fig.subplots_adjust(left=.12, right=.76, bottom=.14, top=.96, hspace=.32)
            figures["market_levels"] = fig

    if not markets.empty and {"date", "name", "close", "group"}.issubset(markets.columns):
        data = markets.copy(); data["date"] = pd.to_datetime(data.date, errors="coerce"); data["close"] = pd.to_numeric(data.close, errors="coerce")
        data = data.dropna(subset=["date", "close"])
        start, end = pd.Timestamp(config.start), pd.Timestamp(config.end)
        data = data.loc[data.date <= end]
        yields = data.loc[data.group.astype(str).str.lower() == "yield"]
        if not yields.empty:
            fig, ax = _new_fig((10, 5))
            for i, (name, part) in enumerate(yields.groupby("name")):
                part = part.sort_values("date"); anchors = part.loc[part.date < start, "close"]
                if anchors.empty: continue
                part = part.loc[part.date.between(start, end)]
                if not part.empty: ax.plot(part.date, part.close - anchors.iloc[-1], label=str(name), color=PALETTE[i % len(PALETTE)], lw=1.6)
            ax.axhline(0, color="#344054", lw=.8); ax.set_title("Yield changes during report period", loc="left", fontweight="bold")
            ax.set_ylabel("Change from prior close (percentage points)"); ax.grid(axis="y", color="#E6E8EB", lw=.7); ax.legend(frameon=False)
            fig.autofmt_xdate(); _caption(ax, f"Requested: {config.start} to {config.end}. Source: markets table; yields are changes in percentage points from the last close before report start.")
            fig.tight_layout(rect=(0, .06, 1, 1)); figures["yield_changes"] = fig
        vix = data.loc[data.name.astype(str).str.contains("VIX", case=False, na=False)]
        if not vix.empty:
            vix = vix.loc[vix.date.between(start, end)].sort_values("date")
            if not vix.empty:
                fig, ax = _new_fig((10, 4.5)); ax.plot(vix.date, vix.close, color=RED, lw=1.7)
                ax.set_title("India VIX", loc="left", fontweight="bold"); ax.set_ylabel("Index points"); ax.grid(axis="y", color="#E6E8EB", lw=.7)
                fig.autofmt_xdate(); _caption(ax, f"Requested: {config.start} to {config.end}; observed: {vix.date.min().date()} to {vix.date.max().date()}. Source: markets table.")
                fig.tight_layout(rect=(0, .06, 1, 1)); figures["india_vix"] = fig

    # Price performance, rebased to 100, supports sector/universe scans without mixing price scales.
    if not prices.empty and {"date", "symbol", "close"}.issubset(prices.columns):
        data = prices.copy()
        data["date"] = pd.to_datetime(data.date, errors="coerce")
        data["close"] = pd.to_numeric(data.close, errors="coerce")
        data = data.dropna(subset=["date", "symbol", "close"])
        start, end = pd.Timestamp(config.start), pd.Timestamp(config.end)
        data = data.loc[data.date <= end]
        if not data.empty:
            full_pivot = data.sort_values("date").pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
            anchor = full_pivot.loc[full_pivot.index < start].tail(1)
            pivot = full_pivot.loc[full_pivot.index >= start]
            ranked = _frame(analysis, "stock_momentum")
            if not ranked.empty and {"symbol", "composite_score_pct"}.issubset(ranked.columns):
                ranked["composite_score_pct"] = pd.to_numeric(ranked.composite_score_pct, errors="coerce")
                ranked = ranked.dropna(subset=["composite_score_pct"]).sort_values("composite_score_pct")
                selected = list(dict.fromkeys(ranked.symbol.head(6).tolist() + ranked.symbol.tail(6).tolist()))
                selected = [symbol for symbol in selected if symbol in pivot.columns]
            else:
                selected = list(pivot.columns[:12])
            if not selected:
                selected = list(pivot.columns[:12])
            if not anchor.empty:
                selected = [s for s in selected if s in anchor.columns and pd.notna(anchor.iloc[0][s])]
                rebased = pivot[selected].divide(anchor.iloc[0][selected]).multiply(100).dropna(how="all") if selected else pd.DataFrame()
            else:
                rebased = pd.DataFrame()
            if not rebased.empty:
                fig, ax = _new_fig((11, 5.5))
                for i, col in enumerate(rebased.columns):
                    ax.plot(rebased.index, rebased[col], lw=1.25, color=PALETTE[i % len(PALETTE)], label=str(col))
                ax.axhline(100, color="#98A2B3", lw=.8, ls="--")
                ax.set_title("Selected constituent price performance", loc="left", fontweight="bold")
                ax.set_ylabel("Indexed close (prior report close = 100)")
                ax.grid(axis="y", color="#E6E8EB", lw=.7)
                ax.legend(frameon=False, ncol=3, fontsize=7)
                fig.autofmt_xdate()
                _caption(ax, f"Requested: {config.start} to {config.end}; observed: {rebased.index.min().date()} to {rebased.index.max().date()}. Source: prices table; selected symbols ordered by momentum where available, rebased to the last close before report start.")
                fig.tight_layout(rect=(0, .06, 1, 1))
                figures["constituent_performance"] = fig

    # Diverging heatmap: label every cell and include units and actual sample period.
    span = _analysis_value(analysis, "correlation_dates", "correlation_period", "correlation_sample")
    if isinstance(span, pd.DataFrame):
        span = "; ".join(f"{key}: {value}" for key, value in span.iloc[0].items()) if not span.empty else None
    if not span and not prices.empty and "date" in prices:
        corr_dates = pd.to_datetime(prices.date, errors="coerce")
        corr_dates = sorted(corr_dates.loc[corr_dates <= pd.Timestamp(config.end)].dropna().unique())[-int(config.correlation_sessions):]
        if corr_dates:
            span = f"{pd.Timestamp(corr_dates[0]).date()} to {pd.Timestamp(corr_dates[-1]).date()}"
    raw = _matrix(_analysis_value(analysis, "correlation_raw", "raw_correlation", "correlation_pearson", "correlation"))
    if raw.shape[0] > 1 and raw.shape[1] > 1:
        raw = raw.apply(pd.to_numeric, errors="coerce")
        fig, ax = _new_fig((max(8, min(17, raw.shape[1] * .55)), max(6, min(15, raw.shape[0] * .48))))
        annot_size = min(9, max(5, 220 / max(raw.shape)))
        sns.heatmap(raw, cmap=sns.diverging_palette(10, 175, s=72, l=48, as_cmap=True), center=0, vmin=-1, vmax=1,
                    annot=True, fmt=".2f", annot_kws={"size": annot_size}, linewidths=.35, linecolor="white", cbar_kws={"label": "Pearson correlation (unitless)"}, ax=ax)
        ax.set_title("Sample return correlation", loc="left", fontweight="bold")
        ax.set_xlabel("Symbols"); ax.set_ylabel("Symbols")
        ax.tick_params(axis="both", labelsize=min(9, max(5, 200 / max(raw.shape))))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        _caption(ax, f"Raw sample correlation; requested window: {getattr(config, 'correlation_sessions', 'n/a')} sessions, minimum observations: {getattr(config, 'min_correlation_observations', 'n/a')}. Sample dates: {span or 'not supplied'}. Source: daily close returns.")
        fig.subplots_adjust(left=.18, bottom=.24, right=.92, top=.93)
        figures["correlation_raw"] = fig

    shrunk = _matrix(_analysis_value(analysis, "correlation_shrunk", "shrunk_correlation", "correlation_shrinkage"))
    if shrunk.shape[0] > 1 and shrunk.shape[1] > 1:
        shrunk = shrunk.apply(pd.to_numeric, errors="coerce")
        fig, ax = _new_fig((max(8, min(17, shrunk.shape[1] * .55)), max(6, min(15, shrunk.shape[0] * .48))))
        annot_size = min(9, max(5, 220 / max(shrunk.shape)))
        sns.heatmap(shrunk, cmap=sns.diverging_palette(10, 175, s=72, l=48, as_cmap=True), center=0, vmin=-1, vmax=1,
                    annot=True, fmt=".2f", annot_kws={"size": annot_size}, linewidths=.35, linecolor="white", cbar_kws={"label": "Correlation after shrinkage (unitless)"}, ax=ax)
        ax.set_title("Correlation after shrinkage", loc="left", fontweight="bold")
        ax.set_xlabel("Symbols"); ax.set_ylabel("Symbols")
        ax.tick_params(axis="both", labelsize=min(9, max(5, 200 / max(shrunk.shape))))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        method = _analysis_value(analysis, "correlation_shrinkage_method")
        if isinstance(method, pd.DataFrame):
            method = "; ".join(f"{key}: {value}" for key, value in method.iloc[0].items()) if not method.empty else None
        if method is None:
            method = "Method not supplied"
        _caption(ax, f"Shrinkage estimate shown separately from raw sample correlation. Sample window: {span or 'dates not supplied'}; requested {getattr(config, 'correlation_sessions', 'n/a')} sessions. Method: {method}.")
        fig.subplots_adjust(left=.18, bottom=.24, right=.92, top=.93)
        figures["correlation_shrunk"] = fig

    # Horizon heatmap uses percentage returns for equity/index series and keeps
    # yield moves in a separate panel measured in percentage points.
    index_momentum = _frame(analysis, "index_momentum")
    horizon_cols = [c for c in index_momentum.columns if c.startswith("return_") and c.endswith("s_pct")]
    if not index_momentum.empty and horizon_cols:
        labels = index_momentum.get("name", index_momentum.get("symbol", pd.Series(index=index_momentum.index, dtype=object))).fillna(index_momentum.get("symbol")).astype(str)
        is_yield = index_momentum.get("group", pd.Series("", index=index_momentum.index)).astype(str).str.lower().eq("yield")
        for label, subset, title, unit in (
            ("index_momentum_heatmap", index_momentum.loc[~is_yield].copy(), "Index and macro momentum by horizon", "Return (%)"),
            ("yield_momentum_heatmap", index_momentum.loc[is_yield].copy(), "Yield changes by horizon", "Change (percentage points)"),
        ):
            if subset.empty:
                continue
            subset[horizon_cols] = subset[horizon_cols].apply(pd.to_numeric, errors="coerce")
            subset["_label"] = labels.loc[subset.index]
            subset = subset.dropna(subset=horizon_cols, how="all").sort_values("_label")
            if subset.empty:
                continue
            matrix = subset.set_index("_label")[horizon_cols]
            fig, ax = _new_fig((max(8, len(horizon_cols) * 1.25), max(4.5, min(14, .34 * len(matrix) + 2))))
            bound = float(np.nanpercentile(np.abs(matrix.to_numpy(dtype=float)), 95)) if matrix.notna().any().any() else 1
            bound = max(bound, .1)
            sns.heatmap(matrix, cmap=sns.diverging_palette(10, 175, s=72, l=48, as_cmap=True), center=0, vmin=-bound, vmax=bound,
                        annot=True, fmt=".1f", annot_kws={"size": min(9, max(6, 180 / max(matrix.shape)))}, linewidths=.35,
                        linecolor="white", cbar_kws={"label": unit}, ax=ax)
            ax.set_title(title, loc="left", fontweight="bold"); ax.set_xlabel("Trading sessions")
            ax.set_ylabel("Series"); ax.tick_params(axis="both", labelsize=8)
            _caption(ax, f"As of {config.end}. Index returns are percentages; yields are percentage-point changes. Horizon construction follows analysis metadata. Source: index_momentum.")
            fig.tight_layout(rect=(0, .05, 1, 1)); figures[label] = fig

    # Long-share and option PCR panels are plotted only when their source fields exist.
    participant = _frame(tables, "participant_oi")
    if not participant.empty and {"date", "participant"}.issubset(participant.columns):
        participant["date"] = pd.to_datetime(participant.date, errors="coerce")
        participant = participant.dropna(subset=["date"]).loc[lambda x: x.date.between(pd.Timestamp(config.start), pd.Timestamp(config.end))].sort_values("date")
        has_long = "long_share_pct" in participant and pd.to_numeric(participant.long_share_pct, errors="coerce").notna().any()
        pcr_col = next((c for c in ("option_oi_pcr", "pcr") if c in participant), None)
        has_pcr = pcr_col is not None and pd.to_numeric(participant[pcr_col], errors="coerce").notna().any()
        if not participant.empty and (has_long or has_pcr):
            fig, axes = plt.subplots(2 if has_long and has_pcr else 1, 1, figsize=(11, 7 if has_long and has_pcr else 4.5), facecolor="white", sharex=True)
            axes = np.atleast_1d(axes)
            if has_long:
                ax = axes[0]
                for i, (name, part) in enumerate(participant.groupby("participant")):
                    part = part.sort_values("date")
                    ax.plot(part.date, pd.to_numeric(part.long_share_pct, errors="coerce"), label=str(name), color=PALETTE[i % len(PALETTE)], lw=1.5)
                ax.set_ylabel("Long share of index-futures OI (%)"); ax.legend(frameon=False, ncol=2, fontsize=8); ax.grid(axis="y", color="#E6E8EB", lw=.7)
                ax.set_title("Participant positioning", loc="left", fontweight="bold")
            if has_pcr:
                ax = axes[1] if has_long else axes[0]
                for i, (name, part) in enumerate(participant.groupby("participant")):
                    part = part.sort_values("date")
                    ax.plot(part.date, pd.to_numeric(part[pcr_col], errors="coerce"), label=str(name), color=PALETTE[i % len(PALETTE)], lw=1.5)
                ax.axhline(1, color="#98A2B3", lw=.8, ls="--"); ax.set_ylabel("Option OI PCR (ratio)"); ax.legend(frameon=False, ncol=2, fontsize=8); ax.grid(axis="y", color="#E6E8EB", lw=.7)
                if not has_long: ax.set_title("Participant option put/call ratio", loc="left", fontweight="bold")
            axes[-1].set_xlabel("Date")
            fig.autofmt_xdate(); _caption(axes[-1], f"Requested: {config.start} to {config.end}; observed: {participant.date.min().date()} to {participant.date.max().date()}. Long-share is index-futures positioning; PCR is option OI put/call ratio. Source: participant_oi.")
            fig.tight_layout(rect=(0, .06, 1, 1)); figures["participant_positioning"] = fig

    # A compact daily return heatmap highlights report-period gainers and losers.
    daily = _frame(analysis, "daily_returns")
    gainers = _frame(analysis, "period_gainers")
    losers = _frame(analysis, "period_losers")
    if not daily.empty and {"date", "symbol", "daily_return_pct"}.issubset(daily.columns):
        daily["date"] = pd.to_datetime(daily.date, errors="coerce")
        daily["daily_return_pct"] = pd.to_numeric(daily.daily_return_pct, errors="coerce")
        daily = daily.dropna(subset=["date", "daily_return_pct"]).loc[lambda x: x.date.between(pd.Timestamp(config.start), pd.Timestamp(config.end))]
        selected = []
        for source in (gainers, losers):
            if not source.empty and "symbol" in source:
                selected.extend(source.symbol.dropna().head(6).astype(str).tolist())
        selected = list(dict.fromkeys(selected))
        if selected:
            daily["symbol"] = daily.symbol.astype(str)
            daily = daily.loc[daily.symbol.isin(selected)]
        if not daily.empty:
            matrix = daily.pivot_table(index="symbol", columns="date", values="daily_return_pct", aggfunc="last")
            bound = float(np.nanpercentile(np.abs(matrix.to_numpy(dtype=float)), 95)) if matrix.notna().any().any() else 1
            bound = max(bound, .1)
            fig, ax = _new_fig((max(9, min(16, matrix.shape[1] * .7)), max(4, min(11, .34 * len(matrix) + 2))))
            sns.heatmap(matrix, cmap=sns.diverging_palette(10, 175, s=72, l=48, as_cmap=True), center=0, vmin=-bound, vmax=bound,
                        annot=True, fmt=".1f", annot_kws={"size": min(8, max(5, 160 / max(matrix.shape)))}, linewidths=.35,
                        linecolor="white", cbar_kws={"label": "Daily return (%)"}, ax=ax)
            ax.set_title("Daily stock returns", loc="left", fontweight="bold"); ax.set_xlabel("Date"); ax.set_ylabel("Symbol")
            ax.tick_params(axis="both", labelsize=7); plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            _caption(ax, f"Requested: {config.start} to {config.end}; observed: {matrix.columns.min().date()} to {matrix.columns.max().date()}. Rows prioritize period gainers and losers where supplied. Source: daily_returns.")
            fig.tight_layout(rect=(0, .06, 1, 1)); figures["daily_return_heatmap"] = fig

    momentum = _analysis_value(analysis, "stock_momentum")
    if isinstance(momentum, pd.DataFrame) and not momentum.empty and "symbol" in momentum:
        score = next((c for c in ("composite_score_pct", "period_return_pct", "return_5s_pct") if c in momentum), None)
        if score:
            mm = momentum.copy(); mm[score] = pd.to_numeric(mm[score], errors="coerce")
            mm = mm.dropna(subset=[score]).sort_values(score).tail(20)
            if not mm.empty:
                labels = mm.get("name", mm.symbol).fillna(mm.symbol).astype(str)
                fig, ax = _new_fig((10, max(5, .3 * len(mm))))
                vals = mm[score].to_numpy()
                ax.barh(np.arange(len(mm)), vals, color=[TEAL if v >= 0 else RED for v in vals])
                ax.axvline(0, color="#344054", lw=.8); ax.set_yticks(np.arange(len(mm)), labels); ax.invert_yaxis()
                ax.set_xlabel(f"{score.replace('_', ' ')} (%)")
                ax.set_title("Stock momentum", loc="left", fontweight="bold")
                ax.grid(axis="x", color="#E6E8EB", lw=.7)
                _caption(ax, f"As of report end {config.end}. Selected highest-scoring observations from stock_momentum. Source: adjusted closing-price returns.")
                fig.tight_layout(rect=(0, .04, 1, 1)); figures["stock_momentum"] = fig

    if isinstance(momentum, pd.DataFrame) and not momentum.empty and {"sector", "period_return_pct"}.issubset(momentum.columns):
        sector_data = momentum.copy()
        sector_data["period_return_pct"] = pd.to_numeric(sector_data.period_return_pct, errors="coerce")
        sector_data = sector_data.dropna(subset=["sector", "period_return_pct"])
        if not sector_data.empty:
            summary = (sector_data.groupby("sector").period_return_pct
                       .agg(median_return_pct="median", constituent_count="count")
                       .sort_values("median_return_pct"))
            y = np.arange(len(summary))
            vals = summary.median_return_pct.to_numpy(dtype=float)
            fig, ax = _new_fig((10.5, max(4.5, .42 * len(summary) + 1.8)))
            ax.barh(y, vals, color=[TEAL if value >= 0 else RED for value in vals], height=.68)
            ax.axvline(0, color="#344054", lw=.8)
            ax.set_yticks(y, summary.index.astype(str)); ax.invert_yaxis()
            ax.set_xlabel("Median constituent period return (%)")
            ax.set_title("Sector median returns", loc="left", fontweight="bold")
            ax.grid(axis="x", color="#E6E8EB", lw=.7)
            extent = max(float(np.nanmax(np.abs(vals))) if len(vals) else 0, 1)
            ax.set_xlim(min(float(np.nanmin(vals)) if len(vals) else -1, 0) - .18 * extent,
                        max(float(np.nanmax(vals)) if len(vals) else 1, 0) + .28 * extent)
            for yi, (value, count) in enumerate(zip(vals, summary.constituent_count.to_numpy())):
                offset = .03 * extent if value >= 0 else -.03 * extent
                ax.text(value + offset, yi, f"{value:+.1f}%  (n={int(count)})", va="center",
                        ha="left" if value >= 0 else "right", fontsize=8, color="#344054")
            _caption(ax, f"Requested period: {config.start} to {config.end}. Each bar is the median period return among constituents with sector metadata and a valid return; n is the observed constituent count. Source: stock_momentum.")
            fig.tight_layout(rect=(0, .05, 1, 1)); figures["sector_median_returns"] = fig

    breadth = _analysis_value(analysis, "breadth")
    if isinstance(breadth, pd.DataFrame) and not breadth.empty and "date" in breadth:
        b = breadth.copy(); b["date"] = pd.to_datetime(b.date, errors="coerce"); b = b.dropna(subset=["date"]).sort_values("date")
        if not b.empty:
            fig, ax = _new_fig((11, 5.2))
            if {"advancers", "decliners"}.issubset(b.columns):
                ax.plot(b.date, pd.to_numeric(b.advancers, errors="coerce"), color=TEAL, label="Advancers", lw=1.6)
                ax.plot(b.date, -pd.to_numeric(b.decliners, errors="coerce"), color=RED, label="Decliners (negative)", lw=1.6)
                ax.set_ylabel("Stock count")
            if "above_50dma_pct" in b.columns:
                ax2 = ax.twinx(); ax2.plot(b.date, pd.to_numeric(b.above_50dma_pct, errors="coerce"), color=BLUE, label="Above 50-day average (%)", lw=1.6)
                ax2.set_ylabel("Share above 50-day average (%)"); lines, labels = ax.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels(); ax.legend(lines+l2, labels+lab2, frameon=False, fontsize=8)
            else:
                ax.legend(frameon=False, fontsize=8)
            ax.axhline(0, color="#344054", lw=.8); ax.set_title("Market breadth", loc="left", fontweight="bold"); ax.grid(axis="y", color="#E6E8EB", lw=.7)
            fig.autofmt_xdate(); _caption(ax, f"Requested: {config.start} to {config.end}; observed: {b.date.min().date()} to {b.date.max().date()}. Advancers, decliners, and moving-average participation are distinct measures. Source: breadth analysis.")
            fig.tight_layout(rect=(0, .06, 1, 1)); figures["breadth"] = fig

    if not flows.empty and {"date", "category", "net_cr"}.issubset(flows.columns):
        data = flows.copy()
        data["date"] = pd.to_datetime(data.date, errors="coerce")
        data["net_cr"] = pd.to_numeric(data.net_cr, errors="coerce")
        data = data.dropna(subset=["date", "net_cr"])
        data = data.loc[data.date.between(pd.Timestamp(config.start), pd.Timestamp(config.end))]
        if not data.empty:
            pivot = data.pivot_table(index="date", columns="category", values="net_cr", aggfunc="sum").sort_index()
            fig, ax = _new_fig((11, 5.2))
            for i, col in enumerate(pivot.columns):
                ax.plot(pivot.index, pivot[col], label=str(col), color=PALETTE[i % len(PALETTE)], lw=1.6)
            ax.axhline(0, color="#344054", lw=.8)
            ax.set_title("Daily institutional cash flows", loc="left", fontweight="bold")
            ax.set_ylabel("Net flow (₹ crore)"); ax.grid(axis="y", color="#E6E8EB", lw=.7)
            ax.legend(frameon=False, fontsize=8)
            fig.autofmt_xdate()
            _caption(ax, f"Requested: {config.start} to {config.end}; observed: {data.date.min().date()} to {data.date.max().date()}. Daily cash flows are separate from quarterly shareholding changes. Source: flows table.")
            fig.tight_layout(rect=(0, .06, 1, 1))
            figures["institutional_flows"] = fig

    if not holdings.empty and {"symbol", "period", "fii_pct", "dii_pct"}.issubset(holdings.columns):
        h = holdings.copy()
        for c in ("fii_pct", "dii_pct"):
            h[c] = pd.to_numeric(h[c], errors="coerce")
        h["period"] = h["period"].astype(str)
        hperiod = pd.to_datetime(h["period"].str.replace(r"[^0-9-]", "", regex=True), errors="coerce")
        if hperiod.notna().any():
            h = h.loc[hperiod <= pd.Timestamp(config.end)]
        if h[["fii_pct", "dii_pct"]].notna().any().any():
            h = h.sort_values(["symbol", "period"]).groupby("symbol", as_index=False).tail(1)
            h = h.sort_values("symbol").head(20)
            fig, ax = _new_fig((10, max(4.5, .28 * len(h))))
            y = np.arange(len(h))
            ax.barh(y, h.fii_pct, color=BLUE, label="FII (%)")
            ax.barh(y, h.dii_pct, left=h.fii_pct.fillna(0), color=TEAL, label="DII (%)")
            ax.set_yticks(y, h.symbol.astype(str)); ax.invert_yaxis()
            ax.set_xlabel("Reported ownership (%)")
            ax.set_title("Latest reported institutional ownership", loc="left", fontweight="bold")
            ax.legend(frameon=False); ax.grid(axis="x", color="#E6E8EB", lw=.7)
            _caption(ax, f"Latest available period by symbol as of {config.end}. Quarterly holdings are distinct from daily cash flows. Source: holdings table.")
            fig.tight_layout(rect=(0, .04, 1, 1))
            figures["institutional_ownership"] = fig

    if not delivery.empty:
        date_col = "date" if "date" in delivery else None
        possible = [c for c in ("delivery_pct", "delivery_percentage", "delivery") if c in delivery]
        sym_col = "symbol" if "symbol" in delivery else None
        if date_col and possible and sym_col:
            col = possible[0]
            d = delivery.copy(); d[date_col] = pd.to_datetime(d[date_col], errors="coerce"); d[col] = pd.to_numeric(d[col], errors="coerce")
            d = d.dropna(subset=[date_col, col]).loc[lambda x: x[date_col].between(pd.Timestamp(config.start), pd.Timestamp(config.end))]
            if not d.empty:
                fig, ax = _new_fig((11, 5.2))
                for i, (sym, part) in enumerate(d.groupby(sym_col)):
                    if i >= 8: break
                    part = part.sort_values(date_col)
                    ax.plot(part[date_col], part[col], label=str(sym), color=PALETTE[i % len(PALETTE)], lw=1.4)
                ax.set_title("Delivery activity", loc="left", fontweight="bold")
                ax.set_ylabel("Delivery percentage (%)"); ax.grid(axis="y", color="#E6E8EB", lw=.7); ax.legend(frameon=False, ncol=2, fontsize=8)
                fig.autofmt_xdate(); _caption(ax, f"Requested: {config.start} to {config.end}; observed: {d[date_col].min().date()} to {d[date_col].max().date()}. Delivery percentage is separate from breadth. Source: delivery table.")
                fig.tight_layout(rect=(0, .06, 1, 1)); figures["delivery"] = fig

    q_latest = _frame(analysis, "quarterly_latest")
    q = q_latest if not q_latest.empty else quarterly.copy()
    if not q.empty and {"symbol", "period", "revenue_yoy_pct", "profit_yoy_pct"}.issubset(q.columns):
        q = q.copy()
        for c in ("revenue_yoy_pct", "profit_yoy_pct"):
            q[c] = pd.to_numeric(q[c], errors="coerce")
        q = q.dropna(subset=["revenue_yoy_pct", "profit_yoy_pct"])
        if q_latest.empty:
            qperiod = pd.to_datetime(q["period"].astype(str).str.replace(r"[^0-9-]", "", regex=True), errors="coerce")
            if qperiod.notna().any():
                q = q.loc[qperiod <= pd.Timestamp(config.end)]
        if not q.empty:
            if q_latest.empty:
                q = q.sort_values("period").groupby("symbol", as_index=False).tail(1)
            q = q.sort_values("symbol").head(30)
            fig, ax = _new_fig((10, max(5, .26 * len(q))))
            y = np.arange(len(q)); width = .38
            ax.barh(y-width/2, q.revenue_yoy_pct, height=width, color=BLUE, label="Revenue YoY (%)")
            ax.barh(y+width/2, q.profit_yoy_pct, height=width, color=TEAL, label="Profit YoY (%)")
            ax.axvline(0, color="#344054", lw=.8)
            labels = q.symbol.astype(str) + " · " + q.period.astype(str)
            ax.set_yticks(y, labels); ax.invert_yaxis()
            ax.set_title("Latest reported quarterly growth", loc="left", fontweight="bold")
            ax.set_xlabel("Year-over-year change (%)"); ax.legend(frameon=False); ax.grid(axis="x", color="#E6E8EB", lw=.7)
            fetched = pd.to_datetime(q["fetched_at"], errors="coerce").dropna() if "fetched_at" in q else pd.Series(dtype="datetime64[ns]")
            fetched_note = f"Snapshot fetched: {fetched.min().date()} to {fetched.max().date()}. " if not fetched.empty else "Fetch date not supplied. "
            periods = ", ".join(q.period.dropna().astype(str).unique()[:8])
            _caption(ax, f"Reported quarters: {periods or 'not supplied'}, selected by period label on or before {config.end}. {fetched_note}Filing or release date is not inferred. Source: quarterly_latest / quarterly table.")
            fig.tight_layout(rect=(0, .04, 1, 1)); figures["quarterly_growth"] = fig

    from .scope_charts import build_scope_charts
    figures.update(build_scope_charts(tables, analysis, config))
    from .theme_charts import build_theme_charts
    figures.update(build_theme_charts(tables, analysis, config))
    from .state_charts import build_state_charts
    figures.update(build_state_charts(tables, analysis, config))
    from .release_charts import build_release_charts
    figures.update(build_release_charts(tables, analysis, config))
    return figures
