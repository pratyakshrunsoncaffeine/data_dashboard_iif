import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns


def build_theme_charts(tables, analysis, config):
    figures = {}
    latest = analysis.get("niche_macro_latest", pd.DataFrame())
    if not latest.empty:
        for window, rows in latest.groupby("window_sessions"):
            for measure, title, bound in (("rho_latest", "Macro correlations", 1), ("rho_change", "Change in macro correlation versus prior non-overlapping window", 2)):
                panel = rows.pivot_table(index="theme", columns="macro_name", values=measure, aggfunc="last").dropna(how="all").sort_index()
                for offset in range(0, len(panel), 24):
                    part = panel.iloc[offset:offset + 24]
                    fig, ax = plt.subplots(figsize=(12, max(5, .32 * len(part) + 2)), facecolor="white")
                    sns.heatmap(part, cmap="RdBu_r", center=0, vmin=-bound, vmax=bound, annot=True, fmt=".2f", annot_kws={"size": 7}, linewidths=.4, cbar_kws={"label": "Pearson r" if measure == "rho_latest" else "Change in r"}, ax=ax)
                    ax.set_title(f"{title}: {int(window)} sessions", loc="left", fontsize=11)
                    ax.set_xlabel("Macro factor"); ax.set_ylabel("IIF research basket")
                    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
                    ax.tick_params(axis="y", labelsize=8)
                    fig.text(.02, .01, f"Through {config.end}. Source: adjusted equity prices and dated macro market series. Yields use percentage-point changes.\nMinimum paired observations and basket membership are in the exported tables. Descriptive association; publication vintages are not inferred.", fontsize=7, color="#555")
                    fig.tight_layout(rect=(0, .07, 1, 1))
                    figures[f"niche_macro_{measure}_{int(window)}_{offset // 24 + 1}"] = fig
    performance = analysis.get("niche_performance", pd.DataFrame())
    measures = [name for name in ("report_return_pct", "return_1d_pct", "return_5d_pct", "return_21d_pct", "return_63d_pct", "return_126d_pct") if name in performance]
    if not performance.empty and measures:
        panel = performance.set_index("theme")[measures].dropna(how="all").sort_index()
        for offset in range(0, len(panel), 24):
            part = panel.iloc[offset:offset + 24].rename(columns={"report_return_pct": "Report period", "return_1d_pct": "1 session", "return_5d_pct": "5 sessions", "return_21d_pct": "21 sessions", "return_63d_pct": "63 sessions", "return_126d_pct": "126 sessions"})
            bound = max(1, float(np.nanpercentile(np.abs(part.to_numpy()), 95)))
            fig, ax = plt.subplots(figsize=(11, max(5, .32 * len(part) + 2)), facecolor="white")
            sns.heatmap(part, cmap="RdBu_r", center=0, vmin=-bound, vmax=bound, annot=True, fmt=".1f", annot_kws={"size": 8}, cbar_kws={"label": "Basket return (%)"}, ax=ax)
            ax.set_title("Niche-sector basket returns", loc="left", fontsize=11)
            ax.set_xlabel("Return horizon"); ax.set_ylabel("IIF research basket"); ax.tick_params(labelsize=8)
            fig.text(.02, .01, f"Report period: {config.start} to {config.end}. Current constituents; equal daily weights with minimum coverage.\nSource: disclosed IIF constituent mapping and observed adjusted equity closes. These are research proxies, with independent weights.", fontsize=7, color="#555")
            fig.tight_layout(rect=(0, .07, 1, 1))
            figures[f"niche_returns_{offset // 24 + 1}"] = fig
    rolling = analysis.get("niche_macro_rolling", pd.DataFrame())
    if not rolling.empty and not latest.empty:
        selected = latest.loc[latest.window_sessions.eq(60)].dropna(subset=["rho_change"]).sort_values("rho_change", key=lambda values: values.abs(), ascending=False).head(6)
        if not selected.empty:
            fig, axes = plt.subplots(len(selected), 1, figsize=(11, 2.8 * len(selected)), squeeze=False, facecolor="white")
            for ax, row in zip(axes[:, 0], selected.itertuples()):
                part = rolling.loc[rolling.theme.eq(row.theme) & rolling.macro_name.eq(row.macro_name) & rolling.window_sessions.eq(60)].copy()
                part["date"] = pd.to_datetime(part.date)
                part = part.sort_values("date")
                ax.plot(part.date, part.rho, color="#355c7d", lw=1.4)
                ax.axhline(0, color="#888", lw=.6); ax.set_ylim(-1, 1)
                ax.axvspan(pd.Timestamp(config.start), pd.Timestamp(config.end), color="#eee", alpha=.7)
                ax.set_title(f"{row.theme} / {row.macro_name}", loc="left", fontsize=9)
                ax.set_ylabel("60-session r"); ax.grid(axis="y", alpha=.25)
                ax.xaxis.set_major_locator(mdates.AutoDateLocator()); ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
            fig.text(.02, .005, f"Six largest absolute changes versus the prior non-overlapping 60-session window at {config.end}; the selection is descriptive.\nShading marks the report period. Source: adjusted equity prices and dated macro market series. See sample-size and coverage tables.", fontsize=7, color="#555")
            fig.tight_layout(rect=(0, .035, 1, 1))
            figures["niche_macro_rolling_changes"] = fig
    return figures
