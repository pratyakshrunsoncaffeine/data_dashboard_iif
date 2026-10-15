"""Streamlit front end for generating and reviewing IIF research runs."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
from io import BytesIO
from pathlib import Path
import sys
import zipfile
import numpy as np

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from iif_research.config import ResearchConfig
from iif_research.pipeline import load_run, run_research


st.set_page_config(page_title="data_dashboard_iif", page_icon="📊", layout="wide")
st.title("data_dashboard_iif")
st.info("Cloud outputs use temporary storage. Download each research pack before ending the session.")
st.markdown("<style>[data-testid='stMetric']{background:#F5F8F5;border:1px solid #E2E9E4;border-radius:10px;padding:12px 14px;min-height:92px} [data-testid='stMetricLabel']{color:#62746B}</style>", unsafe_allow_html=True)


def _runs(config: ResearchConfig) -> list[Path]:
    if not config.output_dir.exists():
        return []
    return sorted((p for p in config.output_dir.iterdir() if p.is_dir() and (p / "run.json").is_file()), key=lambda p: p.stat().st_mtime, reverse=True)


def _frame(mapping, key):
    value = (mapping or {}).get(key)
    return value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value or [])


def _overview_figure(result):
    cfg = result["config"]
    tables = result["tables"]
    analysis = result["analysis"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 7.4), constrained_layout=True, facecolor="white")
    ink, muted, grid, green, teal, blue, red = "#18332B", "#62746B", "#E2E9E4", "#5C9E43", "#168C78", "#416C8A", "#C96B5B"

    markets = _frame(tables, "markets")
    if {"date", "name", "close"}.issubset(markets.columns):
        markets["date"] = pd.to_datetime(markets["date"], errors="coerce")
        markets["close"] = pd.to_numeric(markets["close"], errors="coerce")
        markets = markets.loc[markets.date.between(pd.Timestamp(cfg.start), pd.Timestamp(cfg.end))].dropna(subset=["date", "close"])
        selected = [("Nifty 50", blue), ("Nifty Bank", teal), ("Nifty Midcap 100", green), ("S&P 500", red), ("Nasdaq", "#9B79A8")]
        ax = axes[0, 0]
        plotted = 0
        for name, color in selected:
            frame = markets.loc[markets.name.astype(str).str.casefold().eq(name.casefold())].sort_values("date")
            if not frame.empty:
                values = frame.close / frame.close.iloc[0] * 100
                ax.plot(frame.date, values, linewidth=1.8, color=color, label=name)
                plotted += 1
        ax.axhline(100, color=muted, linestyle=(0, (4, 3)), linewidth=0.8)
        ax.set_title("Relative index performance", loc="left", color=ink, fontweight="bold")
        ax.set_ylabel("Index (start = 100)", color=muted)
        if plotted:
            ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper left")
        else:
            ax.text(.5, .5, "No index history in this report", ha="center", va="center", transform=ax.transAxes, color=muted)
    else:
        ax = axes[0, 0]
        ax.set_title("Relative index performance", loc="left", color=ink, fontweight="bold")
        ax.text(.5, .5, "No index history in this report", ha="center", va="center", transform=ax.transAxes, color=muted)

    breadth = _frame(analysis, "breadth")
    ax = axes[0, 1]
    if {"date", "advancers", "decliners"}.issubset(breadth.columns):
        breadth["date"] = pd.to_datetime(breadth["date"], errors="coerce")
        breadth = breadth.loc[breadth.date.between(pd.Timestamp(cfg.start), pd.Timestamp(cfg.end))].sort_values("date")
        denominator = pd.to_numeric(breadth.advancers, errors="coerce").fillna(0) + pd.to_numeric(breadth.decliners, errors="coerce").fillna(0)
        share = pd.to_numeric(breadth.advancers, errors="coerce").div(denominator.replace(0, np.nan)).mul(100)
        ax.plot(breadth.date, share.rolling(5, min_periods=2).mean(), color=teal, linewidth=1.8, label="5-day avg advancing share")
        if "above_50dma_pct" in breadth:
            ax.plot(breadth.date, pd.to_numeric(breadth.above_50dma_pct, errors="coerce"), color=blue, linewidth=1.5, label="Above 50-day average")
        if "above_200dma_pct" in breadth:
            ax.plot(breadth.date, pd.to_numeric(breadth.above_200dma_pct, errors="coerce"), color=green, linewidth=1.5, label="Above 200-day average")
        ax.set_ylim(0, 100)
        ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    else:
        ax.text(.5, .5, "No breadth observations in this report", ha="center", va="center", transform=ax.transAxes, color=muted)
    ax.set_title("Breadth and trend participation", loc="left", color=ink, fontweight="bold")
    ax.set_ylabel("Share of stocks (%)", color=muted)

    flows = _frame(tables, "flows")
    ax = axes[1, 0]
    if {"date", "category", "net_cr"}.issubset(flows.columns):
        flows["date"] = pd.to_datetime(flows["date"], errors="coerce")
        flows["net_cr"] = pd.to_numeric(flows["net_cr"], errors="coerce")
        flows = flows.loc[flows.date.between(pd.Timestamp(cfg.start), pd.Timestamp(cfg.end))].dropna(subset=["date", "net_cr"])
        for category, color in [("DII", teal), ("FII", blue)]:
            frame = flows.loc[flows.category.astype(str).str.upper().eq(category)].groupby("date", as_index=False).net_cr.sum().sort_values("date")
            if not frame.empty:
                ax.plot(frame.date, frame.net_cr.cumsum() / 1000, color=color, linewidth=1.8, label=category)
        ax.axhline(0, color=muted, linewidth=0.8)
        ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    else:
        ax.text(.5, .5, "No institutional flow data in this report", ha="center", va="center", transform=ax.transAxes, color=muted)
    ax.set_title("Cumulative institutional flows · observed daily history", loc="left", color=ink, fontweight="bold")
    ax.set_ylabel("₹ thousand crore", color=muted)

    niche = _frame(analysis, "niche_performance")
    ax = axes[1, 1]
    if {"theme", "report_return_pct", "availability", "selected_members"}.issubset(niche.columns):
        niche["report_return_pct"] = pd.to_numeric(niche["report_return_pct"], errors="coerce")
        niche = niche.loc[niche.availability.eq("available") & niche.selected_members.ge(3)].dropna(subset=["report_return_pct"])
        if not niche.empty:
            selected_niche = pd.concat([niche.nlargest(4, "report_return_pct"), niche.nsmallest(4, "report_return_pct")]).drop_duplicates("theme").sort_values("report_return_pct")
            colors = [green if value >= 0 else red for value in selected_niche.report_return_pct]
            ax.barh(selected_niche.theme, selected_niche.report_return_pct, color=colors, height=.66)
            ax.axvline(0, color=muted, linewidth=.8)
    else:
        ax.text(.5, .5, "No eligible niche baskets in this report", ha="center", va="center", transform=ax.transAxes, color=muted)
    ax.set_title("Niche-theme return leaders and laggards", loc="left", color=ink, fontweight="bold")
    ax.set_xlabel("Return over report period (%)", color=muted)

    for ax in axes.flat:
        ax.grid(axis="y", color=grid, linewidth=.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=muted, labelsize=7.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        if ax in (axes[0, 0], axes[0, 1]):
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))
    return fig


def _chart(chart_name, title=None):
    result = st.session_state.get("research_result")
    if not result:
        st.info("No report loaded. Run a collection or select an existing report in the sidebar.")
        return
    image = Path(result["output_path"]) / "charts" / f"{chart_name}.png"
    if not image.exists():
        st.info(f"{title or chart_name.replace('_', ' ').title()} is unavailable in this run because the required observations are missing.")
        return
    st.image(str(image), use_container_width=True)


def _table(name, title=None, analysis=False):
    result = st.session_state.get("research_result")
    if not result:
        st.info("No report loaded. Run a collection or select an existing report in the sidebar.")
        return
    frame = _frame(result["analysis"] if analysis else result["tables"], name)
    st.subheader(title or name.replace("_", " ").title())
    if frame.empty:
        st.info("This section is unavailable in the selected run.")
    else:
        st.dataframe(frame, use_container_width=True, hide_index=True)


def _ensure_zip(run_path: Path) -> Path:
    destination = run_path / "download_bundle.zip"
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in run_path.rglob("*"):
            if file.is_file() and file.suffix != ".zip":
                archive.write(file, file.relative_to(run_path))
    return destination


def _bundle_for(run_path: Path) -> Path:
    """Reuse an already finalized run bundle; create a fallback only if absent."""
    for filename in ("research_pack.zip", "download_bundle.zip"):
        candidate = run_path / filename
        if candidate.is_file():
            return candidate
    return _ensure_zip(run_path)


def _chart_png_bundle(run_path: Path) -> bytes:
    out = BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted((run_path / "charts").glob("*.png")):
            archive.write(file, file.name)
    return out.getvalue()


today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
default_start = (pd.Timestamp(today) - pd.DateOffset(months=4)).date().isoformat()
base = ResearchConfig(start=default_start, end=today.isoformat())
existing = _runs(base)
# Load the newest checksum-valid saved report once per browser session. This is
# local file I/O only: no collector or external source is called at startup.
if not st.session_state.get("research_result") and not st.session_state.get("initial_run_load_checked"):
    st.session_state["initial_run_load_checked"] = True
    for candidate in existing:
        try:
            loaded = load_run(candidate)
            loaded["zip_path"] = _bundle_for(candidate)
            st.session_state["research_result"] = loaded
            break
        except Exception:
            continue
with st.sidebar:
    st.header("Research run")
    with st.form("run_form"):
        universe = st.selectbox("Universe", ["nifty50", "nifty100", "nifty200", "nifty500"], index=3)
        start = st.date_input("Start date", value=pd.Timestamp(base.start).date(), max_value=today)
        end = st.date_input("End date", value=min(pd.Timestamp(base.end).date(), today), max_value=today)
        fundamental_count = st.number_input("Fundamental companies", min_value=1, max_value=500, value=500, step=1,
                                             help="Defaults to all 500 current constituents; lower this count for faster focused runs.")
        correlation_count = st.number_input("Correlation stocks", min_value=2, max_value=40, value=24, step=1)
        correlation_sessions = st.number_input("Correlation lookback sessions", min_value=60, max_value=756, value=126, step=1)
        refresh = st.checkbox("Refresh cached data", value=False)
        submitted = st.form_submit_button("Collect and build report", type="primary", use_container_width=True)
    st.divider()
    st.subheader("Existing reports")
    current_runs = _runs(base)
    choices = ["Select a saved report"] + [f"{p.name}  ·  {p}" for p in current_runs]
    selected = st.selectbox("Load existing report", choices, index=0)
    load_clicked = st.button("Load selected report", use_container_width=True, disabled=selected == choices[0])

if submitted:
    if start > end:
        st.error("Start date must be on or before end date.")
    else:
        try:
            cfg = ResearchConfig(start=start.isoformat(), end=end.isoformat(), universe=universe,
                                 fundamental_count=int(fundamental_count),
                                 correlation_count=int(correlation_count),
                                 correlation_sessions=int(correlation_sessions), refresh=refresh)
            progress_area = st.status("Starting research run", expanded=True)
            def progress(message):
                progress_area.write(message)
            result = run_research(cfg, progress=progress)
            # Rebuild bundle after pipeline has written checksummed saved_run data and run.json.
            zip_path = _bundle_for(Path(result["output_path"]))
            result["zip_path"] = zip_path
            st.session_state["research_result"] = result
            progress_area.update(label="Research report ready", state="complete", expanded=False)
            st.success(f"Report created: {result['output_path']}")
        except Exception as exc:
            st.exception(exc)

if load_clicked:
    selected_path = Path(selected.split("  ·  ", 1)[1])
    try:
        loaded = load_run(selected_path)
        loaded["zip_path"] = _bundle_for(selected_path)
        st.session_state["research_result"] = loaded
        st.success(f"Loaded saved report: {selected_path.name}")
    except Exception as exc:
        st.exception(exc)

result = st.session_state.get("research_result")
if result:
    run_path = Path(result["output_path"])
    if not result.get("zip_path"):
        result["zip_path"] = _bundle_for(run_path)
    zip_path = Path(result["zip_path"])
    with st.sidebar:
        st.markdown(f"**Loaded:** `{run_path.name}`")
        st.download_button("Download full report ZIP", data=zip_path.read_bytes(), file_name=f"{run_path.name}.zip", mime="application/zip", use_container_width=True)
        pngs = sorted((run_path / "charts").glob("*.png"))
        if pngs:
            st.download_button("Download chart PNGs (300 dpi)", data=_chart_png_bundle(run_path), file_name=f"{run_path.name}_charts_300dpi.zip", mime="application/zip", use_container_width=True)

sections = ["Overview", "Institutions", "Momentum", "Results", "Correlation", "Breadth/Delivery", "Global/Macro", "Niche/Macro", "Events", "Market state", "Data coverage"]
section = st.radio("Dashboard sections", sections, horizontal=True, label_visibility="collapsed", key="active_dashboard_section")

if section == "Overview":
    st.header("Market dashboard")
    if result:
        cfg = result["config"]
        prices = _frame(result["tables"], "prices")
        covered = 0
        if {"date", "symbol", "close"}.issubset(prices):
            observed = pd.to_datetime(prices.date, errors="coerce").between(pd.Timestamp(cfg.start), pd.Timestamp(cfg.end))
            covered = prices.loc[observed & pd.to_numeric(prices.close, errors="coerce").notna(), "symbol"].nunique()
        members = len(_frame(result["tables"], "members"))
        index_momentum = _frame(result["analysis"], "index_momentum")
        nifty = index_momentum.loc[index_momentum.get("symbol", pd.Series(dtype=str)).astype(str).eq("^NSEI")]
        nifty_return = pd.to_numeric(nifty.get("period_return_pct", pd.Series(dtype=float)), errors="coerce").dropna()
        flow_summary = _frame(result["analysis"], "flow_summary")
        dii = pd.to_numeric(flow_summary.loc[flow_summary.get("category", pd.Series(dtype=str)).eq("DII"), "net_cr"], errors="coerce") if {"category", "net_cr"}.issubset(flow_summary.columns) else pd.Series(dtype=float)
        fii = pd.to_numeric(flow_summary.loc[flow_summary.get("category", pd.Series(dtype=str)).eq("FII"), "net_cr"], errors="coerce") if {"category", "net_cr"}.issubset(flow_summary.columns) else pd.Series(dtype=float)
        flow_dates = pd.to_datetime(_frame(result["tables"], "flows").get("date", pd.Series(dtype=str)), errors="coerce").dropna()
        flow_period = f"{flow_dates.min():%d %b}–{flow_dates.max():%d %b}" if not flow_dates.empty else "no observations"
        breadth = _frame(result["analysis"], "breadth")
        breadth_latest = breadth.sort_values("date").iloc[-1] if not breadth.empty and "date" in breadth else None
        st.caption(f"{cfg.universe.upper()}  ·  {pd.Timestamp(cfg.start):%d %b} to {pd.Timestamp(cfg.end):%d %b %Y}  ·  {covered:,}/{members:,} constituents with price data")
        cards = st.columns(5)
        cards[0].metric("Nifty 50 · period", f"{nifty_return.iloc[-1]:+.1f}%" if not nifty_return.empty else "—")
        cards[1].metric(f"DII net flows · {flow_period}", f"₹{dii.iloc[-1] / 1000:+.1f}k cr" if not dii.empty else "—")
        cards[2].metric(f"FII net flows · {flow_period}", f"₹{fii.iloc[-1] / 1000:+.1f}k cr" if not fii.empty else "—")
        cards[3].metric("Above 50-day avg.", f"{float(breadth_latest['above_50dma_pct']):.0f}%" if breadth_latest is not None and "above_50dma_pct" in breadth_latest else "—")
        cards[4].metric("Price coverage", f"{covered:,}/{members:,}")
        st.divider()
        overview_fig = _overview_figure(result)
        st.pyplot(overview_fig, width="stretch")
        plt.close(overview_fig)
        with st.expander("Open full-period market context"):
            _chart("market_levels", "Global and macro levels")
            _chart("constituent_performance", "Constituent performance")
        html_report = run_path / "report.html"
        if html_report.exists():
            st.download_button("Download print-friendly HTML report", data=html_report.read_bytes(), file_name="iif_research_report.html", mime="text/html")
        notes = run_path / "newsletter_notes.md"
        if notes.exists():
            with st.expander("Newsletter notes: deterministic data facts"):
                st.markdown(notes.read_text(encoding="utf-8"))
    else:
        st.info("Run the collector or load an existing report. The dashboard does not collect data on startup.")

if section == "Institutions":
    st.header("Institutions")
    _chart("institutional_flows", "Daily cash market flows")
    _chart("participant_positioning", "Participant long-share positioning and option PCR")
    _table("participant_oi", "Daily participant open interest")
    st.caption("Daily cash flows and quarterly ownership disclosures are separate data series.")
    _table("flows", "Daily institutional cash flows")
    _chart("institutional_ownership", "Latest quarterly ownership")
    _table("holdings_changes", "Quarterly ownership changes", analysis=True)
    _table("official_ownership_changes", "Ownership changes from dated NSE disclosures", analysis=True)
    _table("official_holdings", "Official ownership disclosures and filing dates")
    _table("holdings", "Reported holdings")

if section == "Momentum":
    st.header("Momentum")
    _chart("stock_momentum", "Stock momentum")
    _chart("index_momentum_heatmap", "Index and macro horizon returns")
    _chart("yield_momentum_heatmap", "Yield changes by horizon")
    _table("stock_momentum", "Derived stock momentum", analysis=True)
    _table("index_momentum", "Index and macro momentum", analysis=True)

if section == "Results":
    st.header("Results")
    _table("period_prices", "Period price performance", analysis=True)
    _chart("daily_return_heatmap", "Daily stock returns")
    _table("period_gainers", "Period gainers", analysis=True)
    _table("period_losers", "Period losers", analysis=True)
    _table("highs_lows", "Highs and lows", analysis=True)
    _table("volume_anomalies", "Volume anomalies", analysis=True)
    st.subheader("Quarterly results and fundamentals")
    st.caption("These are quarterly snapshots selected by reporting-period labels. Collection timestamps describe when data was fetched; they do not verify an earnings release date and do not provide point-in-time release coverage.")
    _chart("quarterly_growth", "Latest reported quarterly growth")
    _table("quarterly_latest", "Latest quarter by symbol", analysis=True)
    _table("official_reported_results", "Numeric NSE results released in the selected date range", analysis=True)
    _chart("official_earnings_map", "Reported earnings growth and margin")
    _table("official_quarterly_latest", "Latest official quarterly earnings and exact prior-year comparisons", analysis=True)
    _table("official_quarterly_history", "Official quarterly financial history")
    _table("official_filings", "Dated NSE financial filing inventory")
    _table("quarterly_history", "Quarterly history", analysis=True)

if section == "Correlation":
    st.header("Correlation")
    _chart("correlation_raw", "Raw sample correlation")
    _table("correlation_pair_summary", "Correlation pair statistics", analysis=True)
    _chart("correlation_shrunk", "Correlation after shrinkage")
    _table("correlation_universe", "Correlation universe and exclusions", analysis=True)
    _table("rolling_pair_correlation", "Rolling pair correlation", analysis=True)
    if result:
        st.caption(f"Requested window: {result['config'].correlation_sessions} sessions; minimum observations: {result['config'].min_correlation_observations}. Sample correlations use available paired daily returns and may cover a shorter actual date span.")

if section == "Breadth/Delivery":
    st.header("Breadth and delivery")
    _chart("breadth", "Market breadth")
    _table("breadth", "Daily breadth indicators", analysis=True)
    _chart("delivery", "Delivery activity")
    _table("delivery", "Daily delivery observations")

if section == "Global/Macro":
    st.header("Global and macro")
    st.caption("All series are displayed as dated observations with their source units. Publication availability and fetched timestamps are preserved in the source tables; these are research records rather than live quotes.")
    _chart("global_gdp_latest", "Latest GDP observations by country")
    _chart("global_annual_cpi_latest", "Latest annual CPI observations by country")
    _chart("us_monthly_cpi", "United States monthly CPI")
    _chart("us_monthly_unemployment", "United States monthly unemployment")
    _chart("us_monthly_policy_rate", "United States monthly policy rate")
    _chart("india_monthly_cpi", "India monthly CPI inflation")
    _chart("india_monthly_iip", "India monthly industrial production")
    _chart("india_quarterly_gdp", "India quarterly GDP growth")
    _chart("india_monthly_bond_yield", "India monthly 10-year government bond yield")
    _table("global_economy", "Global economic observations")
    _table("economic_calendar", "Economic release calendar")
    st.subheader("Market and index observations")
    _chart("nse_index_return_heatmap", "NSE all-index returns at the latest snapshot date")
    _table("index_snapshot", "NSE all-index snapshot")
    _chart("market_levels", "Global and Indian market performance")
    _chart("yield_changes", "Yield changes")
    _chart("india_vix", "India VIX")
    _table("macro_correlations", "Macro correlations", analysis=True)
    _table("markets", "Market and macro observations")
    _table("macro", "Macro data")
    st.subheader("Long history and breadth by size")
    _chart("nifty_drawdown", "Nifty 50 drawdown from running peak")
    _chart("nifty_calendar_year_seasonality", "Nifty 50 calendar-year seasonality")
    _chart("breadth_by_size", "Observed breadth by size group")
    _table("long_index_history", "Long index history")
    _table("members_by_size", "Published members by size group")

if section == "Niche/Macro":
    st.header("Niche themes and macro correlations")
    if not result:
        st.info("Load or generate a report to inspect niche baskets and macro correlations.")
    else:
        niche = result.get("analysis", {})
        latest = _frame(niche, "niche_macro_latest")
        rolling = _frame(niche, "niche_macro_rolling")
        performance = _frame(niche, "niche_performance")
        coverage = _frame(niche, "theme_coverage")
        membership = _frame(niche, "niche_membership")
        method = _frame(niche, "niche_macro_method")
        st.caption("These are analyst-curated equal-weight research proxies from the configured issuer mapping. The minimum basket rule is 3 members and 80% observed coverage. They do not reproduce TJI constituents, weights, or returns.")

        if not coverage.empty:
            available_dates = pd.to_datetime(coverage.get("date", pd.Series(dtype="object")), errors="coerce").dropna()
            cfg_start = pd.Timestamp(result["config"].start).date()
            cfg_end = pd.Timestamp(result["config"].end).date()
            default_start = max(cfg_start, available_dates.min().date()) if not available_dates.empty else cfg_start
            default_end = min(cfg_end, available_dates.max().date()) if not available_dates.empty else cfg_end
        else:
            default_start = pd.Timestamp(result["config"].start).date()
            default_end = pd.Timestamp(result["config"].end).date()
        if default_start > default_end:
            default_start = default_end
        bounds = st.date_input("Date range for theme coverage and rolling correlations", value=(default_start, default_end), key="niche_date_bounds")
        if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
            date_start, date_end = pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])
        else:
            date_start = date_end = pd.Timestamp(bounds)
        if date_start > date_end:
            st.warning("Choose a start date on or before the end date.")
        else:
            lo, hi = date_start.normalize(), date_end.normalize()
            if not coverage.empty and "date" in coverage:
                coverage_dates = pd.to_datetime(coverage.date, errors="coerce")
                coverage_view = coverage.loc[coverage_dates.between(lo, hi)].copy()
            else:
                coverage_view = coverage
            st.subheader("Theme availability and coverage")
            if coverage_view.empty:
                st.info("No daily theme coverage observations are available in the selected date range. Themes without current constituents remain explicit in the membership mapping; no TJI membership is inferred.")
            else:
                st.dataframe(coverage_view, use_container_width=True, hide_index=True)
                st.download_button("Download filtered theme coverage CSV", data=coverage_view.to_csv(index=False).encode("utf-8"),
                                   file_name="theme_coverage.csv", mime="text/csv", key="niche_coverage_csv")
                if {"theme", "available"}.issubset(coverage_view.columns):
                    latest_coverage = coverage_view.sort_values("date").groupby("theme", as_index=False).tail(1)
                    is_available = latest_coverage.available.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
                    unavailable = latest_coverage.loc[~is_available]
                    if not unavailable.empty:
                        st.warning("Unavailable or below threshold: " + "; ".join(
                            f"{r.theme}: {getattr(r, 'reason', 'minimum membership/coverage not met')}"
                            for r in unavailable.itertuples(index=False)
                        ))
            with st.expander("Source judgments and proxy membership", expanded=False):
                if membership.empty:
                    st.info("No theme membership map is available. Check config/niche_themes.csv and the saved run manifest.")
                else:
                    if "theme" in membership:
                        selected_themes = sorted(membership.theme.dropna().astype(str).unique())
                        theme_scope = st.multiselect("Themes to inspect", selected_themes, default=selected_themes, key="niche_membership_themes")
                        membership = membership.loc[membership.theme.astype(str).isin(theme_scope)]
                    st.dataframe(membership, use_container_width=True, hide_index=True)
                    st.download_button("Download selected proxy membership CSV", data=membership.to_csv(index=False).encode("utf-8"),
                                       file_name="niche_membership.csv", mime="text/csv", key="niche_membership_csv")
                if not method.empty:
                    st.markdown("**Method and limitations**")
                    st.dataframe(method, use_container_width=True, hide_index=True)
                st.caption("Classification and source links are analyst judgments. Empty rows mean no defensible current-universe proxy was selected; thin themes are shown with their actual counts. Current constituents create survivorship limitations for historical analyses.")

        st.subheader("Saved theme return and macro-correlation figures")
        charts_dir = Path(result["output_path"]) / "charts"
        with st.expander("Open saved heatmaps and overview plots", expanded=False):
            saved_names = sorted(
                p.stem for p in charts_dir.glob("*.png")
                if p.stem.startswith("niche_macro_") or p.stem.startswith("niche_returns_")
            ) if charts_dir.exists() else []
            if saved_names:
                for name in saved_names:
                    _chart(name, name.replace("niche_", "").replace("_", " ").title())
            else:
                st.info("Niche figures are unavailable in this run because there are no eligible mapped theme observations.")

        st.subheader("Selected rolling relationship")
        if rolling.empty or not {"theme", "macro_name", "window_sessions", "date", "rho"}.issubset(rolling.columns):
            st.info("Rolling theme/macro observations are unavailable. Check the coverage table for the basket eligibility reason.")
        else:
            rolling["date"] = pd.to_datetime(rolling.date, errors="coerce").dt.normalize()
            rolling = rolling.dropna(subset=["date"])
            rolling = rolling.loc[rolling.date.le(hi)] if date_start <= date_end else rolling.iloc[0:0]
            choices_frame = rolling.loc[rolling.date.ge(lo)] if not rolling.empty else rolling
            if choices_frame.empty:
                st.info("No rolling observations fall within the selected date range. Expand the date range to inspect the available window.")
            else:
                c1, c2, c3 = st.columns(3)
                theme_options = sorted(choices_frame.theme.dropna().astype(str).unique())
                selected_theme = c1.selectbox("Theme", theme_options, key="niche_roll_theme")
                theme_frame = choices_frame.loc[choices_frame.theme.astype(str).eq(selected_theme)]
                macro_options = sorted(theme_frame.macro_name.dropna().astype(str).unique())
                selected_macro = c2.selectbox("Macro factor", macro_options, key="niche_roll_macro")
                window_options = sorted(pd.to_numeric(theme_frame.loc[theme_frame.macro_name.astype(str).eq(selected_macro), "window_sessions"], errors="coerce").dropna().astype(int).unique())
                selected_window = c3.selectbox("Window (sessions)", window_options, key="niche_roll_window")

                pair_all = rolling.loc[
                    rolling.theme.astype(str).eq(selected_theme)
                    & rolling.macro_name.astype(str).eq(selected_macro)
                    & pd.to_numeric(rolling.window_sessions, errors="coerce").eq(int(selected_window))
                ].sort_values("date")
                pair_shown = pair_all.loc[pair_all.date.between(lo, hi)].copy()
                if pair_shown.empty:
                    st.info("No selected-pair observations fall within the date range.")
                else:
                    latest_pair = pair_all.loc[pair_all.date.le(hi)]
                    if not latest_pair.empty:
                        current = latest_pair.iloc[-1]
                        prior_position = len(pair_all.loc[pair_all.date.le(hi)]) - 1 - int(selected_window)
                        previous_n = int(pair_all.iloc[prior_position].get("joint_observations", 0)) if prior_position >= 0 else 0
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Current-window r", f"{current.rho:.3f}" if pd.notna(current.rho) else "Unavailable")
                        prior_r = current.get("rho_previous_nonoverlap", float("nan"))
                        m2.metric("Prior non-overlap r", f"{prior_r:.3f}" if pd.notna(prior_r) else "Unavailable")
                        m3.metric("Current joint observations", f"{int(current.get('joint_observations', 0))} / {int(selected_window)}")
                        m4.metric("Prior joint observations", f"{previous_n} / {int(selected_window)}")
                        st.caption(f"Latest selected observation date: {current.date:%Y-%m-%d}. Prior non-overlapping window ends {int(selected_window)} benchmark sessions earlier; counts are paired available returns.")
                    fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
                    ax.plot(pair_shown.date, pair_shown.rho, color="#355c7d", linewidth=1.4)
                    ax.axhline(0, color="#888888", linewidth=.8)
                    ax.set_ylim(-1, 1)
                    ax.set_ylabel(f"Pearson r ({int(selected_window)} sessions)")
                    ax.set_xlabel("Date")
                    ax.set_title(f"{selected_theme} / {selected_macro}", loc="left", fontsize=11)
                    ax.grid(axis="y", alpha=.25)
                    locator = mdates.AutoDateLocator()
                    ax.xaxis.set_major_locator(locator)
                    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                    csv = pair_shown.to_csv(index=False).encode("utf-8")
                    st.download_button("Download selected rolling series CSV", data=csv,
                                       file_name=f"{selected_theme}_{selected_macro}_{int(selected_window)}s_rolling.csv",
                                       mime="text/csv", key="niche_selected_rolling_csv")

        with st.expander("Theme performance and latest macro statistics", expanded=False):
            if not performance.empty:
                if "latest_date" in performance:
                    pdates = pd.to_datetime(performance.latest_date, errors="coerce")
                    performance_view = performance.loc[pdates.between(lo, hi)]
                else:
                    performance_view = performance
                st.dataframe(performance_view, use_container_width=True, hide_index=True)
                st.download_button("Download filtered theme performance CSV", data=performance_view.to_csv(index=False).encode("utf-8"),
                                   file_name="niche_performance.csv", mime="text/csv", key="niche_performance_csv")
            else:
                st.info("Theme performance is unavailable for this run.")
            if not latest.empty:
                if "last_date" in latest:
                    ldates = pd.to_datetime(latest.last_date, errors="coerce")
                    latest_view = latest.loc[ldates.between(lo, hi)]
                else:
                    latest_view = latest
                st.dataframe(latest_view, use_container_width=True, hide_index=True)
                st.download_button("Download filtered latest macro statistics CSV", data=latest_view.to_csv(index=False).encode("utf-8"),
                                   file_name="niche_macro_latest.csv", mime="text/csv", key="niche_latest_csv")
            else:
                st.info("Latest theme/macro statistics are unavailable for this run.")

if section == "Events":
    st.header("Events")
    _chart("official_release_timeline", "Official economic release dates")
    _table("economic_calendar", "Economic release calendar")
    _table("events", "Corporate and market events")
    _table("bse_events", "BSE announcements")

if section == "Market state":
    st.header("Market state")
    st.caption("Observed market participation, sector leadership, risk and delivery activity. Each figure uses the dates and coverage shown in its caption; unavailable inputs are identified below.")
    if not result:
        st.info("Load or generate a report to inspect market-state charts.")
    else:
        state_names = (
            ("sector_rotation", "Sector rotation observations"),
            ("stock_risk_return", "Risk and return observations"),
            ("advance_decline", "Daily breadth and phase observations"),
            ("stock_price_volume_delivery", "Delivery and liquidity observations"),
            ("nifty_market_state", "Nifty market state observations"),
            ("institution_absorption", "Institutional absorption observations"),
            ("factor_spreads", "Factor spread observations"),
            ("regime_snapshot", "Market regime snapshot"),
        )
        state_frames = [(name, title, _frame(result.get("analysis", {}), name)) for name, title in state_names]
        available = [(name, title, frame) for name, title, frame in state_frames if not frame.empty]
        if not available:
            st.info("Market-state charts are unavailable in this saved report. Generate a report with the current analysis to add them.")
        else:
            _chart("state_sector_rotation", "Sector relative strength and momentum")
            _chart("state_risk_return", "Observed 63-session risk and return")
            _chart("state_breadth_trajectory", "Cumulative and moving-average breadth")
            _chart("state_breadth_phase", "Breadth phase and participation")
            _chart("state_return_distribution", "Empirical return distribution")
            _chart("state_delivery_participation", "Delivery participation and traded value")
            _chart("state_volatility_history", "Nifty realized volatility and India VIX")
            _chart("state_nifty_candlestick", "Nifty 50 OHLC and moving averages")
            _chart("state_institution_absorption", "Institutional flow and domestic absorption")
            _chart("state_factor_spreads", "Observed factor spreads")
            for name, title, _ in available:
                _table(name, title, analysis=True)

if section == "Data coverage":
    st.header("Data coverage")
    if result:
        for name, frame in result["tables"].items():
            data = _frame(result["tables"], name)
            with st.expander(f"{name}: {len(data)} rows", expanded=data.empty):
                if data.empty:
                    st.warning("Unavailable: no rows returned for this configured source table.")
                else:
                    sources = sorted(data.source.dropna().astype(str).unique()) if "source" in data else []
                    st.caption("Sources: " + (", ".join(sources) if sources else "not supplied"))
                    st.dataframe(data.head(50), use_container_width=True, hide_index=True)
        if result.get("quality"):
            st.subheader("Quality checks")
            st.dataframe(pd.DataFrame(result["quality"]), use_container_width=True, hide_index=True)
    else:
        st.info("Load or generate a report to inspect source coverage.")
