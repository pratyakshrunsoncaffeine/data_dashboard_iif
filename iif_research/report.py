"""Portable, print-friendly research report and artifact export."""
from __future__ import annotations

import base64
from datetime import datetime
import html
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from .charts import build_charts
from .config import TABLE_NAMES


def _df(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if value is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(value)
    except (ValueError, TypeError):
        return pd.DataFrame()


def _jsonable(value: Any):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if hasattr(value, "item"):
        try: return value.item()
        except Exception: pass
    if pd.isna(value) if not isinstance(value, (dict, list, tuple, set)) else False:
        return None
    return str(value)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name)).strip("_.") or "table"


def _excel_frame(frame):
    result = frame.copy()
    for column in result:
        if isinstance(result[column].dtype, pd.DatetimeTZDtype):
            result[column] = result[column].map(lambda value: value.isoformat() if pd.notna(value) else None)
        elif result[column].dtype == object:
            result[column] = result[column].map(lambda value: value.isoformat() if isinstance(value, (pd.Timestamp, datetime)) and value.tzinfo is not None else value)
    return result


def _summary_table(name: str, frame: pd.DataFrame) -> dict[str, Any]:
    dates = {}
    for column in ("date", "fetched_at", "period"):
        if column in frame.columns and not frame[column].dropna().empty:
            vals = frame[column].dropna().astype(str)
            dates[column] = {"first": vals.min(), "last": vals.max()}
    sources = sorted(frame["source"].dropna().astype(str).unique().tolist()) if "source" in frame else []
    return {"rows": int(len(frame)), "columns": list(frame.columns), "date_coverage": dates, "sources": sources,
            "status": "available" if not frame.empty else "unavailable", "missing_values": {str(c): int(frame[c].isna().sum()) for c in frame.columns}}


def _fact_notes(tables, analysis, quality, config, coverage):
    lines = ["# IIF Research — Data Notes", "", "This file records data facts from the generated run. It does not add qualitative market interpretation.", "",
             f"- Report period: {config.start} to {config.end}", f"- Universe: {config.universe}", f"- Correlation sample request: {config.correlation_sessions} sessions; minimum {config.min_correlation_observations} observations", "",
             "## Coverage"]
    for name, info in coverage.items():
        lines.append(f"- {name}: {info['status']}; {info['rows']} rows; sources: {', '.join(info['sources']) if info['sources'] else 'not supplied'}")
    lines.extend(["", "## Deterministic observations"])
    mom = _df((analysis or {}).get("stock_momentum"))
    if not mom.empty:
        score = next((c for c in ("composite_score_pct", "period_return_pct") if c in mom), None)
        if score:
            s = pd.to_numeric(mom[score], errors="coerce").dropna()
            if not s.empty:
                lines.append(f"- Stock momentum observations: {len(s)}; median {s.median():.2f}% using `{score}`.")
    gainers = _df((analysis or {}).get("period_gainers"))
    if not gainers.empty and {"symbol", "period_return_pct"}.issubset(gainers.columns):
        top = gainers.dropna(subset=["period_return_pct"]).head(5)
        if not top.empty:
            items = "; ".join(f"{row.symbol} {float(row.period_return_pct):+.2f}%" for row in top.itertuples())
            lines.append(f"- Top period gainers: {items}.")
    index = _df((analysis or {}).get("index_momentum"))
    if not index.empty and {"name", "period_return_pct"}.issubset(index.columns):
        if "group" in index:
            index = index.loc[~index.group.astype(str).str.lower().eq("yield")]
        top = index.dropna(subset=["period_return_pct"]).sort_values("period_return_pct", ascending=False).head(5)
        if not top.empty:
            items = "; ".join(f"{row['name']} {float(row['period_return_pct']):+.2f}%" for _, row in top.iterrows())
            lines.append(f"- Top index and market series by period return: {items}.")
    if not mom.empty and {"sector", "period_return_pct"}.issubset(mom.columns):
        sectors = mom.dropna(subset=["sector", "period_return_pct"]).groupby("sector").period_return_pct.agg(["median", "count"]).sort_values("median", ascending=False).head(5)
        if not sectors.empty:
            items = "; ".join(f"{sector} median {row['median']:+.2f}% (n={int(row['count'])})" for sector, row in sectors.iterrows())
            lines.append(f"- Sector return leaders by median constituent period return: {items}.")
    breadth = _df((analysis or {}).get("breadth"))
    if not breadth.empty and {"advancers", "decliners"}.issubset(breadth.columns):
        latest = breadth.sort_values("date").iloc[-1] if "date" in breadth else breadth.iloc[-1]
        lines.append(f"- Latest supplied breadth row: {int(latest['advancers']) if pd.notna(latest['advancers']) else 'missing'} advancers; {int(latest['decliners']) if pd.notna(latest['decliners']) else 'missing'} decliners.")
    flow_summary = _df((analysis or {}).get("flow_summary"))
    if not flow_summary.empty and {"category", "net_cr"}.issubset(flow_summary.columns):
        lines.append("- Daily institutional cash flows (₹ crore):")
        for row in flow_summary.itertuples():
            lines.append(f"  - {row.category}: net {float(row.net_cr):+,.2f}; buy {float(row.buy_cr):,.2f}; sell {float(row.sell_cr):,.2f}.")
        lines.append("  These daily cash flows are distinct from quarterly shareholding changes.")
    holdings = _df((analysis or {}).get("holdings_changes"))
    if not holdings.empty:
        lines.append(f"- Quarterly ownership-change observations supplied: {len(holdings)}; changes are percentage points between the displayed periods.")
        for column, label in (("fii_change_pp", "FII"), ("dii_change_pp", "DII")):
            if column in holdings:
                changed = holdings.dropna(subset=[column]).sort_values(column, key=lambda s: pd.to_numeric(s, errors="coerce")).tail(3)
                if not changed.empty:
                    changes = "; ".join(f"{r.symbol} {float(getattr(r, column)):+.2f} pp" for r in changed.itertuples())
                    lines.append(f"  - Largest reported {label} ownership increases: {changes}.")
    qlatest = _df((analysis or {}).get("quarterly_latest"))
    if not qlatest.empty and "period" in qlatest:
        periods = sorted(qlatest.period.dropna().astype(str).unique())
        fetched = pd.to_datetime(qlatest.get("fetched_at", pd.Series(dtype=str)), errors="coerce").dropna()
        lines.append(f"- Quarterly fundamentals represent reported period label(s): {', '.join(periods)}. Collection timestamp{'' if len(fetched) == 1 else 's'}: "
                     + (f"{fetched.min().isoformat()} to {fetched.max().isoformat()}" if not fetched.empty else "not supplied")
                     + ". Collection timestamps do not establish earnings release dates; this is not a point-in-time release feed.")
    official = _df((analysis or {}).get("official_reported_results"))
    if not official.empty:
        companies = official.symbol.nunique() if "symbol" in official else 0
        lines.append(f"- Official numeric earnings disclosures released within the report window: {len(official)} company/period/basis rows across {companies} companies. Fiscal period and exchange release date are retained separately; standalone and consolidated rows are separate disclosures.")
    official_ownership = _df((analysis or {}).get("official_ownership_changes"))
    if not official_ownership.empty:
        lines.append(f"- Official quarterly ownership comparisons: {len(official_ownership)} companies with two available NSE XBRL periods. Original filing links and broadcast dates are exported.")
    official_latest = _df((analysis or {}).get("official_quarterly_latest"))
    if not official_latest.empty:
        comparable = int(pd.to_numeric(official_latest.revenue_yoy_pct, errors="coerce").notna().sum())
        periods = ", ".join(sorted(official_latest.period.astype(str).unique()))
        lines.append(f"- Official earnings history: {len(official_latest)} companies with a latest numeric quarter available by report end; {comparable} exact same-basis prior-year revenue/income comparisons. Latest fiscal periods: {periods}.")
        lines.append("- Bank income and operating-company revenue retain their XBRL concept labels. Profit growth is omitted when prior profit is nonpositive; absolute profit change remains available. First-publication vintages are not reconstructed.")
    state = _df((analysis or {}).get("regime_snapshot"))
    for row in state.itertuples():
        if isinstance(row.value, (int, float)) and pd.notna(row.value):
            lines.append(f"- {row.metric}: {float(row.value):.2f} {row.unit}; observed {pd.Timestamp(row.as_of).date()}.")
    if not any(line.startswith(("- Stock momentum", "- Latest supplied", "- Daily institutional cash flows", "- Quarterly ownership", "- Top period gainers", "- Top index", "- Sector return leaders")) for line in lines):
        lines.append("- No derived numeric observation was available for the sections above.")
    missing = [name for name, info in coverage.items() if info["rows"] == 0]
    lines.extend(["", "## Missing sections"])
    if missing:
        lines.extend(f"- {name}: unavailable in this run." for name in missing)
    else:
        lines.append("- No empty configured source tables.")
    if quality:
        lines.extend(["", "## Data quality checks"])
        from collections import Counter
        counts = Counter((str(item.get("dataset", "quality check")), str(item.get("status", "reported"))) for item in quality if isinstance(item, dict))
        lines.extend(f"- {dataset}: {status}; {count} source records." for (dataset, status), count in sorted(counts.items()))
        lines.append("- Full per-source and per-symbol details are in csv/quality_records.csv and sources_coverage_manifest.json.")
    return "\n".join(lines) + "\n"


def export_report(tables, analysis, quality: list[dict], config) -> Path:
    """Write a self-contained report bundle and return its run directory."""
    output_root = Path(config.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / f"{stamp}_{_safe_name(config.universe)}"
    run_dir.mkdir(parents=True, exist_ok=False)
    assets = run_dir / "charts"
    tables_dir = run_dir / "csv"
    assets.mkdir(); tables_dir.mkdir()

    table_frames = {name: _df((tables or {}).get(name)) for name in TABLE_NAMES}
    # Include any additional pipeline tables, retaining configured tables even when absent.
    for name, value in (tables or {}).items():
        if name not in table_frames:
            table_frames[str(name)] = _df(value)
    derived_frames = {str(k): _df(v) for k, v in (analysis or {}).items() if isinstance(v, pd.DataFrame)}
    coverage = {name: _summary_table(name, frame) for name, frame in table_frames.items()}

    for name, frame in {**table_frames, **{f"analysis_{k}": v for k, v in derived_frames.items()}}.items():
        frame.to_csv(tables_dir / f"{_safe_name(name)}.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(quality or []).to_csv(tables_dir / "quality_records.csv", index=False, encoding="utf-8-sig")

    workbook = run_dir / "research_tables.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        used_sheets = set()
        for name, frame in table_frames.items():
            sheet = _safe_name(name)[:31]
            base, suffix = sheet[:27], 1
            while sheet.casefold() in used_sheets:
                sheet = f"{base}_{suffix}"[:31]; suffix += 1
            used_sheets.add(sheet.casefold())
            _excel_frame(frame).iloc[:1_048_575].to_excel(writer, sheet_name=sheet, index=False)
        for name, frame in derived_frames.items():
            if name == "niche_macro_rolling" and "date" in frame:
                frame = frame.loc[pd.to_datetime(frame.date).between(pd.Timestamp(config.start), pd.Timestamp(config.end))]
            sheet = ("A_" + _safe_name(name))[:31]
            base, suffix = sheet[:27], 1
            while sheet.casefold() in used_sheets:
                sheet = f"{base}_{suffix}"[:31]; suffix += 1
            used_sheets.add(sheet.casefold())
            _excel_frame(frame).iloc[:1_048_575].to_excel(writer, sheet_name=sheet, index=False)

    figures = build_charts(tables or {}, analysis or {}, config)
    images = {}
    for name, fig in figures.items():
        path = assets / _safe_name(name)
        png = path.with_suffix(".png")
        fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
        images[name] = base64.b64encode(png.read_bytes()).decode("ascii")
        plt.close(fig)

    manifest = {
        "report_period": {"start": config.start, "end": config.end}, "universe": config.universe,
        "correlation": {"count": config.correlation_count, "requested_sessions": config.correlation_sessions,
                        "minimum_observations": config.min_correlation_observations,
                        "sample_period": _jsonable((analysis or {}).get("correlation_period", (analysis or {}).get("correlation_dates")))},
        "tables": coverage, "quality": quality or [], "charts": sorted(figures),
        "generated_at_local": datetime.now().astimezone().isoformat(),
    }
    (run_dir / "sources_coverage_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=_jsonable), encoding="utf-8")
    notes = _fact_notes(tables or {}, analysis or {}, quality or [], config, coverage)
    (run_dir / "newsletter_notes.md").write_text(notes, encoding="utf-8")

    title = f"IIF Research | {config.universe} | {config.start} to {config.end}"
    sections = []
    order = ["market_levels", "yield_changes", "india_vix", "constituent_performance", "stock_momentum", "index_momentum_heatmap", "yield_momentum_heatmap",
             "daily_return_heatmap", "institutional_flows", "institutional_ownership", "participant_positioning",
             "correlation_raw", "correlation_shrunk", "breadth", "delivery", "quarterly_growth"]
    labels = {"market_levels":"Global and Indian market performance", "yield_changes":"Yield changes", "india_vix":"India VIX", "constituent_performance":"Constituent performance", "stock_momentum":"Stock momentum",
              "index_momentum_heatmap":"Index momentum by horizon", "yield_momentum_heatmap":"Yield changes by horizon", "daily_return_heatmap":"Daily stock returns",
              "institutional_flows":"Institutional cash flows", "institutional_ownership":"Quarterly ownership", "correlation_raw":"Raw sample correlation",
              "participant_positioning":"Participant positioning", "correlation_shrunk":"Correlation after shrinkage", "breadth":"Breadth", "delivery":"Delivery activity", "quarterly_growth":"Quarterly reported growth"}
    for name in order + sorted(set(images) - set(order)):
        if name in images:
            label = labels.get(name, name.replace("_", " ").title())
            sections.append(f'<section><h2>{html.escape(label)}</h2><img src="data:image/png;base64,{images[name]}" alt="{html.escape(label)} chart"></section>')
    for name, info in coverage.items():
        if not info["rows"]:
            sections.append(f'<section class="unavailable"><h2>{html.escape(name.replace("_", " ").title())}</h2><p>Unavailable in this run; source table is empty.</p></section>')
    from collections import Counter
    quality_counts = Counter((str(item.get("dataset", "quality check")), str(item.get("status", "reported"))) for item in quality or [] if isinstance(item, dict))
    q_html = "".join(f"<li>{html.escape(dataset)}: {html.escape(status)}; {count} source records</li>" for (dataset, status), count in sorted(quality_counts.items()))
    coverage_rows = "".join(f"<tr><td>{html.escape(name)}</td><td>{info['rows']}</td><td>{html.escape(', '.join(info['sources']) or 'Not supplied')}</td></tr>" for name, info in coverage.items())
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title><style>
body{{font:15px/1.5 Arial,sans-serif;color:#111;margin:36px auto;max-width:1100px;padding:0 24px;background:#fff}} h1{{font-size:28px;color:#111}}h2{{font-size:19px;margin-bottom:8px;color:#111}}section{{page-break-inside:avoid;margin:30px 0}}img{{display:block;width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #d8dee6;text-align:left;padding:7px}}.muted{{color:#667085}}.unavailable{{padding:0;background:#fff}}@media print{{body{{margin:12mm;max-width:none}}h1,h2{{page-break-after:avoid}}section{{page-break-inside:avoid}}}}
 </style></head><body><h1>{html.escape(title)}</h1><p class="muted">Data report generated {html.escape(datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z'))}. Figures include source and unit captions.</p>
 <h2>Coverage and sources</h2><table><thead><tr><th>Section</th><th>Rows</th><th>Sources</th></tr></thead><tbody>{coverage_rows}</tbody></table>
 {''.join(sections)}<section><h2>Data quality</h2>{('<ul>'+q_html+'</ul>') if q_html else '<p>No quality findings supplied.</p>'}</section>
 <p class="muted">Quarterly fundamentals and macro series are fetched snapshots; observation periods do not establish first-publication dates. Current stock membership can introduce survivorship bias. Ownership changes are quarterly disclosures; daily cash flows are separate. Niche baskets use independent IIF constituent judgments and equal daily weights. Workbook rolling-correlation rows cover the report period; complete history is in CSV and saved observations.</p>
 </body></html>"""
    (run_dir / "report.html").write_text(document, encoding="utf-8")
    zip_path = run_dir / "download_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in run_dir.rglob("*"):
            if file.is_file() and file != zip_path:
                zf.write(file, file.relative_to(run_dir))
    return run_dir
