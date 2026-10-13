"""Bounded NSE quarterly XBRL history for current research-universe members.

This adapter is separate from the date-window filing collector and from
Screener's quarterly snapshot table. It retains filing availability dates and
only selects XBRL filings that were public by the configured report end.
"""
from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .providers import NSE_HOME, _parse_nse_financial_xbrl, _request

METADATA_COLUMNS = [
    "symbol", "fiscal_period", "period_end", "basis", "filing_type", "audited",
    "broadcast_date", "creation_date", "revised_date", "available_at",
    "source_xbrl_url", "isin", "source",
]
RESULT_COLUMNS = [
    "symbol", "period", "fiscal_period", "basis", "fiscal_year", "release_date", "available_at",
    "revenue_cr", "net_profit_cr", "eps",
    "revenue_concept", "profit_concept", "eps_concept", "currency", "amount_unit",
    "eps_unit", "source_xbrl_url", "source", "fetched_at", "parse_status",
]
HISTORY_START = "01-04-2025"  # NSE's Integrated Filing-Financial XBRL era.
PARSER_VERSION = "nse-integrated-quarter-history-v2-bank-concepts"


def _parse_date(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce", dayfirst=not bool(re.match(r"^\d{4}-\d{2}-\d{2}", str(value))))


def _release_time(row: dict[str, Any]) -> pd.Timestamp:
    # Keep availability conservative for revised filings: the row is not
    # usable until the latest timestamp attached to that filing record.
    dates = [_parse_date(row.get(key)) for key in ("broadcast_Date", "creation_Date", "revised_Date")]
    dates = [d for d in dates if pd.notna(d)]
    return max(dates) if dates else pd.NaT


def _metadata_record(row: dict[str, Any], requested_symbol: str, end: pd.Timestamp) -> dict[str, Any] | None:
    available = _release_time(row)
    if pd.isna(available) or available.normalize() > end:
        return None
    period = str(row.get("qe_Date") or row.get("toDate") or "").strip()
    period_end = _parse_date(period)
    if pd.isna(period_end) or period_end.normalize() > end:
        return None
    return {
        "symbol": str(row.get("symbol") or requested_symbol).strip().upper(),
        "fiscal_period": period,
        "period_end": period_end,
        "basis": str(row.get("consolidated") or "").strip().title(),
        "filing_type": str(row.get("type_Sub") or "Financial results").strip(),
        "audited": str(row.get("audited") or "").strip(),
        "broadcast_date": row.get("broadcast_Date"),
        "creation_date": row.get("creation_Date"),
        "revised_date": row.get("revised_Date"),
        "available_at": available,
        "source_xbrl_url": str(row.get("xbrl") or row.get("ixbrl") or ""),
        "isin": row.get("isin"),
        "source": "NSE Integrated Filing-Financials metadata",
        "_raw": row,
    }


def _valid_xbrl(url: str) -> bool:
    return bool(url) and url.lower().endswith((".xml", ".xhtml", ".html")) and not url.rstrip("/").endswith("/-")


def _cache_root(config: Any) -> Path:
    path = Path(config.cache_dir) / "official_history" / PARSER_VERSION
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, default=str, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def _fetch_symbol_metadata(symbol: str, end: pd.Timestamp, cache: Path, refresh: bool) -> tuple[str, list[dict], str]:
    path = cache / "metadata" / f"{end.date().isoformat()}_{symbol}.json"
    if not refresh:
        cached = _read_json(path)
        if isinstance(cached, list):
            return symbol, cached, ""
    params = {
        "index": "equities", "type": "Integrated Filing- Financials",
        "from_date": HISTORY_START, "to_date": end.strftime("%d-%m-%Y"),
        "page": 1, "size": 100, "symbol": symbol,
    }
    try:
        payload = _request(f"{NSE_HOME}/api/integrated-filing-results", params=params,
                           referer=NSE_HOME, timeout=15, retries=0).json()
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("unexpected NSE integrated-filings metadata schema")
        rows = payload["data"]
        _write_json(path, rows)
        return symbol, rows, ""
    except Exception as exc:
        return symbol, [], str(exc)


def _selected_context_facts(xml: bytes, period: str) -> dict[str, tuple[str, str]]:
    """Return concept -> (unitRef, unit measure) for the selected quarter context."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml)
    units = {}
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] == "unit":
            units[node.attrib.get("id", "")] = " ".join(
                (child.text or "").strip() for child in node.iter()
                if child.tag.rsplit("}", 1)[-1] == "measure")
    target = _parse_date(period)
    candidates = []
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] != "context":
            continue
        if any(c.tag.rsplit("}", 1)[-1] in {"scenario", "segment", "explicitMember", "typedMember"}
               for c in node.iter()):
            continue
        period_node = next((c for c in node if c.tag.rsplit("}", 1)[-1] == "period"), None)
        if period_node is None:
            continue
        start = next((c.text for c in period_node if c.tag.rsplit("}", 1)[-1] == "startDate"), None)
        end = next((c.text for c in period_node if c.tag.rsplit("}", 1)[-1] == "endDate"), None)
        if not start or not end or pd.isna(target) or end != target.date().isoformat():
            continue
        start_date = _parse_date(start)
        days = (target - start_date).days + 1 if pd.notna(start_date) else 0
        if 45 <= days <= 125:
            candidates.append((days, node.attrib.get("id", "")))
    if not candidates:
        return {}
    context_id = min(candidates)[1]
    facts = {}
    for node in root.iter():
        if node.attrib.get("contextRef") == context_id:
            local = node.tag.rsplit("}", 1)[-1]
            unit_ref = node.attrib.get("unitRef", "")
            if unit_ref and local not in facts:
                facts[local] = (unit_ref, units.get(unit_ref, unit_ref))
    return facts


def _first_concept(facts: dict[str, tuple[str, str]], names: tuple[str, ...]) -> str:
    return next((name for name in names if name in facts), "")


REVENUE_CONCEPTS = ("RevenueFromOperations", "Income", "TotalIncome", "OperatingIncome")
PROFIT_CONCEPTS = (
    "ProfitLossForPeriod", "ProfitLossForPeriodFromContinuingOperations",
    "ProfitLossForThePeriod", "ProfitLossFromOrdinaryActivitiesAfterTax",
    "ProfitLossAfterTaxesMinorityInterestAndShareOfProfitLossOfAssociates",
    "ProfitLossAfterTax", "ProfitLossAfterTaxAndExtraordinaryItems",
)
EPS_CONCEPTS = (
    "BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
    "BasicEarningsLossPerShareFromContinuingOperations", "EarningsPerShareBasic",
    "BasicEarningsPerShareAfterExtraordinaryItems", "BasicEarningsPerShareBeforeExtraordinaryItems",
    "BasicAndDilutedEPSAfterExtraordinaryItemsNetOfTaxExpenseForThePeriodNotToBeAnnualized",
    "BasicAndDilutedEPSBeforeExtraordinaryItemsNetOfTaxExpenseForThePeriodNotToBeAnnualized",
)


def _basis_label(value: str) -> str:
    normalized = value.strip().lower()
    return "Consolidated" if normalized == "consolidated" else "Standalone" if normalized in {"standalone", "non-consolidated"} else value


def _choose_filings(metadata: pd.DataFrame, max_periods_per_symbol: int) -> pd.DataFrame:
    if metadata.empty:
        return metadata.copy()
    d = metadata.copy()
    d["period_end"] = pd.to_datetime(d.period_end, errors="coerce")
    d["basis"] = d.basis.map(_basis_label)
    d = d.loc[d.period_end.notna() & d.period_end.dt.is_quarter_end & d.source_xbrl_url.map(_valid_xbrl)].copy()
    if d.empty:
        return d
    d = d.sort_values(["symbol", "period_end", "available_at"])
    d = d.drop_duplicates(["symbol", "period_end", "basis"], keep="last")
    selected = []
    for symbol, group in d.groupby("symbol", sort=True):
        latest_period = group.period_end.max()
        latest_rows = group.loc[group.period_end.eq(latest_period)]
        basis = "Consolidated" if latest_rows.basis.eq("Consolidated").any() else "Standalone"
        group = group.loc[group.basis.eq(basis)].sort_values("period_end", ascending=False)
        periods = list(group.period_end.drop_duplicates())
        desired = [latest_period]
        prior_year = latest_period - pd.DateOffset(years=1)
        if prior_year in periods:
            desired.append(prior_year)
        previous = next((p for p in periods if p < latest_period and p not in desired), None)
        if previous is not None:
            desired.append(previous)
        desired = desired[:max_periods_per_symbol]
        for period in periods:
            if len(desired) >= max_periods_per_symbol:
                break
            if period not in desired:
                desired.append(period)
        selected.extend(group.loc[group.period_end.isin(desired)].to_dict("records"))
    return pd.DataFrame(selected, columns=d.columns)


def _parse_filing(filing: dict[str, Any], cache: Path, refresh: bool) -> dict[str, Any]:
    url = str(filing["source_xbrl_url"])
    key = hashlib.sha256(f"{PARSER_VERSION}|{url}".encode()).hexdigest()
    path = cache / "numeric" / f"{key}.json"
    if not refresh:
        cached = _read_json(path)
        if isinstance(cached, dict):
            return cached
    result = {col: None for col in RESULT_COLUMNS}
    result.update({
        "symbol": filing["symbol"], "period": filing["period_end"],
        "fiscal_period": filing["fiscal_period"], "basis": filing["basis"],
        "fiscal_year": str(filing.get("_raw", {}).get("financialYear") or ""),
        "release_date": filing["available_at"],
        "available_at": filing["available_at"], "source_xbrl_url": url,
        "source": "NSE Integrated Filing-Financials XBRL",
        "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    })
    try:
        xml = _request(url, referer=NSE_HOME, timeout=15, retries=0).content
        parsed = _parse_nse_financial_xbrl(xml, result["symbol"], result["fiscal_period"], result["basis"], str(result["available_at"]))
        facts = _selected_context_facts(xml, result["fiscal_period"])
        revenue_name = _first_concept(facts, REVENUE_CONCEPTS)
        profit_name = _first_concept(facts, PROFIT_CONCEPTS)
        eps_name = _first_concept(facts, EPS_CONCEPTS)
        if not parsed.empty:
            values = parsed.iloc[0]
            for column in ("revenue_cr", "net_profit_cr", "eps"):
                result[column] = values.get(column)
            result["parse_status"] = "parsed" if any(pd.notna(result[c]) for c in ("revenue_cr", "net_profit_cr", "eps")) else "no supported facts"
        else:
            result["parse_status"] = "no supported quarter facts"
        result["revenue_concept"] = revenue_name
        result["profit_concept"] = profit_name
        result["eps_concept"] = eps_name
        money_ref = (facts.get(revenue_name) or facts.get(profit_name) or ("", ""))[1]
        eps_ref = facts.get(eps_name, ("", ""))[1]
        result["amount_unit"] = money_ref
        result["eps_unit"] = eps_ref
        result["currency"] = "INR" if "inr" in money_ref.lower() or "indianrupee" in money_ref.lower() else ""
    except Exception as exc:
        result["parse_status"] = f"error: {exc}"
    _write_json(path, result)
    return result


def collect_official_history(tables: dict[str, pd.DataFrame], config: Any,
                             progress: Callable[[str], None] | None = None, *,
                             max_periods_per_symbol: int = 3,
                             workers: int = 4) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """Fetch available quarter history for current members; limit downloads to selected periods.

    For each member this queries one bounded NSE metadata endpoint. It downloads
    at most ``max_periods_per_symbol`` XBRL files at four workers by default,
    selecting the latest quarter, exact same-quarter prior fiscal year when
    present, then the preceding available quarter. Results stay in separate
    tables and never overwrite Screener snapshots.
    """
    empty = {"official_financial_history_filings": pd.DataFrame(columns=METADATA_COLUMNS),
             "official_quarterly_history": pd.DataFrame(columns=RESULT_COLUMNS)}
    members = tables.get("members", pd.DataFrame())
    if members.empty or "symbol" not in members:
        return empty, [{"dataset": "official_history", "status": "unavailable", "rows": 0,
                        "message": "No member symbols supplied."}]
    supplied = list(dict.fromkeys(members.symbol.dropna().astype(str).str.strip().str.upper()))
    supplied = [symbol for symbol in supplied if symbol not in {"", "NAN"}]
    symbols = sorted(supplied[:max(1, int(getattr(config, "fundamental_count", len(supplied))))])
    end = pd.Timestamp(config.end).normalize()
    cache = _cache_root(config)
    metadata_rows: list[dict[str, Any]] = []
    metadata_errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 4))) as pool:
        futures = {pool.submit(_fetch_symbol_metadata, symbol, end, cache, bool(config.refresh)): symbol for symbol in symbols}
        for index, future in enumerate(as_completed(futures), 1):
            symbol, raw_rows, error = future.result()
            if error:
                metadata_errors.append(f"{symbol}: {error}")
            for row in raw_rows:
                parsed = _metadata_record(row, symbol, end)
                if parsed is not None:
                    metadata_rows.append(parsed)
            if progress and (index % 25 == 0 or index == len(symbols)):
                progress(f"Official NSE quarterly metadata: {index}/{len(symbols)} members; {len(metadata_rows)} in-range filings")
    metadata = pd.DataFrame(metadata_rows)
    if not metadata.empty:
        metadata = metadata.drop_duplicates(["symbol", "fiscal_period", "basis", "available_at", "source_xbrl_url"])
    selected = _choose_filings(metadata, max(1, int(max_periods_per_symbol)))
    selected_records = selected.to_dict("records") if not selected.empty else []
    parsed_rows: list[dict[str, Any]] = []
    parse_errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 4))) as pool:
        futures = [pool.submit(_parse_filing, item, cache, bool(config.refresh)) for item in selected_records]
        for index, future in enumerate(as_completed(futures), 1):
            try:
                row = future.result()
                parsed_rows.append(row)
                if str(row.get("parse_status", "")).startswith("error:"):
                    parse_errors.append(f"{row['symbol']} {row['fiscal_period']}: {row['parse_status']}")
            except Exception as exc:
                parse_errors.append(str(exc))
            if progress and (index % 25 == 0 or index == len(futures)):
                progress(f"Official NSE quarterly XBRL: {index}/{len(futures)} selected filings")
    results = pd.DataFrame(parsed_rows, columns=RESULT_COLUMNS)
    if not results.empty:
        results["period"] = pd.to_datetime(results.period, errors="coerce", format="mixed")
        results["release_date"] = pd.to_datetime(results.release_date, errors="coerce", format="mixed")
        results["available_at"] = pd.to_datetime(results.available_at, errors="coerce", format="mixed")
    errors = metadata_errors + parse_errors
    parsed_mask = results.parse_status.eq("parsed") if not results.empty else pd.Series(dtype=bool)
    revenue_profit_mask = (pd.to_numeric(results.revenue_cr, errors="coerce").notna()
                           & pd.to_numeric(results.net_profit_cr, errors="coerce").notna()) if not results.empty else pd.Series(dtype=bool)
    nonparsed_rows = int((~parsed_mask).sum()) if len(parsed_mask) else 0
    if nonparsed_rows:
        errors.append(f"{nonparsed_rows} selected filings had no supported quarter facts")
    quality = [{
        "dataset": "official_history", "status": "downloaded" if not errors and not results.empty else "partial" if not metadata.empty else "unavailable",
        "rows": len(results), "symbols_requested": len(symbols),
        "symbols_with_metadata": int(metadata.symbol.nunique()) if not metadata.empty else 0,
        "symbols_with_attempted_filings": int(results.symbol.nunique()) if not results.empty else 0,
        "symbols_with_parsed_numeric_facts": int(results.loc[parsed_mask, "symbol"].nunique()) if not results.empty else 0,
        "symbols_with_revenue_and_profit": int(results.loc[revenue_profit_mask, "symbol"].nunique()) if not results.empty else 0,
        "filings_selected": len(selected_records),
        "parsed_rows": int(parsed_mask.sum()) if not results.empty else 0,
        "rows_with_revenue_and_profit": int(revenue_profit_mask.sum()) if not results.empty else 0,
        "source": "NSE Integrated Filing-Financials historical metadata and XBRL",
        "first_period": str(results.period.min().date()) if not results.empty and results.period.notna().any() else "",
        "last_period": str(results.period.max().date()) if not results.empty and results.period.notna().any() else "",
        "message": f"As of {config.end}; selected latest quarter, same quarter prior year when available, and preceding available quarter, preferring consolidated basis. Screener history remains separate."
                    + (f"; {len(metadata_errors)} metadata errors; {len(parse_errors)} parse errors; {nonparsed_rows} selected filings had no supported quarter facts" if errors else ""),
        "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }]
    output = {
        "official_financial_history_filings": metadata.reindex(columns=METADATA_COLUMNS) if not metadata.empty else empty["official_financial_history_filings"],
        "official_quarterly_history": results.reindex(columns=RESULT_COLUMNS) if not results.empty else empty["official_quarterly_history"],
    }
    return output, quality
