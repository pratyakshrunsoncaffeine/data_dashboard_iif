"""Public data providers for the IIF date-range research run.

Parsing patterns for Screener quarterly/shareholding tables and official NSE
index history follow the user's existing Screener_Momentum and Caelion
handover implementations. No data is bundled: every row is fetched or read
from an exact-request cache during the run.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .config import MARKETS, TABLE_NAMES, ResearchConfig

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125 Safari/537.36")
TIMEOUT = 18
RETRIES = 2
MAX_WORKERS = 4
NSE_INDICES_URL = "https://www.niftyindices.com/BackPage/getHistoricaldatatabletoString"
NSE_INDICES_PAGE = "https://www.niftyindices.com/reports/historical-data"
NSE_HOME = "https://www.nseindia.com"

SCHEMAS: dict[str, list[str]] = {
    "prices": ["date", "symbol", "close", "raw_close", "open", "raw_open", "volume", "high", "low", "source"],
    "markets": ["date", "symbol", "close", "raw_close", "open", "raw_open", "volume", "high", "low", "source", "name", "group", "unit"],
    "members": ["symbol", "name", "sector"],
    "flows": ["date", "category", "buy_cr", "sell_cr", "net_cr", "source"],
    "participant_oi": ["date", "participant", "index_long", "index_short", "long_share_pct",
                       "option_call_long", "option_call_short", "option_put_long", "option_put_short", "option_oi_pcr", "source"],
    "delivery": ["date", "symbol", "delivery_pct", "traded_qty", "delivered_qty", "source"],
    "events": ["date", "symbol", "event", "details", "url", "source"],
    "quarterly": ["symbol", "period", "basis", "revenue_cr", "net_profit_cr", "eps", "revenue_yoy_pct", "profit_yoy_pct", "source", "fetched_at"],
    "holdings": ["symbol", "period", "fii_pct", "dii_pct", "source", "fetched_at"],
    "valuations": ["date", "name", "pe", "pb", "dividend_yield_pct", "source"],
    "macro": ["date", "name", "value", "unit", "source"],
    # Exchange filing data keeps disclosure availability separate from the
    # fiscal period to which the reported numbers relate.
    "official_filings": ["release_date", "symbol", "fiscal_period", "basis", "filing_type",
                          "source_xbrl_url", "in_universe", "source"],
    "official_results": ["release_date", "symbol", "fiscal_period", "basis", "revenue_cr",
                          "net_profit_cr", "eps", "currency", "amount_unit", "eps_unit",
                          "revenue_concept", "profit_concept", "eps_concept",
                          "source_xbrl_url", "source", "fetched_at"],
    "official_holdings": ["symbol", "period", "fii_pct", "dii_pct", "submission_date",
                          "broadcast_date", "source_xbrl_url", "source", "fetched_at"],
}

LIQUID_30 = [
    "RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "BHARTIARTL", "ITC", "SBIN", "TCS",
    "LT", "AXISBANK", "KOTAKBANK", "HINDUNILVR", "BAJFINANCE", "MARUTI", "SUNPHARMA",
    "TITAN", "NTPC", "TATAMOTORS", "ONGC", "ULTRACEMCO", "POWERGRID", "M&M", "ADANIPORTS",
    "TATASTEEL", "WIPRO", "JSWSTEEL", "COALINDIA", "GRASIM", "HCLTECH", "ASIANPAINT",
]


def _empty(name: str) -> pd.DataFrame:
    return pd.DataFrame(columns=SCHEMAS[name])


def _append_observations(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if incoming is None or incoming.empty:
        return existing
    if existing is None or existing.empty:
        return incoming.copy()
    present_columns = [column for column in incoming.columns if column not in existing.columns or incoming[column].notna().any()]
    combined = pd.concat([existing, incoming.loc[:, present_columns]], ignore_index=True)
    return combined.reindex(columns=list(dict.fromkeys([*existing.columns, *incoming.columns])))


def _request(url: str, *, method: str = "get", params: dict | None = None,
             json_body: dict | None = None, referer: str | None = None,
             timeout: int = TIMEOUT, retries: int = RETRIES) -> requests.Response:
    headers = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}
    if referer:
        headers["Referer"] = referer
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.request(method, url, params=params, json=json_body,
                                        headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last = exc
            if attempt < retries:
                time.sleep(0.35 * (attempt + 1))
    raise RuntimeError(f"{url}: {last}")


def _cache_key(dataset: str, config: ResearchConfig, **extra: Any) -> str:
    # Include the exact range and run date so as-of snapshots cannot silently age.
    payload = {"dataset": dataset, "start": config.start, "end": config.end,
               "as_of": date.today().isoformat(), **extra}
    if dataset == "flows":
        # Prevent older caches, created when Upstox's ambiguous DII header was
        # treated as net, from being mistaken for correctly parsed observations.
        payload["parser_version"] = "explicit-gross-net-v2"
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _read_cache(dataset: str, config: ResearchConfig, **extra: Any) -> pd.DataFrame | None:
    path = Path(config.cache_dir) / f"{dataset}_{_cache_key(dataset, config, **extra)}.csv"
    if config.refresh or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None


def _write_cache(frame: pd.DataFrame, dataset: str, config: ResearchConfig, **extra: Any) -> None:
    path = Path(config.cache_dir) / f"{dataset}_{_cache_key(dataset, config, **extra)}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _fundamental_cache(dataset: str, symbol: str, config: ResearchConfig) -> pd.DataFrame | None:
    """Reuse prior smaller-cap caches, then promote them to this cap's key."""
    extras = {"symbol": symbol, "universe": config.universe,
              "fundamental_cap": config.fundamental_count, "parser_version": "quarters-section-v2"}
    frame = _read_cache(dataset, config, **extras)
    if frame is not None and not frame.empty:
        return frame
    if config.fundamental_count > 30:
        old_extras = {**extras, "fundamental_cap": 30}
        frame = _read_cache(dataset, config, **old_extras)
        if frame is not None and not frame.empty:
            _write_cache(frame, dataset, config, **extras)
            return frame
    return None


def _quality(dataset: str, frame: pd.DataFrame, source: str, status: str, message: str = "",
             symbol: str = "") -> dict[str, Any]:
    dates = pd.to_datetime(frame["date"], errors="coerce") if "date" in frame and not frame.empty else pd.Series(dtype="datetime64[ns]")
    return {"dataset": dataset, "symbol": symbol, "status": status, "rows": len(frame),
            "source": source, "first_date": dates.min().date().isoformat() if dates.notna().any() else "",
            "last_date": dates.max().date().isoformat() if dates.notna().any() else "",
            "message": message, "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds")}


def _progress(progress: Callable | None, completed: int, total: int, label: str) -> None:
    if progress:
        try:
            progress(completed, total, label)
        except TypeError:
            progress(label)


def _is_connection_timeout(exc: BaseException) -> bool:
    message = str(exc).lower()
    return isinstance(exc, (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout,
                            TimeoutError)) or any(token in message for token in
                                                   ("connect timed out", "connection timed out", "read timed out",
                                                    "connecttimeouterror", "readtimeouterror"))


def _fetch_members(config: ResearchConfig) -> tuple[pd.DataFrame, str, str]:
    idx = config.universe.removeprefix("nifty")
    path = f"https://www.niftyindices.com/IndexConstituent/ind_nifty{idx}list.csv"
    try:
        response = _request(path, referer="https://www.niftyindices.com/indices/equity/broad-based-indices")
        frame = pd.read_csv(io.BytesIO(response.content))
        columns = {str(c).strip().lower(): c for c in frame.columns}
        symbol_col = next(columns[c] for c in columns if "symbol" in c)
        name_col = next((columns[c] for c in columns if "company" in c or "name" in c), symbol_col)
        sector_col = next((columns[c] for c in columns if "industry" in c or "sector" in c), None)
        out = pd.DataFrame({"symbol": frame[symbol_col].astype(str).str.strip().str.upper(),
                            "name": frame[name_col].astype(str).str.strip(),
                            "sector": frame[sector_col].astype(str).str.strip() if sector_col else ""})
        out = out[out.symbol.ne("") & ~out.symbol.str.startswith("DUMMY")].drop_duplicates("symbol").reset_index(drop=True)
        if out.empty:
            raise ValueError("official constituent file was empty")
        return out, "NSE Indices constituent CSV", "downloaded; current constituent snapshot"
    except Exception as exc:
        # Seed is intentionally identified as a fallback universe, never as an official member list.
        return pd.DataFrame({"symbol": LIQUID_30, "name": LIQUID_30, "sector": ""}), "IIF liquid 30-name fallback", f"official constituent CSV unavailable ({exc}); fallback seed is not a complete {config.universe} snapshot"


def _yahoo_chart(symbol: str, start: str, end: str) -> tuple[pd.DataFrame, str]:
    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
    end_ts = int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='^=.-')}"
    payload = _request(url, params={"period1": start_ts, "period2": end_ts, "interval": "1d", "events": "div,splits"}).json()
    result = payload.get("chart", {}).get("result")
    if not result:
        raise ValueError(payload.get("chart", {}).get("error") or "Yahoo returned no chart result")
    obj = result[0]
    try:
        tz = ZoneInfo(obj.get("meta", {}).get("exchangeTimezoneName") or "UTC")
    except Exception:
        tz = ZoneInfo("UTC")
    timestamps = obj.get("timestamp", [])
    quote_data = obj.get("indicators", {}).get("quote", [{}])[0]
    adj = obj.get("indicators", {}).get("adjclose", [{}])[0].get("adjclose", [])
    raw_close = quote_data.get("close", [])
    raw_open = quote_data.get("open", [])
    rows = []
    for i, ts in enumerate(timestamps):
        close = raw_close[i] if i < len(raw_close) else None
        adjusted = adj[i] if i < len(adj) and adj[i] is not None else close
        # Yahoo supplies adjusted close only; apply its ratio to OHLC so price fields agree.
        ratio = adjusted / close if close not in (None, 0) and adjusted is not None else 1.0
        open_value = raw_open[i] if i < len(raw_open) else None
        rows.append({"date": pd.to_datetime(ts, unit="s", utc=True).tz_convert(tz).date().isoformat(),
                     "close": adjusted, "raw_close": close,
                     "open": open_value * ratio if open_value is not None else None, "raw_open": open_value,
                     "volume": quote_data.get("volume", [None] * len(timestamps))[i],
                     "high": (quote_data.get("high", [None] * len(timestamps))[i] * ratio) if quote_data.get("high", [None] * len(timestamps))[i] is not None else None,
                     "low": (quote_data.get("low", [None] * len(timestamps))[i] * ratio) if quote_data.get("low", [None] * len(timestamps))[i] is not None else None})
    frame = pd.DataFrame(rows)
    frame = frame[frame["date"].between(start, end)].reset_index(drop=True)
    if frame.empty:
        raise ValueError("Yahoo returned no daily bars")
    return frame, "Yahoo Finance chart (adjusted close; OHLC ratio-adjusted)"


def _prices_for(yahoo_symbol: str, start: str, end: str) -> tuple[str, pd.DataFrame, str]:
    frame, source = _yahoo_chart(yahoo_symbol, start, end)
    symbol = yahoo_symbol.removesuffix(".NS")
    frame["symbol"] = symbol
    frame["source"] = source
    return symbol, frame[SCHEMAS["prices"]], source


def _nse_index_history(name: str, start: str, end: str) -> pd.DataFrame:
    payload = {"cinfo": json.dumps({"name": name.upper(), "startDate": pd.Timestamp(start).strftime("%d-%b-%Y"),
                                    "endDate": pd.Timestamp(end).strftime("%d-%b-%Y"), "indexName": name})}
    response = _request(NSE_INDICES_URL, method="post", json_body=payload, referer=NSE_INDICES_PAGE)
    data: Any = response.json()
    if isinstance(data, dict) and "d" in data:
        data = data["d"]
    if isinstance(data, str):
        data = json.loads(data)
        if isinstance(data, dict) and "d" in data:
            data = data["d"]
    if not isinstance(data, list):
        raise ValueError("Unexpected NSE historical response")
    raw = pd.DataFrame(data)
    if raw.empty:
        return _empty("markets")
    def find(*names: str, default=None):
        return next((raw[c] for want in names for c in raw.columns if str(c).lower() == want.lower()), default)
    dates = find("HistoricalDate", "Date", "date")
    close = find("CLOSE", "Close", "close")
    open_ = find("OPEN", "Open", "open")
    out = pd.DataFrame({"date": pd.to_datetime(dates, errors="coerce", dayfirst=True).dt.date.astype("string"),
                        "symbol": name, "close": pd.to_numeric(pd.Series(close).astype(str).str.replace(",", "", regex=False), errors="coerce"),
                        "raw_close": pd.NA,
                        "open": pd.to_numeric(pd.Series(open_).astype(str).str.replace(",", "", regex=False), errors="coerce") if open_ is not None else pd.NA,
                        "raw_open": pd.to_numeric(pd.Series(open_).astype(str).str.replace(",", "", regex=False), errors="coerce") if open_ is not None else pd.NA,
                        "volume": pd.NA, "high": pd.NA, "low": pd.NA,
                        "source": "NSE Indices historical data", "name": name, "group": "India index", "unit": "index points"})
    out = out.dropna(subset=["date", "close"])
    return out[out["date"].between(start, end)].reset_index(drop=True)


def _clean_num(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    s = str(value).strip().replace(",", "").replace("%", "")
    if not s or s in {"-", "—", "nan"}:
        return None
    match = re.search(r"\(?-?\d+(?:\.\d+)?\)?", s)
    if not match:
        return None
    value = float(match.group().replace("(", "-").replace(")", ""))
    if "Cr" in str(value):
        return value
    return value


def _parse_cash_flow_tables(tables: list[pd.DataFrame], start: str, end: str, source: str) -> pd.DataFrame:
    """Parse visible daily FII/DII rows; monthly aggregates are deliberately ignored."""
    rows: list[dict[str, Any]] = []
    for table in tables:
        if table.empty:
            continue
        # Flatten multi-row headers returned by Moneycontrol's grouped table.
        cols = [" ".join(str(x) for x in col if str(x) != "nan").lower() if isinstance(col, tuple)
                else str(col).lower() for col in table.columns]
        frame = table.copy()
        frame.columns = cols
        date_col = next((c for c in cols if "date" in c), None)
        if date_col is None:
            continue
        buy_cols = {kind: next((c for c in cols if label in c and ("gross purchase" in c or "buy" in c)), None)
                    for kind, label in (("FII", "fii"), ("DII", "dii"))}
        sell_cols = {kind: next((c for c in cols if label in c and ("gross sales" in c or "sell" in c)), None)
                     for kind, label in (("FII", "fii"), ("DII", "dii"))}
        # Only the explicit net-purchase/sales column is unambiguous. Upstox's
        # DII header "Net purchase" is actually gross buy; its final
        # "Net purchase/sales" column is the reported net.
        net_cols = {}
        for kind, label in (("FII", "fii"), ("DII", "dii")):
            matches = [c for c in cols if label in c and re.search(r"net\s+purchase\s*/\s*sales", c)]
            net_cols[kind] = matches[0] if matches else None
        # Avoid mixing derivatives tables into cash activity. Gross buy may be
        # derived only from an explicit net + sales pair.
        if not (buy_cols["FII"] and sell_cols["FII"] and sell_cols["DII"] and
                (buy_cols["DII"] or net_cols["DII"])):
            continue
        for _, item in frame.iterrows():
            raw_date = re.sub(r"\s+", " ", str(item[date_col])).strip()
            date_match = re.search(r"\d{4}-\d{2}-\d{2}|\d{1,2}[-/][A-Za-z0-9]+[-/]\d{2,4}", raw_date)
            if date_match:
                raw_date = date_match.group(0)
            parsed = pd.to_datetime(raw_date, errors="coerce", dayfirst=True)
            if pd.isna(parsed):
                continue  # Reject month totals and labels such as "Month till date".
            day = parsed.date().isoformat()
            if not start <= day <= end:
                continue
            for category in ("FII", "DII"):
                buy_col, sell_col = buy_cols[category], sell_cols[category]
                buy = _clean_num(item.get(buy_col)) if buy_col else None
                sell = _clean_num(item.get(sell_col)) if sell_col else None
                reported_net = _clean_num(item.get(net_cols[category])) if net_cols[category] else None
                derived_buy = False
                if buy is None and reported_net is not None and sell is not None:
                    buy = reported_net + sell
                    derived_buy = True
                if buy is None and sell is None:
                    continue
                computed_net = buy - sell if buy is not None and sell is not None else None
                # Reject inconsistent publisher rows rather than silently
                # overriding the source's explicit net field.
                if (reported_net is not None and computed_net is not None and
                        abs(computed_net - reported_net) > 0.02):
                    continue
                rows.append({"date": day, "category": category, "buy_cr": buy, "sell_cr": sell,
                             "net_cr": reported_net if reported_net is not None else computed_net,
                             "source": source + ("; gross buy derived as explicit net plus sales" if derived_buy else "")})
    if not rows:
        return _empty("flows")
    return pd.DataFrame(rows, columns=SCHEMAS["flows"]).drop_duplicates(["date", "category"], keep="first").sort_values(["date", "category"]).reset_index(drop=True)


def _fetch_cash_flows(config: ResearchConfig) -> tuple[pd.DataFrame, str, str]:
    """Try public daily pages; retain their true observation dates and visible coverage."""
    sources = [
        ("https://www.moneycontrol.com/stocks/marketstats/fii_dii_activity/", "Moneycontrol FII/DII daily table"),
        ("https://upstox.com/fii-dii-data/", "Upstox FII/DII daily table"),
    ]
    errors = []
    for url, source in sources:
        try:
            html = _request(url).text
            parsed = pd.read_html(io.StringIO(html))
            frame = _parse_cash_flow_tables(parsed, config.start, config.end, source)
            if not frame.empty:
                return frame, source, "visible dated rows only; page may expose a limited recent window, no dates were filled"
        except Exception as exc:
            errors.append(f"{source}: {exc}")
    return _empty("flows"), "Moneycontrol/Upstox public daily tables", "; ".join(errors) or "no daily rows for requested range"


def _screener_company(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    url = f"https://www.screener.in/company/{quote(symbol, safe='')}/consolidated/"
    response = _request(url, timeout=8, retries=0)
    basis = "consolidated"
    quarterly_tables, holding_tables = _screener_history_tables(response.text)
    if not quarterly_tables:
        alt = f"https://www.screener.in/company/{quote(symbol, safe='')}/"
        alt_resp = _request(alt, timeout=8, retries=0)
        alt_quarterly, alt_holding = _screener_history_tables(alt_resp.text)
        if alt_quarterly or alt_holding:
            quarterly_tables, holding_tables = alt_quarterly, alt_holding
            basis, url = "standalone", alt
    quarters = []
    holdings_by_period: dict[str, dict[str, Any]] = {}
    fetched = datetime.now().astimezone().isoformat(timespec="seconds")
    for table in quarterly_tables:
        if table.empty:
            continue
        first_col = table.columns[0]
        metric_col = table[first_col].astype(str).str.strip().str.lower()
        # Only quarter section + quarter period labels qualify; this excludes
        # annual P&L and TTM tables even if they contain Sales and Net Profit.
        periods = [str(c).strip() for c in table.columns[1:]]
        valid = [(i, p) for i, p in enumerate(periods) if re.fullmatch(r"(?:Mar|Jun|Sep|Dec)\s+\d{4}", p, re.I)]
        if not valid:
            continue
        sales = _metric_row(table, metric_col, ("sales", "revenue", "operating revenue"))
        profit = _metric_row(table, metric_col, ("net profit", "profit after tax", "pat"))
        eps = _metric_row(table, metric_col, ("eps in rs", "eps"))
        for i, period in valid:
            revenue = _clean_num(sales[i]) if sales is not None and len(sales) > i else None
            net_profit = _clean_num(profit[i]) if profit is not None and len(profit) > i else None
            eps_value = _clean_num(eps[i]) if eps is not None and len(eps) > i else None
            if revenue is not None or net_profit is not None or eps_value is not None:
                quarters.append({"symbol": symbol, "period": period, "basis": basis,
                                 "revenue_cr": revenue, "net_profit_cr": net_profit, "eps": eps_value,
                                 "revenue_yoy_pct": None, "profit_yoy_pct": None,
                                 "source": f"Screener.in ({basis}): {url}", "fetched_at": fetched})
    for table in holding_tables:
        if table.empty:
            continue
        periods = [str(c).strip() for c in table.columns[1:]]
        values: dict[str, list[Any]] = {}
        for _, row in table.iterrows():
            label = str(row.iloc[0]).strip().lower()
            if "fii" in label or "foreign institutional" in label or "foreign portfolio" in label:
                values["fii"] = list(row.iloc[1:])
            elif "dii" in label or "domestic institutional" in label:
                values["dii"] = list(row.iloc[1:])
        for i, period in enumerate(periods):
            if not re.fullmatch(r"[A-Za-z]{3}\s+\d{4}", period):
                continue
            fii = _clean_num(values.get("fii", [None] * len(periods))[i])
            dii = _clean_num(values.get("dii", [None] * len(periods))[i])
            if fii is None and dii is None:
                continue
            existing = holdings_by_period.get(period)
            if existing is None or (existing["fii_pct"] is None and fii is not None) or (existing["dii_pct"] is None and dii is not None):
                holdings_by_period[period] = {"symbol": symbol, "period": period, "fii_pct": fii, "dii_pct": dii,
                                              "source": f"Screener.in ({basis}): {url}", "fetched_at": fetched}
    # De-duplicate quarter tabs that Screener may render more than once.
    quarters = list({row["period"]: row for row in quarters}.values())
    quarter_values = {}
    for row in quarters:
        parsed = pd.to_datetime(row["period"], format="%b %Y", errors="coerce")
        if pd.notna(parsed):
            quarter_values[parsed.to_period("Q")] = row
    for quarter, row in quarter_values.items():
        previous = quarter_values.get(quarter - 4)
        if previous:
            for value_key, growth_key in (("revenue_cr", "revenue_yoy_pct"), ("net_profit_cr", "profit_yoy_pct")):
                current, prior = row[value_key], previous[value_key]
                if current is not None and prior not in (None, 0):
                    row[growth_key] = (current / prior - 1) * 100
    return (pd.DataFrame(quarters, columns=SCHEMAS["quarterly"]),
            pd.DataFrame(list(holdings_by_period.values()), columns=SCHEMAS["holdings"]), basis, url)


def _parse_nse_shareholding_xbrl(content: bytes | str, symbol: str, period: str,
                                 filing_date: str = "") -> pd.DataFrame:
    """Extract official NSE XBRL domestic/foreign institutional holdings facts."""
    root = ET.fromstring(content)
    facts: dict[tuple[str, str], list[str]] = {}
    for node in root.iter():
        local = node.tag.rsplit("}", 1)[-1]
        context = node.attrib.get("contextRef", "")
        if local == "ShareholdingAsAPercentageOfTotalNumberOfShares" and context.endswith("_ContextI"):
            facts.setdefault((local, context), []).append((node.text or "").strip())

    def fact(context: str) -> float | None:
        vals = facts.get(("ShareholdingAsAPercentageOfTotalNumberOfShares", context + "_ContextI"), [])
        for raw in vals:
            value = _clean_num(raw)
            if value is not None:
                return value * 100.0  # NSE XBRL stores share percentages as fractions.
        return None

    fii = fact("InstitutionsForeign")
    dii = fact("InstitutionsDomestic")
    if fii is None and dii is None:
        return _empty("holdings")
    stamp = pd.to_datetime(period, errors="coerce", dayfirst=not bool(re.match(r"^\d{4}-\d{2}-\d{2}", str(period))))
    if pd.isna(stamp):
        return _empty("holdings")
    return pd.DataFrame([{"symbol": symbol, "period": stamp.date().isoformat(), "fii_pct": fii, "dii_pct": dii,
                          "source": f"NSE shareholding pattern XBRL; filed {filing_date or 'date unavailable'}",
                          "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds")}],
                        columns=SCHEMAS["holdings"])


def _parse_nse_financial_xbrl(content: bytes | str, symbol: str, period: str, basis: str,
                              filing_date: str = "") -> pd.DataFrame:
    """Read quarterly revenue, PAT, and EPS from an NSE integrated-filing XBRL."""
    root = ET.fromstring(content)
    contexts = {}
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] != "context":
            continue
        period_node = next((c for c in node if c.tag.rsplit("}", 1)[-1] == "period"), None)
        if period_node is None:
            continue
        start_node = next((c for c in period_node if c.tag.rsplit("}", 1)[-1] == "startDate"), None)
        end_node = next((c for c in period_node if c.tag.rsplit("}", 1)[-1] in {"endDate", "instant"}), None)
        if end_node is None:
            continue
        # Dimensional segment facts are not the company-wide result summary.
        has_dimensions = any(c.tag.rsplit("}", 1)[-1] in {"scenario", "segment", "explicitMember", "typedMember"}
                             for c in node.iter())
        contexts[node.attrib.get("id", "")] = (start_node.text if start_node is not None else None,
                                                  end_node.text, has_dimensions)
    unit_measures = {}
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] == "unit":
            unit_measures[node.attrib.get("id", "")] = " ".join(
                (child.text or "").strip() for child in node.iter()
                if child.tag.rsplit("}", 1)[-1] == "measure")
    end_date = pd.to_datetime(period, errors="coerce", dayfirst=not bool(re.match(r"^\d{4}-\d{2}-\d{2}", str(period))))
    if pd.isna(end_date):
        return _empty("quarterly")
    candidates = []
    for context_id, (start, end, dimensional) in contexts.items():
        if dimensional or end != end_date.date().isoformat() or not start:
            continue
        start_date = pd.to_datetime(start, errors="coerce")
        if pd.isna(start_date):
            continue
        days = (end_date - start_date).days + 1
        if 45 <= days <= 125:
            candidates.append((days, context_id))
    if not candidates:
        return _empty("quarterly")
    _, context_id = min(candidates)
    facts = {}
    for node in root.iter():
        local = node.tag.rsplit("}", 1)[-1]
        if node.attrib.get("contextRef") == context_id and node.text:
            facts.setdefault(local, []).append((node.text.strip(), node.attrib.get("unitRef", "")))

    def pick(*fact_names):
        for name in fact_names:
            for raw, unit in facts.get(name, []):
                number = _clean_num(raw)
                if number is None:
                    continue
                unit_lower = unit.lower()
                if name in {"RevenueFromOperations", "Income", "TotalIncome", "OperatingIncome",
                            "ProfitLossForPeriod", "ProfitLossForPeriodFromContinuingOperations",
                            "ProfitLossForThePeriod", "ProfitLossFromOrdinaryActivitiesAfterTax",
                            "ProfitLossAfterTaxesMinorityInterestAndShareOfProfitLossOfAssociates",
                            "ProfitLossAfterTax", "ProfitLossAfterTaxAndExtraordinaryItems"}:
                    resolved = unit_measures.get(unit, unit).lower()
                    if "crore" in resolved:
                        return number
                    if "lakh" in resolved:
                        return number / 100.0
                    if resolved in {"inr", "indianrupee", "iso4217:inr"} or resolved.endswith(":inr"):
                        return number / 10_000_000.0
                    # Only normalize supported, explicitly declared INR units;
                    # unknown currency/unit facts must not masquerade as crore.
                    return None
                return number
        return None

    # Bank taxonomy filings use `Income` and `ProfitLossForThePeriod`; they
    # otherwise share the same duration contexts and INR units as non-banks.
    # Insurance taxonomy top-line concepts are retained under their actual
    # label. OperatingIncome is used only when the standard income concepts
    # are absent; it is not relabeled as conventional sales revenue.
    revenue = pick("RevenueFromOperations", "Income", "TotalIncome", "OperatingIncome")
    profit = pick("ProfitLossForPeriod", "ProfitLossForPeriodFromContinuingOperations",
                  "ProfitLossForThePeriod", "ProfitLossFromOrdinaryActivitiesAfterTax",
                  "ProfitLossAfterTaxesMinorityInterestAndShareOfProfitLossOfAssociates",
                  "ProfitLossAfterTax", "ProfitLossAfterTaxAndExtraordinaryItems")
    # EPS has a distinct per-share unit and must be explicitly declared in INR.
    eps_names = ("BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
                 "BasicEarningsLossPerShareFromContinuingOperations", "EarningsPerShareBasic",
                 "BasicEarningsPerShareAfterExtraordinaryItems", "BasicEarningsPerShareBeforeExtraordinaryItems",
                 "BasicAndDilutedEPSAfterExtraordinaryItemsNetOfTaxExpenseForThePeriodNotToBeAnnualized",
                 "BasicAndDilutedEPSBeforeExtraordinaryItemsNetOfTaxExpenseForThePeriodNotToBeAnnualized")
    eps = None
    for eps_name in eps_names:
        for raw, unit in facts.get(eps_name, []):
            resolved = unit_measures.get(unit, unit).lower().replace(" ", "")
            if "inr" not in resolved and "indianrupee" not in resolved:
                continue
            if "share" not in resolved:
                continue
            eps = _clean_num(raw)
            if eps is not None:
                break
        if eps is not None:
            break
    if revenue is None and profit is None and eps is None:
        return _empty("quarterly")
    return pd.DataFrame([{"symbol": symbol, "period": end_date.date().isoformat(), "basis": basis,
                          "revenue_cr": revenue, "net_profit_cr": profit, "eps": eps,
                          "revenue_yoy_pct": None, "profit_yoy_pct": None,
                          "source": f"NSE Integrated Filing XBRL; filed {filing_date or 'date unavailable'}",
                          "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds")}],
                        columns=SCHEMAS["quarterly"])


def _fetch_nse_integrated_results(config: ResearchConfig) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Fetch dated NSE integrated financial filings and their numeric XBRL results.

    The standard financial-results API is no longer populated for current
    filings; NSE routes March-2025 onward results through Integrated Filing.
    Pagination and XBRL downloads are bounded to ten pages and four workers.
    """
    url = f"{NSE_HOME}/api/integrated-filing-results"
    params = {"index": "equities", "type": "Integrated Filing- Financials",
              "from_date": pd.Timestamp(config.start).strftime("%d-%m-%Y"),
              "to_date": pd.Timestamp(config.end).strftime("%d-%m-%Y"), "page": 1, "size": 100}
    records, errors, total = [], [], None
    for page in range(1, 11):
        try:
            page_params = {**params, "page": page}
            payload = _request(url, params=page_params, referer=NSE_HOME, timeout=12, retries=0).json()
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                raise ValueError("unexpected NSE integrated-filing response schema")
            if total is None:
                total = pd.to_numeric(payload.get("totalCount"), errors="coerce")
            batch = payload["data"]
            if not batch:
                break
            records.extend(batch)
            if len(records) >= (total if pd.notna(total) else len(records)):
                break
        except Exception as exc:
            errors.append(str(exc))
            break
    events, latest = [], {}
    for item in records:
        raw_release = item.get("broadcast_Date") or item.get("creation_Date") or item.get("revised_Date")
        release = pd.to_datetime(raw_release, errors="coerce", dayfirst=True)
        if pd.isna(release):
            continue
        release_day = release.date().isoformat()
        if not (config.start <= release_day <= config.end):
            continue
        symbol = str(item.get("symbol") or "").upper()
        period = str(item.get("qe_Date") or "")
        basis = str(item.get("consolidated") or "").title()
        details = f"{item.get('type_Sub') or 'Financial filing'}; {basis or 'basis not stated'}; period ended {period}"
        events.append({"date": release_day, "symbol": symbol, "event": "Financial results announcement",
                       "details": details, "url": item.get("xbrl") or item.get("ixbrl") or "",
                       "source": "NSE Integrated Filing Financials date-range API"})
        key = (symbol, period, basis)
        if symbol and period and item.get("xbrl"):
            prior = latest.get(key)
            if prior is None or release > prior[0]:
                latest[key] = (release, item)
    def fetch_numeric(pair):
        (symbol, period, basis), (release, item) = pair
        try:
            content = _request(str(item["xbrl"]), referer=NSE_HOME, timeout=12, retries=0).content
            frame = _parse_nse_financial_xbrl(content, symbol, period, basis, release.isoformat())
            return frame, ""
        except Exception as exc:
            return _empty("quarterly"), f"{symbol} {period}: {exc}"
    numeric = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(fetch_numeric, pair) for pair in latest.items()]
        for future in as_completed(futures):
            frame, error = future.result()
            if not frame.empty:
                numeric.append(frame)
            elif error:
                errors.append(error)
    event_frame = pd.DataFrame(events, columns=SCHEMAS["events"]).drop_duplicates().reset_index(drop=True)
    quarter_frame = pd.concat(numeric, ignore_index=True) if numeric else _empty("quarterly")
    if not quarter_frame.empty:
        quarter_frame = quarter_frame.drop_duplicates(["symbol", "period", "basis"], keep="last").reset_index(drop=True)
    expected = int(total) if total is not None and pd.notna(total) else len(records)
    note = f"NSE Integrated Filing returned {len(records)} metadata rows ({len(event_frame)} in-range filings); parsed {len(quarter_frame)} numeric quarterly results"
    if expected > len(records):
        note += f"; bounded pagination partial ({len(records)} of {expected} rows, 10-page cap)"
    if errors:
        note += f"; {len(errors)} page/XBRL errors"
    return event_frame, quarter_frame, note


def collect_official_filings(config: ResearchConfig, symbols: list[str] | None = None
                             ) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Collect date-eligible NSE filings as disclosure records and numeric XBRL.

    ``official_filings`` retains every in-range metadata row (including items
    outside the requested stock universe); ``in_universe`` marks relevance.
    Numeric XBRL is parsed for current-universe rows only. Fiscal period and
    exchange release date remain distinct fields. This helper intentionally
    does not alter the legacy ``quarterly`` Screener snapshot table.
    """
    url = f"{NSE_HOME}/api/integrated-filing-results"
    universe = {str(s).upper() for s in (symbols or [])}
    symbol_key = hashlib.sha256("|".join(sorted(universe)).encode()).hexdigest()[:16]
    cache_args = {"symbols_key": symbol_key, "parser_version": "nse-integrated-xbrl-v1"}
    cached_filings = _read_cache("official_filings_range", config, **cache_args)
    cached_results = _read_cache("official_results_range", config, **cache_args)
    if cached_filings is not None and not cached_filings.empty and cached_results is not None and not cached_results.empty:
        tables = {"official_filings": cached_filings.reindex(columns=SCHEMAS["official_filings"]),
                  "official_results": cached_results.reindex(columns=SCHEMAS["official_results"])}
        dates = pd.to_datetime(tables["official_filings"]["release_date"], errors="coerce")
        first, last = (dates.min().date().isoformat(), dates.max().date().isoformat()) if dates.notna().any() else ("", "")
        quality = [{"dataset": name, "symbol": "", "status": "cached", "rows": len(frame),
                    "source": "exact-request NSE filing cache", "first_date": first, "last_date": last,
                    "message": "Exact date range, current universe, and parser-version cache.",
                    "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds")}
                   for name, frame in tables.items()]
        return tables, quality
    params = {"index": "equities", "type": "Integrated Filing- Financials",
              "from_date": pd.Timestamp(config.start).strftime("%d-%m-%Y"),
              "to_date": pd.Timestamp(config.end).strftime("%d-%m-%Y"), "page": 1, "size": 100}
    records: list[dict] = []
    errors: list[str] = []
    expected = None
    for page in range(1, 11):
        try:
            response = _request(url, params={**params, "page": page}, referer=NSE_HOME,
                                timeout=12, retries=0)
            payload = response.json()
            if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                raise ValueError("unexpected integrated-filing API response schema")
            if expected is None:
                expected = int(_clean_num(payload.get("totalCount")) or 0)
            batch = payload["data"]
            if not batch:
                break
            records.extend(batch)
            if len(records) >= expected:
                break
        except Exception as exc:
            errors.append(f"metadata page {page}: {exc}")
            break

    filing_rows = []
    eligible = []
    for item in records:
        raw_release = item.get("broadcast_Date") or item.get("creation_Date") or item.get("revised_Date")
        release = pd.to_datetime(raw_release, errors="coerce", dayfirst=True)
        if pd.isna(release):
            continue
        release_day = release.date().isoformat()
        if not config.start <= release_day <= config.end:
            continue
        symbol = str(item.get("symbol") or "").strip().upper()
        period = str(item.get("qe_Date") or "").strip()
        basis = str(item.get("consolidated") or "").strip().title()
        xbrl = str(item.get("xbrl") or item.get("ixbrl") or "")
        in_universe = not universe or symbol in universe
        filing_rows.append({"release_date": release.isoformat(), "symbol": symbol,
                            "fiscal_period": period, "basis": basis,
                            "filing_type": str(item.get("type_Sub") or "Financial filing"),
                            "source_xbrl_url": xbrl, "in_universe": in_universe,
                            "source": "NSE Integrated Filing Financials date-range API"})
        if in_universe and symbol and period and xbrl:
            eligible.append((release, item, symbol, period, basis, xbrl))

    # Latest revision per stock, fiscal period, and basis; other filing rows
    # remain present in official_filings for auditability.
    latest: dict[tuple[str, str, str], tuple] = {}
    for row in eligible:
        release, item, symbol, period, basis, xbrl = row
        key = symbol, period, basis
        if key not in latest or release > latest[key][0]:
            latest[key] = row

    def parse_one(entry):
        release, item, symbol, period, basis, xbrl = entry
        try:
            xml = _request(xbrl, referer=NSE_HOME, timeout=12, retries=0).content
            quarter = _parse_nse_financial_xbrl(xml, symbol, period, basis, release.isoformat())
            if quarter.empty:
                return None, ""
            root = ET.fromstring(xml)
            units = {}
            for unit_node in root.iter():
                if unit_node.tag.rsplit("}", 1)[-1] == "unit":
                    uid = unit_node.attrib.get("id", "")
                    units[uid] = " ".join((node.text or "").strip() for node in unit_node.iter()
                                           if node.tag.rsplit("}", 1)[-1] == "measure")
            # The parser normalizes monetary facts only when the XBRL fact has
            # a declared INR/lakh/crore unit. Keep the declared unit evidence.
            target_end = pd.to_datetime(period, errors="coerce", dayfirst=True)
            quarter_contexts = []
            for context_node in root.iter():
                if context_node.tag.rsplit("}", 1)[-1] != "context":
                    continue
                period_node = next((c for c in context_node if c.tag.rsplit("}", 1)[-1] == "period"), None)
                if period_node is None or any(c.tag.rsplit("}", 1)[-1] in {"scenario", "segment", "explicitMember", "typedMember"}
                                              for c in context_node.iter()):
                    continue
                start_node = next((c for c in period_node if c.tag.rsplit("}", 1)[-1] == "startDate"), None)
                end_node = next((c for c in period_node if c.tag.rsplit("}", 1)[-1] in {"endDate", "instant"}), None)
                if start_node is None or end_node is None or pd.isna(target_end) or end_node.text != target_end.date().isoformat():
                    continue
                start_date = pd.to_datetime(start_node.text, errors="coerce")
                days = (target_end - start_date).days + 1 if pd.notna(start_date) else 0
                if 45 <= days <= 125:
                    quarter_contexts.append((days, context_node.attrib.get("id", "")))
            selected_context = min(quarter_contexts)[1] if quarter_contexts else ""
            facts = {}
            for node in root.iter():
                if node.attrib.get("contextRef") == selected_context:
                    facts.setdefault(node.tag.rsplit("}", 1)[-1], node.attrib.get("unitRef", ""))
            revenue_names = ("RevenueFromOperations", "Income", "TotalIncome", "OperatingIncome")
            profit_names = ("ProfitLossForPeriod", "ProfitLossForPeriodFromContinuingOperations",
                            "ProfitLossForThePeriod", "ProfitLossFromOrdinaryActivitiesAfterTax",
                            "ProfitLossAfterTaxesMinorityInterestAndShareOfProfitLossOfAssociates",
                            "ProfitLossAfterTax", "ProfitLossAfterTaxAndExtraordinaryItems")
            eps_names = ("BasicEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
                         "BasicEarningsLossPerShareFromContinuingOperations", "EarningsPerShareBasic",
                         "BasicEarningsPerShareAfterExtraordinaryItems", "BasicEarningsPerShareBeforeExtraordinaryItems",
                         "BasicAndDilutedEPSAfterExtraordinaryItemsNetOfTaxExpenseForThePeriodNotToBeAnnualized",
                         "BasicAndDilutedEPSBeforeExtraordinaryItemsNetOfTaxExpenseForThePeriodNotToBeAnnualized")
            revenue_concept = next((name for name in revenue_names if name in facts), "")
            profit_concept = next((name for name in profit_names if name in facts), "")
            eps_concept = next((name for name in eps_names if name in facts), "")
            money_ref = facts.get(revenue_concept) or facts.get(profit_concept) or ""
            eps_ref = facts.get(eps_concept, "")
            declared = units.get(money_ref, money_ref)
            normalized = "INR crore" if "inr" in declared.lower() or "indianrupee" in declared.lower() else declared
            row = quarter.iloc[0].to_dict()
            return {"release_date": release.isoformat(), "symbol": symbol, "fiscal_period": period,
                    "basis": basis, "revenue_cr": row.get("revenue_cr"),
                    "net_profit_cr": row.get("net_profit_cr"), "eps": row.get("eps"),
                    "currency": "INR" if "inr" in declared.lower() or "indianrupee" in declared.lower() else "",
                    "amount_unit": normalized, "eps_unit": units.get(eps_ref, eps_ref),
                    "revenue_concept": revenue_concept, "profit_concept": profit_concept,
                    "eps_concept": eps_concept,
                    "source_xbrl_url": xbrl, "source": "NSE Integrated Filing XBRL",
                    "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds")}, ""
        except Exception as exc:
            return None, f"{symbol} {period}: {exc}"

    parsed = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(parse_one, item) for item in latest.values()]
        for future in as_completed(futures):
            row, error = future.result()
            if row:
                parsed.append(row)
            elif error:
                errors.append(error)
    tables = {
        "official_filings": pd.DataFrame(filing_rows, columns=SCHEMAS["official_filings"]),
        "official_results": pd.DataFrame(parsed, columns=SCHEMAS["official_results"]),
    }
    if not tables["official_filings"].empty and not tables["official_results"].empty:
        _write_cache(tables["official_filings"], "official_filings_range", config, **cache_args)
        _write_cache(tables["official_results"], "official_results_range", config, **cache_args)
    status = "downloaded" if filing_rows else "unavailable"
    note = f"NSE integrated financial filings: {len(filing_rows)} actual in-range metadata rows; {len(parsed)} numeric XBRL results for requested universe"
    if expected is not None and len(records) < expected:
        note += f"; bounded pagination returned {len(records)} of {expected} rows"
    if errors:
        note += f"; {len(errors)} retrieval/parse errors"
    quality = [{"dataset": name, "symbol": "", "status": status if name == "official_filings" else ("downloaded" if parsed else "unavailable"),
                "rows": len(frame), "source": "NSE Integrated Filing Financials API / XBRL",
                "first_date": min((str(row["release_date"])[:10] for row in filing_rows), default=""),
                "last_date": max((str(row["release_date"])[:10] for row in filing_rows), default=""),
                "message": note, "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds")}
               for name, frame in tables.items()]
    return tables, quality


def collect_official_holdings(config: ResearchConfig, symbols: list[str], *, max_periods: int = 2,
                              progress: Callable | None = None, reuse_frame: pd.DataFrame | None = None
                              ) -> tuple[pd.DataFrame, list[dict]]:
    """Fetch latest two calendar-quarter disclosures available by report end.

    Reporting periods must be quarter ends; revisions are resolved by latest
    submission/broadcast within a period, then periods are selected by fiscal
    period date. Intraperiod allotment dates are never presented as quarters.
    """
    selected = [str(s).upper() for s in symbols[:config.fundamental_count]]
    rows, quality, consecutive_errors = [], [], 0
    reuse = reuse_frame.copy() if reuse_frame is not None else pd.DataFrame(columns=SCHEMAS["official_holdings"])
    def one(symbol):
        cache_args = {"symbol": symbol, "max_periods": max_periods, "parser_version": "nse-shareholding-xbrl-v2-quarter-end"}
        cached = _read_cache("official_holdings_symbol", config, **cache_args)
        if cached is not None and not cached.empty:
            return cached.reindex(columns=SCHEMAS["official_holdings"]).to_dict("records"), len(cached), True
        url = f"{NSE_HOME}/api/corporate-share-holdings-master"
        payload = _request(url, params={"index": "equities", "symbol": symbol}, referer=NSE_HOME,
                          timeout=10, retries=0).json()
        if not isinstance(payload, list):
            raise ValueError("unexpected shareholding metadata schema")
        eligible = []
        for item in payload:
            raw_period = str(item.get("date") or "").strip()
            period_stamp = pd.to_datetime(raw_period, errors="coerce", dayfirst=True)
            if pd.isna(period_stamp) or not period_stamp.is_quarter_end or period_stamp.date().isoformat() > config.end:
                continue
            available = item.get("broadcastDate") or item.get("submissionDate") or item.get("systemDate")
            stamp = pd.to_datetime(available, errors="coerce", dayfirst=True)
            if pd.isna(stamp) or stamp.date().isoformat() > config.end:
                continue
            eligible.append((stamp, period_stamp, item))
        by_period = {}
        for stamp, period_stamp, item in eligible:
            period = period_stamp.date().isoformat()
            previous = by_period.get(period)
            if previous is None or stamp > previous[0]:
                by_period[period] = (stamp, period_stamp, item)
        ordered_all = sorted(by_period.values(), key=lambda item: item[1])
        take = max(0, int(max_periods))
        ordered = ordered_all[-take:] if take else []
        result = []
        for stamp, period_stamp, item in ordered:
            xbrl = str(item.get("xbrl") or "")
            if not xbrl:
                continue
            prior_match = reuse.loc[(reuse.get("symbol", pd.Series(dtype=str)).astype(str).str.upper() == symbol)
                                    & (reuse.get("period", pd.Series(dtype=str)).astype(str) == period_stamp.date().isoformat())
                                    & (reuse.get("source_xbrl_url", pd.Series(dtype=str)).astype(str) == xbrl)]
            if not prior_match.empty:
                reused = prior_match.iloc[-1].to_dict()
                reused.update({"submission_date": str(item.get("submissionDate") or ""),
                               "broadcast_date": str(item.get("broadcastDate") or ""),
                               "source": "NSE shareholding pattern XBRL (reused; filing URL revalidated)",
                               "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds")})
                result.append(reused)
                continue
            xml = _request(xbrl, referer=NSE_HOME, timeout=10, retries=0).content
            base = _parse_nse_shareholding_xbrl(xml, symbol, period_stamp.date().isoformat(), stamp.isoformat())
            if base.empty:
                continue
            b = base.iloc[0].to_dict()
            result.append({"symbol": symbol, "period": b["period"], "fii_pct": b["fii_pct"], "dii_pct": b["dii_pct"],
                           "submission_date": str(item.get("submissionDate") or ""),
                           "broadcast_date": str(item.get("broadcastDate") or ""),
                           "source_xbrl_url": xbrl, "source": "NSE shareholding pattern XBRL",
                           "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds")})
        if result:
            _write_cache(pd.DataFrame(result, columns=SCHEMAS["official_holdings"]),
                         "official_holdings_symbol", config, **cache_args)
        return result, len(ordered), False
    # Iterate in bounded groups to avoid a large concurrent burst and stop if
    # NSE starts refusing all requests; successful symbols are retained.
    for offset in range(0, len(selected), MAX_WORKERS):
        batch = selected[offset:offset + MAX_WORKERS]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(one, symbol): symbol for symbol in batch}
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    got, periods, cached = future.result()
                    rows.extend(got)
                    consecutive_errors = 0
                    quality.append({"dataset": "official_holdings", "symbol": symbol,
                                    "status": "cached" if cached else ("downloaded" if got else ("partial" if periods else "unavailable")), "rows": len(got),
                                    "source": "exact-request official ownership cache" if cached else "NSE corporate shareholding master + XBRL",
                                    "first_date": min((r["period"] for r in got), default=""),
                                    "last_date": max((r["period"] for r in got), default=""),
                                    "message": f"latest {periods} calendar-quarter periods by period date, with latest in-period revision; availability by {config.end} verified",
                                    "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds")})
                except Exception as exc:
                    consecutive_errors += 1
                    quality.append({"dataset": "official_holdings", "symbol": symbol, "status": "unavailable", "rows": 0,
                                    "source": "NSE corporate shareholding master + XBRL", "first_date": "", "last_date": "",
                                    "message": str(exc), "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds")})
        if progress:
            current = pd.DataFrame(rows, columns=SCHEMAS["official_holdings"])
            try:
                progress(min(offset + len(batch), len(selected)), len(selected),
                         f"Official shareholding XBRL: {min(offset + len(batch), len(selected))}/{len(selected)}",
                         current, list(quality))
            except TypeError:
                _progress(progress, min(offset + len(batch), len(selected)), len(selected),
                          f"Official shareholding XBRL: {min(offset + len(batch), len(selected))}/{len(selected)}")
        if consecutive_errors >= 4:
            for symbol in selected[offset + len(batch):]:
                quality.append({"dataset": "official_holdings", "symbol": symbol,
                                "status": "deferred_due_source_outage", "rows": 0,
                                "source": "NSE corporate shareholding master + XBRL", "first_date": "", "last_date": "",
                                "message": "Stopped after four consecutive NSE retrieval failures; no values inferred.",
                                "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds")})
            break
    return pd.DataFrame(rows, columns=SCHEMAS["official_holdings"]), quality


def _nse_shareholding_company(symbol: str, *, max_periods: int | None = None) -> tuple[pd.DataFrame, str]:
    """Fetch a company's dated NSE shareholding history and numeric XBRL facts.

    The endpoint returns filing metadata for all available reporting periods;
    individual XBRL disclosures are fetched only for those returned records.
    Callers should bound concurrent symbols and may cap periods for a quick run.
    """
    url = f"{NSE_HOME}/api/corporate-share-holdings-master"
    records = _request(url, params={"index": "equities", "symbol": symbol}, referer=NSE_HOME,
                       timeout=12, retries=0).json()
    if not isinstance(records, list):
        raise ValueError("NSE shareholding endpoint returned a non-list payload")
    # Revisions can expose multiple XBRLs for one quarter; keep the latest
    # exchange broadcast for that quarter, but retain every distinct period.
    by_period = {}
    for item in records:
        period = str(item.get("date") or "").strip()
        if not period:
            continue
        previous = by_period.get(period)
        key = str(item.get("broadcastDate") or item.get("systemDate") or "")
        if previous is None or key > str(previous.get("broadcastDate") or previous.get("systemDate") or ""):
            by_period[period] = item
    ordered = sorted(by_period.values(), key=lambda item: pd.to_datetime(item.get("date"), errors="coerce", dayfirst=True))
    if max_periods is not None:
        ordered = ordered[-max(0, int(max_periods)):]
    rows = []
    errors = []
    for item in ordered:
        file_url = item.get("xbrl")
        if not file_url:
            errors.append(f"{item.get('date')}: no XBRL file URL")
            continue
        try:
            xml = _request(str(file_url), referer=NSE_HOME, timeout=12, retries=0).content
            frame = _parse_nse_shareholding_xbrl(xml, symbol, str(item.get("date")),
                                                  str(item.get("broadcastDate") or item.get("submissionDate") or ""))
            if not frame.empty:
                rows.extend(frame.to_dict("records"))
            else:
                errors.append(f"{item.get('date')}: XBRL lacks institutional shareholding summary facts")
        except Exception as exc:
            errors.append(f"{item.get('date')}: {exc}")
    frame = pd.DataFrame(rows, columns=SCHEMAS["holdings"]).drop_duplicates(["symbol", "period"], keep="last")
    note = f"NSE disclosed {len(frame)} numeric periods from {len(ordered)} filing metadata rows"
    if errors:
        note += f"; {len(errors)} XBRL period(s) unavailable or lacked summary facts"
    return frame.reset_index(drop=True), note


def _screener_history_tables(html: str) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
    """Read Screener's dedicated quarters and shareholding sections only."""
    soup = BeautifulSoup(html, "lxml")
    quarters = soup.select_one("section#quarters, #quarters")
    shareholding = soup.select_one("section#shareholding, #shareholding")
    def read_section(section):
        if section is None:
            return []
        try:
            return pd.read_html(io.StringIO(str(section)))
        except ValueError:
            return []
    return read_section(quarters), read_section(shareholding)


def _table_label(table: pd.DataFrame) -> str:
    return " ".join([str(c) for c in table.columns] + [str(v) for v in table.iloc[:, 0].head(12).tolist()]).lower()


def _metric_row(table: pd.DataFrame, metric_col: pd.Series, labels: tuple[str, ...]):
    for i, label in enumerate(metric_col):
        if label.strip().lower() in labels:
            return list(table.iloc[i, 1:])
    return None


def _fetch_nse_archives(config: ResearchConfig, members: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Fetch exact-date public daily files where NSE archives expose them."""
    deliveries, participant, flows = [], [], []
    d = pd.Timestamp(config.start).date()
    last = pd.Timestamp(config.end).date()
    requests_to_make = []
    while d <= last:
        stamp = d.strftime("%d%m%Y")
        requests_to_make.extend([
            (d, "mto", "NSE security-wise delivery position MTO archive",
             f"https://nsearchives.nseindia.com/archives/equities/mto/MTO_{stamp}.DAT",
             f"https://archives.nseindia.com/archives/equities/mto/MTO_{stamp}.DAT"),
            (d, "oi", "NSE participant OI archive",
             f"https://archives.nseindia.com/content/nsccl/fao_participant_oi_{stamp}.csv",
             f"https://nsearchives.nseindia.com/content/nsccl/fao_participant_oi_{stamp}.csv"),
        ])
        d += timedelta(days=1)

    def download(spec):
        day, parser, label, url, fallback_url = spec
        try:
            try:
                content = _request(url, timeout=8, retries=1).content
            except Exception:
                content = _request(fallback_url, timeout=8, retries=1).content
            # Participant OI has a one-row title; MTO has three metadata rows.
            if parser == "mto":
                # NSE's MTO header omits the Series field although each detail
                # row includes it; provide the seven-column layout explicitly.
                frame = pd.read_csv(io.BytesIO(content), skiprows=3, header=None,
                    names=["Record Type", "Sr No", "Name of Security", "Series", "Quantity Traded",
                           "Deliverable Quantity(gross across client level)",
                           "% of Deliverable Quantity to Traded Quantity"])
            else:
                frame = pd.read_csv(io.BytesIO(content), skiprows=1 if parser == "oi" else 0)
            return day, parser, label, frame
        except Exception:
            return day, parser, label, None

    # Each calendar date is independent; bound requests to four workers so
    # unavailable weekends and blocked hosts do not make a one-run report stall.
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        downloaded = list(pool.map(download, requests_to_make))
    for d, parser, label, frame in downloaded:
        if frame is None:
            continue
        try:
            target = deliveries if parser in {"delivery", "mto"} else participant
            frame.columns = [str(c).strip().upper() for c in frame.columns]
            if parser == "mto":
                series = next((c for c in frame if c == "SERIES"), None)
                symbol_col = next((c for c in frame if c in {"NAME OF SECURITY", "SYMBOL"}), None)
                qty = next((c for c in frame if c == "QUANTITY TRADED"), None)
                delivered = next((c for c in frame if "DELIVERABLE QUANTITY" in c), None)
                pct = next((c for c in frame if c.startswith("% OF DELIVERABLE")), None)
                if series and symbol_col and qty and delivered and pct:
                    eligible = set(members.symbol.dropna().astype(str).str.upper())
                    stock_rows = frame.loc[frame[series].astype(str).str.strip().str.upper().eq("EQ")]
                    for _, row in stock_rows.iterrows():
                        symbol = str(row[symbol_col]).strip().upper()
                        if symbol not in eligible:
                            continue
                        target.append({"date": d.isoformat(), "symbol": symbol,
                                       "delivery_pct": _clean_num(row[pct]),
                                       "traded_qty": _clean_num(row[qty]),
                                       "delivered_qty": _clean_num(row[delivered]),
                                       "source": label})
            elif parser == "delivery":
                sym = next((c for c in frame if c in {"SYMBOL", "TCKRSYMB"}), None)
                qty = next((c for c in frame if "TTL_TRD_QNTY" in c or "TOTTRDQTY" in c), None)
                delivered = next((c for c in frame if "DELIV_QTY" in c), None)
                pct = next((c for c in frame if "DELIV_PER" in c), None)
                if sym:
                    for _, row in frame.iterrows():
                        target.append({"date": d.isoformat(), "symbol": str(row[sym]).upper(),
                                       "delivery_pct": _clean_num(row.get(pct)) if pct else None,
                                       "traded_qty": _clean_num(row.get(qty)) if qty else None,
                                       "delivered_qty": _clean_num(row.get(delivered)) if delivered else None,
                                       "source": label})
            else:
                part = next((c for c in frame if "CLIENT TYPE" in c or "PARTICIPANT" in c), None)
                def oi_column(*terms: str):
                    return next((c for c in frame if all(term in c for term in terms)), None)
                idxlong = oi_column("FUTIDX", "LONG") or oi_column("INDEX FUTURES", "LONG") or oi_column("FUTURE INDEX", "LONG")
                idxshort = oi_column("FUTIDX", "SHORT") or oi_column("INDEX FUTURES", "SHORT") or oi_column("FUTURE INDEX", "SHORT")
                calllong = oi_column("OPTIDX", "CE", "LONG") or oi_column("OPTION INDEX CALL", "LONG")
                callshort = oi_column("OPTIDX", "CE", "SHORT") or oi_column("OPTION INDEX CALL", "SHORT")
                putlong = oi_column("OPTIDX", "PE", "LONG") or oi_column("OPTION INDEX PUT", "LONG")
                putshort = oi_column("OPTIDX", "PE", "SHORT") or oi_column("OPTION INDEX PUT", "SHORT")
                if part and idxlong and idxshort:
                    for _, row in frame.iterrows():
                        longv, shortv = _clean_num(row[idxlong]), _clean_num(row[idxshort])
                        total = (longv or 0) + (shortv or 0)
                        call_value = _clean_num(row[calllong]) if calllong else None
                        put_value = _clean_num(row[putlong]) if putlong else None
                        participant.append({"date": d.isoformat(), "participant": str(row[part]),
                                            "index_long": longv, "index_short": shortv,
                                            "long_share_pct": longv / total * 100 if total else None,
                                            "option_call_long": call_value,
                                            "option_call_short": _clean_num(row[callshort]) if callshort else None,
                                            "option_put_long": put_value,
                                            "option_put_short": _clean_num(row[putshort]) if putshort else None,
                                            "option_oi_pcr": put_value / call_value if call_value else None,
                                            "source": f"{label}; index option PCR=put-long/call-long by participant"})
        except Exception:
            continue
    return {"delivery": pd.DataFrame(deliveries, columns=SCHEMAS["delivery"]),
            "participant_oi": pd.DataFrame(participant, columns=SCHEMAS["participant_oi"]),
            "flows": pd.DataFrame(flows, columns=SCHEMAS["flows"])}


def _fetch_events(config: ResearchConfig) -> pd.DataFrame:
    url = f"{NSE_HOME}/api/corporate-announcements"
    params = {"index": "equities", "from_date": pd.Timestamp(config.start).strftime("DD-MM-YYYY"),
              "to_date": pd.Timestamp(config.end).strftime("DD-MM-YYYY")}
    # NSE expects real formatted dates; strftime's DD token is not supported by Python.
    params["from_date"] = pd.Timestamp(config.start).strftime("%d-%m-%Y")
    params["to_date"] = pd.Timestamp(config.end).strftime("%d-%m-%Y")
    payload = _request(url, params=params, referer=NSE_HOME).json()
    if isinstance(payload, dict):
        payload = payload.get("data", [])
    rows = []
    for item in payload if isinstance(payload, list) else []:
        item = {str(k).lower(): v for k, v in item.items()}
        rows.append({"date": pd.to_datetime(item.get("an_dt") or item.get("date"), errors="coerce").date().isoformat(),
                     "symbol": item.get("symbol", ""), "event": item.get("desc", item.get("subject", "Corporate announcement")),
                     "details": item.get("attchmnttext", item.get("details", "")),
                     "url": item.get("attchmntfile", item.get("url", "")), "source": "NSE corporate announcements"})
    return pd.DataFrame(rows, columns=SCHEMAS["events"])


def _fetch_financial_results(config: ResearchConfig) -> pd.DataFrame:
    """Use NSE's dated results-announcement feed; company-page quarters stay separate."""
    url = f"{NSE_HOME}/api/corporates-financial-results"
    params = {"index": "equities", "from_date": pd.Timestamp(config.start).strftime("%d-%m-%Y"),
              "to_date": pd.Timestamp(config.end).strftime("%d-%m-%Y")}
    payload = _request(url, params=params, referer=NSE_HOME).json()
    if isinstance(payload, dict):
        payload = payload.get("data", [])
    rows = []
    for item in payload if isinstance(payload, list) else []:
        item = {str(k).lower(): v for k, v in item.items()}
        raw_date = item.get("an_dt") or item.get("date") or item.get("broadcastdate")
        parsed_date = pd.to_datetime(raw_date, errors="coerce", dayfirst=True)
        if pd.isna(parsed_date):
            continue
        d = parsed_date.date().isoformat()
        if not (config.start <= d <= config.end):
            continue
        rows.append({"date": d, "symbol": item.get("symbol", ""),
                     "event": "Financial results announcement",
                     "details": item.get("desc") or item.get("subject") or item.get("period") or "",
                     "url": item.get("attchmntfile") or item.get("url") or "",
                     "source": "NSE corporates-financial-results date-range API"})
    return pd.DataFrame(rows, columns=SCHEMAS["events"])


def _fetch_corporate_actions(config: ResearchConfig) -> pd.DataFrame:
    url = f"{NSE_HOME}/api/corporates-corporateActions"
    params = {"index": "equities", "from_date": pd.Timestamp(config.start).strftime("%d-%m-%Y"),
              "to_date": pd.Timestamp(config.end).strftime("%d-%m-%Y")}
    payload = _request(url, params=params, referer=NSE_HOME).json()
    if isinstance(payload, dict):
        payload = payload.get("data", [])
    rows = []
    for item in payload if isinstance(payload, list) else []:
        item = {str(k).lower(): v for k, v in item.items()}
        raw_date = item.get("exdate") or item.get("date") or item.get("bc_start_date")
        parsed_date = pd.to_datetime(raw_date, errors="coerce", dayfirst=True)
        if pd.isna(parsed_date):
            continue
        d = parsed_date.date().isoformat()
        if not (config.start <= d <= config.end):
            continue
        rows.append({"date": d, "symbol": item.get("symbol", ""),
                     "event": "Corporate action",
                     "details": " | ".join(str(item.get(k)) for k in ("purpose", "series", "subject") if item.get(k)),
                     "url": item.get("attchmntfile") or "", "source": "NSE corporate actions date-range API"})
    return pd.DataFrame(rows, columns=SCHEMAS["events"])


def collect(config: ResearchConfig, progress: Callable | None = None) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Collect requested daily public data and return all named tables and health records."""
    tables = {name: _empty(name) for name in TABLE_NAMES}
    quality: list[dict] = []
    start = (pd.Timestamp(config.start) - pd.Timedelta(days=400)).date().isoformat()

    cached = _read_cache("members", config, universe=config.universe)
    if cached is not None:
        members = cached.reindex(columns=SCHEMAS["members"])
        members = members.loc[~members.symbol.astype(str).str.upper().str.startswith("DUMMY")].drop_duplicates("symbol").reset_index(drop=True)
        member_source, member_note = "exact-request cache", "cached current constituent snapshot"
    else:
        members, member_source, member_note = _fetch_members(config)
        _write_cache(members, "members", config, universe=config.universe)
    tables["members"] = members
    quality.append(_quality("members", members, member_source, "downloaded" if "fallback" not in member_source else "partial", member_note))

    # Fetch a sector-spread fundamental sample when an official sector column exists.
    if "sector" in members and not members.sector.fillna("").eq("").all():
        sector_groups = [group.symbol.dropna().astype(str).tolist() for _, group in
                         members.assign(_sector=members.sector.fillna("Other")).groupby("_sector", sort=True)]
        symbols = []
        while any(sector_groups):
            for group in sector_groups:
                if group:
                    symbols.append(group.pop(0))
    else:
        symbols = members.symbol.dropna().astype(str).tolist()

    # Cache prices per symbol so a stopped run can resume without redownloading completed names.
    results: list[tuple[str, pd.DataFrame, str]] = []
    missing_symbols = []
    for symbol in symbols:
        cached_one = _read_cache("prices_symbol", config, symbol=symbol, yahoo_symbol=f"{symbol}.NS", lookback_start=start)
        if cached_one is not None and not cached_one.empty:
            frame = cached_one.reindex(columns=SCHEMAS["prices"])
            results.append((symbol, frame, "exact-request per-symbol cache"))
            quality.append(_quality("prices", frame, "exact-request per-symbol cache", "cached", symbol=symbol))
        else:
            missing_symbols.append(symbol)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_prices_for, f"{symbol}.NS", start, config.end): symbol for symbol in missing_symbols}
        for done, future in enumerate(as_completed(futures), 1):
            symbol = futures[future]
            try:
                _, frame, source = future.result()
                results.append((symbol, frame, source))
                _write_cache(frame, "prices_symbol", config, symbol=symbol, yahoo_symbol=f"{symbol}.NS", lookback_start=start)
                quality.append(_quality("prices", frame, source, "downloaded", symbol=symbol))
            except Exception as exc:
                quality.append(_quality("prices", _empty("prices"), "Yahoo Finance chart", "failed", str(exc), symbol))
            _progress(progress, done, len(futures), f"Stock prices {done}/{len(futures)}")
    tables["prices"] = pd.concat([r[1] for r in results], ignore_index=True) if results else _empty("prices")

    # Only India indices use the slower official NSE history endpoint. Cache
    # each series independently and fetch no more than four series at a time.
    def load_market(item):
        name, (symbol, group, unit) = item
        saved = _read_cache("market", config, name=name, symbol=symbol, lookback_start=start)
        if saved is not None:
            return name, symbol, saved.reindex(columns=SCHEMAS["markets"]), "exact-request cache", "cached", "cached dated history"
        frame = _empty("markets")
        nse_error = ""
        source = "Yahoo Finance chart"
        if group in {"India", "Sector", "Volatility"}:
            try:
                frame = _nse_index_history(name, start, config.end)
                frame["symbol"], frame["group"], frame["unit"] = symbol, group, unit
                source = "NSE Indices historical data"
            except Exception as exc:
                nse_error = str(exc)
        if frame.empty:
            try:
                raw, yahoo_source = _yahoo_chart(symbol, start, config.end)
                frame = raw.assign(symbol=symbol, source=yahoo_source, name=name, group=group, unit=unit).reindex(columns=SCHEMAS["markets"])
                source = yahoo_source
            except Exception as exc:
                return name, symbol, frame, source, "failed", f"{nse_error}; Yahoo: {exc}".strip("; ")
        _write_cache(frame, "market", config, name=name, symbol=symbol, lookback_start=start)
        return name, symbol, frame.reindex(columns=SCHEMAS["markets"]), source, "downloaded", "dated historical observations"

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(load_market, item) for item in MARKETS.items()]
        for done, future in enumerate(as_completed(futures), 1):
            name, symbol, frame, source, status, message = future.result()
            if not frame.empty:
                tables["markets"] = pd.concat([tables["markets"], frame], ignore_index=True)
            quality.append(_quality("markets", frame, source, status, message, symbol))
            _progress(progress, done, len(futures), f"Market series {done}/{len(futures)}")

    archives = _fetch_nse_archives(config, members)
    for table_name, frame in archives.items():
        tables[table_name] = frame
        msg = "dated daily archive rows" if not frame.empty else "public date-specific archive unavailable or blocked for requested dates"
        quality.append(_quality(table_name, frame, "NSE public daily archive", "downloaded" if not frame.empty else "unavailable", msg))
    flows, flow_source, flow_note = _fetch_cash_flows(config)
    tables["flows"] = flows
    quality.append(_quality("flows", flows, flow_source,
                            "partial" if not flows.empty else "unavailable", flow_note))

    event_frames = []
    event_errors = []
    for loader in (_fetch_events, _fetch_financial_results, _fetch_corporate_actions):
        try:
            frame = loader(config)
            if not frame.empty:
                event_frames.append(frame)
        except Exception as exc:
            event_errors.append(f"{loader.__name__}: {exc}")
    if event_frames:
        tables["events"] = pd.concat(event_frames, ignore_index=True).drop_duplicates().reset_index(drop=True)
    event_source = "NSE corporate announcements and financial results date-range APIs"
    event_msg = ("includes results actually announced in requested range" if not tables["events"].empty
                 else "no returned announcements in range; " + ("; ".join(event_errors) if event_errors else "feeds returned no rows"))
    quality.append(_quality("events", tables["events"], event_source,
                            "downloaded" if not tables["events"].empty else "unavailable", event_msg))

    # Full history is available from the published company page. Bound to config cap.
    fundamental_symbols = symbols[:config.fundamental_count]
    missing_fundamentals = []
    partial_fundamental_cache: dict[str, tuple[pd.DataFrame | None, pd.DataFrame | None]] = {}
    for sym in fundamental_symbols:
        qcache = _fundamental_cache("quarterly_symbol", sym, config)
        hcache = _fundamental_cache("holdings_symbol", sym, config)
        if qcache is not None and hcache is not None:
            qframe = qcache.reindex(columns=SCHEMAS["quarterly"]) if qcache is not None else _empty("quarterly")
            hframe = hcache.reindex(columns=SCHEMAS["holdings"]) if hcache is not None else _empty("holdings")
            tables["quarterly"] = _append_observations(tables["quarterly"], qframe)
            tables["holdings"] = _append_observations(tables["holdings"], hframe)
            quality.append(_quality("quarterly", qframe, "exact-request cache", "cached", symbol=sym))
            quality.append(_quality("holdings", hframe, "exact-request cache", "cached", symbol=sym))
        else:
            partial_fundamental_cache[sym] = (qcache, hcache)
            missing_fundamentals.append(sym)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_screener_company, sym): sym for sym in missing_fundamentals}
        completed = 0
        consecutive_connection_errors = 0
        for future in as_completed(futures):
            completed += 1
            sym = futures[future]
            try:
                qframe, hframe, basis, url = future.result()
                consecutive_connection_errors = 0
                qcache = _fundamental_cache("quarterly_symbol", sym, config)
                hcache = _fundamental_cache("holdings_symbol", sym, config)
                if qcache is not None:
                    qframe = qcache.reindex(columns=SCHEMAS["quarterly"])
                if hcache is not None:
                    hframe = hcache.reindex(columns=SCHEMAS["holdings"])
                # Published Screener tables are in INR crore and all available quarters/periods are retained.
                tables["quarterly"] = _append_observations(tables["quarterly"], qframe)
                tables["holdings"] = _append_observations(tables["holdings"], hframe)
                if not qframe.empty:
                    _write_cache(qframe, "quarterly_symbol", config, symbol=sym, universe=config.universe,
                                 fundamental_cap=config.fundamental_count, parser_version="quarters-section-v2")
                if not hframe.empty:
                    _write_cache(hframe, "holdings_symbol", config, symbol=sym, universe=config.universe,
                                 fundamental_cap=config.fundamental_count, parser_version="quarters-section-v2")
                status = "downloaded" if not qframe.empty and not hframe.empty else "partial"
                msg = f"{basis} company page; all displayed history; fetched {datetime.now().date().isoformat()}"
                quality.append(_quality("quarterly", qframe, f"Screener.in ({basis})", status, msg, sym))
                quality.append(_quality("holdings", hframe, f"Screener.in ({basis})", status, msg, sym))
            except Exception as exc:
                cached_q, cached_h = partial_fundamental_cache.get(sym, (None, None))
                for dataset, cached_frame in (("quarterly", cached_q), ("holdings", cached_h)):
                    if cached_frame is not None:
                        cached_frame = cached_frame.reindex(columns=SCHEMAS[dataset])
                        tables[dataset] = _append_observations(tables[dataset], cached_frame)
                        quality.append(_quality(dataset, cached_frame, "exact-request cache", "cached",
                                                "retained after sibling Screener dataset fetch failed", sym))
                    else:
                        cancelled_by_breaker = future.cancelled()
                        status = ("deferred_due_source_outage" if cancelled_by_breaker or _is_connection_timeout(exc)
                                  else "unavailable")
                        message = ("Cancelled after three consecutive Screener connection timeouts; no data inferred."
                                   if cancelled_by_breaker else str(exc))
                        quality.append(_quality(dataset, _empty(dataset), "Screener.in", status, message, sym))
                if future.cancelled():
                    consecutive_connection_errors = 0
                elif _is_connection_timeout(exc):
                    consecutive_connection_errors += 1
                else:
                    consecutive_connection_errors = 0
            _progress(progress, completed, len(futures), f"Fundamentals {completed}/{len(futures)}")
            if consecutive_connection_errors >= 3:
                remaining = [f for f in futures if f not in {future} and not f.done()]
                for pending in remaining:
                    pending.cancel()

    # Snapshot endpoints are included only when their observation date falls inside the requested range.
    today = date.today().isoformat()
    if config.start <= today <= config.end:
        try:
            payload = _request(f"{NSE_HOME}/api/allIndices", referer=NSE_HOME).json()
            rows = payload.get("data", []) if isinstance(payload, dict) else []
            valuation_rows = []
            for item in rows:
                name = item.get("index") or item.get("indexName")
                observation = pd.to_datetime(item.get("lastUpdateTime") or item.get("date"), errors="coerce", dayfirst=True)
                if pd.isna(observation):
                    continue
                observation_date = observation.date().isoformat()
                if not (config.start <= observation_date <= config.end):
                    continue
                pe, pb, div = item.get("pe"), item.get("pb"), item.get("divYield")
                valuation_rows.append({"date": observation_date, "name": name, "pe": _clean_num(pe), "pb": _clean_num(pb),
                                      "dividend_yield_pct": _clean_num(div), "source": "NSE allIndices snapshot"})
            tables["valuations"] = pd.DataFrame(valuation_rows, columns=SCHEMAS["valuations"])
            quality.append(_quality("valuations", tables["valuations"], "NSE allIndices", "downloaded" if valuation_rows else "unavailable",
                                    "dated single snapshot observations; rows without an in-range published timestamp omitted"))
        except Exception as exc:
            quality.append(_quality("valuations", tables["valuations"], "NSE allIndices", "unavailable", str(exc)))
    else:
        quality.append(_quality("valuations", tables["valuations"], "NSE allIndices", "unavailable",
                                f"current snapshot date {today} lies outside requested range; no historical valuation endpoint used"))
    quality.append(_quality("macro", tables["macro"], "RBI public time series", "unavailable",
                            "No dated RBI observation-series endpoint configured; no values inferred from current snapshots"))

    # Cache each raw table under the exact request and as-of date. This prevents a
    # previous range or previous run date from masquerading as the current request.
    for name in TABLE_NAMES:
        if name in {"members", "prices", "markets", "quarterly", "holdings"}:
            continue  # These are cached at universe or individual-symbol granularity above.
        cached_frame = _read_cache(name, config)
        if cached_frame is not None:
            tables[name] = cached_frame.reindex(columns=SCHEMAS[name])
        else:
            _write_cache(tables[name], name, config)

    # Enforce stable schemas for downstream report code, including sources that failed.
    for name in TABLE_NAMES:
        tables[name] = tables[name].reindex(columns=SCHEMAS[name])
    return tables, quality
