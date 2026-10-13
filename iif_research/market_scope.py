"""Additive MarketShack-shaped public market coverage for newsletter research.

Uses NSE Indices/NSE/NSEIX and Yahoo public data. MarketShack's own historical
series are not reused; its methodology and visible index list informed scope.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from .config import ResearchConfig

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125 Safari/537.36"
NSE = "https://www.nseindia.com"
NSE_HISTORY = "https://www.niftyindices.com/BackPage/getHistoricaldatatabletoString"
NSE_HISTORY_PAGE = "https://www.niftyindices.com/reports/historical-data"
MAX_WORKERS = 4

SCHEMAS = {
    "index_snapshot": ["date", "name", "group", "last", "change_pct", "month_return_pct", "year_return_pct", "pe", "pb", "advances", "declines", "source"],
    "sector_history": ["date", "name", "close", "source"],
    "long_index_history": ["date", "name", "close", "source"],
    "members_by_size": ["symbol", "name", "group", "source"],
    "bse_events": ["date", "symbol", "event", "details", "url", "source"],
    "gift_snapshot": ["date", "instrument", "last", "change", "change_pct", "expiry", "volume", "contracts", "source"],
}

MEMBER_LISTS = {
    "Nifty 50": "ind_nifty50list.csv",
    "Nifty 100": "ind_nifty100list.csv",
    "Nifty 200": "ind_nifty200list.csv",
    "Nifty 500": "ind_nifty500list.csv",
    "Nifty Midcap 150": "ind_niftymidcap150list.csv",
    "Nifty Smallcap 250": "ind_niftysmallcap250list.csv",
}


def _empty(name: str) -> pd.DataFrame:
    return pd.DataFrame(columns=SCHEMAS[name])


def _request(url: str, *, params: dict | None = None, method: str = "get", json_body: dict | None = None,
             referer: str | None = None, timeout: int = 10, retries: int = 1) -> requests.Response:
    headers = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}
    if referer:
        headers["Referer"] = referer
    error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.request(method, url, params=params, json=json_body, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            error = exc
            # Permission/rate-limit responses are not usefully retried.
            if getattr(getattr(exc, "response", None), "status_code", None) in (403, 429):
                break
            if attempt < retries:
                time.sleep(.25 * (attempt + 1))
    raise RuntimeError(f"{url}: {error}")


def _key(dataset: str, config: ResearchConfig, **extra: Any) -> str:
    payload = {"dataset": dataset, "start": config.start, "end": config.end,
               "as_of": date.today().isoformat(), "version": 1, **extra}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:24]


def _cache_read(dataset: str, config: ResearchConfig, **extra: Any) -> pd.DataFrame | None:
    path = Path(config.cache_dir) / f"market_scope_{dataset}_{_key(dataset, config, **extra)}.csv"
    if config.refresh or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None


def _cache_write(frame: pd.DataFrame, dataset: str, config: ResearchConfig, **extra: Any) -> None:
    path = Path(config.cache_dir) / f"market_scope_{dataset}_{_key(dataset, config, **extra)}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _quality(dataset: str, frame: pd.DataFrame, source: str, status: str, message: str = "") -> dict[str, Any]:
    values = pd.to_datetime(frame["date"], errors="coerce") if "date" in frame and not frame.empty else pd.Series(dtype="datetime64[ns]")
    return {"dataset": dataset, "status": status, "rows": len(frame), "source": source,
            "first_date": values.min().date().isoformat() if values.notna().any() else "",
            "last_date": values.max().date().isoformat() if values.notna().any() else "",
            "message": message, "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds")}


def _progress(progress: Callable | None, done: int, total: int, message: str) -> None:
    if progress:
        try:
            progress(done, total, message)
        except TypeError:
            progress(message)


def _published_date(payload: dict) -> date | None:
    raw = payload.get("timestamp")
    stamp = pd.to_datetime(raw, errors="coerce", dayfirst=True)
    if pd.isna(stamp):
        return None
    return stamp.date()


def _nse_index_snapshot(config: ResearchConfig) -> tuple[pd.DataFrame, str]:
    payload = _request(f"{NSE}/api/allIndices", referer=NSE).json()
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("NSE allIndices returned an unexpected response")
    observed = _published_date(payload)
    if observed is None:
        raise ValueError("NSE allIndices did not publish an observation timestamp")
    end = pd.Timestamp(config.end).date()
    if observed > end:
        raise ValueError(f"NSE snapshot date {observed} is after requested end {end}")
    rows = []
    for item in payload["data"]:
        name = item.get("index") or item.get("indexSymbol")
        if not name:
            continue
        group = str(item.get("key", "NSE index")).replace(" INDICES", "").replace("INDICES ELIGIBLE IN DERIVATIVES", "Derivatives Eligible").title()
        rows.append({"date": observed.isoformat(), "name": str(name).strip(), "group": group,
                     "last": item.get("last"), "change_pct": item.get("percentChange"),
                     "month_return_pct": item.get("perChange30d"), "year_return_pct": item.get("perChange365d"),
                     "pe": item.get("pe"), "pb": item.get("pb"),
                     "advances": item.get("advances"), "declines": item.get("declines"),
                     "source": f"NSE allIndices snapshot ({observed.isoformat()})"})
    frame = pd.DataFrame(rows, columns=SCHEMAS["index_snapshot"])
    if frame.empty:
        raise ValueError("NSE allIndices returned no named indices")
    if observed < pd.Timestamp(config.start).date():
        msg = f"latest published snapshot is {observed.isoformat()}, before requested range; snapshot is context only, not a range observation"
        status = "prior_snapshot"
    else:
        msg = "published NSE snapshot timestamp is within requested range"
        status = "downloaded"
    return frame, f"{status}: {msg}"


def _nse_history(name: str, start: str, end: str) -> pd.DataFrame:
    body = {"cinfo": json.dumps({"name": name.upper(), "startDate": pd.Timestamp(start).strftime("%d-%b-%Y"),
                                 "endDate": pd.Timestamp(end).strftime("%d-%b-%Y"), "indexName": name})}
    payload = _request(NSE_HISTORY, method="post", json_body=body, referer=NSE_HISTORY_PAGE).json()
    data: Any = payload.get("d") if isinstance(payload, dict) and "d" in payload else payload
    if isinstance(data, str):
        data = json.loads(data)
        data = data.get("d", data) if isinstance(data, dict) else data
    if not isinstance(data, list):
        raise ValueError("NSE historical index response was not a row list")
    raw = pd.DataFrame(data)
    if raw.empty:
        return _empty("sector_history")
    date_col = next((c for c in raw if str(c).lower() in ("historicaldate", "date")), None)
    close_col = next((c for c in raw if str(c).lower() == "close"), None)
    if date_col is None or close_col is None:
        raise ValueError("NSE index history is missing date or close")
    frame = pd.DataFrame({"date": pd.to_datetime(raw[date_col], errors="coerce", dayfirst=True).dt.date.astype("string"),
                          "name": name, "close": pd.to_numeric(raw[close_col].astype(str).str.replace(",", "", regex=False), errors="coerce"),
                          "source": "NSE Indices historical data"})
    return frame.dropna(subset=["date", "close"]).loc[lambda f: f.date.between(start, end)].reset_index(drop=True)


def _yahoo_history(symbol: str, name: str, start: str, end: str, table: str) -> pd.DataFrame:
    p1 = int(pd.Timestamp(start, tz="UTC").timestamp())
    p2 = int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='^=.-')}"
    payload = _request(url, params={"period1": p1, "period2": p2, "interval": "1d", "events": "div,splits"}).json()
    result = payload.get("chart", {}).get("result")
    if not result:
        raise ValueError(payload.get("chart", {}).get("error") or "Yahoo returned no history")
    obj = result[0]
    try:
        tz = ZoneInfo(obj.get("meta", {}).get("exchangeTimezoneName") or "UTC")
    except Exception:
        tz = ZoneInfo("UTC")
    values = obj.get("indicators", {}).get("quote", [{}])[0].get("close", [])
    rows = []
    for stamp, close in zip(obj.get("timestamp", []), values):
        if close is None:
            continue
        day = pd.to_datetime(stamp, unit="s", utc=True).tz_convert(tz).date().isoformat()
        if start <= day <= end:
            rows.append({"date": day, "name": name, "close": close, "source": "Yahoo Finance chart"})
    return pd.DataFrame(rows, columns=SCHEMAS[table])


def _read_constituents(group: str, filename: str) -> tuple[pd.DataFrame, str]:
    url = f"https://www.niftyindices.com/IndexConstituent/{filename}"
    response = _request(url, referer="https://www.niftyindices.com/indices/equity/broad-based-indices")
    raw = pd.read_csv(io.BytesIO(response.content))
    cols = {str(c).strip().lower(): c for c in raw.columns}
    symbol_col = next((cols[k] for k in cols if "symbol" in k), None)
    name_col = next((cols[k] for k in cols if "company" in k or "name" in k), symbol_col)
    if symbol_col is None:
        raise ValueError(f"{filename} has no symbol column")
    symbols = raw[symbol_col].astype(str).str.strip().str.upper()
    mask = ~symbols.str.startswith("DUMMY")
    sector_col = next((cols[k] for k in cols if "industry" in k or "sector" in k), None)
    frame = pd.DataFrame({"symbol": symbols[mask],
                          "name": raw.loc[mask, name_col].astype(str).str.strip() if name_col else symbols[mask],
                          "group": group, "source": "NSE Indices official constituent CSV"})
    frame = frame[frame.symbol.ne("")].drop_duplicates("symbol").reset_index(drop=True)
    return frame, f"{url}; DUMMY* placeholders excluded" if (symbols.str.startswith("DUMMY")).any() else url


def _bse_events(config: ResearchConfig) -> tuple[pd.DataFrame, str]:
    url = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"
    params = {"strCat": "-1", "strPrevDate": pd.Timestamp(config.start).strftime("%Y%m%d"),
              "strScrip": "", "strSearch": "P", "strToDate": pd.Timestamp(config.end).strftime("%Y%m%d"),
              "strType": "C", "subcategory": "-1"}
    records, errors, total_count = [], [], None
    for page in range(1, 4):
        try:
            payload = _request(url, params={**params, "pageno": page}, referer="https://www.bseindia.com/", timeout=8, retries=1).json()
            table = payload.get("Table", []) if isinstance(payload, dict) else []
            meta = payload.get("Table1", []) if isinstance(payload, dict) else []
            if meta and isinstance(meta[0], dict):
                total_count = pd.to_numeric(meta[0].get("ROWCNT"), errors="coerce")
            if not table:
                break
            records.extend(table)
            if len(table) < 50:
                break
        except Exception as exc:
            errors.append(str(exc))
            break
    rows = []
    for item in records:
        raw_date = item.get("NEWS_DT") or item.get("DT_TM") or item.get("NEWS_DT")
        raw_text = str(raw_date or "")
        stamp = pd.to_datetime(raw_text, errors="coerce", dayfirst=not bool(re.match(r"^\d{4}-\d{2}-\d{2}", raw_text)))
        if pd.isna(stamp):
            continue
        day = stamp.date().isoformat()
        if not (config.start <= day <= config.end):
            continue
        attachment = item.get("ATTACHMENTNAME") or item.get("ATTACHMENT") or item.get("NSURL") or ""
        if attachment and str(attachment).startswith("/"):
            attachment = "https://www.bseindia.com" + str(attachment)
        rows.append({"date": day, "symbol": str(item.get("SCRIP_CD") or item.get("SCRIP_CODE") or ""),
                     "event": item.get("NEWSSUB") or item.get("SUBJECT") or "BSE filing",
                     "details": item.get("HEADLINE") or item.get("MORE") or "",
                     "url": attachment, "source": "BSE corporate announcements"})
    frame = pd.DataFrame(rows, columns=SCHEMAS["bse_events"]).drop_duplicates().reset_index(drop=True)
    note = f"retrieved {len(frame)} dated announcements across at most 3 pages"
    if total_count is not None and total_count > len(records):
        note += f"; partial page window ({len(records)} of {int(total_count)} source rows)"
    if errors:
        note += "; API access error: " + errors[-1]
    return frame, note


def _gift_snapshot(config: ResearchConfig) -> tuple[pd.DataFrame, str]:
    """Parse a dated public GIFT quote; never turn a live quote into history."""
    from bs4 import BeautifulSoup
    nseix = "https://www.nseix.com/"
    nseix_error = ""
    try:
        html = _request(nseix, referer=nseix, timeout=8, retries=0).text
        text = re.sub(r"\s+", " ", BeautifulSoup(html, "lxml").get_text(" ", strip=True))
        match = re.search(r"Intra Day Price\s*-\s*Near month GIFT NIFTY Future\s*(.*?)(?:Index Futures|$)", text, re.I)
        numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", match.group(1)) if match else []
        if numbers:
            day = datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()
            if config.start <= day <= config.end:
                row = {"date": day, "instrument": "Near month GIFT NIFTY Future", "last": float(numbers[0].replace(",", "")),
                       "change": float(numbers[1].replace(",", "")) if len(numbers) > 1 else None,
                       "change_pct": float(numbers[2].replace(",", "")) if len(numbers) > 2 else None,
                       "expiry": "", "volume": None, "contracts": None,
                       "source": "NSE IX public page; retrieval date (quote timestamp unavailable in HTML)"}
                return pd.DataFrame([row], columns=SCHEMAS["gift_snapshot"]), "NSE IX quote has no published timestamp; retrieval date labeled and not used as historical series"
        nseix_error = "NSE IX page has no parseable public quote"
    except Exception as exc:
        nseix_error = str(exc)

    # The MarketShack public landing page embeds this numeric quote with its
    # own as-of timestamp and explicitly says it does not retain history.
    fallback_url = "https://themarketshack.com/"
    html = _request(fallback_url, referer=fallback_url, timeout=10, retries=0).text
    key_match = re.search(r'"key"\s*:\s*"gift-nifty"', html, re.I)
    if not key_match:
        raise ValueError(f"NSE IX quote unavailable ({nseix_error}); dated MarketShack numeric quote not found")
    # The public card includes a nested change object, so read named scalar
    # fields from the bounded card window instead of assuming flat JSON.
    card = html[key_match.start():key_match.start() + 1200]
    def scalar(name: str, default=None):
        match = re.search(r'"' + re.escape(name) + r'"\s*:\s*("([^"\\]*(?:\\.[^"\\]*)*)"|[-+]?\d+(?:\.\d+)?|true|false|null)', card, re.I)
        if not match:
            return default
        token = match.group(1)
        if token.startswith('"'):
            return json.loads(token)
        if token.lower() == "null":
            return default
        if token.lower() in ("true", "false"):
            return token.lower() == "true"
        return float(token)
    raw_day = scalar("asof") or scalar("time")
    stamp = pd.to_datetime(raw_day, errors="coerce", utc=True)
    if pd.isna(stamp):
        raise ValueError("MarketShack GIFT quote lacks a valid published as-of date")
    day = stamp.date().isoformat()
    if day > config.end:
        raise ValueError(f"published GIFT quote date {day} is after requested end {config.end}")
    if day < config.start:
        raise ValueError(f"latest published GIFT quote ({day}) predates requested range ({config.start})")
    last, change, previous = scalar("value"), scalar("pts"), scalar("prev")
    if last is None or change is None:
        raise ValueError("MarketShack GIFT quote lacks last or point change")
    change_pct = float(change) / float(previous) * 100 if previous not in (None, 0, "0") else None
    row = {"date": day, "instrument": scalar("name", "GIFT NIFTY"), "last": last,
           "change": change, "change_pct": change_pct, "expiry": scalar("expiry", ""),
           "volume": scalar("volume"), "contracts": scalar("contracts"),
           "source": "MarketShack public embedded GIFT NIFTY numeric snapshot; no historical series retained"}
    return pd.DataFrame([row], columns=SCHEMAS["gift_snapshot"]), f"dated snapshot only ({day}); NSE IX quote unavailable ({nseix_error}); MarketShack does not retain history"


def collect_market_scope(config: ResearchConfig, progress: Callable | None = None) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Return additional index, size-universe, global-history, and BSE tables."""
    tables = {name: _empty(name) for name in SCHEMAS}
    quality = []

    snapshot = _cache_read("index_snapshot", config)
    snapshot_note = "exact request/as-of cache"
    if snapshot is None:
        try:
            snapshot, snapshot_note = _nse_index_snapshot(config)
            _cache_write(snapshot, "index_snapshot", config)
        except Exception as exc:
            snapshot_note = f"NSE allIndices unavailable: {exc}"
            snapshot = _empty("index_snapshot")
    tables["index_snapshot"] = snapshot.reindex(columns=SCHEMAS["index_snapshot"])
    status = "cached" if "cache" in snapshot_note else ("downloaded" if "downloaded" in snapshot_note else ("prior_snapshot" if "prior_snapshot" in snapshot_note else "unavailable"))
    quality.append(_quality("index_snapshot", tables["index_snapshot"], "NSE allIndices", status, snapshot_note))

    # Full broad-market and sectoral price-index coverage (38 series) comes from
    # NSE's official history endpoint, sourced from its currently published list.
    if not snapshot.empty:
        broad = snapshot.loc[snapshot.group.str.lower().eq("broad market"), "name"].tolist()
        sectoral = snapshot.loc[snapshot.group.str.lower().eq("sectoral"), "name"].tolist()
        index_names = list(dict.fromkeys(sectoral + [name for name in broad if "vix" not in name.lower()]))[:38]
    else:
        index_names = []
    start = (pd.Timestamp(config.start) - pd.Timedelta(days=400)).date().isoformat()
    def load_sector(name):
        cached = _cache_read("sector_history", config, name=name, range_start=start)
        if cached is not None:
            return name, cached.reindex(columns=SCHEMAS["sector_history"]), "cached", "exact request/as-of cache"
        try:
            frame = _nse_history(name, start, config.end)
            if frame.empty:
                raise ValueError("NSE returned no daily bars")
            _cache_write(frame, "sector_history", config, name=name, range_start=start)
            return name, frame, "downloaded", "official dated NSE price-index history"
        except Exception as exc:
            return name, _empty("sector_history"), "unavailable", str(exc)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = [pool.submit(load_sector, name) for name in index_names]
        for done, future in enumerate(as_completed(futures), 1):
            name, frame, status, message = future.result()
            if not frame.empty:
                tables["sector_history"] = pd.concat([tables["sector_history"], frame], ignore_index=True)
            quality.append(_quality("sector_history", frame, "NSE Indices history", status, f"{name}: {message}"))
            _progress(progress, done, len(futures), f"NSE sector/broad index histories {done}/{len(futures)}")
    if not index_names:
        quality.append(_quality("sector_history", tables["sector_history"], "NSE Indices history", "unavailable",
                                "Index catalogue unavailable because NSE allIndices snapshot could not be read"))

    end5 = pd.Timestamp(config.end).date()
    start5 = end5 - timedelta(days=5 * 365 + 2)
    long_cache = _cache_read("long_index_history", config, symbol="^NSEI", start=start5.isoformat())
    if long_cache is None:
        try:
            long_cache = _yahoo_history("^NSEI", "Nifty 50", start5.isoformat(), config.end, "long_index_history")
            _cache_write(long_cache, "long_index_history", config, symbol="^NSEI", start=start5.isoformat())
            long_status, long_note = "downloaded", "five-year Yahoo daily history; range bounded to requested end"
        except Exception as exc:
            long_cache = _empty("long_index_history")
            long_status, long_note = "unavailable", str(exc)
    else:
        long_status, long_note = "cached", "exact request/as-of cache"
    tables["long_index_history"] = long_cache.reindex(columns=SCHEMAS["long_index_history"])
    quality.append(_quality("long_index_history", tables["long_index_history"], "Yahoo Finance chart", long_status, long_note))

    member_frames = []
    member_statuses = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_read_constituents, group, filename): (group, filename)
                   for group, filename in MEMBER_LISTS.items()}
        for done, future in enumerate(as_completed(futures), 1):
            group, filename = futures[future]
            try:
                frame, source = future.result()
                member_frames.append(frame)
                status, message = "downloaded", f"{group}: {len(frame)} members; {source}"
            except Exception as exc:
                frame = _empty("members_by_size")
                status, message = "unavailable", f"{group}: {exc}"
            member_statuses.append((frame, status, message))
            _progress(progress, done, len(futures), f"Official size-universe lists {done}/{len(futures)}")
    tables["members_by_size"] = pd.concat(member_frames, ignore_index=True) if member_frames else _empty("members_by_size")
    for frame, status, message in member_statuses:
        quality.append(_quality("members_by_size", frame, "NSE Indices constituent CSV", status, message))

    bse_cache = _cache_read("bse_events", config)
    if bse_cache is None:
        try:
            bse_cache, bse_note = _bse_events(config)
            bse_status = "downloaded" if not bse_cache.empty else "unavailable"
            _cache_write(bse_cache, "bse_events", config)
        except Exception as exc:
            bse_cache, bse_status, bse_note = _empty("bse_events"), "unavailable", str(exc)
    else:
        bse_status, bse_note = "cached", "exact request/as-of cache"
    tables["bse_events"] = bse_cache.reindex(columns=SCHEMAS["bse_events"])
    quality.append(_quality("bse_events", tables["bse_events"], "BSE corporate announcements API", bse_status, bse_note))

    gift_cache = _cache_read("gift_snapshot", config)
    if gift_cache is None or gift_cache.empty:
        try:
            gift_cache, gift_note = _gift_snapshot(config)
            gift_status = "downloaded" if not gift_cache.empty else "unavailable"
            _cache_write(gift_cache, "gift_snapshot", config)
        except Exception as exc:
            gift_cache, gift_status, gift_note = _empty("gift_snapshot"), "unavailable", str(exc)
    else:
        gift_status, gift_note = "cached", "exact request/as-of cache"
    tables["gift_snapshot"] = gift_cache.reindex(columns=SCHEMAS["gift_snapshot"])
    quality.append(_quality("gift_snapshot", tables["gift_snapshot"], "NSE IX public quote", gift_status, gift_note))

    return tables, quality
