"""Public macroeconomic data for the IIF newsletter.

The collector uses official MoSPI, FRED and World Bank endpoints. Most official
series are revised histories; ``available_at`` is deliberately left unknown so
an observation date is never confused with a release/vintage date.
"""
from __future__ import annotations

from datetime import date
from io import StringIO
from pathlib import Path
import hashlib
import re
import ssl
from typing import Any

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ECONOMY_COLUMNS = ["date", "country", "indicator", "value", "unit", "source", "frequency", "period", "available_at", "fetched_at"]
CALENDAR_COLUMNS = ["date", "country", "event", "details", "url", "source"]
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
MOSPI_ROOT = "https://api.mospi.gov.in/api/"
WORLD_BANK_ROOT = "https://api.worldbank.org/v2/country"
MOSPI_RELEASE_CALENDAR_URL = "https://www.mospi.gov.in/advance-release-calendar"
BLS_ICS_URL = "https://www.bls.gov/schedule/news_release/bls.ics"

COUNTRIES = {
    "IN": "India", "US": "United States", "CN": "China", "JP": "Japan",
    "GB": "United Kingdom", "DE": "Germany", "EMU": "Euro area (EU)",
    "AU": "Australia", "CA": "Canada", "BR": "Brazil", "RU": "Russia",
    "KR": "Korea, Rep.", "FR": "France", "IT": "Italy", "ID": "Indonesia",
    "MX": "Mexico", "ZA": "South Africa", "TR": "Turkey", "SA": "Saudi Arabia",
    "AR": "Argentina",
}
WB_INDICATORS = {
    "NY.GDP.MKTP.KD.ZG": "Real GDP growth",
    "FP.CPI.TOTL.ZG": "CPI inflation",
}
WB_COUNTRY_LABELS = {
    "russian federation": "Russia",
    "korea, rep.": "South Korea",
    "korea, rep": "South Korea",
    "turkiye": "Turkey",
    "euro area": "Euro area (EU)",
}
FRED_SERIES = {
    "INDIRLTLT01STM": ("India 10-year government bond yield", "monthly", "percent", 0, "level"),
    "CPIAUCSL": ("US CPI inflation", "monthly", "percent", 12, "index"),
    "UNRATE": ("US unemployment rate", "monthly", "percent", 0, "level"),
    "FEDFUNDS": ("US federal funds rate", "monthly", "percent", 0, "level"),
    "DGS10": ("US 10-year Treasury yield", "daily", "percent", 0, "level"),
    # FRED returns 404 for the requested CUSR0000SA0L1E identifier; CPILFESL is
    # its active, seasonally adjusted U.S. CPI less food and energy series.
    "CPILFESL": ("US core CPI inflation", "monthly", "percent", 12, "index"),
    "GDPC1": ("US real GDP growth", "quarterly", "percent", 4, "index"),
}
MOSPI_SERIES = {
    "india_cpi": {
        "indicator": "CPI inflation", "frequency": "monthly", "unit": "percent",
        "url": MOSPI_ROOT + "cpi/getCPIIndex",
        "params": {"base_year": "2012", "series": "Current", "state_code": 99,
                   "sector_code": 3, "group_code": "0", "subgroup_code": "0.99"},
        "value_fields": ("index_value", "index", "value"), "transform_lag": 12,
    },
    "india_iip": {
        "indicator": "IIP growth", "frequency": "monthly", "unit": "percent",
        "url": MOSPI_ROOT + "iip/getIIPMonthly",
        "params": {"base_year": "2011-12", "category_code": 4, "type": "General"},
        "value_fields": ("index", "index_value", "value"), "transform_lag": 12,
    },
    "india_real_gdp": {
        "indicator": "Real GDP growth", "frequency": "quarterly", "unit": "percent",
        "url": MOSPI_ROOT + "nas/getNASData",
        "params": {"base_year": "2011-12", "series": "Current", "frequency_code": "Quarterly", "indicator_code": 5},
        "value_fields": ("constant_price", "constantPrice", "real_gdp", "value"), "transform_lag": 4,
    },
}


class _LegacyMoSPIAdapter(HTTPAdapter):
    """Allow MoSPI's legacy TLS renegotiation while retaining certificate checks."""
    def init_poolmanager(self, *args, **kwargs):
        context = ssl.create_default_context()
        context.options |= 0x4  # OP_LEGACY_SERVER_CONNECT; verify_mode remains CERT_REQUIRED.
        kwargs["ssl_context"] = context
        return super().init_poolmanager(*args, **kwargs)


def _session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = "IIF-Research/1.0 (public-data collector)"
    retry = Retry(total=1, connect=1, read=1, status=1, backoff_factor=.25,
                  status_forcelist=(429, 500, 502, 503, 504), allowed_methods=frozenset(["GET"]))
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.mount("https://api.mospi.gov.in/", _LegacyMoSPIAdapter(max_retries=retry))
    return session


def _fetched_at() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def _period_end_month(value: Any) -> pd.Timestamp:
    text = str(value).strip()
    dt = pd.to_datetime(text, errors="coerce")
    if pd.isna(dt):
        dt = pd.to_datetime(text, format="%Y-%B", errors="coerce")
    if pd.isna(dt):
        dt = pd.to_datetime(text, format="%Y-%b", errors="coerce")
    return pd.Timestamp(dt).to_period("M").to_timestamp("M") if pd.notna(dt) else pd.NaT


def _record_period(record: dict[str, Any], frequency: str) -> pd.Timestamp:
    lower = {str(k).lower(): v for k, v in record.items()}
    if frequency == "quarterly":
        quarter = str(lower.get("quarter", lower.get("qtr", "")))
        year = str(lower.get("year", lower.get("financial_year", "")))
        match = re.search(r"Q\s*([1-4])", quarter, re.I)
        if not match:
            match = re.search(r"Q\s*([1-4])", year, re.I)
        if not match:
            return pd.NaT
        q = int(match.group(1))
        y = re.search(r"(20\d{2})", year)
        if not y:
            return pd.NaT
        fy_start = int(y.group(1))
        if "-" in year or "/" in year:
            yr, month = (fy_start + (1 if q == 4 else 0), ((q + 1) * 3) % 12 or 12)
            if q < 4:
                yr = fy_start
        else:
            yr, month = fy_start, q * 3
        return pd.Timestamp(year=yr, month=month, day=1) + pd.offsets.MonthEnd(0)
    month = lower.get("month")
    year = lower.get("year")
    if month is not None and year is not None:
        dt = _period_end_month(f"{year}-{month}")
        if pd.notna(dt):
            return dt
    for field in ("date", "period", "time_period", "reference_period"):
        if lower.get(field):
            dt = pd.to_datetime(lower[field], errors="coerce")
            if pd.notna(dt):
                return dt.to_period("M").to_timestamp("M")
    return pd.NaT


def _value(record: dict[str, Any], fields: tuple[str, ...]) -> float:
    lower = {str(k).lower(): v for k, v in record.items()}
    for field in fields:
        candidate = lower.get(field.lower())
        value = pd.to_numeric(candidate, errors="coerce")
        if pd.notna(value):
            return float(value)
    return np.nan


def _parse_mospi_records(records: list[dict[str, Any]], frequency: str, fields: tuple[str, ...]) -> pd.Series:
    pairs = []
    for row in records:
        if not isinstance(row, dict):
            continue
        period, value = _record_period(row, frequency), _value(row, fields)
        if pd.notna(period) and pd.notna(value):
            pairs.append((period, value))
    if not pairs:
        return pd.Series(dtype=float)
    pair_frame = pd.DataFrame(pairs, columns=["period", "value"])
    if pair_frame.period.duplicated().any():
        raise ValueError("MoSPI response has multiple values per national period.")
    series = pair_frame.set_index("period").value.sort_index()
    return series


def _fetch_mospi(session: requests.Session, definition: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    records: list[dict[str, Any]] = []
    page = 1
    while True:
        params = {**definition["params"], "Format": "JSON", "limit": 100, "page": page}
        response = session.get(definition["url"], params=params, timeout=12)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            rows = payload.get("data")
            metadata = payload.get("meta_data", payload.get("metadata", {})) or {}
        elif isinstance(payload, list):
            rows, metadata = payload, {}
        else:
            rows, metadata = None, {}
        if not isinstance(rows, list):
            raise ValueError("MoSPI returned a non-observation response.")
        records.extend(rows)
        pages = int(metadata.get("totalPages", metadata.get("total_pages", 1)) or 1)
        if page >= pages:
            break
        if page >= 4:
            raise ValueError("MoSPI response exceeds four pages; bounded fetch stopped.")
        page += 1
    series = _parse_mospi_records(records, definition["frequency"], definition["value_fields"])
    if series.empty:
        raise ValueError("MoSPI returned no usable national observations.")
    # One year of warm-up is necessary for monthly/quarterly year-on-year rates.
    warm_start = start - pd.DateOffset(years=1)
    return series.loc[(series.index >= warm_start) & (series.index <= end)]


def _fetch_fred(session: requests.Session, code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    response = session.get(FRED_CSV_URL, params={"id": code, "cosd": (start - pd.DateOffset(years=1)).date().isoformat(),
                                                  "coed": end.date().isoformat()}, timeout=12)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    if frame.shape[1] < 2:
        raise ValueError("FRED CSV did not contain date/value columns.")
    dates = pd.to_datetime(frame.iloc[:, 0], errors="coerce")
    values = pd.to_numeric(frame.iloc[:, 1], errors="coerce")
    series = pd.Series(values.to_numpy(), index=dates).dropna().sort_index()
    series = series.loc[~series.index.isna()]
    if series.empty:
        raise ValueError("FRED returned no numeric observations.")
    return series


def _fetch_world_bank(session: requests.Session, indicator: str, start_year: int, end_year: int) -> list[dict[str, Any]]:
    codes = ";".join(COUNTRIES)
    url = f"{WORLD_BANK_ROOT}/{codes}/indicator/{indicator}"
    response = session.get(url, params={"format": "json", "date": f"{start_year}:{end_year}", "per_page": 2000}, timeout=12)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        raise ValueError("World Bank API returned no observation page.")
    return payload[1]


def _world_bank_cache_key(indicator: str) -> str:
    """Version WDI caches when the requested country scope changes."""
    digest = hashlib.sha256(";".join(sorted(COUNTRIES)).encode("utf-8")).hexdigest()[:12]
    normalized = indicator.lower().replace(".", "_")
    return f"worldbank_v2_{digest}_{normalized}"


def _transform(series: pd.Series, lag: int) -> pd.Series:
    series = series.sort_index().astype(float)
    if lag:
        return series.pct_change(periods=lag, fill_method=None).mul(100).replace([np.inf, -np.inf], np.nan).dropna()
    return series.dropna()


def _table_rows(series: pd.Series, *, country: str, indicator: str, unit: str,
                source: str, frequency: str, fetched_at: pd.Timestamp) -> pd.DataFrame:
    if series.empty:
        return pd.DataFrame(columns=ECONOMY_COLUMNS)
    dates = pd.to_datetime(series.index).tz_localize(None) if getattr(series.index, "tz", None) is not None else pd.to_datetime(series.index)
    if frequency == "monthly":
        period = dates.to_period("M").astype(str)
    elif frequency == "quarterly":
        period = dates.to_period("Q").astype(str)
    elif frequency == "annual":
        period = dates.year.astype(str)
    else:
        period = dates.strftime("%Y-%m-%d")
    return pd.DataFrame({"date": dates, "country": country, "indicator": indicator,
                         "value": pd.to_numeric(series.to_numpy(), errors="coerce"), "unit": unit,
                         "source": source, "frequency": frequency, "period": period,
                         "available_at": pd.NaT, "fetched_at": fetched_at}, columns=ECONOMY_COLUMNS)


def _load_cache(path: Path, end: pd.Timestamp) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path, parse_dates=["date", "available_at", "fetched_at"])
        if not set(ECONOMY_COLUMNS).issubset(frame.columns):
            return None
        frame["period"] = frame.period.astype(str)
        return frame.loc[frame.date.le(end), ECONOMY_COLUMNS]
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def _save_cache(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _collect_calendar(session: requests.Session, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Read actual upcoming BLS release items from its official iCalendar feed."""
    quality: list[dict[str, str]] = []
    try:
        response = session.get(BLS_ICS_URL, timeout=12)
        response.raise_for_status()
        text = response.text
        blocks = re.findall(r"BEGIN:VEVENT\s*(.*?)\s*END:VEVENT", text, flags=re.S)
        rows = []
        window_end = end + pd.Timedelta(days=45)
        for block in blocks:
            start_match = re.search(r"^DTSTART(?:;[^:]*)?:(\d{8})", block, flags=re.M)
            summary = re.search(r"^SUMMARY:(.+)$", block, flags=re.M)
            if not start_match or not summary:
                continue
            event_date = pd.to_datetime(start_match.group(1), format="%Y%m%d", errors="coerce")
            title = summary.group(1).replace("\\,", ",").replace("\\n", " ").strip()
            if pd.notna(event_date) and start <= event_date <= window_end:
                rows.append({"date": event_date, "country": "United States", "event": title,
                             "details": "Scheduled BLS release date; occurrence is not inferred from the schedule.",
                             "url": "https://www.bls.gov/schedule/", "source": "U.S. Bureau of Labor Statistics"})
        frame = pd.DataFrame(rows, columns=CALENDAR_COLUMNS).drop_duplicates(["date", "event"])
        if frame.empty:
            quality.append({"source": "BLS release calendar", "status": "No matching events", "message": "No scheduled BLS entries fell within the report end date plus 45 days."})
        else:
            quality.append({"source": "BLS release calendar", "status": "Loaded", "message": f"Loaded {len(frame)} scheduled U.S. release events; coverage is limited to BLS."})
        return frame.sort_values(["date", "event"]).reset_index(drop=True), quality
    except Exception as exc:
        quality.append({"source": "BLS release calendar", "status": "Unavailable", "message": str(exc)[:240]})
        return pd.DataFrame(columns=CALENDAR_COLUMNS), quality


def collect_economy(config: Any, progress=None) -> tuple[dict[str, pd.DataFrame], list[dict[str, str]]]:
    """Fetch five-year macro context and release-calendar rows.

    The latest annual World Bank observation may predate ``config.end``; its
    observation year is preserved. All displayed series are cut off at end.
    """
    end = pd.Timestamp(config.end).normalize()
    output_start = end - pd.DateOffset(years=5)
    fetch_start = output_start - pd.DateOffset(years=1)
    cache_dir = Path(getattr(config, "cache_dir", Path("data/cache"))) / "economy"
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = _session()
    quality: list[dict[str, str]] = [{"source": "availability caveat", "status": "Unverified",
        "message": "available_at is blank: fetched/revised history does not establish when each observation was first published; this is not point-in-time data."}]
    frames: list[pd.DataFrame] = []
    jobs = [("india_cpi", "MoSPI CPI"), ("india_iip", "MoSPI IIP"), ("india_real_gdp", "MoSPI GDP")]
    jobs += [(f"fred_{code.lower()}", f"FRED {meta[0]}") for code, meta in FRED_SERIES.items()]
    wb_cache_keys = {_world_bank_cache_key(indicator): indicator for indicator in WB_INDICATORS}
    jobs += [(_world_bank_cache_key(indicator), f"World Bank {name}") for indicator, name in WB_INDICATORS.items()]
    total = len(jobs) + 1
    for index, (cache_key, label) in enumerate(jobs):
        if progress:
            progress(f"Global economy {index + 1}/{total}: {label}")
        cached = None if getattr(config, "refresh", False) else _load_cache(cache_dir / f"{cache_key}.csv", end)
        if cached is not None and not cached.empty:
            frames.append(cached)
            quality.append({"source": label, "status": "Cached", "message": f"Loaded {len(cached)} cached observations."})
            continue
        try:
            fetched = _fetched_at()
            if cache_key.startswith("india_"):
                definition = MOSPI_SERIES[cache_key]
                raw = _fetch_mospi(session, definition, fetch_start, end)
                series = _transform(raw, definition["transform_lag"])
                series = series.loc[series.index >= output_start]
                frame = _table_rows(series, country="India", indicator=definition["indicator"], unit=definition["unit"],
                                    source="Ministry of Statistics and Programme Implementation (MoSPI)",
                                    frequency=definition["frequency"], fetched_at=fetched)
                source_url = definition["url"]
            elif cache_key.startswith("fred_"):
                code = cache_key.removeprefix("fred_").upper()
                # Cache keys map to full FRED identifiers (some contain digits).
                code = next(c for c in FRED_SERIES if c.lower() == cache_key.removeprefix("fred_"))
                name, frequency, unit, lag, kind = FRED_SERIES[code]
                raw = _fetch_fred(session, code, fetch_start, end)
                series = _transform(raw, lag)
                series = series.loc[(series.index >= output_start) & (series.index <= end)]
                frame = _table_rows(series, country="India" if code == "INDIRLTLT01STM" else "United States", indicator=name, unit=unit,
                                    source=f"Federal Reserve Bank of St. Louis FRED ({code})", frequency=frequency, fetched_at=fetched)
                source_url = f"https://fred.stlouisfed.org/series/{code}"
            else:
                indicator = wb_cache_keys[cache_key]
                records = _fetch_world_bank(session, indicator, fetch_start.year, end.year)
                rows = []
                for record in records:
                    if record.get("value") is None or str(record.get("obs_status", "")).upper() in {"F", "P"}:
                        continue
                    country_code = str(record.get("countryiso3code", ""))
                    # WB response countryiso3code uses ISO-3, so resolve via code/slug metadata.
                    country_meta = record.get("country", {}) or {}
                    raw_name = str(country_meta.get("value", ""))
                    country = WB_COUNTRY_LABELS.get(raw_name.lower(), raw_name)
                    period_year = int(record["date"])
                    obs_date = pd.Timestamp(year=period_year, month=12, day=31)
                    if obs_date < output_start or obs_date > end:
                        continue
                    rows.append({"date": obs_date, "country": country, "indicator": WB_INDICATORS[indicator],
                                 "value": float(record["value"]), "unit": "percent", "source": f"World Bank WDI ({indicator})",
                                 "frequency": "annual", "period": str(period_year), "available_at": pd.NaT, "fetched_at": fetched})
                frame = pd.DataFrame(rows, columns=ECONOMY_COLUMNS)
                source_url = f"https://api.worldbank.org/v2/country/{';'.join(COUNTRIES)}/indicator/{indicator}"
                if frame.empty:
                    raise ValueError("World Bank returned no observations in the requested five-year window.")
            _save_cache(frame, cache_dir / f"{cache_key}.csv")
            frames.append(frame)
            quality.append({"source": label, "status": "Loaded", "message": f"Loaded {len(frame)} rows from {source_url}."})
        except Exception as exc:
            # Keep usable cache if refresh failed, so one source outage does not erase a previous run.
            fallback = _load_cache(cache_dir / f"{cache_key}.csv", end)
            if fallback is not None and not fallback.empty:
                frames.append(fallback)
                status = "Cached fallback"
                message = f"Source failed; used {len(fallback)} cached rows. {str(exc)[:160]}"
            else:
                status = "Unavailable"
                message = str(exc)[:240]
            quality.append({"source": label, "status": status, "message": message})

    # Stable country display names are based on API country metadata; suppress
    # aggregates not explicitly requested (but retain the Euro area aggregate).
    economy = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=ECONOMY_COLUMNS)
    if not economy.empty:
        for col in ("date", "available_at", "fetched_at"):
            economy[col] = pd.to_datetime(economy[col], errors="coerce")
        # Apply display-name normalization to both fresh pulls and cached rows.
        economy["country"] = economy.country.map(
            lambda name: WB_COUNTRY_LABELS.get(str(name).lower(), name)
        )
        economy["period"] = economy.period.astype(str)
        economy["value"] = pd.to_numeric(economy.value, errors="coerce")
        economy = economy.loc[economy.date.between(output_start, end) & economy.value.notna()]
        economy = economy.drop_duplicates(["country", "indicator", "period", "source"], keep="last")
        economy = economy.sort_values(["country", "indicator", "date"]).reset_index(drop=True)
    calendar, calendar_quality = _collect_calendar(session, pd.Timestamp(config.start).normalize(), end)
    quality.extend(calendar_quality)
    if progress:
        progress("Global economy data complete")
    return {"global_economy": economy[ECONOMY_COLUMNS], "economic_calendar": calendar[CALENDAR_COLUMNS]}, quality
