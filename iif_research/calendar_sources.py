"""Supplemental public economic-release calendars.

This module deliberately stays separate from :mod:`economy`: each publisher's
release calendar is date-stamped and remains distinct from the period dates of
its statistical series.  Only official publisher pages are requested.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


CALENDAR_COLUMNS = ["date", "country", "event", "details", "url", "source", "event_status", "fetched_at"]
SOURCES = {
    "BEA": ("https://www.bea.gov/news/schedule/full", "U.S. Bureau of Economic Analysis (BEA)", "United States"),
    "Federal Reserve": ("https://www.federalreserve.gov/newsevents/{year}-{month_name}.htm", "Federal Reserve Board", "United States"),
    "ONS": ("https://www.ons.gov.uk/releasecalendar", "Office for National Statistics (ONS)", "United Kingdom"),
}
TIMEOUT = 12


def _soup(html: str | bytes) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def _in_range(value: Any, start: pd.Timestamp, end: pd.Timestamp) -> pd.Timestamp | None:
    dt = pd.to_datetime(value, errors="coerce")
    if pd.isna(dt):
        return None
    dt = pd.Timestamp(dt).normalize()
    return dt if start.normalize() <= dt <= end.normalize() else None


def _parse_ons_calendar(html: str | bytes, start: Any, end: Any, *, base_url: str = SOURCES["ONS"][0]) -> pd.DataFrame:
    """Parse ONS release-card data attributes, preserving the published status."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    rows: list[dict[str, Any]] = []
    for link in _soup(html).select("a[data-gtm-release-date][data-gtm-release-title]"):
        date_text = link.get("data-gtm-release-date", "")
        if not re.fullmatch(r"\d{8}", date_text):
            continue
        date = _in_range(pd.to_datetime(date_text, format="%Y%m%d", errors="coerce"), start_ts, end_ts)
        if date is None:
            continue
        parent = link.find_parent("li")
        status_text = parent.get_text(" ", strip=True) if parent else ""
        if "Cancelled" in status_text:
            status = "Cancelled"
        elif "Published" in status_text:
            status = "Published"
        elif "Upcoming" in status_text:
            status = "Scheduled"
        else:
            status = "Date listed by ONS"
        time_text = link.get("data-gtm-release-time", "")
        title = link.get("data-gtm-release-title", link.get_text(" ", strip=True))
        rows.append({
            "date": date,
            "country": "United Kingdom",
            "event": title,
            "details": f"ONS release calendar entry; listed time {time_text} UK time; status: {status}.",
            "url": urljoin(base_url, link.get("href", "")),
            "source": "Office for National Statistics (ONS) release calendar",
            "event_status": status,
        })
    return pd.DataFrame(rows, columns=CALENDAR_COLUMNS[:-1])


def _parse_bea_schedule(html: str | bytes, start: Any, end: Any, *, base_url: str = SOURCES["BEA"][0]) -> pd.DataFrame:
    """Parse the official BEA schedule table; dates remain scheduled dates."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    rows: list[dict[str, Any]] = []
    soup = _soup(html)
    for table in soup.select("table"):
        table_year = None
        for heading in table.select("th"):
            match = re.search(r"\bYear\s+(20\d{2})\b", heading.get_text(" ", strip=True), re.I)
            if match:
                table_year = int(match.group(1))
                break
        for tr in table.select("tr"):
            date_cell = tr.select_one("td.scheduled-date")
            title_cell = tr.select_one("td.release-title")
            if date_cell is None or title_cell is None:
                continue
            date_text = date_cell.select_one(".release-date")
            if date_text is None:
                continue
            date_label = date_text.get_text(" ", strip=True)
            row_year = re.search(r"\b(20\d{2})\b", date_label)
            year = int(row_year.group(1)) if row_year else table_year
            # Never infer a missing calendar year from the requested interval:
            # doing so can move a current-year schedule into a historical year.
            if year is None:
                continue
            date_value = date_label if row_year else f"{date_label} {year}"
            date = _in_range(date_value, start_ts, end_ts)
            if date is None:
                continue
            title = title_cell.get_text(" ", strip=True)
            clock = date_cell.find("small")
            time_text = clock.get_text(" ", strip=True) if clock else "time not listed"
            link = tr.select_one("td.views-field-field-scheduled-release-url a[href]")
            rows.append({
                "date": date,
                "country": "United States",
                "event": title,
                "details": f"Official BEA release schedule; scheduled time {time_text} Eastern Time.",
                "url": urljoin(base_url, link["href"]) if link else base_url,
                "source": "U.S. Bureau of Economic Analysis (BEA) release schedule",
                "event_status": "Scheduled date (official schedule)",
            })
    return pd.DataFrame(rows, columns=CALENDAR_COLUMNS[:-1])


def _parse_fed_calendar(html: str | bytes, year: int, month: int, start: Any, end: Any,
                        *, base_url: str) -> pd.DataFrame:
    """Parse official Federal Reserve monthly release-date lists.

    Speeches are omitted. FOMC, Beige Book, and statistical release sections are
    retained, with multiple listed release dates expanded into separate rows.
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    rows: list[dict[str, Any]] = []
    soup = _soup(html)
    for header in soup.select(".row.cal-nojs__rowTitle h4"):
        section = header.get_text(" ", strip=True)
        if section not in {"FOMC Meetings", "Beige Book", "Statistical Releases"}:
            continue
        row = header.find_parent("div", class_="row")
        if row is None:
            continue
        for sibling in row.find_next_siblings("div", class_="row"):
            if sibling.select_one(".row.cal-nojs__rowTitle h4"):
                break
            panel = sibling.select_one(".panel-body")
            if panel is None:
                continue
            columns = panel.select(".row > div")
            if len(columns) < 3:
                continue
            time_text = columns[0].get_text(" ", strip=True)
            event_col = columns[1]
            event = next((p.get_text(" ", strip=True) for p in event_col.find_all("p")
                          if p.get_text(" ", strip=True) and "Watch Live" not in p.get_text(" ", strip=True)), "")
            dates_text = columns[-1].get_text(" ", strip=True)
            if not event:
                continue
            # Release-date cells are either comma-separated day numbers or a
            # day range for a meeting. Expand explicit listed days only.
            for token in re.findall(r"(?<!\d)(\d{1,2})(?!\d)", dates_text):
                day = int(token)
                try:
                    dt = pd.Timestamp(year=year, month=month, day=day)
                except ValueError:
                    continue
                date = _in_range(dt, start_ts, end_ts)
                if date is None:
                    continue
                rows.append({
                    "date": date,
                    "country": "United States",
                    "event": event,
                    "details": f"Federal Reserve {section} schedule; listed time {time_text} Eastern Time. Actual publication is not independently verified.",
                    "url": base_url,
                    "source": "Federal Reserve Board monthly calendar",
                    "event_status": "Scheduled date (official calendar)",
                })
    return pd.DataFrame(rows, columns=CALENDAR_COLUMNS[:-1])


def _session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = "IIF-Research/1.0 (public release calendar collector)"
    return session


def _cache_path(config: Any, source: str) -> Path:
    key = sha256(f"{source}|{config.start}|{config.end}".encode()).hexdigest()[:16]
    return Path(config.cache_dir) / f"calendar_sources_{source.lower().replace(' ', '_')}_{key}.csv"


def _fetch_one(session: requests.Session, url: str, params: dict[str, Any] | None = None) -> str:
    response = session.get(url, params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.text


def _ons_next_page(html: str | bytes, base_url: str = SOURCES["ONS"][0]) -> str | None:
    soup = _soup(html)
    for link in soup.select("a[href]"):
        label = " ".join((link.get_text(" ", strip=True), str(link.get("aria-label", "")))).strip().lower()
        if label == "next" or label.startswith("next ") or link.get("rel") == ["next"]:
            return urljoin(base_url, link["href"])
    return None


def _fetch_ons_pages(session: requests.Session, params: dict[str, Any], start: Any, end: Any,
                     *, max_pages: int = 10) -> tuple[pd.DataFrame, int, bool]:
    """Follow ONS's published pagination links, with an explicit page cap."""
    url = SOURCES["ONS"][0]
    next_params: dict[str, Any] | None = params
    frames: list[pd.DataFrame] = []
    partial = False
    pages = 0
    while url and pages < max_pages:
        html = _fetch_one(session, url, next_params)
        pages += 1
        frames.append(_parse_ons_calendar(html, start, end))
        next_url = _ons_next_page(html, SOURCES["ONS"][0])
        if not next_url:
            url = ""
        elif pages == max_pages:
            partial = True
            url = next_url
        else:
            url, next_params = next_url, None
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CALENDAR_COLUMNS[:-1])
    return combined.drop_duplicates(["date", "country", "event", "source"]).reset_index(drop=True), pages, partial


def collect_calendar_sources(config: Any, progress=None) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Collect dated BEA, Federal Reserve, and ONS release-calendar entries.

    Each publisher has its own range/as-of-independent cache and quality result.
    Failed sources do not replace a prior successful cache or block others.
    """
    start, end = pd.Timestamp(config.start), pd.Timestamp(config.end)
    session = _session()
    fetched_at = pd.Timestamp.now(tz="UTC").tz_localize(None)
    frames: list[pd.DataFrame] = []
    quality: list[dict[str, Any]] = []
    ons_params = {"after-day": start.day, "after-month": start.month, "after-year": start.year,
                  "before-day": end.day, "before-month": end.month, "before-year": end.year,
                  "release-type": "type-published", "limit": 100, "page": 1, "sort": "date-newest"}
    months = list(pd.period_range(start=start.to_period("M"), end=end.to_period("M"), freq="M"))
    max_fed_months = 12
    fed_months = months[:max_fed_months]
    source_fetchers = {
        "BEA": lambda: _parse_bea_schedule(_fetch_one(session, SOURCES["BEA"][0]), start, end),
        "Federal Reserve": lambda: pd.concat([
            _parse_fed_calendar(
                _fetch_one(session, SOURCES["Federal Reserve"][0].format(year=period.year, month_name=period.strftime("%B").lower())),
                period.year, period.month, start, end,
                base_url=SOURCES["Federal Reserve"][0].format(year=period.year, month_name=period.strftime("%B").lower()),
            ) for period in fed_months
        ], ignore_index=True) if fed_months else pd.DataFrame(columns=CALENDAR_COLUMNS[:-1]),
        "ONS": lambda: _fetch_ons_pages(session, ons_params, start, end),
    }
    for source, fetch_source in source_fetchers.items():
        path = _cache_path(config, source)
        cache_status = False
        source_partial = False
        source_note = ""
        try:
            if path.exists() and not getattr(config, "refresh", False):
                frame = pd.read_csv(path, parse_dates=["date", "fetched_at"])
                cache_status = True
            else:
                downloaded = fetch_source()
                if source == "ONS":
                    frame, page_count, source_partial = downloaded
                    source_note = f"Fetched {page_count} ONS result page(s)."
                else:
                    frame = downloaded
                if source == "Federal Reserve" and len(months) > max_fed_months:
                    source_partial = True
                    source_note = f"Federal Reserve calendar capped at {max_fed_months} months; later months were not fetched."
                if "fetched_at" not in frame:
                    frame["fetched_at"] = fetched_at
                if not source_partial:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    frame.to_csv(path, index=False)
            if "fetched_at" not in frame:
                frame["fetched_at"] = fetched_at
            frames.append(frame.reindex(columns=CALENDAR_COLUMNS))
            dates = pd.to_datetime(frame["date"], errors="coerce") if not frame.empty else pd.Series(dtype="datetime64[ns]")
            quality.append({
                "dataset": "economic_calendar", "symbol": "", "status": "cached" if cache_status else ("partial" if source_partial else ("downloaded" if not frame.empty else "empty")),
                "rows": len(frame), "source": source, "first_date": dates.min() if not dates.empty else None,
                "last_date": dates.max() if not dates.empty else None,
                "message": ("Official dated release-calendar entries; schedules and publication statuses are kept explicit. " + source_note + ("Pagination cap reached; additional ONS pages remain." if source == "ONS" and source_partial else "")) if not frame.empty else "Official source reachable but supplied no in-range records.",
                "fetched_at": fetched_at,
            })
        except Exception as exc:
            quality.append({
                "dataset": "economic_calendar", "symbol": "", "status": "unavailable", "rows": 0,
                "source": source, "first_date": None, "last_date": None,
                "message": f"Official source unavailable: {type(exc).__name__}: {exc}", "fetched_at": fetched_at,
            })
        if progress:
            progress("economic_calendar", source, quality[-1]["status"], quality[-1]["rows"])
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CALENDAR_COLUMNS)
    if not result.empty:
        result["date"] = pd.to_datetime(result["date"], errors="coerce")
        result["fetched_at"] = pd.to_datetime(result["fetched_at"], errors="coerce")
        result = result.drop_duplicates(["date", "country", "event", "source"]).sort_values(["date", "country", "source", "event"]).reset_index(drop=True)
    return result.reindex(columns=CALENDAR_COLUMNS), quality
