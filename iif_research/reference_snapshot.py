import json
import re
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import requests


def parse_snapshots(html, config):
    decoder = json.JSONDecoder()
    rows = []
    for match in re.finditer(r'\{"(?:chg|key)":', html):
        try:
            item, _ = decoder.raw_decode(html[match.start():])
        except ValueError:
            continue
        if not {"key", "name", "value"}.issubset(item):
            continue
        raw_date = item.get("asof") or item.get("time")
        stamp = pd.to_datetime(raw_date, errors="coerce", utc=True)
        day = stamp.tz_convert("Asia/Kolkata").date().isoformat() if pd.notna(stamp) else None
        if day and day > config.end:
            continue
        if day is None and config.end != datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat():
            continue
        changes = item.get("chg", {})
        row = {"date": day, "key": item["key"], "name": item["name"], "group": item.get("group"), "value": item["value"], "kind": item.get("kind", "market quote"), "asof_text": item.get("asofTxt"), "expiry": item.get("expiry"), "source": "https://themarketshack.com/ public embedded numeric snapshot", "availability_note": "Published snapshot date; may predate report start" if day else "Published year/date unknown; retrieval context only", "fetched_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()}
        row.update({f"return_{key}_pct": changes.get(key) for key in ("d1", "w1", "m1", "y1", "y3", "y5", "ytd")})
        row.update({f"rate_{key}_pct": item.get(key) for key in ("sdf", "msf", "bank")})
        rows.append(row)
    return pd.DataFrame(rows).drop_duplicates("key") if rows else pd.DataFrame()


def collect_reference_snapshot(config):
    try:
        response = requests.get("https://themarketshack.com/", timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        frame = parse_snapshots(response.text, config)
        return {"reference_snapshots": frame}, [{"dataset": "reference_snapshots", "source": "https://themarketshack.com/", "status": "snapshot" if not frame.empty else "unavailable", "rows": len(frame), "message": "Public numeric quote and rate board; source gaps stay blank. This snapshot does not provide a historical time series or first-publication vintages."}]
    except (requests.RequestException, ValueError) as error:
        return {"reference_snapshots": pd.DataFrame()}, [{"dataset": "reference_snapshots", "source": "https://themarketshack.com/", "status": "unavailable", "rows": 0, "message": str(error)}]
