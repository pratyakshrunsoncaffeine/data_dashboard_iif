from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo
import re
import xml.etree.ElementTree as ET
import pandas as pd
import requests
from bs4 import BeautifulSoup

NEWS_FEEDS = {
    "BBC Business": "https://feeds.bbci.co.uk/news/business/rss.xml",
    "RBI press releases": "https://www.rbi.org.in/pressreleases_rss.xml",
    "Government of India releases": "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3",
}

def parse_news(xml, start, end, source):
    rows = []
    for item in ET.fromstring(xml).findall(".//item"):
        raw_date = item.findtext("pubDate") or item.findtext("date")
        if not raw_date:
            continue
        try:
            day = parsedate_to_datetime(raw_date).astimezone(ZoneInfo("Asia/Kolkata")).date().isoformat()
        except (ValueError, TypeError):
            continue
        if start <= day <= end:
            rows.append({"date": day, "symbol": "", "event": item.findtext("title") or "News release", "details": BeautifulSoup(item.findtext("description") or "", "html.parser").get_text(" ", strip=True)[:1200], "url": item.findtext("link") or "", "source": source})
    return pd.DataFrame(rows, columns=["date", "symbol", "event", "details", "url", "source"])

def collect_supplementary(config):
    events, quality = [], []
    def one(entry):
        name, url = entry
        fetched = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()
        try:
            response = requests.get(url, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            frame = parse_news(response.content, config.start, config.end, name)
            return frame, {"dataset": "news", "symbol": name, "status": "partial" if len(frame) else "unavailable", "rows": len(frame), "source": url, "first_date": frame.date.min() if len(frame) else None, "last_date": frame.date.max() if len(frame) else None, "message": "RSS only retains recent releases. Not a complete historical news archive.", "fetched_at": fetched}
        except Exception as exc:
            return pd.DataFrame(), {"dataset": "news", "symbol": name, "status": "unavailable", "rows": 0, "source": url, "message": str(exc), "fetched_at": fetched}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for frame, record in pool.map(one, NEWS_FEEDS.items()):
            quality.append(record)
            if len(frame):
                events.append(frame)
    macro = pd.DataFrame(columns=["date", "name", "value", "unit", "source"])
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()
    if config.start <= today <= config.end:
        try:
            response = requests.get("https://www.rbi.org.in/", timeout=12, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            text = BeautifulSoup(response.text, "html.parser").get_text(" ", strip=True)
            match = re.search(r"(?:Policy\s+)?Repo\s+Rate\s*[:\-]?\s*(\d+(?:\.\d+)?)\s*%", text, flags=re.I)
            if match:
                macro = pd.DataFrame([{"date": today, "name": "Policy repo rate (retrieval-date snapshot)", "value": float(match.group(1)), "unit": "percent", "source": "https://www.rbi.org.in/"}])
            quality.append({"dataset": "macro", "status": "snapshot" if match else "unavailable", "rows": len(macro), "source": "https://www.rbi.org.in/", "message": "Observed at retrieval date. Policy effective date and historical rates are not inferred.", "fetched_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()})
        except Exception as exc:
            quality.append({"dataset": "macro", "status": "unavailable", "rows": 0, "source": "https://www.rbi.org.in/", "message": str(exc)})
    return pd.concat(events, ignore_index=True) if events else pd.DataFrame(), macro, quality
