import json
import hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import requests


def parse_nifty_ohlc(payload, config):
    result = payload.get("chart", {}).get("result") or []
    if not result:
        return pd.DataFrame()
    obj = result[0]
    quotes = obj.get("indicators", {}).get("quote", [{}])[0]
    dates = pd.to_datetime(obj.get("timestamp", []), unit="s", utc=True).tz_convert("Asia/Kolkata").tz_localize(None).normalize()
    frame = pd.DataFrame({"date": dates})
    for column in ("open", "high", "low", "close", "volume"):
        values = quotes.get(column, [])
        frame[column] = pd.to_numeric(pd.Series(values).reindex(range(len(frame))), errors="coerce").to_numpy()
    frame["name"] = "Nifty 50"
    frame["symbol"] = "^NSEI"
    frame["source"] = "Yahoo Finance public chart; unadjusted OHLC"
    frame["fetched_at"] = datetime.now(ZoneInfo("Asia/Kolkata")).isoformat()
    return frame.loc[frame.date.le(pd.Timestamp(config.end))].drop_duplicates("date").reset_index(drop=True)


def collect_price_context(config):
    payload_key = f"nifty_raw_ohlc_v1|{config.start}|{config.end}"
    digest = hashlib.sha256(payload_key.encode()).hexdigest()[:24]
    path = config.cache_dir / f"nifty_raw_ohlc_{digest}.json"
    try:
        if path.exists() and not config.refresh:
            frame = pd.read_json(path, orient="split")
            status = "cached"
        else:
            start = pd.Timestamp(config.start) - pd.Timedelta(days=400)
            response = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI", params={"period1": int(start.tz_localize("UTC").timestamp()), "period2": int((pd.Timestamp(config.end) + pd.Timedelta(days=1)).tz_localize("UTC").timestamp()), "interval": "1d"}, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
            response.raise_for_status()
            frame = parse_nifty_ohlc(response.json(), config)
            if not frame.empty:
                path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_json(path, orient="split", date_format="iso")
            status = "downloaded" if not frame.empty else "unavailable"
        quality = {"dataset": "nifty_ohlc", "status": status, "rows": len(frame), "source": "Yahoo Finance public chart", "message": "Observed unadjusted Nifty OHLC; open is never inferred from previous close. History includes warm-up sessions."}
        return {"nifty_ohlc": frame}, [quality]
    except (requests.RequestException, ValueError, OSError) as error:
        return {"nifty_ohlc": pd.DataFrame()}, [{"dataset": "nifty_ohlc", "status": "unavailable", "rows": 0, "source": "Yahoo Finance public chart", "message": str(error)}]
