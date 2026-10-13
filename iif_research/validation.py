import pandas as pd

def coverage_checks(tables, config):
    records = []
    markets = tables.get("markets", pd.DataFrame())
    expected = set()
    if not markets.empty and {"name", "date", "close"}.issubset(markets):
        subset = markets.loc[markets.name.eq("Nifty 50") & markets.close.notna()].copy()
        days = pd.to_datetime(subset.date, errors="coerce").dt.strftime("%Y-%m-%d")
        expected = set(days.loc[days.between(config.start, config.end)].dropna())
    for dataset in ("flows", "participant_oi", "delivery"):
        frame = tables.get(dataset, pd.DataFrame())
        actual = set(pd.to_datetime(frame.date, errors="coerce").dt.strftime("%Y-%m-%d").dropna()) if "date" in frame else set()
        missing = sorted(expected - actual)
        records.append({"dataset": dataset, "status": "partial" if missing else "checked" if expected else "unverified", "rows": len(frame), "source": "IIF coverage check against Nifty 50 observed sessions", "message": f"Observed {len(actual & expected)}/{len(expected)} benchmark sessions. Missing: {', '.join(missing) or 'none'}. Exchange holidays inferred from actual benchmark observations."})
    prices = tables.get("prices", pd.DataFrame())
    members = tables.get("members", pd.DataFrame())
    if not prices.empty and "symbol" in prices:
        matched = prices.copy()
        matched["date"] = pd.to_datetime(matched.date, errors="coerce").dt.strftime("%Y-%m-%d")
        available = matched.loc[matched.date.between(config.start, config.end) & matched.close.notna(), "symbol"].nunique()
        records.append({"dataset": "breadth_universe", "status": "checked" if available == len(members) else "partial", "rows": int(available), "source": "IIF universe coverage check", "message": f"{available}/{len(members)} constituent symbols have a close in the requested period. Current membership, not point-in-time membership."})
    records.extend([
        {"dataset": "proprietary_feeds", "status": "not_connected", "rows": 0, "source": "Reference email scope", "message": "Reference Fear & Greed score, Tijori custom indices and management commentary are not replicated proprietary feeds. Public breadth, volatility, official events and linked releases are supplied separately."},
        {"dataset": "historical_fundamentals", "status": "snapshot", "rows": len(tables.get("quarterly", [])), "source": "Screener company pages", "message": "Quarterly values and shareholding are retrieval-date snapshots filtered by fiscal period, not a point-in-time database of what was known at report end. Dated exchange announcement events are separate."},
    ])
    return records
