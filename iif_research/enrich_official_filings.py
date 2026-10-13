"""Persist a bounded official NSE filing/ownership pass into collected_run."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import ResearchConfig
from .providers import SCHEMAS, collect_official_filings, collect_official_holdings


def run(config: ResearchConfig, *, holdings_limit: int = 24) -> dict:
    checkpoint = config.cache_dir / "collected_run"
    checkpoint.mkdir(parents=True, exist_ok=True)
    members_path = checkpoint / "members.json"
    members = pd.read_json(members_path, orient="split") if members_path.exists() else pd.DataFrame(columns=["symbol"])
    symbols = members.get("symbol", pd.Series(dtype=str)).dropna().astype(str).tolist()
    symbols = symbols[:config.fundamental_count]

    extra, quality = collect_official_filings(config, symbols)
    for name, frame in extra.items():
        frame.to_json(checkpoint / f"{name}.json", orient="split", date_format="iso")
    official_holdings, hold_quality = collect_official_holdings(config, symbols[:holdings_limit], max_periods=2)
    official_holdings.to_json(checkpoint / "official_holdings.json", orient="split", date_format="iso")
    quality.extend(hold_quality)

    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {"config": {}, "quality": []}
    # The earlier Screener fetch ended after repeated connection timeouts.
    # Replace queued/unresolved labels with the actual state; preserve usable rows.
    for item in metadata.get("quality", []):
        if item.get("dataset") in {"quarterly", "holdings"} and item.get("status") in {"pending", "unavailable"}:
            item["status"] = "deferred_due_source_outage"
            item["message"] = "Screener.in connection timed out repeatedly at www.screener.in and screener.in; collection was stopped. No values inferred."
    reference_path = checkpoint / "reference_snapshots.json"
    reference_rows = 0
    if reference_path.exists():
        raw = json.loads(reference_path.read_text(encoding="utf-8"))
        reference_rows = len(raw.get("data", [])) if isinstance(raw, dict) else len(raw)
    metadata.setdefault("quality", [])
    metadata["quality"] = [q for q in metadata["quality"] if q.get("dataset") != "reference_snapshots"]
    metadata["quality"].append({"dataset": "reference_snapshots", "symbol": "", "status": "snapshot" if reference_rows else "unavailable",
                                "rows": reference_rows, "source": "MarketShack public numeric board",
                                "first_date": config.end if reference_rows else "", "last_date": config.end if reference_rows else "",
                                "message": "Public numeric quote/rate snapshot; not a historical time series or publication-vintage feed.",
                                "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds")})
    metadata["quality"].extend(quality)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"filings": len(extra["official_filings"]), "results": len(extra["official_results"]),
            "holdings": len(official_holdings), "holding_symbols": len(hold_quality), "quality": quality}


def continue_official_holdings(config: ResearchConfig) -> dict:
    """Refresh ownership periods for every member using quarter-end rules.

    Existing numeric XBRL is reused only when fresh NSE metadata selects the
    identical XBRL URL and period. Every symbol's metadata is revalidated.
    """
    checkpoint = config.cache_dir / "collected_run"
    members = pd.read_json(checkpoint / "members.json", orient="split")
    symbols = members["symbol"].dropna().astype(str).tolist()[:config.fundamental_count]
    metadata_path = checkpoint / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    holdings_path = checkpoint / "official_holdings.json"
    prior = pd.read_json(holdings_path, orient="split") if holdings_path.exists() else pd.DataFrame()
    current_quality: list[dict] = []
    metadata["quality"] = [q for q in metadata.get("quality", []) if q.get("dataset") != "official_holdings"]
    metadata["quality"].append({"dataset": "official_holdings_refresh", "symbol": "", "status": "collecting",
                                "rows": 0, "source": "NSE corporate shareholding master + XBRL",
                                "first_date": "", "last_date": "",
                                "message": "Revalidating all member filing metadata with quarter-end selection; checkpoint updates every four symbols.",
                                "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds")})
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def persist(_done, _total, _label, batch_frame, batch_quality):
        nonlocal current_quality
        processed = {q.get("symbol") for q in batch_quality}
        kept_prior = (prior.loc[~prior.symbol.astype(str).isin(processed)] if not prior.empty else prior)
        combined = pd.concat([kept_prior, batch_frame], ignore_index=True) if not kept_prior.empty else batch_frame.copy()
        combined = combined.reindex(columns=SCHEMAS["official_holdings"])
        if not combined.empty:
            combined = combined.drop_duplicates(["symbol", "period"], keep="last")
        combined.to_json(holdings_path, orient="split", date_format="iso")
        current_quality = list(batch_quality)
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        meta["quality"] = [q for q in meta.get("quality", []) if q.get("dataset") not in {"official_holdings", "official_holdings_refresh"}]
        meta["quality"].extend(current_quality)
        meta["quality"].append({"dataset": "official_holdings_refresh", "symbol": "", "status": "collecting",
                                "rows": len(combined), "source": "NSE corporate shareholding master + XBRL",
                                "first_date": str(combined.period.min()) if not combined.empty else "",
                                "last_date": str(combined.period.max()) if not combined.empty else "",
                                "message": f"Revalidated {len(processed)}/{len(symbols)} symbols; checkpoint updated by four-symbol batches.",
                                "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds")})
        metadata_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"checkpointed quarter-end official holdings {_done}/{len(symbols)}; rows={len(combined)}", flush=True)

    from .providers import collect_official_holdings
    final, new_quality = collect_official_holdings(config, symbols, max_periods=2, progress=persist, reuse_frame=prior)
    if not final.empty:
        final = final.drop_duplicates(["symbol", "period"], keep="last")
    final.to_json(holdings_path, orient="split", date_format="iso")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["quality"] = [q for q in metadata.get("quality", []) if q.get("dataset") not in {"official_holdings", "official_holdings_refresh"}]
    metadata["quality"].extend(new_quality)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"requested_members": len(symbols), "revalidated": len(new_quality), "attempted_now": len(symbols),
            "rows": len(final), "symbols_with_data": final.symbol.nunique() if not final.empty else 0,
            "new_quality": new_quality}


if __name__ == "__main__":
    result = run(ResearchConfig())
    print(json.dumps({k: v for k, v in result.items() if k != "quality"}, indent=2))
    print(json.dumps(result["quality"], indent=2))
