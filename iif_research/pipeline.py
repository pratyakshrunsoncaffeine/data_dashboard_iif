from dataclasses import asdict
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import hashlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from queue import Empty, Queue
import pandas as pd
from .config import ResearchConfig, TABLE_NAMES


def definition_digest():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "iif_research").glob("*.py")) + [root / "config" / "niche_themes.csv"]
    digest = hashlib.sha256()
    for path in paths:
        if path.exists():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _collect_concurrently(providers, config, progress):
    progress_messages = Queue()

    def queue_progress(*parts):
        progress_messages.put(" | ".join(map(str, parts)))

    def publish_progress():
        while True:
            try:
                progress(progress_messages.get_nowait())
            except Empty:
                return

    with ThreadPoolExecutor(max_workers=len(providers)) as executor:
        futures = {executor.submit(provider, config, progress=queue_progress): provider for provider in providers}
        pending = set(futures)
        results = {}
        while pending:
            completed, pending = wait(pending, timeout=0.25, return_when=FIRST_COMPLETED)
            publish_progress()
            for future in completed:
                results[futures[future]] = future.result()
        publish_progress()

    return [results[provider] for provider in providers]


def run_research(config=None, progress=None):
    from .providers import collect
    from .analysis import analyze
    from .report import export_report
    config = config or ResearchConfig()
    progress = progress or (lambda message: None)
    progress("Collecting source observations")
    tables, quality = collect(config, progress=progress)
    from .economy import collect_economy
    from .market_scope import collect_market_scope
    progress("Collecting global economies and the wider index board")
    providers = (collect_economy, collect_market_scope)
    for extra_tables, extra_quality in _collect_concurrently(providers, config, progress):
        for name, frame in extra_tables.items():
            tables[name] = pd.concat([tables[name], frame], ignore_index=True) if name in tables and not tables[name].empty else frame
        for record in extra_quality:
            record.setdefault("dataset", "global_economy" if "World Bank" in record.get("source", "") or "FRED" in record.get("source", "") or "MoSPI" in record.get("source", "") else "extended_scope")
            record.setdefault("rows", None)
        quality.extend(extra_quality)
    from .supplementary import collect_supplementary
    progress("Collecting public news and policy observations")
    news, macro, extra_quality = collect_supplementary(config)
    quality.extend(extra_quality)
    from .reference_snapshot import collect_reference_snapshot
    reference_tables, reference_quality = collect_reference_snapshot(config)
    tables.update(reference_tables)
    quality.extend(reference_quality)
    from .price_context import collect_price_context
    price_tables, price_quality = collect_price_context(config)
    tables.update(price_tables)
    quality.extend(price_quality)
    from .providers import collect_official_filings, collect_official_holdings
    progress("Collecting dated official earnings and ownership disclosures")
    symbols = tables.get("members", pd.DataFrame()).get("symbol", pd.Series(dtype=str)).tolist()
    official_tables, official_quality = collect_official_filings(config, symbols=symbols)
    tables.update(official_tables)
    quality.extend(official_quality)
    official_holdings, official_quality = collect_official_holdings(config, symbols=symbols, progress=progress)
    tables["official_holdings"] = official_holdings
    quality.extend(official_quality)
    from .official_history import collect_official_history
    progress("Collecting official quarterly financial history")
    history_tables, history_quality = collect_official_history(tables, config, progress=progress)
    tables.update(history_tables)
    quality.extend(history_quality)
    from .calendar_sources import collect_calendar_sources
    progress("Collecting additional official economic release calendars")
    calendar, calendar_quality = collect_calendar_sources(config, progress=lambda *parts: progress(" | ".join(map(str, parts))))
    if not calendar.empty:
        tables["economic_calendar"] = pd.concat([tables.get("economic_calendar", pd.DataFrame()), calendar], ignore_index=True).drop_duplicates(["date", "country", "event", "source"])
    quality.extend(calendar_quality)
    if not news.empty:
        tables["events"] = pd.concat([tables.get("events", pd.DataFrame()), news], ignore_index=True)
    if not macro.empty:
        tables["macro"] = pd.concat([tables.get("macro", pd.DataFrame()), macro], ignore_index=True)
    for name in TABLE_NAMES:
        tables.setdefault(name, pd.DataFrame())
    from .validation import coverage_checks
    quality.extend(coverage_checks(tables, config))
    checkpoint = config.cache_dir / "collected_run"
    checkpoint.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_json(checkpoint / f"{name}.json", orient="split", date_format="iso")
    (checkpoint / "metadata.json").write_text(json.dumps({"config": asdict(config), "quality": quality}, default=str), encoding="utf-8")
    return build_run(tables, quality, config, progress)


def build_run(tables, quality, config, progress=None):
    from .analysis import analyze
    from .report import export_report
    progress = progress or (lambda message: None)
    tables, quality = complete_dated_snapshots(tables, quality, config)
    progress("Calculating research tables")
    analysis = analyze(tables, config)
    from .context_analysis import context_analysis
    analysis.update(context_analysis(tables, analysis, config))
    from .themes import analyze_themes
    analysis.update(analyze_themes(tables, config))
    from .market_state import analyze_market_state
    analysis.update(analyze_market_state(tables, analysis, config))
    from .official_analysis import analyze_official
    analysis.update(analyze_official(tables, config))
    from .quarter_history_analysis import analyze_quarter_history
    analysis.update(analyze_quarter_history(tables, config))
    progress("Exporting newsletter report and charts")
    output_path = Path(export_report(tables, analysis, quality, config))
    manifest = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "generated_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
        "definition_digest": definition_digest(),
        "tables": {}, "analysis": {}, "quality": quality,
    }
    saved = output_path / "saved_run"
    saved.mkdir(parents=True, exist_ok=True)
    for kind, frames in (("tables", tables), ("analysis", analysis)):
        for key, frame in frames.items():
            path = saved / f"{kind}_{key}.{'parquet' if len(frame) > 100000 else 'json'}"
            if path.suffix == ".parquet":
                frame.to_parquet(path, index=True)
            else:
                frame.to_json(path, orient="split", date_format="iso", date_unit="ms")
            manifest[kind][key] = {"file": str(path.relative_to(output_path)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (output_path / "run.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    import zipfile
    zip_path = output_path / "research_pack.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in output_path.rglob("*"):
            if file.is_file() and file.suffix != ".zip":
                archive.write(file, file.relative_to(output_path))
    progress("Research pack ready")
    return {"config": config, "tables": tables, "analysis": analysis, "quality": quality, "output_path": output_path}


def complete_dated_snapshots(tables, quality, config):
    tables, quality = dict(tables), list(quality)
    snapshot = tables.get("index_snapshot", pd.DataFrame()).copy()
    if tables.get("valuations", pd.DataFrame()).empty and not snapshot.empty:
        snapshot["date"] = pd.to_datetime(snapshot.date, errors="coerce")
        snapshot = snapshot.loc[snapshot.date.between(pd.Timestamp(config.start), pd.Timestamp(config.end))]
        for column in ("pe", "pb"):
            snapshot[column] = pd.to_numeric(snapshot[column], errors="coerce").where(lambda values: values.gt(0))
        values = snapshot.dropna(subset=["pe", "pb"], how="all")[["date", "name", "pe", "pb", "source"]].copy()
        values["dividend_yield_pct"] = float("nan")
        tables["valuations"] = values
        quality.append({"dataset": "valuations", "status": "snapshot" if not values.empty else "unavailable", "rows": len(values), "source": "NSE dated all-indices board", "message": "PE and PB retained from the published dated index board. Nonpositive source placeholders are unavailable; dividend yield is not inferred."})
    reference = tables.get("reference_snapshots", pd.DataFrame()).copy()
    if tables.get("gift_snapshot", pd.DataFrame()).empty and not reference.empty:
        reference["date"] = pd.to_datetime(reference.date, errors="coerce")
        gift = reference.loc[reference.key.eq("gift-nifty") & reference.date.between(pd.Timestamp(config.start), pd.Timestamp(config.end)) & reference.value.notna()]
        if not gift.empty:
            tables["gift_snapshot"] = pd.DataFrame({"date": gift.date, "instrument": gift.name, "last": gift.value, "change": float("nan"), "change_pct": gift.return_d1_pct, "expiry": gift.expiry, "volume": float("nan"), "contracts": None, "source": gift.source}).reset_index(drop=True)
            quality.append({"dataset": "gift_snapshot", "status": "snapshot", "rows": len(gift), "source": "Market Shack public numeric quote board", "message": "Fallback quote uses its published date and expiry. No historical futures path, cash/futures spread or unknown turnover is inferred."})
    if "nifty_ohlc" in tables and not any(item.get("dataset") == "nifty_ohlc" for item in quality):
        quality.append({"dataset": "nifty_ohlc", "status": "cached" if not tables["nifty_ohlc"].empty else "unavailable", "rows": len(tables["nifty_ohlc"]), "source": "Yahoo Finance public chart", "message": "Observed unadjusted OHLC; warm-up sessions included; open is never inferred."})
    from .validation import coverage_checks
    check_sources = {"IIF coverage check against Nifty 50 observed sessions", "IIF universe coverage check", "Reference email scope", "Screener company pages"}
    quality = [item for item in quality if item.get("source") not in check_sources]
    quality.extend(coverage_checks(tables, config))
    return tables, quality


def rebuild_collected(config, progress=None):
    checkpoint = config.cache_dir / "collected_run"
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    for key in ("start", "end", "universe", "fundamental_count", "correlation_count", "correlation_sessions"):
        if metadata["config"][key] != getattr(config, key):
            raise ValueError(f"Collected checkpoint does not match requested {key}")
    tables = {}
    for file in checkpoint.glob("*.json"):
        if file.name == "metadata.json":
            continue
        frame = pd.read_json(file, orient="split", convert_dates=False)
        for column in ("date", "fetched_at", "available_at"):
            if column in frame:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        tables[file.stem] = frame
    return build_run(tables, metadata["quality"], config, progress)


def load_run(path):
    path = Path(path).resolve()
    manifest = json.loads((path / "run.json").read_text(encoding="utf-8"))
    result = {"config": ResearchConfig(**manifest["config"]), "quality": manifest["quality"], "output_path": path}
    for kind in ("tables", "analysis"):
        result[kind] = {}
        for name, entry in manifest[kind].items():
            file = (path / entry["file"]).resolve()
            if not file.is_relative_to(path):
                raise ValueError("Saved run contains an unsafe file path")
            if hashlib.sha256(file.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"Saved data checksum failed: {name}")
            frame = pd.read_parquet(file) if file.suffix == ".parquet" else pd.read_json(file, orient="split", convert_dates=False)
            if "date" in frame:
                frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            result[kind][name] = frame
    return result
