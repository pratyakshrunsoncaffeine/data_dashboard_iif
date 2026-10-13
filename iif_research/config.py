from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

@dataclass
class ResearchConfig:
    start: str = "2026-10-01"
    end: str = "2026-10-11"
    universe: str = "nifty500"
    correlation_count: int = 24
    correlation_sessions: int = 126
    min_correlation_observations: int = 60
    fundamental_count: int = 500
    cache_dir: Path = ROOT / "data" / "cache"
    output_dir: Path = ROOT / "output"
    refresh: bool = False

    def __post_init__(self):
        self.start = pd.Timestamp(self.start).date().isoformat()
        self.end = pd.Timestamp(self.end).date().isoformat()
        if self.start > self.end:
            raise ValueError("Start must be on or before end")
        if pd.Timestamp(self.end).date() > datetime.now(ZoneInfo("Asia/Kolkata")).date():
            raise ValueError("End date cannot be in the future")
        if not 2 <= self.correlation_count <= 40:
            raise ValueError("Correlation universe must contain 2 to 40 stocks")
        if self.correlation_sessions < self.min_correlation_observations:
            raise ValueError("Correlation window is shorter than the minimum sample")
        if not 1 <= self.fundamental_count <= 500:
            raise ValueError("Fundamental count must be 1 to 500")
        if self.universe not in ("nifty50", "nifty100", "nifty200", "nifty500"):
            raise ValueError("Choose nifty50, nifty100, nifty200 or nifty500")
        self.cache_dir = Path(self.cache_dir)
        self.output_dir = Path(self.output_dir)

MARKETS = {
    "Nifty 50": ("^NSEI", "India", "index points"),
    "Nifty Bank": ("^NSEBANK", "India", "index points"),
    "Sensex": ("^BSESN", "India", "index points"),
    "Nifty Next 50": ("^NSMIDCP", "India", "index points"),
    "Nifty Midcap 100": ("NIFTY_MIDCAP_100.NS", "India", "index points"),
    "Nifty Smallcap 100": ("NIFTY_SMLCAP_100.NS", "India", "index points"),
    "Nifty Midcap 150": ("NIFTYMIDCAP150.NS", "India", "index points"),
    "Nifty Smallcap 250": ("NIFTYSMLCAP250.NS", "India", "index points"),
    "Nifty IT": ("^CNXIT", "Sector", "index points"),
    "Nifty Auto": ("^CNXAUTO", "Sector", "index points"),
    "Nifty FMCG": ("^CNXFMCG", "Sector", "index points"),
    "Nifty Metal": ("^CNXMETAL", "Sector", "index points"),
    "Nifty Pharma": ("^CNXPHARMA", "Sector", "index points"),
    "Nifty Realty": ("^CNXREALTY", "Sector", "index points"),
    "Nifty Energy": ("^CNXENERGY", "Sector", "index points"),
    "India VIX": ("^INDIAVIX", "Volatility", "index points"),
    "S&P 500": ("^GSPC", "Global", "index points"),
    "Nasdaq": ("^IXIC", "Global", "index points"),
    "FTSE 100": ("^FTSE", "Global", "index points"),
    "Nikkei 225": ("^N225", "Global", "index points"),
    "Hang Seng": ("^HSI", "Global", "index points"),
    "Shanghai Composite": ("000001.SS", "Global", "index points"),
    "DAX": ("^GDAXI", "Global", "index points"),
    "Dow Jones": ("^DJI", "Global", "index points"),
    "Brent": ("BZ=F", "Commodity", "USD/barrel futures"),
    "WTI": ("CL=F", "Commodity", "USD/barrel futures"),
    "Gold": ("GC=F", "Commodity", "USD/oz futures"),
    "Silver": ("SI=F", "Commodity", "USD/oz futures"),
    "Copper": ("HG=F", "Commodity", "USD/lb futures"),
    "USD/INR": ("INR=X", "Currency", "INR per USD"),
    "Dollar index": ("DX-Y.NYB", "Currency", "index points"),
    "US 10Y yield": ("^TNX", "Yield", "percent"),
}

TABLE_NAMES = ("prices", "markets", "members", "flows", "participant_oi", "delivery", "events", "quarterly", "holdings", "valuations", "macro")
