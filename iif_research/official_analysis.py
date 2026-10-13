import numpy as np
import pandas as pd


def analyze_official(tables, config):
    results = tables.get("official_results", pd.DataFrame()).copy()
    if not results.empty:
        results["release_date"] = pd.to_datetime(results.release_date, errors="coerce")
        results = results.loc[results.release_date.dt.normalize().between(pd.Timestamp(config.start), pd.Timestamp(config.end))]
        revenue = pd.to_numeric(results.revenue_cr, errors="coerce")
        profit = pd.to_numeric(results.net_profit_cr, errors="coerce")
        results["net_profit_margin_pct"] = (100 * profit / revenue.where(revenue.ne(0))).replace([np.inf, -np.inf], np.nan)
    holdings = tables.get("official_holdings", pd.DataFrame()).copy()
    changes = []
    if not holdings.empty:
        holdings["period_date"] = pd.to_datetime(holdings.period, errors="coerce")
        broadcast = pd.to_datetime(holdings.broadcast_date, errors="coerce", format="mixed", dayfirst=True)
        submitted = pd.to_datetime(holdings.get("submission_date", pd.Series(index=holdings.index, dtype=str)), errors="coerce", format="mixed", dayfirst=True)
        holdings["available_date"] = broadcast.fillna(submitted)
        holdings = holdings.loc[holdings.period_date.dt.is_quarter_end & holdings.period_date.le(pd.Timestamp(config.end)) & holdings.available_date.dt.normalize().le(pd.Timestamp(config.end))].sort_values(["symbol", "period_date", "available_date"])
        holdings = holdings.drop_duplicates(["symbol", "period_date"], keep="last")
        for symbol, rows in holdings.groupby("symbol"):
            if len(rows) < 2:
                continue
            prior, current = rows.iloc[-2], rows.iloc[-1]
            quarter_gap = current.period_date.to_period("Q").ordinal - prior.period_date.to_period("Q").ordinal
            fii = pd.to_numeric(pd.Series([prior.fii_pct, current.fii_pct]), errors="coerce")
            dii = pd.to_numeric(pd.Series([prior.dii_pct, current.dii_pct]), errors="coerce")
            changes.append({"symbol": symbol, "prior_period": prior.period, "latest_period": current.period,
                            "prior_fii_pct": prior.fii_pct, "latest_fii_pct": current.fii_pct,
                            "fii_change_pp": fii.iloc[1] - fii.iloc[0] if quarter_gap == 1 else np.nan,
                            "prior_dii_pct": prior.dii_pct, "latest_dii_pct": current.dii_pct,
                            "dii_change_pp": dii.iloc[1] - dii.iloc[0] if quarter_gap == 1 else np.nan,
                            "quarters_between": quarter_gap,
                            "comparison_status": "consecutive quarters" if quarter_gap == 1 else "prior fiscal quarter unavailable; QoQ change omitted",
                            "broadcast_date": current.broadcast_date, "source_xbrl_url": current.source_xbrl_url,
                            "source": "NSE shareholding pattern XBRL"})
    return {"official_reported_results": results, "official_ownership_changes": pd.DataFrame(changes)}
