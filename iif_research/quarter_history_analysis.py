from __future__ import annotations

import numpy as np
import pandas as pd


def analyze_quarter_history(tables, config):
    history = tables.get("official_quarterly_history", pd.DataFrame()).copy()
    if history.empty:
        return {"official_quarterly_latest": pd.DataFrame()}
    history["period_date"] = pd.to_datetime(history.period, errors="coerce", format="mixed")
    history["release_date"] = pd.to_datetime(history.release_date, errors="coerce", format="mixed")
    history = history.loc[history.period_date.dt.is_quarter_end & history.period_date.le(pd.Timestamp(config.end))
                          & history.release_date.dt.normalize().le(pd.Timestamp(config.end))].copy()
    history = history.sort_values(["symbol", "period_date", "release_date"])
    history = history.drop_duplicates(["symbol", "period", "basis"], keep="last")
    rows = []
    for symbol, company in history.groupby("symbol", sort=True):
        latest_filing_period = company.period_date.max()
        numeric_columns = [column for column in ("revenue_cr", "net_profit_cr", "eps") if column in company]
        numeric = company[numeric_columns].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
        company = company.loc[numeric]
        if company.empty:
            continue
        latest_period = company.period_date.max()
        candidates = company.loc[company.period_date.eq(latest_period)].copy()
        preferred = candidates.basis.astype(str).str.lower().str.contains("consolidated").astype(int)
        latest = candidates.assign(_preferred=preferred).sort_values(["_preferred", "release_date"]).iloc[-1]
        row = latest.drop(labels=["_preferred", "period_date"]).to_dict()
        row["period"] = latest_period.date().isoformat()
        row["latest_filing_period"] = latest_filing_period.date().isoformat()
        row["numeric_lags_latest_filing"] = latest_period < latest_filing_period
        year_ago = (latest_period.to_period("Q") - 4).end_time.normalize()
        prior = company.loc[company.period_date.eq(year_ago) & company.basis.eq(latest.basis)]
        row.update({"prior_year_period": year_ago.date().isoformat(), "prior_revenue_cr": np.nan,
                    "prior_net_profit_cr": np.nan, "revenue_yoy_pct": np.nan, "profit_yoy_pct": np.nan,
                    "profit_change_cr": np.nan, "net_profit_margin_pct": np.nan,
                    "prior_net_profit_margin_pct": np.nan, "margin_change_pp": np.nan,
                    "comparison_note": "Same-basis prior-year fiscal quarter unavailable."})
        revenue = pd.to_numeric(latest.get("revenue_cr"), errors="coerce")
        profit = pd.to_numeric(latest.get("net_profit_cr"), errors="coerce")
        if pd.notna(revenue) and revenue > 0 and pd.notna(profit):
            row["net_profit_margin_pct"] = profit / revenue * 100
        if not prior.empty:
            old = prior.iloc[-1]
            old_revenue = pd.to_numeric(old.get("revenue_cr"), errors="coerce")
            old_profit = pd.to_numeric(old.get("net_profit_cr"), errors="coerce")
            row["prior_revenue_cr"], row["prior_net_profit_cr"] = old_revenue, old_profit
            revenue_concepts = (str(latest.get("revenue_concept", "")), str(old.get("revenue_concept", "")))
            profit_concepts = (str(latest.get("profit_concept", "")), str(old.get("profit_concept", "")))
            revenue_match = all(revenue_concepts) and revenue_concepts[0] == revenue_concepts[1]
            profit_match = all(profit_concepts) and profit_concepts[0] == profit_concepts[1]
            notes = []
            if revenue_match and pd.notna(old_revenue) and old_revenue > 0 and pd.notna(revenue):
                row["revenue_yoy_pct"] = (revenue / old_revenue - 1) * 100
            else:
                notes.append("Revenue growth unavailable: positive prior value and matching XBRL concept required.")
            if profit_match and pd.notna(old_profit) and pd.notna(profit):
                row["profit_change_cr"] = profit - old_profit
                if old_profit > 0:
                    row["profit_yoy_pct"] = (profit / old_profit - 1) * 100
                else:
                    notes.append("Prior profit is nonpositive; percentage profit growth omitted, absolute change retained.")
            else:
                notes.append("Profit comparison unavailable: matching XBRL concept and both values required.")
            if revenue_match and profit_match and pd.notna(old_revenue) and old_revenue > 0 and pd.notna(old_profit):
                row["prior_net_profit_margin_pct"] = old_profit / old_revenue * 100
                row["margin_change_pp"] = row["net_profit_margin_pct"] - row["prior_net_profit_margin_pct"]
            row["comparison_note"] = " ".join(notes) or "Exact prior-year fiscal quarter, same basis and XBRL concepts."
        rows.append(row)
    latest = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    members = tables.get("members", pd.DataFrame())
    if not latest.empty and {"symbol", "name", "sector"}.issubset(members):
        latest = latest.merge(members[["symbol", "name", "sector"]].drop_duplicates("symbol"), on="symbol", how="left")
    return {"official_quarterly_latest": latest}
