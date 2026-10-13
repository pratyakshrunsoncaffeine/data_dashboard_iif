import numpy as np
import pandas as pd


def context_analysis(tables, analysis, config):
    result = {}
    prices = tables.get("prices", pd.DataFrame())
    members = tables.get("members_by_size", pd.DataFrame())
    if not prices.empty:
        data = prices.copy()
        data["date"] = pd.to_datetime(data.date).dt.normalize()
        data = data.loc[data.date.le(pd.Timestamp(config.end))]
        levels = data.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").sort_index()
        averages = {n: levels.shift(1).rolling(n, min_periods=n).mean() for n in (50, 200)}
        rows = []
        for group, cohort in members.groupby("group") if "group" in members else []:
            symbols = sorted(set(levels.columns).intersection(cohort.symbol))
            for date in levels.index[levels.index >= pd.Timestamp(config.start)]:
                for n, average in averages.items():
                    eligible = levels.loc[date, symbols].notna() & average.loc[date, symbols].notna()
                    count = int(eligible.sum())
                    above = int((levels.loc[date, symbols].gt(average.loc[date, symbols]) & eligible).sum())
                    rows.append({"date": date, "group": group, "window": n, "members": len(cohort.symbol.unique()), "observed_members": len(symbols), "eligible_count": count, "above_count": above, "above_pct": 100 * above / count if count else np.nan})
        result["size_breadth"] = pd.DataFrame(rows)
        if not levels.empty:
            latest = levels.iloc[-1]
            prior = averages[200].iloc[-1]
            distance = pd.DataFrame({"symbol": latest.index, "date": levels.index[-1], "close": latest.values, "prior_200_session_average": prior.values, "distance_pct": (latest / prior - 1).mul(100).values})
            result["distance_from_200ma"] = distance.merge(tables.get("members", pd.DataFrame()), on="symbol", how="left").sort_values("distance_pct", ascending=False)
    momentum = analysis.get("stock_momentum", pd.DataFrame())
    if not momentum.empty and {"sector", "period_return_pct"}.issubset(momentum):
        result["sector_period_returns"] = momentum.groupby("sector").period_return_pct.agg(median_return_pct="median", mean_return_pct="mean", observed_count="count").reset_index()
    economy = tables.get("global_economy", pd.DataFrame())
    if not economy.empty:
        result["economy_latest"] = economy.sort_values("date").groupby(["country", "indicator", "frequency", "source"], dropna=False).tail(1).reset_index(drop=True)
    markets = tables.get("markets", pd.DataFrame())
    if not markets.empty:
        markets = markets.copy()
        markets["date"] = pd.to_datetime(markets.date, errors="coerce").dt.normalize()
        markets = markets.dropna(subset=["date"])
        panel = markets.pivot_table(index="date", columns="name", values="close", aggfunc="last").sort_index()
        rows = []
        for numerator, denominator, factor, unit in (("Gold", "Silver", 1, "ratio of USD/oz prices"), ("Copper", "Gold", 2000, "USD/short ton copper divided by USD/oz gold")):
            if numerator not in panel or denominator not in panel:
                continue
            ratio = factor * panel[numerator] / panel[denominator]
            ratio = ratio.loc[(ratio.index >= pd.Timestamp(config.start)) & (ratio.index <= pd.Timestamp(config.end))].dropna()
            rows.extend({"date": date, "ratio": f"{numerator}/{denominator}", "value": value, "unit": unit, "source": "Yahoo commodity futures; descriptive front-contract ratio"} for date, value in ratio.items())
        result["commodity_ratios"] = pd.DataFrame(rows)
    return result
