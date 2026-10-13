# data_dashboard_iif

Streamlit dashboard for IIF public market, company filing, institutional flow, global macro, and niche sector research.

## Run locally

Install requirements and run `streamlit run dashboard.py`.

Select the inclusive date range, choose a stock universe, and click **Collect and build report**. The dashboard does not collect data at startup. Each run provides figures, tables, source coverage, a report, and downloadable research bundles.

Cloud runs use temporary storage. Download each research pack before the session ends.

## Sources

The app collects from public data endpoints documented in `iif_research`. Missing observations and blocked feeds remain disclosed in the report coverage table. It does not contain a saved report or source cache.
