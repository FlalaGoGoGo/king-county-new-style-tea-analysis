Dashboard Data Reference Bundle

Recommended review order:
1) zip_master.csv
2) store_master.csv
3) brand_summary.csv
4) license_trend_top10.csv
5) kpi_summary.csv
6) demo_storyboard.csv
7) review_samples.csv
8) store_logo_map.csv
9) brand_logo_map.csv
10) reviews_logos_summary.json

Primary join keys:
- zip5 (ZIP-level tables)
- store_id (store + Google metadata)
- brand (brand summary)

Notes:
- Core ranking logic uses tea-forward stores only.
- Substitute competitor layer is risk-only; do not use it to recalculate core opportunity score.
