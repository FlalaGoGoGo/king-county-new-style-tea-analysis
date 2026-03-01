# Scripts

This folder contains the public-facing analysis scripts that power the King County New Style Tea project.

## Prerequisites

- Python 3.11+
- `pip install -r requirements.txt`
- Optional: `GOOGLE_MAPS_API_KEY` only for the Google Places refresh scripts

## Suggested reading order

1. `01_fetch_license_data.py`
2. `02_build_supply_by_area.py`
3. `03a_fetch_trends_last5y.py`
4. `04_build_demand_supply_gap.py`
5. `05_build_license_timeseries.py`
6. `07_text_sentiment_topics.py`
7. `08_build_opportunity_score.py`
8. `12_build_brand_landscape.py`
9. `17_weight_sensitivity_analysis.py`
10. `34_build_visual_dashboard.py`

## Common public rebuild flow

From the repository root:

```bash
python3 scripts/01_fetch_license_data.py
python3 scripts/02_build_supply_by_area.py
python3 scripts/03a_fetch_trends_last5y.py
python3 scripts/04_build_demand_supply_gap.py
python3 scripts/05_build_license_timeseries.py
python3 scripts/07_text_sentiment_topics.py
python3 scripts/08_build_opportunity_score.py
python3 scripts/12_build_brand_landscape.py
python3 scripts/17_weight_sensitivity_analysis.py
python3 scripts/34_build_visual_dashboard.py --output docs/index.html
```

## API-backed scripts

These scripts enrich public data but require your own Google Maps Platform key:

- `25_fetch_google_places_reviews.py`
- `35_fetch_google_place_coordinates.py`

Additional integration scripts for Google Places metadata are included for transparency.
