# Reference Data Dictionary

This folder contains the dashboard-ready tables that support the published site in `docs/`.

## Primary Join Keys

- `zip5`: ZIP-level joins across ranking, scoring, and trend views
- `store_id`: store-level joins across store cards, reviews, and coordinates
- `brand`: brand-level joins across brand summaries and logo mappings

## Core Dashboard Tables

| File | Grain | What it contains | Main downstream use |
| --- | --- | --- | --- |
| `zip_master.csv` | 1 row per ZIP | Final ZIP scoring panel with demand, gap, momentum, text, feasibility, rank, and action note | Opportunity matrix, demand vs supply chart, focused ZIP insight |
| `store_master.csv` | 1 row per store | Master store table with location, category, Google metadata, ratings, and review coverage | Brand -> Store Explorer, real review panel, store counts |
| `brand_summary.csv` | 1 row per brand | Brand-level footprint summary across stores, cities, ZIPs, rating, and operating model | Context view, brand concentration, logo panel |
| `review_samples.csv` | 1 row per review | Curated real review pool with source URL, rating, review date, and review text | Real Review Evidence and text explainability layer |
| `store_logo_map.csv` | 1 row per store | Store-to-logo mapping used by the dashboard avatar tiles | Store cards and detail panels |
| `brand_logo_map.csv` | 1 row per brand | Brand-to-logo mapping plus source metadata | Brand logo browser and fallbacks |

## Support Tables

| File | Grain | What it contains | Main downstream use |
| --- | --- | --- | --- |
| `license_trend_top10.csv` | 1 row per ZIP-year | Opening trend table for the top ZIPs shown in the dashboard | Context view -> License Trend |
| `kpi_summary.csv` | 1 row per metric | Small summary table for headline KPIs | Quick KPI cards and snapshot checks |
| `demo_storyboard.csv` | 1 row per demo scenario | Suggested click path and support metrics for live walkthroughs | Presentation prep and demo script alignment |
| `reviews_logos_summary.json` | 1 JSON document | QA summary of review and logo matching coverage | Internal QA and release checks |

## Important Columns

### `zip_master.csv`

- `zip5`: ZIP code in King County scope
- `demand_score`: normalized demand score based on population, income, and young-adult share
- `supply_score`: normalized supply pressure score from tea-forward store density
- `gap_score`: normalized demand minus supply score
- `momentum_score`: normalized opening momentum score from recent license activity
- `text_signal_score`: normalized text evidence score from review sentiment/topics
- `feasibility_score`: normalized affordability and execution score
- `opportunity_score`: final weighted opportunity score
- `rank`: overall ZIP ranking
- `quadrant`: demand/supply quadrant label
- `recommended_strategy`: short strategy label used by the dashboard
- `action_note`: plain-language action note shown in the focused ZIP panel

### `store_master.csv`

- `store_id`: stable store identifier used across the repo
- `brand`: normalized brand name
- `store_name`: storefront name shown in the dashboard
- `city`, `zip5`: location fields
- `primary_beverage_category`: level-1 beverage category
- `hero_products`: manually curated hero product cues
- `business_model`: independent vs franchise-style operating cue
- `google_rating`, `google_user_ratings_total`: Google Places rating fields
- `review_count`, `avg_rating`, `date_min`, `date_max`: local review coverage summary

### `review_samples.csv`

- `store_id`: joins each review back to `store_master.csv`
- `review_rank`: within-store review order used for the review timeline
- `review_date`: review timestamp used in the evidence panel
- `rating`: source rating when available
- `source_platform`: review source label
- `source_url`: direct link to the source review context
- `review_text`: full review text used by the dashboard
- `review_text_short`: shortened text shown in compact timeline cards

## Relationship To Other Data Folders

- `data/raw/`: selected source tables that are safe to share publicly
- `data/processed/`: intermediate model outputs used to build the final dashboard-ready tables
- `data/interim/`: small helper tables used by selected scripts
- `data/external/geo/`: geographic boundary files used for the map layer

## Notes

- The published dashboard reads from the reference bundle, not directly from every raw source file.
- Coffee-led substitute competitors are used as a risk layer, not as core tea-forward supply.
- Large raw exports, local caches, API error dumps, and backup files are intentionally excluded from this public repository.
