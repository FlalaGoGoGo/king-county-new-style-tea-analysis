# GitHub Upload Package (King County New Style Tea Dashboard)

This folder is pre-organized for publishing your dashboard via GitHub Pages.

## 1) Required files for online dashboard
- `docs/king_county_new_style_tea_dashboard.html`
- `docs/.nojekyll`

These are enough to publish the interactive dashboard.

## 2) Optional reference data (for transparency in class)
- `data_reference/dashboard_data/*`

These CSV/JSON/TXT files are not required for rendering the page (the current HTML already embeds data),
but they are useful if classmates/professor want to inspect your source tables.

## 3) Optional screenshot
- `screenshots/dashboard_preview.png`

Useful for README preview.

## 4) How to publish
1. Create a new GitHub repository (public or private).
2. Upload all files/folders inside this `GitHub` folder to the repo root.
3. In GitHub repo: `Settings -> Pages`.
4. Set `Source` to `Deploy from a branch`.
5. Select branch `main` and folder `/docs`.
6. Save and wait 1-2 minutes.
7. Your URL will be:
   - `https://<your-github-username>.github.io/<repo-name>/`

## Notes
- Current `king_county_new_style_tea_dashboard.html` loads Plotly and Google Fonts from CDN.
- If a viewer's network blocks CDN, charts/fonts may not fully load.
- If you want, we can generate a fully offline self-contained page (bundle Plotly locally) in the next step.
