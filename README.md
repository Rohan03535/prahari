# PRAHARI — Parking Intelligence Command System

AI-driven parking enforcement for Bengaluru Traffic Police (Flipkart Gridlock 2.0, Theme 1).

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (free public URL)

1. Push this repo to GitHub (include `data/*.parquet` — processed bundles, ~30MB total).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Connect your GitHub repo.
4. Set **Main file path:** `app.py`
5. Deploy. Your public URL will be `https://<app-name>.streamlit.app`

### Required files in repo

- `app.py` — entrypoint
- `requirements.txt`
- `config.py`
- `src/` — all modules
- `data/parking.parquet`, `data/junction_summary.parquet`, `data/cell_agg.parquet`, `data/recurrence.parquet`

### Do NOT commit

- Raw CSV (`data/raw/` — 105MB, over GitHub/HackerEarth limits)
- `.venv/`, secrets, local caches

## Re-build data bundles (optional)

If you have the raw hackathon CSV locally:

```bash
python scripts/bundle_data.py
```
