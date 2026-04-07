# Sentinel-2 Crop Health Monitor

A Streamlit dashboard for agricultural field monitoring using Google Earth Engine and Sentinel-2 satellite imagery.

Built as an extension of the crop health classification pipeline developed for an Earth observation startup focused on agricultural monitoring and deforestation detection.

You can access it here: https://gee-dashboard-fsbujrzugxgyps4fybrqiv.streamlit.app/

## Features

- **Vegetation health mapping** — NDVI + EVI weighted classification (Healthy / Moderate Stress / High Stress)
- **Water status mapping** — NDWI + NDMI moisture analysis (Adequate / Mild Deficit / Strong Deficit)
- **Anomaly & hotspot detection** — Isolation Forest + DBSCAN spatial clustering across all four spectral indices
- **Rainfall context** — CHIRPS 30-day cumulative precipitation with drought vs. irrigation interpretation
- **Quadrant analysis** — Spatial stress breakdown for targeted field intervention
- **Automated agronomic report** — Plain-language field condition summary

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Authentication

The app supports three GEE auth modes (tried in order):

1. **Service account** via `st.secrets["GEE_SERVICE_ACCOUNT_JSON"]` (recommended for deployment)
2. **Service account** via `GEE_SERVICE_ACCOUNT_JSON` environment variable
3. **Application default credentials** (local dev — runs `ee.Initialize()` directly)

For local development, authenticate once via:
```bash
earthengine authenticate
```

Then set your project ID in the sidebar.

## Project Structure

```
gee_dashboard/
├── app.py              # Streamlit entry point
├── auth.py             # GEE authentication (service account + fallback)
├── data/
│   └── pipeline.py     # Core GEE pipeline (AOI, S2 loading, indices, classification, hotspots)
├── viz/
│   └── charts.py       # Plotly chart components
├── requirements.txt
└── README.md
```

## Data Sources

| Source | Description |
|--------|-------------|
| `COPERNICUS/S2_SR_HARMONIZED` | Sentinel-2 Level-2A surface reflectance |
| `UCSB-CHG/CHIRPS/DAILY` | Daily precipitation at 0.05° resolution |

## Spectral Indices

| Index | Formula | Purpose |
|-------|---------|---------|
| NDVI | (B8−B4)/(B8+B4) | Vegetation density & photosynthetic activity |
| EVI | 2.5×(B8−B4)/(B8+6B4−7.5B2+1) | NDVI saturation correction in dense canopy |
| NDWI | (B8−B11)/(B8+B11) | Canopy liquid water content (Gao 1996) |
| NDMI | (B8−B12)/(B8+B12) | Plant & soil moisture stress |
