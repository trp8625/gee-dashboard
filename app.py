"""
app.py
------
Streamlit dashboard wrapping the GEE crop health & water status pipeline.

Run locally:
    streamlit run app.py

Auth:
    Set GEE_PROJECT in the sidebar or via st.secrets.
    Service account JSON can be set via GEE_SERVICE_ACCOUNT_JSON in st.secrets
    or as an environment variable. Falls back to application default credentials.
"""

import streamlit as st
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Health Monitor",
    layout="wide",
)

# ── Imports (deferred so page config runs first) ──────────────────────────────
from auth import initialize_gee
from data.pipeline import (
    build_aoi,
    load_sentinel2,
    compute_indices,
    download_indices,
    get_rainfall_context,
    classify_health,
    classify_water,
    compute_uniformity,
    detect_hotspots,
)
from viz.charts import (
    plot_health_map,
    plot_water_map,
    plot_ndvi_hotspots,
    plot_class_breakdown,
    plot_ndvi_histogram,
    plot_rainfall_gauge,
)

# ── Dark theme CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #0f1a0f !important;
    color: #d4e6c3 !important;
}
[data-testid="stSidebar"] {
    background-color: #0a120a !important;
    border-right: 1px solid #1e3a1e;
}
[data-testid="stSidebar"] * { color: #a8c897 !important; }
.dash-header { padding: 1.2rem 0 0.4rem 0; border-bottom: 1px solid #1e3a1e; margin-bottom: 1.2rem; }
.dash-header h1 { color: #7dbf6a !important; font-size: 1.6rem !important; font-weight: 700; letter-spacing: 0.02em; margin: 0; }
.dash-header p { color: #5a8050 !important; font-size: 0.82rem; margin: 0.2rem 0 0 0; }
.kpi-row { display: flex; gap: 12px; margin-bottom: 1.2rem; }
.kpi-card { background: #132213; border: 1px solid #1e3a1e; border-radius: 8px; padding: 14px 18px; flex: 1; min-width: 0; }
.kpi-label { font-size: 0.72rem; color: #5a8050; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.kpi-value { font-size: 1.25rem; font-weight: 600; color: #7dbf6a; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.kpi-sub { font-size: 0.75rem; color: #c0392b; margin-top: 2px; }
h2, h3, [data-testid="stHeading"] { color: #7dbf6a !important; font-size: 1rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.06em; border-bottom: 1px solid #1e3a1e; padding-bottom: 6px; margin-top: 1.4rem !important; }
.stPlotlyChart { background: #132213 !important; border: 1px solid #1e3a1e !important; border-radius: 8px !important; padding: 4px; }
[data-testid="stDataFrame"] { background: #132213 !important; border: 1px solid #1e3a1e !important; border-radius: 8px !important; }
[data-testid="stButton"] button[kind="primary"] { background-color: #2d6a2d !important; border: none !important; color: #d4e6c3 !important; font-weight: 600 !important; }
[data-testid="stButton"] button[kind="primary"]:hover { background-color: #3d8a3d !important; }
[data-testid="stProgress"] > div > div { background-color: #7dbf6a !important; }
hr { border-color: #1e3a1e !important; }
.report-box { background: #132213; border: 1px solid #1e3a1e; border-radius: 8px; padding: 16px 20px; font-size: 0.88rem; line-height: 1.7; color: #a8c897; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('''
<div class="dash-header">
  <h1>Sentinel-2 Crop Health Monitor</h1>
  <p>Agricultural field monitoring · Vegetation health, water status & anomaly detection · Google Earth Engine</p>
</div>
''', unsafe_allow_html=True)

# ── Sidebar: inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    gee_project = st.text_input(
        "GEE Project ID",
        value="projecttestrun",
        help="Your Google Earth Engine project ID.",
    )

    st.subheader("Field Location")
    lat = st.number_input("Latitude",  value=45.15, format="%.4f", step=0.01)
    lon = st.number_input("Longitude", value=11.80, format="%.4f", step=0.01)
    buffer_km = st.slider(
        "Buffer radius (km)", min_value=0.5, max_value=5.0, value=2.0, step=0.5,
        help="Radius of the square bounding box around the center point."
    )

    st.subheader("Date")
    import datetime as dt
    date_center = st.date_input(
        "Analysis date",
        value=dt.date(2024, 7, 15),
        min_value=dt.date(2017, 1, 1),
        max_value=dt.date.today(),
    ).strftime("%Y-%m-%d")
    window_days = st.slider(
        "Date window (± days)", min_value=5, max_value=30, value=10, step=5,
        help="Number of days before/after the analysis date for the composite."
    )

    st.subheader("Advanced")
    cloud_threshold = st.slider("Max cloud cover (%)", 5, 50, 20, step=5)
    scale = st.selectbox(
        "Download resolution (m/px)",
        options=[10, 20, 30],
        index=1,
        help="Lower = finer detail but slower. 20m is a good balance."
    )
    contamination = st.slider(
        "Anomaly sensitivity", 0.01, 0.15, 0.05, step=0.01,
        help="Fraction of pixels flagged as anomalies by Isolation Forest."
    )

    run_button = st.button("Run Analysis", type="primary", use_container_width=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

# ── Main: run pipeline ────────────────────────────────────────────────────────
if run_button:
    with st.spinner("Initializing GEE..."):
        auth_ok = initialize_gee(project=gee_project)

    if not auth_ok:
        st.error("GEE authentication failed. Check your project ID and credentials.")
        st.stop()

    progress = st.progress(0, text="Loading Sentinel-2 data...")

    try:
        import ee

        # 1. Build AOI
        aoi = build_aoi(lat, lon, buffer_km)
        progress.progress(10, text="AOI defined. Loading Sentinel-2...")

        # 2. Load composite
        composite, n_images = load_sentinel2(aoi, date_center, window_days, cloud_threshold)
        progress.progress(30, text=f"Loaded {n_images} scenes. Computing indices...")

        # 3. Compute spectral indices
        indices_image = compute_indices(composite)
        progress.progress(45, text="Downloading index arrays...")

        # 4. Download arrays
        arrays = download_indices(indices_image, aoi, scale=scale)
        ndvi, evi, ndwi, ndmi = arrays["NDVI"], arrays["EVI"], arrays["NDWI"], arrays["NDMI"]

        # Valid pixel mask (non-nan, non-zero)
        valid_mask = (
            np.isfinite(ndvi) & np.isfinite(evi) &
            np.isfinite(ndwi) & np.isfinite(ndmi) &
            (ndvi != 0)
        )

        progress.progress(60, text="Classifying health & water status...")

        # 5. Classification
        health_map, health_pcts, health_quad = classify_health(ndvi, evi, valid_mask=valid_mask)
        water_map,  water_pcts,  water_quad  = classify_water(ndwi, ndmi, valid_mask=valid_mask)

        progress.progress(72, text="Computing uniformity & hotspots...")

        # 6. Uniformity + hotspots
        cluster_map, unif_score, unif_label = compute_uniformity(ndvi, valid_mask)
        anomaly_mask, hotspot_map = detect_hotspots(
            ndvi, evi, ndwi, ndmi,
            valid_mask=valid_mask,
            contamination=contamination,
        )

        progress.progress(85, text="Fetching CHIRPS rainfall data...")

        # 7. Rainfall
        rainfall = get_rainfall_context(aoi, date_center)

        progress.progress(100, text="Done!")

        # Store in session state
        st.session_state.results = {
            "n_images": n_images,
            "ndvi": ndvi, "evi": evi, "ndwi": ndwi, "ndmi": ndmi,
            "valid_mask": valid_mask,
            "health_map": health_map, "health_pcts": health_pcts, "health_quad": health_quad,
            "water_map": water_map,   "water_pcts": water_pcts,   "water_quad": water_quad,
            "cluster_map": cluster_map, "unif_score": unif_score, "unif_label": unif_label,
            "anomaly_mask": anomaly_mask, "hotspot_map": hotspot_map,
            "rainfall": rainfall,
            "date_center": date_center,
            "lat": lat, "lon": lon, "buffer_km": buffer_km,
        }

    except ValueError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.stop()

# ── Display results ───────────────────────────────────────────────────────────
if st.session_state.results is not None:
    r = st.session_state.results

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    st.subheader("Field Summary")
    st.markdown(f'''
<div class="kpi-row">
  <div class="kpi-card"><div class="kpi-label">Analysis Date</div><div class="kpi-value">{r["date_center"]}</div></div>
  <div class="kpi-card"><div class="kpi-label">Scenes Used</div><div class="kpi-value">{r["n_images"]}</div></div>
  <div class="kpi-card"><div class="kpi-label">Vegetation Health</div><div class="kpi-value">{r["health_pcts"]["Healthy"]:.0f}% Healthy</div><div class="kpi-sub">{r["health_pcts"]["High Stress"]:.0f}% High Stress</div></div>
  <div class="kpi-card"><div class="kpi-label">Water Status</div><div class="kpi-value">{r["water_pcts"]["Adequate"]:.0f}% Adequate</div><div class="kpi-sub">{r["water_pcts"]["Strong Deficit"]:.0f}% Strong Deficit</div></div>
  <div class="kpi-card"><div class="kpi-label">Field Uniformity</div><div class="kpi-value">{r["unif_label"]}</div><div class="kpi-sub">Score: {r["unif_score"]}/100</div></div>
</div>
''', unsafe_allow_html=True)

    st.divider()

    # ── Maps row ──────────────────────────────────────────────────────────────
    st.subheader("Spatial Maps")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.plotly_chart(
            plot_health_map(r["health_map"], r["valid_mask"]),
            use_container_width=True,
        )

    with col2:
        st.plotly_chart(
            plot_water_map(r["water_map"], r["valid_mask"]),
            use_container_width=True,
        )

    with col3:
        st.plotly_chart(
            plot_ndvi_hotspots(
                r["ndvi"], r["anomaly_mask"], r["hotspot_map"],
                r["unif_score"], r["unif_label"], r["valid_mask"],
            ),
            use_container_width=True,
        )

    st.divider()

    # ── Charts row ────────────────────────────────────────────────────────────
    st.subheader("Analysis")
    col_a, col_b, col_c = st.columns([2, 1.5, 1.5])

    with col_a:
        st.plotly_chart(
            plot_class_breakdown(r["health_pcts"], r["water_pcts"]),
            use_container_width=True,
        )

    with col_b:
        st.plotly_chart(
            plot_ndvi_histogram(r["ndvi"], r["valid_mask"]),
            use_container_width=True,
        )

    with col_c:
        rf = r["rainfall"]
        if rf["total_mm"] is not None:
            st.plotly_chart(
                plot_rainfall_gauge(rf["total_mm"], rf["daily_mm"], rf["rainfall_class"]),
                use_container_width=True,
            )
        else:
            st.info("Rainfall data unavailable for this region/date.")

    st.divider()

    # ── Quadrant analysis table ───────────────────────────────────────────────
    st.subheader("Quadrant Stress Breakdown")
    quad_names = {"NW": "Northwest", "NE": "Northeast", "SW": "Southwest", "SE": "Southeast"}
    class_labels = {1: "Healthy", 2: "Moderate Stress", 3: "High Stress"}

    import pandas as pd
    quad_rows = []
    for q_key, q_label in quad_names.items():
        h_q = r["health_quad"].get(q_key, {})
        w_q = r["water_quad"].get(q_key, {})
        quad_rows.append({
            "Quadrant": q_label,
            "Dominant Health Class": class_labels.get(h_q.get("dominant", 0), "N/A"),
            "Health Stress %": h_q.get("stress_pct", "N/A"),
            "Water Stress %": w_q.get("stress_pct", "N/A"),
        })

    st.dataframe(pd.DataFrame(quad_rows), use_container_width=True, hide_index=True)

    # ── Agronomic report ──────────────────────────────────────────────────────
    st.subheader("Automated Field Report")
    rf = r["rainfall"]

    worst_health_quad = max(
        r["health_quad"].items(), key=lambda x: x[1].get("stress_pct", 0), default=(None, {})
    )
    worst_water_quad = max(
        r["water_quad"].items(), key=lambda x: x[1].get("stress_pct", 0), default=(None, {})
    )
    quad_full = {"NW": "northwest", "NE": "northeast", "SW": "southwest", "SE": "southeast"}
    n_anomalies = int(r["anomaly_mask"].sum())

    report_lines = [
        f"**Field:** {r['lat']:.3f}°N, {r['lon']:.3f}°E · {r['buffer_km']} km buffer",
        f"**Date:** {r['date_center']} (composite from {r['n_images']} Sentinel-2 scenes)",
        "",
        f"**Vegetation health:** {r['health_pcts']['Healthy']:.1f}% of the field is healthy. "
        f"{r['health_pcts']['High Stress']:.1f}% shows high stress. "
        + (
            f"Highest stress concentration is in the {quad_full.get(worst_health_quad[0], 'unknown')} quadrant "
            f"({worst_health_quad[1].get('stress_pct', 0):.1f}% stressed pixels)."
            if worst_health_quad[0] else ""
        ),
        "",
        f"**Water status:** {r['water_pcts']['Adequate']:.1f}% of pixels show adequate water. "
        f"{r['water_pcts']['Strong Deficit']:.1f}% show strong deficit. "
        + (
            f"Worst water stress is in the {quad_full.get(worst_water_quad[0], 'unknown')} quadrant."
            if worst_water_quad[0] else ""
        ),
        "",
        f"**Field uniformity:** {r['unif_label']} (score: {r['unif_score']}/100). "
        f"{n_anomalies} anomalous pixels detected via Isolation Forest + DBSCAN clustering.",
        "",
    ]

    if rf["total_mm"] is not None:
        report_lines.append(
            f"**Rainfall context:** {rf['total_mm']} mm cumulative precipitation over the past 30 days "
            f"({rf['daily_mm']} mm/day avg) — classified as **{rf['rainfall_class']}**. "
            + (
                "Water deficit likely reflects drought stress or poor irrigation."
                if rf["rainfall_class"] == "Low" and r["water_pcts"]["Strong Deficit"] > 20
                else "Water deficit may indicate drainage or soil issues despite adequate rainfall."
                if rf["rainfall_class"] == "Adequate" and r["water_pcts"]["Strong Deficit"] > 20
                else "Conditions appear consistent with rainfall levels."
            )
        )

    st.markdown('<div class="report-box">' + "<br>".join(report_lines) + '</div>', unsafe_allow_html=True)

else:
    # ── Empty state ───────────────────────────────────────────────────────────
    st.info(
        "Configure your field location and date in the sidebar, then click Run Analysis."
    )

    col1, col2, col3 = st.columns(3)
    col1.markdown("#### Vegetation Health\nNDVI + EVI classification into Healthy / Moderate Stress / High Stress zones.")
    col2.markdown("#### Water Status\nNDWI + NDMI moisture analysis with CHIRPS 30-day rainfall context.")
    col3.markdown("#### Anomaly Detection\nIsolation Forest + DBSCAN hotspot detection across all four spectral indices.")
