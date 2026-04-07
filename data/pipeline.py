"""
data/pipeline.py
----------------
Core GEE data pipeline extracted from crop_health_classifier_sentinel2.ipynb.

All functions are stateless and accept explicit parameters so they can be
called directly from Streamlit with user-supplied inputs.
"""

import ee
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# AOI
# ---------------------------------------------------------------------------

def build_aoi(lat: float, lon: float, buffer_km: float) -> ee.Geometry:
    """Square bounding box centered on (lat, lon) with buffer_km radius."""
    delta_deg = buffer_km / 111.0
    return ee.Geometry.BBox(
        lon - delta_deg,
        lat - delta_deg,
        lon + delta_deg,
        lat + delta_deg,
    )


# ---------------------------------------------------------------------------
# Sentinel-2 loading & cloud masking
# ---------------------------------------------------------------------------

def _mask_s2_clouds(image: ee.Image) -> ee.Image:
    """QA60 bitmask cloud mask + scale factor."""
    qa = image.select("QA60")
    cloud_bit  = 1 << 10
    cirrus_bit = 1 << 11
    mask = (
        qa.bitwiseAnd(cloud_bit).eq(0)
        .And(qa.bitwiseAnd(cirrus_bit).eq(0))
    )
    return image.updateMask(mask).divide(10000)


def load_sentinel2(
    aoi: ee.Geometry,
    date_center: str,
    window_days: int = 10,
    cloud_threshold: int = 20,
) -> tuple[ee.Image, int]:
    """
    Load a cloud-masked Sentinel-2 SR median composite.

    Returns (composite ee.Image, n_scenes).
    Raises ValueError if no scenes found.
    """
    center_dt  = datetime.strptime(date_center, "%Y-%m-%d")
    start_date = (center_dt - timedelta(days=window_days)).strftime("%Y-%m-%d")
    end_date   = (center_dt + timedelta(days=window_days)).strftime("%Y-%m-%d")

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_threshold))
        .map(_mask_s2_clouds)
    )

    n_images = collection.size().getInfo()
    if n_images == 0:
        raise ValueError(
            f"No Sentinel-2 scenes found between {start_date} and {end_date} "
            f"with cloud cover < {cloud_threshold}%. "
            "Try increasing the date window or cloud threshold."
        )

    composite = collection.median().clip(aoi)
    return composite, n_images


# ---------------------------------------------------------------------------
# Spectral indices
# ---------------------------------------------------------------------------

def compute_indices(image: ee.Image) -> ee.Image:
    """
    Compute NDVI, EVI, NDWI (Gao 1996), NDMI from a Sentinel-2 SR image.
    Returns ee.Image with bands ['NDVI','EVI','NDWI','NDMI'].
    """
    blue  = image.select("B2")
    red   = image.select("B4")
    nir   = image.select("B8")
    swir1 = image.select("B11")
    swir2 = image.select("B12")

    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    evi  = (
        nir.subtract(red).multiply(2.5)
        .divide(
            nir.add(red.multiply(6))
            .subtract(blue.multiply(7.5))
            .add(1)
        )
        .rename("EVI")
    )
    ndwi = nir.subtract(swir1).divide(nir.add(swir1)).rename("NDWI")
    ndmi = nir.subtract(swir2).divide(nir.add(swir2)).rename("NDMI")

    return ee.Image([ndvi, evi, ndwi, ndmi])


def download_band(
    image: ee.Image,
    band_name: str,
    aoi: ee.Geometry,
    scale: int = 10,
) -> np.ndarray:
    """Download a single band as a 2D numpy array via getRegion."""
    import requests, io, csv

    url = image.select(band_name).getDownloadURL(
        {
            "scale": scale,
            "region": aoi,
            "format": "GEO_TIFF",
            "filePerBand": False,
        }
    )
    # Use ee_to_numpy pattern via getRegion for small AOIs
    data = image.select(band_name).reduceRegion(
        reducer=ee.Reducer.toList(),
        geometry=aoi,
        scale=scale,
        maxPixels=1e7,
    ).get(band_name).getInfo()

    # Determine approximate shape from AOI bounds
    bounds = aoi.bounds().getInfo()["coordinates"][0]
    lons = [p[0] for p in bounds]
    lats = [p[1] for p in bounds]
    lon_range = max(lons) - min(lons)
    lat_range = max(lats) - min(lats)
    cols = max(1, int(lon_range * 111000 / scale))
    rows = max(1, int(lat_range * 111000 / scale))

    arr = np.array(data, dtype=np.float32)
    # Reshape as best we can; pad/trim if needed
    target = rows * cols
    if len(arr) < target:
        arr = np.pad(arr, (0, target - len(arr)), constant_values=np.nan)
    else:
        arr = arr[:target]
    return arr.reshape(rows, cols)


def download_indices(
    indices_image: ee.Image,
    aoi: ee.Geometry,
    scale: int = 20,
) -> dict[str, np.ndarray]:
    """
    Download all four index bands as numpy arrays.
    Returns dict with keys NDVI, EVI, NDWI, NDMI.
    """
    result = {}
    for band in ["NDVI", "EVI", "NDWI", "NDMI"]:
        result[band] = download_band(indices_image, band, aoi, scale=scale)
    return result


# ---------------------------------------------------------------------------
# CHIRPS rainfall
# ---------------------------------------------------------------------------

def get_rainfall_context(
    aoi: ee.Geometry,
    date_center: str,
    lookback_days: int = 30,
) -> dict:
    """
    Retrieve mean cumulative CHIRPS precipitation over AOI for the
    lookback_days window before date_center.

    Returns dict with keys: total_mm, daily_mm, rainfall_class.
    """
    center_dt  = datetime.strptime(date_center, "%Y-%m-%d")
    start_date = (center_dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    chirps = (
        ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
        .filterBounds(aoi)
        .filterDate(start_date, date_center)
        .select("precipitation")
    )

    total_precip = chirps.sum().clip(aoi)
    mean_dict = total_precip.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=5000,
        maxPixels=1e6,
    )

    total_mm = mean_dict.get("precipitation").getInfo()
    if total_mm is None:
        return {"total_mm": None, "daily_mm": None, "rainfall_class": "Unknown"}

    daily_mm = total_mm / lookback_days

    if total_mm < 20:
        rainfall_class = "Low"
    elif total_mm < 60:
        rainfall_class = "Moderate"
    else:
        rainfall_class = "Adequate"

    return {
        "total_mm": round(total_mm, 1),
        "daily_mm": round(daily_mm, 2),
        "rainfall_class": rainfall_class,
        "start_date": start_date,
        "end_date": date_center,
    }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_health(
    ndvi: np.ndarray,
    evi: np.ndarray,
    thresholds: list = [0.5, 0.3],
    valid_mask: np.ndarray = None,
) -> tuple[np.ndarray, dict, dict]:
    """
    Classify vegetation health into 3 classes (1=Healthy, 2=Moderate, 3=High Stress).
    Returns (health_map, percentages dict, quadrant_stats dict).
    """
    h_thresh, m_thresh = thresholds
    health_score = 0.6 * ndvi + 0.4 * evi

    health_map = np.zeros_like(health_score, dtype=np.int8)
    mask = valid_mask if valid_mask is not None else np.ones_like(health_score, dtype=bool)

    health_map[mask & (health_score >= h_thresh)] = 1
    health_map[mask & (health_score >= m_thresh) & (health_score < h_thresh)] = 2
    health_map[mask & (health_score < m_thresh)] = 3

    valid_pixels = max(mask.sum(), 1)
    percentages = {
        "Healthy":         round(100 * (health_map == 1).sum() / valid_pixels, 1),
        "Moderate Stress": round(100 * (health_map == 2).sum() / valid_pixels, 1),
        "High Stress":     round(100 * (health_map == 3).sum() / valid_pixels, 1),
    }

    quadrant_stats = _compute_quadrant_stats(health_map, mask)
    return health_map, percentages, quadrant_stats


def classify_water(
    ndwi: np.ndarray,
    ndmi: np.ndarray,
    thresholds: list = [0.0, -0.3],
    valid_mask: np.ndarray = None,
) -> tuple[np.ndarray, dict, dict]:
    """
    Classify water status into 3 classes (1=Adequate, 2=Mild Deficit, 3=Strong Deficit).
    Returns (water_map, percentages dict, quadrant_stats dict).
    """
    a_thresh, s_thresh = thresholds
    water_score = 0.5 * ndwi + 0.5 * ndmi

    water_map = np.zeros_like(water_score, dtype=np.int8)
    mask = valid_mask if valid_mask is not None else np.ones_like(water_score, dtype=bool)

    water_map[mask & (water_score >= a_thresh)] = 1
    water_map[mask & (water_score >= s_thresh) & (water_score < a_thresh)] = 2
    water_map[mask & (water_score < s_thresh)] = 3

    valid_pixels = max(mask.sum(), 1)
    percentages = {
        "Adequate":       round(100 * (water_map == 1).sum() / valid_pixels, 1),
        "Mild Deficit":   round(100 * (water_map == 2).sum() / valid_pixels, 1),
        "Strong Deficit": round(100 * (water_map == 3).sum() / valid_pixels, 1),
    }

    quadrant_stats = _compute_quadrant_stats(water_map, mask)
    return water_map, percentages, quadrant_stats


def _compute_quadrant_stats(class_map: np.ndarray, mask: np.ndarray) -> dict:
    rows, cols = class_map.shape
    hmid, vmid = rows // 2, cols // 2
    quadrants = {
        "NW": (class_map[:hmid, :vmid], mask[:hmid, :vmid]),
        "NE": (class_map[:hmid, vmid:], mask[:hmid, vmid:]),
        "SW": (class_map[hmid:, :vmid], mask[hmid:, :vmid]),
        "SE": (class_map[hmid:, vmid:], mask[hmid:, vmid:]),
    }
    stats = {}
    for q_name, (q_arr, q_mask) in quadrants.items():
        valid = q_arr[q_mask & (q_arr > 0)]
        if len(valid) > 0:
            counts = np.bincount(valid, minlength=4)[1:]
            dominant_idx = int(np.argmax(counts)) + 1
            stress_pct = round(100 * (valid == 3).sum() / len(valid), 1)
            stats[q_name] = {"dominant": dominant_idx, "stress_pct": stress_pct}
    return stats


# ---------------------------------------------------------------------------
# Uniformity & hotspot detection
# ---------------------------------------------------------------------------

def compute_uniformity(
    ndvi: np.ndarray,
    valid_mask: np.ndarray = None,
    n_clusters: int = 3,
    uniformity_bins: list = [70, 40],
) -> tuple[np.ndarray, int, str]:
    """
    K-Means uniformity score on NDVI pixel values.
    Returns (cluster_map, score 0-100, label).
    """
    mask = valid_mask if valid_mask is not None else np.ones_like(ndvi, dtype=bool)
    valid_pixels = ndvi[mask].reshape(-1, 1)

    if len(valid_pixels) < n_clusters:
        return np.zeros_like(ndvi, dtype=np.int8), 0, "Insufficient data"

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(valid_pixels)

    counts = np.bincount(labels)
    dominant_pct = counts.max() / counts.sum() * 100
    score = int(dominant_pct)

    if score >= uniformity_bins[0]:
        label = "Uniform"
    elif score >= uniformity_bins[1]:
        label = "Mixed"
    else:
        label = "Patchy"

    cluster_map = np.full(ndvi.shape, -1, dtype=np.int8)
    cluster_map[mask] = labels
    return cluster_map, score, label


def detect_hotspots(
    ndvi: np.ndarray,
    evi: np.ndarray,
    ndwi: np.ndarray,
    ndmi: np.ndarray,
    valid_mask: np.ndarray = None,
    contamination: float = 0.05,
    dbscan_eps: float = 0.001,
    dbscan_min_samples: int = 5,
    n_hotspots: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Isolation Forest + DBSCAN hotspot detection.
    Returns (anomaly_mask 2D bool, hotspot_cluster_map 2D int).
    """
    mask = valid_mask if valid_mask is not None else np.ones_like(ndvi, dtype=bool)

    rows, cols = ndvi.shape
    row_idx, col_idx = np.where(mask)
    features = np.column_stack([
        ndvi[mask], evi[mask], ndwi[mask], ndmi[mask]
    ])

    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    iso = IsolationForest(contamination=contamination, random_state=42)
    preds = iso.fit_predict(X_scaled)  # -1 = anomaly

    anomaly_mask = np.zeros_like(ndvi, dtype=bool)
    anomaly_mask[row_idx, col_idx] = preds == -1

    # DBSCAN on anomaly pixel coordinates to find coherent hotspot zones
    anom_rows, anom_cols = np.where(anomaly_mask)
    hotspot_map = np.full(ndvi.shape, -1, dtype=np.int8)

    if len(anom_rows) >= dbscan_min_samples:
        coords = np.column_stack([
            anom_rows / rows,  # normalize to [0,1]
            anom_cols / cols,
        ])
        db = DBSCAN(eps=dbscan_eps * 100, min_samples=dbscan_min_samples)
        cluster_labels = db.fit_predict(coords)
        hotspot_map[anom_rows, anom_cols] = cluster_labels

    return anomaly_mask, hotspot_map
