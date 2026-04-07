"""
viz/charts.py
-------------
Reusable Plotly chart components for the GEE crop health dashboard.
All functions return plotly Figure objects for rendering in Streamlit.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ---------------------------------------------------------------------------
# Color palettes (consistent with notebook)
# ---------------------------------------------------------------------------

HEALTH_COLORS = {
    "Healthy":         "#2d6a2d",
    "Moderate Stress": "#f0c040",
    "High Stress":     "#c0392b",
}

WATER_COLORS = {
    "Adequate":       "#1a6fa8",
    "Mild Deficit":   "#f5a623",
    "Strong Deficit": "#c0392b",
}


# ---------------------------------------------------------------------------
# Map plots (imshow via plotly)
# ---------------------------------------------------------------------------

def plot_classification_map(
    class_map: np.ndarray,
    title: str,
    color_map: dict,
    class_labels: list,
    valid_mask: np.ndarray = None,
) -> go.Figure:
    """
    Render a 2D integer class map as an annotated heatmap.

    class_map: 2D int array (values 1,2,3; 0=nodata)
    color_map: dict mapping label -> hex color
    class_labels: list of labels ordered by class value [1,2,3]
    """
    display = class_map.astype(float)
    if valid_mask is not None:
        display[~valid_mask] = np.nan

    colors = list(color_map.values())
    colorscale = [
        [0.0,  "#1a2e1a"],   # nodata shown as dark
        [0.33, colors[0]],
        [0.34, colors[0]],
        [0.66, colors[1]],
        [0.67, colors[1]],
        [1.0,  colors[2]],
    ]

    fig = go.Figure(
        go.Heatmap(
            z=display,
            colorscale=colorscale,
            zmin=1,
            zmax=3,
            showscale=False,
            hovertemplate="Row: %{y}<br>Col: %{x}<br>Class: %{z:.0f}<extra></extra>",
        )
    )

    # Manual legend via invisible scatter traces
    for label, color in zip(class_labels, colors):
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=12, color=color, symbol="square"),
                name=label,
                showlegend=True,
            )
        )

    fig.update_layout(

    paper_bgcolor="#132213",
    plot_bgcolor="#132213",
    font=dict(color="#ffffff", size=12),
        title=dict(text=title, font=dict(size=13, color="#ffffff")),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, autorange="reversed"),
        legend=dict(
            orientation="v",
            yanchor="top", y=1,
            xanchor="left", x=1.02,
            font=dict(size=11, color="#ffffff"),
        ),
        margin=dict(l=10, r=100, t=40, b=10),
        height=380,
    )
    return fig


def plot_health_map(health_map, valid_mask=None) -> go.Figure:
    return plot_classification_map(
        class_map=health_map,
        title="Vegetation Health Map",
        color_map=HEALTH_COLORS,
        class_labels=["Healthy", "Moderate Stress", "High Stress"],
        valid_mask=valid_mask,
    )


def plot_water_map(water_map, valid_mask=None) -> go.Figure:
    return plot_classification_map(
        class_map=water_map,
        title="Water Status Map",
        color_map=WATER_COLORS,
        class_labels=["Adequate", "Mild Deficit", "Strong Deficit"],
        valid_mask=valid_mask,
    )


# ---------------------------------------------------------------------------
# NDVI continuous map with hotspot overlay
# ---------------------------------------------------------------------------

def plot_ndvi_hotspots(
    ndvi: np.ndarray,
    anomaly_mask: np.ndarray,
    hotspot_map: np.ndarray,
    uniformity_score: int,
    uniformity_label: str,
    valid_mask: np.ndarray = None,
) -> go.Figure:
    display_ndvi = ndvi.astype(float)
    if valid_mask is not None:
        display_ndvi[~valid_mask] = np.nan

    fig = go.Figure(
        go.Heatmap(
            z=display_ndvi,
            colorscale="RdYlGn",
            zmin=0,
            zmax=1,
            colorbar=dict(title="NDVI", thickness=15),
            hovertemplate="NDVI: %{z:.3f}<extra></extra>",
        )
    )

    # Overlay hotspot pixels
    anom_rows, anom_cols = np.where(anomaly_mask)
    if len(anom_rows) > 0:
        fig.add_trace(
            go.Scatter(
                x=anom_cols,
                y=anom_rows,
                mode="markers",
                marker=dict(size=3, color="red", opacity=0.5),
                name="Anomaly pixels",
                showlegend=True,
            )
        )

    fig.update_layout(

    paper_bgcolor="#132213",
    plot_bgcolor="#132213",
    font=dict(color="#ffffff", size=12),
        title=dict(text=f"NDVI + Hotspots — Uniformity: {uniformity_label} ({uniformity_score}/100)", font=dict(size=13, color="#ffffff")),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, autorange="reversed"),
        legend=dict(
            orientation="v",
            yanchor="top", y=1,
            xanchor="left", x=1.02,
            font=dict(size=11, color="#ffffff"),
        ),
        margin=dict(l=10, r=100, t=40, b=10),
        height=380,
    )
    return fig


# ---------------------------------------------------------------------------
# Class breakdown bar charts
# ---------------------------------------------------------------------------

def plot_class_breakdown(health_pcts: dict, water_pcts: dict) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Vegetation Health", "Water Status"),
    )

    # Health
    fig.add_trace(
        go.Bar(
            x=list(health_pcts.keys()),
            y=list(health_pcts.values()),
            marker_color=list(HEALTH_COLORS.values()),
            text=[f"{v:.1f}%" for v in health_pcts.values()],
            textposition="outside",
            name="Health",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Water
    fig.add_trace(
        go.Bar(
            x=list(water_pcts.keys()),
            y=list(water_pcts.values()),
            marker_color=list(WATER_COLORS.values()),
            text=[f"{v:.1f}%" for v in water_pcts.values()],
            textposition="outside",
            name="Water",
            showlegend=False,
        ),
        row=1, col=2,
    )

    fig.update_yaxes(title_text="Area (%)", range=[0, 110], row=1, col=1)
    fig.update_yaxes(range=[0, 110], row=1, col=2)
    fig.update_layout(

    paper_bgcolor="#132213",
    plot_bgcolor="#132213",
    font=dict(color="#ffffff", size=12),
        height=350,
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# NDVI distribution histogram
# ---------------------------------------------------------------------------

def plot_ndvi_histogram(ndvi: np.ndarray, valid_mask: np.ndarray = None) -> go.Figure:
    vals = ndvi[valid_mask].flatten() if valid_mask is not None else ndvi.flatten()
    vals = vals[~np.isnan(vals)]

    fig = go.Figure(
        go.Histogram(
            x=vals,
            nbinsx=50,
            marker_color="#2d6a2d",
            opacity=0.8,
            name="NDVI",
        )
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="#f0c040",
                  annotation_text="Healthy (0.5)",
                  annotation_position="bottom right",
                  annotation_font_size=10)
    fig.add_vline(x=0.3, line_dash="dash", line_color="#c0392b",
                  annotation_text="Stress (0.3)",
                  annotation_position="bottom left",
                  annotation_font_size=10)
    fig.update_layout(

    paper_bgcolor="#132213",
    plot_bgcolor="#132213",
    font=dict(color="#ffffff", size=12),
        title=dict(text="NDVI Distribution", font=dict(color="#ffffff")),
        xaxis_title="NDVI",
        yaxis_title="Pixel Count",
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
        showlegend=False,
    )
    return fig


# ---------------------------------------------------------------------------
# Rainfall context gauge
# ---------------------------------------------------------------------------

def plot_rainfall_gauge(total_mm: float, daily_mm: float, rainfall_class: str) -> go.Figure:
    color = {"Low": "#c0392b", "Moderate": "#f0c040", "Adequate": "#2d6a2d"}.get(
        rainfall_class, "#888888"
    )

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=total_mm,
            title={"text": f"30-Day Rainfall ({rainfall_class})<br><span style='font-size:0.8em'>{daily_mm} mm/day avg</span>"},
            gauge={
                "axis": {"range": [0, 120]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 20],  "color": "#fde8e8"},
                    {"range": [20, 60], "color": "#fef9e8"},
                    {"range": [60, 120],"color": "#e8f5e8"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 2},
                    "thickness": 0.75,
                    "value": total_mm,
                },
            },
            number={"suffix": " mm"},
        )
    )
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
    return fig
