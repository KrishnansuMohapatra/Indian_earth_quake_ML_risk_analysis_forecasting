import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from aftershock_engine import generate_forecast
from data_processing import load_data
from models import (
    train_classification_model,
    train_warning_model,
    get_seismic_zone_risk,
    blend_scores,
)
from visualization import *
from forecast_visualization import *
from streamlit_folium import st_folium

# ── PAGE SETUP ────────────────────────────────────────────────────
st.set_page_config(page_title="SeismoIQ", layout="wide", page_icon="🌍")

# ── DARK THEME CSS (includes folium white-gap fix) ────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] {
      background: radial-gradient(ellipse at 20% 20%, #0d1220 0%, #060810 60%, #000000 100%) !important;
  }
  [data-testid="stHeader"]  { background: transparent !important; }
  [data-testid="stSidebar"] { background: #0a0c14 !important; }
  .block-container { padding-top: 2rem; }

  .stTabs [data-baseweb="tab-list"] {
      background: #111520;
      border-radius: 10px;
      padding: 4px;
      gap: 4px;
  }
  .stTabs [data-baseweb="tab"] {
      background: transparent;
      color: #7a8299;
      border-radius: 8px;
      font-size: 13px;
  }
  .stTabs [aria-selected="true"] {
      background: #1e2a44 !important;
      color: #4a90e2 !important;
  }
  [data-testid="metric-container"] {
      background: #111520;
      border: 1px solid rgba(74,144,226,0.15);
      border-radius: 12px;
      padding: 14px 18px;
  }
  [data-testid="stMetricValue"] { color: #e8eaf0 !important; }
  [data-testid="stMetricLabel"] { color: #7a8299 !important; }
  h2, h3 { color: #c8d0e8 !important; }

  /* ── Folium white-gap fixes ── */
  iframe {
      border: none !important;
      display: block !important;
      background: #060810 !important;
  }
  .stCustomComponentV1 {
      background: transparent !important;
      border: none !important;
  }
  [data-testid="stCustomComponentV1"] > div {
      background: #060810 !important;
  }
  /* Remove any white padding/margin around map iframes */
  .element-container:has(iframe) {
      background: #060810 !important;
      border-radius: 10px;
      overflow: hidden;
  }
</style>
""", unsafe_allow_html=True)

# ── SHARED PLOTLY DARK LAYOUT ─────────────────────────────────────
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#0e1422",
    font=dict(color="#c8d0e8", family="sans-serif"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)"),
    margin=dict(l=40, r=20, t=30, b=40),
)

# ── LOAD DATA (cached) ────────────────────────────────────────────
@st.cache_data(show_spinner="Loading earthquake data…")
def load_data_cached():
    return load_data()

df     = load_data_cached()
df_fc  = prepare_data(df)

# Downsampled df for maps — keep top 2 000 most significant events
MAX_MAP_POINTS = 2000
df_map = df.nlargest(MAX_MAP_POINTS, "magnitude")

st.markdown("## 🌍 SeismoIQ — Earthquake Intelligence System")

# ── TABS ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🗺 Maps", "🤖 ML Prediction", "📈 Forecast", "🌋 Aftershock"
])


# ═════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═════════════════════════════════════════════════════════════════
with tab1:

    # ── Metric row ───────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Events",    len(df))
    col2.metric("Avg Magnitude",   round(df["magnitude"].mean(), 2))
    col3.metric("Avg Depth (km)",  round(df["depth_km"].mean(), 2))
    col4.metric("Max Magnitude",   round(df["magnitude"].max(), 2))

    st.markdown("---")

    # ── Row 1: magnitude histogram + events over time ─────────────
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Magnitude distribution")
        fig1 = px.histogram(
            df, x="magnitude", nbins=30,
            color_discrete_sequence=["#4a90e2"],
        )
        fig1.update_traces(marker_line_color="#0a0c14", marker_line_width=0.5)
        fig1.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.subheader("Events per year")
        yearly = (
            df.groupby("year")
              .size()
              .reset_index(name="count")
        )
        fig2 = px.area(
            yearly, x="year", y="count",
            color_discrete_sequence=["#4ae2a0"],
        )
        fig2.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row 2: depth scatter + magnitude category breakdown ───────
    c3, c4 = st.columns(2)

    with c3:
        st.subheader("Depth vs magnitude")
        fig3 = px.scatter(
            df, x="magnitude", y="depth_km",
            color="magnitude",
            color_continuous_scale="Turbo",
            opacity=0.55,
            labels={"depth_km": "Depth (km)", "magnitude": "Magnitude"},
        )
        fig3.update_layout(**DARK_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        st.subheader("Magnitude category breakdown")
        cat_counts = df["mag_category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        fig4 = px.bar(
            cat_counts, x="count", y="category",
            orientation="h",
            color="count",
            color_continuous_scale="Blues",
            labels={"count": "Events", "category": ""},
        )
        fig4.update_layout(**DARK_LAYOUT, coloraxis_showscale=False)
        fig4.update_yaxes(autorange="reversed", gridcolor="rgba(255,255,255,0.05)")
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row 3: cumulative frequency (logN vs magnitude) ───────────
    st.subheader("Gutenberg–Richter: cumulative frequency")
    fig5 = px.scatter(
        df.drop_duplicates("magnitude_bin").sort_values("magnitude_bin"),
        x="magnitude_bin", y="logN",
        trendline="ols",
        color_discrete_sequence=["#e25c4a"],
        labels={"magnitude_bin": "Magnitude bin", "logN": "log₁₀(N)"},
    )
    fig5.update_layout(**DARK_LAYOUT)
    st.plotly_chart(fig5, use_container_width=True)

    # ── Raw data expander ─────────────────────────────────────────
    with st.expander("Raw data preview"):
        st.dataframe(df.head(100), use_container_width=True)


# ═════════════════════════════════════════════════════════════════
# TAB 2 — MAPS  (lazy-loaded + cached + returned_objects=[])
# ═════════════════════════════════════════════════════════════════

# Cached map builders — recomputed only when df_map changes
@st.cache_data(show_spinner=False, ttl=3600)
def _satellite_map(_df):
    return satellite_map(_df)

@st.cache_data(show_spinner=False, ttl=3600)
def _depth_map(_df):
    return depth_visualization(_df)

@st.cache_data(show_spinner=False, ttl=3600)
def _timeline_map(_df, year):
    return timeline_map(_df, year)

@st.cache_data(show_spinner=False, ttl=3600)
def _seismic_zone_map():
    return seismic_zone_map()

with tab2:

    # Lazy-load gate — avoids building all 4 heavy maps on startup
    if "maps_loaded" not in st.session_state:
        st.session_state.maps_loaded = False

    if not st.session_state.maps_loaded:
        st.info(
            f"Maps use the top **{MAX_MAP_POINTS}** events by magnitude "
            "for performance. Click below to render."
        )
        if st.button("🗺 Load Maps", type="primary"):
            st.session_state.maps_loaded = True
            st.rerun()
    else:
        st.subheader("Satellite map")
        st_folium(
            _satellite_map(df_map),
            height=500,
            returned_objects=[],   # ← stops map state triggering reruns
            use_container_width=True,
        )

        st.subheader("Depth visualisation")
        st_folium(
            _depth_map(df_map),
            height=500,
            returned_objects=[],
            use_container_width=True,
        )

        st.subheader("Timeline")
        year = st.slider("Year", int(df["year"].min()), int(df["year"].max()))
        st_folium(
            _timeline_map(df_map, year),
            height=500,
            returned_objects=[],
            use_container_width=True,
        )

        st.subheader("Seismic zones")
        st_folium(
            _seismic_zone_map(),
            height=500,
            returned_objects=[],
            use_container_width=True,
        )

        if st.button("🔄 Reload Maps"):
            st.cache_data.clear()
            st.session_state.maps_loaded = False
            st.rerun()


# ═════════════════════════════════════════════════════════════════
# TAB 3 — ML PREDICTION
# ═════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("ML Seismic Risk Estimation")

    # ── Load all models (cached) ──────────────────────────────────
    @st.cache_resource(show_spinner="Training models…")
    def get_models():
        df_ = load_data_cached()
        cls_model,  cls_scaler                              = train_classification_model(df_)
        warn_model, warn_scaler, mag_threshold, \
            features, zone_counts                           = train_warning_model(df_)
        return (df_, cls_model, cls_scaler,
                warn_model, warn_scaler,
                mag_threshold, features, zone_counts)

    (df_model, cls_model, cls_scaler,
     warn_model, warn_scaler,
     mag_threshold, feat_names, zone_counts) = get_models()

    # ── Info banner ───────────────────────────────────────────────
    st.info(
        f"📊 Risk threshold: magnitude ≥ **{mag_threshold:.2f}** "
        f"(top 40% of historical events)"
    )

    # ── Input fields ──────────────────────────────────────────────
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        lat   = st.number_input("Latitude",   value=28.6,
                                min_value=-90.0,  max_value=90.0)
    with col_b:
        lon   = st.number_input("Longitude",  value=77.2,
                                min_value=-180.0, max_value=180.0)
    with col_c:
        depth = st.number_input("Depth (km)", value=10.0,
                                min_value=0.0,    max_value=700.0)

    if st.button("Analyse Risk", type="primary"):

        with st.spinner("Calculating seismic risk..."):

            # ── 1. Historical risk score ──────────────────────────
            hist_score, hist_details, nearby_count = get_seismic_zone_risk(
                lat, lon, df_model
            )

            # ── 2. Zone lookup for ML features ────────────────────
            lat_bin  = (lat // 2) * 2
            lon_bin  = (lon // 2) * 2
            zone_row = zone_counts[
                (zone_counts["lat_bin"] == lat_bin) &
                (zone_counts["lon_bin"] == lon_bin)
            ]

            if not zone_row.empty:
                count    = float(zone_row["count"].values[0])
                mean_mag = float(zone_row["mean_mag"].values[0])
                max_mag  = float(zone_row["max_mag"].values[0])
                std_mag  = float(zone_row["std_mag"].values[0])
            else:
                count = mean_mag = max_mag = std_mag = 0.0

            shallow = int(depth <  70)
            deep    = int(depth > 300)

            # days_since_last: unknown for new input → use zone median
            zone_data = df_model[
                (df_model["lat"].between(lat_bin, lat_bin + 2)) &
                (df_model["lon"].between(lon_bin, lon_bin + 2))
            ]
            if not zone_data.empty and "time" in zone_data.columns:
                zone_data        = zone_data.copy()
                zone_data["time"] = pd.to_datetime(zone_data["time"])
                zone_data        = zone_data.sort_values("time")
                time_diffs       = zone_data["time"].diff().dt.days.dropna()
                days_since       = float(time_diffs.median()) if not time_diffs.empty else 999.0
            else:
                days_since = 999.0

            # ── 3. ML prediction ──────────────────────────────────
            X_input = [[
                lat, lon, depth,
                shallow, deep,
                count, mean_mag, max_mag, std_mag,
                days_since,
            ]]

            prob_ml = float(
                warn_model.predict_proba(
                    warn_scaler.transform(X_input)
                )[0][1]
            )

            cat = cls_model.predict(
                cls_scaler.transform([[lat, lon, depth]])
            )[0]

            # ── 4. Dynamic blended final score ────────────────────
            prob_final, w_ml, w_hist = blend_scores(
                prob_ml, hist_score, nearby_count
            )

            # ── Nearby events dataframe ───────────────────────────
            nearby = df_model[
                (df_model["lat"].between(lat - 2, lat + 2)) &
                (df_model["lon"].between(lon - 2, lon + 2))
            ]

        # ── Metrics row ───────────────────────────────────────────
        st.markdown("---")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("🎯 Final Risk",          f"{prob_final * 100:.1f}%")
        m2.metric("🤖 ML Probability",      f"{prob_ml    * 100:.1f}%")
        m3.metric("📜 Historical Score",    f"{hist_score:.1f}%")
        m4.metric("📂 Category",            cat)
        m5.metric("📍 Nearby Events (±2°)", nearby_count)

        # ── Weight banner ─────────────────────────────────────────
        st.info(
            f"⚖ Blending weights — "
            f"ML: **{w_ml*100:.0f}%**  |  "
            f"Historical: **{w_hist*100:.0f}%**  |  "
            f"Based on **{nearby_count}** nearby events"
        )

        st.markdown("---")

        # ── Gauge + Zone summary ──────────────────────────────────
        g1, g2 = st.columns([2, 1])

        with g1:
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(prob_final * 100, 1),
                title={"text": "Seismic Risk %",
                       "font": {"color": "#c8d0e8"}},
                gauge={
                    "axis": {"range": [0, 100],
                             "tickcolor": "#7a8299"},
                    "bar":  {"color": "#4a90e2"},
                    "steps": [
                        {"range": [0,   35], "color": "#0e2a1a"},
                        {"range": [35,  65], "color": "#2a1e0a"},
                        {"range": [65, 100], "color": "#2a0a0a"},
                    ],
                    "threshold": {
                        "line":  {"color": "#e25c4a", "width": 3},
                        "value": prob_final * 100,
                    },
                },
                number={"suffix": "%",
                        "font": {"color": "#e8eaf0"}},
            ))
            fig_g.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#c8d0e8",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_g, use_container_width=True)

        with g2:
            st.markdown("<br><br>", unsafe_allow_html=True)

            if prob_final > 0.65:
                st.error(
                    "⚠ HIGH RISK\n\n"
                    "Significant seismic activity likely in this region."
                )
            elif prob_final > 0.35:
                st.warning(
                    "⚡ MODERATE RISK\n\n"
                    "Some seismic activity historically present."
                )
            else:
                st.success(
                    "✔ LOW RISK\n\n"
                    "Limited seismic activity in this region."
                )

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**Zone Summary**")
            st.markdown(f"- Events in zone  : **{int(count)}**")
            st.markdown(f"- Avg magnitude   : **{mean_mag:.2f}**")
            st.markdown(f"- Max magnitude   : **{max_mag:.2f}**")
            st.markdown(f"- Std magnitude   : **{std_mag:.2f}**")
            st.markdown(f"- Days since last : **{days_since:.0f}**")
            st.markdown(
                f"- Depth type      : **"
                f"{'Shallow' if shallow else 'Deep' if deep else 'Intermediate'}"
                f"**"
            )

        st.markdown("---")

        # ── Feature importance + Donut ────────────────────────────
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("📊 Feature Importance")
            importances = warn_model.feature_importances_
            sorted_idx  = np.argsort(importances)
            fig_fi = go.Figure(go.Bar(
                x=importances[sorted_idx],
                y=[feat_names[i] for i in sorted_idx],
                orientation="h",
                marker=dict(
                    color=importances[sorted_idx],
                    colorscale="Blues",
                    showscale=False,
                ),
            ))
            fig_fi.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor ="rgba(0,0,0,0)",
                font_color   ="#c8d0e8",
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(title="Importance", gridcolor="#2a2f45"),
                yaxis=dict(gridcolor="#2a2f45"),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

        with c2:
            st.subheader("🥧 Score Breakdown")
            fig_pie = go.Figure(go.Pie(
                labels=[
                    f"ML Model ({w_ml*100:.0f}%)",
                    f"Historical ({w_hist*100:.0f}%)",
                ],
                values=[
                    prob_ml    * w_ml   * 100,
                    hist_score * w_hist,
                ],
                hole=0.55,
                marker=dict(colors=["#4a90e2", "#e2a94a"]),
                textinfo="label+percent",
                textfont=dict(color="#c8d0e8"),
            ))
            fig_pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color   ="#c8d0e8",
                height=350,
                margin=dict(l=10, r=10, t=10, b=10),
                showlegend=False,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("---")

        # ── Nearby historical events ──────────────────────────────
        st.subheader("🗂 Nearby Historical Events (±2°)")

        if nearby.empty:
            st.info("No historical events found within ±2° of this location.")
        else:
            nearby_display = (
                nearby[["time", "magnitude", "depth_km", "place", "mag_category"]]
                .sort_values("magnitude", ascending=False)
                .head(20)
                .reset_index(drop=True)
            )
            st.dataframe(nearby_display, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            h1, h2 = st.columns(2)

            with h1:
                st.subheader("📈 Magnitude Distribution")
                fig_mhist = go.Figure(go.Histogram(
                    x=nearby["magnitude"],
                    nbinsx=20,
                    marker_color="#4a90e2",
                    opacity=0.85,
                ))
                fig_mhist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor ="rgba(0,0,0,0)",
                    font_color   ="#c8d0e8",
                    height=300,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(title="Magnitude",   gridcolor="#2a2f45"),
                    yaxis=dict(title="Event Count", gridcolor="#2a2f45"),
                )
                st.plotly_chart(fig_mhist, use_container_width=True)

            with h2:
                st.subheader("📉 Depth Distribution")
                fig_dhist = go.Figure(go.Histogram(
                    x=nearby["depth_km"],
                    nbinsx=20,
                    marker_color="#e2a94a",
                    opacity=0.85,
                ))
                fig_dhist.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor ="rgba(0,0,0,0)",
                    font_color   ="#c8d0e8",
                    height=300,
                    margin=dict(l=10, r=10, t=10, b=10),
                    xaxis=dict(title="Depth (km)",  gridcolor="#2a2f45"),
                    yaxis=dict(title="Event Count", gridcolor="#2a2f45"),
                )
                st.plotly_chart(fig_dhist, use_container_width=True)

        st.markdown("---")

        # ── Debug expander ────────────────────────────────────────
        with st.expander("🔍 Full Debug Info"):
            d1, d2 = st.columns(2)
            with d1:
                st.markdown("**Historical Analysis by Radius**")
                st.json(hist_details)
            with d2:
                st.markdown("**Zone Stats + Weights Used**")
                st.write({
                    "zone_event_count"  : int(count),
                    "zone_mean_mag"     : round(mean_mag,   2),
                    "zone_max_mag"      : round(max_mag,    2),
                    "zone_std_mag"      : round(std_mag,    2),
                    "days_since_last"   : round(days_since, 1),
                    "shallow_quake"     : bool(shallow),
                    "deep_quake"        : bool(deep),
                    "lat_bin"           : lat_bin,
                    "lon_bin"           : lon_bin,
                    "nearby_count_2deg" : nearby_count,
                    "weight_ml"         : f"{w_ml    * 100:.0f}%",
                    "weight_historical" : f"{w_hist  * 100:.0f}%",
                    "raw_ml_prob"       : f"{prob_ml  * 100:.1f}%",
                    "hist_score"        : f"{hist_score:.1f}%",
                    "final_blended"     : f"{prob_final * 100:.1f}%",
                })


# ═════════════════════════════════════════════════════════════════
# TAB 4 — FORECAST
# ═════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Earthquake Forecast")

    months = st.slider("Months to forecast", 6, 24, 12)
    st.pyplot(forecast_frequency(df_fc, periods=months))

    years = st.slider("Years trend", 2, 10, 5)
    st.pyplot(forecast_magnitude_trend(df_fc, forecast_years=years))

    if st.button("Generate Risk Heatmap"):
        st_folium(
            forecast_risk_heatmap(df_fc),
            height=500,
            returned_objects=[],
            use_container_width=True,
        )

# ═════════════════════════════════════════════════════════════════
# TAB 5 — AFTERSHOCK
# ═════════════════════════════════════════════════════════════════
from core.data_loader import load_all_data
from forecast.aftershock import compute_aftershock
import folium
import datetime as _dt

# Indian Standard Time = UTC+5:30
IST = _dt.timezone(_dt.timedelta(hours=5, minutes=30))

def to_ist(t):
    """Convert a datetime / ISO string to a readable IST string."""
    if isinstance(t, str):
        t = pd.to_datetime(t, utc=True)
    elif isinstance(t, pd.Timestamp):
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
    t_ist = t.astimezone(IST)
    return t_ist.strftime("%d %b %Y  %I:%M:%S %p IST")

with tab5:
    st.subheader("🌋 Aftershock Analysis (Real-Time)")
    combined   = load_all_data(df)
    aftershock = compute_aftershock(combined)

    if aftershock:
        col1, col2 = st.columns(2)
        col1.metric("Last Major Magnitude", aftershock["mag"])
        col2.metric("Days Since Event",     aftershock["days_since"])

        st.write("📍 Location:", aftershock["lat"], aftershock["lon"])
        st.write("🕒 Time (IST):", to_ist(aftershock["time"]))

        # Map
        st.subheader("Mainshock location")
        m = folium.Map(
            location=[aftershock["lat"], aftershock["lon"]],
            zoom_start=6,
            tiles="CartoDB dark_matter",
            prefer_canvas=True,          # ← faster canvas renderer
        )
        folium.Marker(
            location=[aftershock["lat"], aftershock["lon"]],
            popup=f"Mainshock  Mag: {aftershock['mag']}",
            icon=folium.Icon(color="red", icon="warning-sign"),
        ).add_to(m)
        folium.Circle(
            location=[aftershock["lat"], aftershock["lon"]],
            radius=200_000,
            color="#e25c4a",
            fill=True,
            fill_opacity=0.15,
        ).add_to(m)
        st_folium(
            m,
            height=420,
            returned_objects=[],         # ← no state ping-pong
            use_container_width=True,
        )

        # Aftershock probability curve
        st.subheader("Aftershock probability curve")
        prob_df = pd.DataFrame(aftershock["curve"])
        prob_df["prob"] = prob_df["prob"] * 100

        fig_as = px.area(
            prob_df, x="day", y="prob",
            labels={"day": "Day after mainshock", "prob": "Probability (%)"},
            color_discrete_sequence=["#e25c4a"],
        )
        fig_as.update_layout(**DARK_LAYOUT)
        st.plotly_chart(fig_as, use_container_width=True)

    else:
        st.info("No recent major earthquake detected.")