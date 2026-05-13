import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import folium
from prophet import Prophet
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


def prepare_data(df):
    """Prepares dataframe with required columns."""
    df = df.copy()
    df = df.drop(columns=[c for c in df.columns if "Unnamed" in str(c)], errors="ignore")
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["mag_category"] = pd.cut(
        df["magnitude"],
        bins=[0, 3, 4, 5, 10],
        labels=["Low", "Moderate", "High", "Severe"]
    )
    return df


# ─────────────────────────────────────────────
# 1. MONTHLY FREQUENCY FORECAST
# ─────────────────────────────────────────────
def forecast_frequency(df, periods=12):
    """
    Forecasts monthly earthquake frequency using Prophet.
    Returns a matplotlib figure showing historical + forecast.
    Args:
        df      : prepared dataframe
        periods : number of months to forecast ahead (default 12)
    """
    # Aggregate by month
    monthly = (
        df.groupby(df["time"].dt.to_period("M"))
        .size()
        .reset_index(name="y")
    )
    monthly["ds"] = monthly["time"].dt.to_timestamp()
    monthly = monthly[["ds", "y"]]

    # Train Prophet
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.1
    )
    model.fit(monthly)

    # Forecast
    future = model.make_future_dataframe(periods=periods, freq="MS")
    forecast = model.predict(future)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    # Historical
    ax.plot(monthly["ds"], monthly["y"],
            color="#00d4ff", linewidth=1.8, label="Historical", zorder=3)
    ax.scatter(monthly["ds"], monthly["y"],
               color="#00d4ff", s=18, zorder=4)

    # Forecast line
    future_fc = forecast[forecast["ds"] > monthly["ds"].max()]
    ax.plot(future_fc["ds"], future_fc["yhat"],
            color="#ff6b35", linewidth=2.2, linestyle="--", label="Forecast", zorder=3)

    # Confidence band
    ax.fill_between(future_fc["ds"],
                    future_fc["yhat_lower"].clip(0),
                    future_fc["yhat_upper"],
                    color="#ff6b35", alpha=0.2, label="95% Confidence")

    # Divider
    ax.axvline(monthly["ds"].max(), color="white", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(monthly["ds"].max(), ax.get_ylim()[1] * 0.95,
            " Forecast →", color="white", fontsize=9, alpha=0.7)

    ax.set_title("Monthly Earthquake Frequency Forecast (India)",
                 color="white", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Date", color="#aaaaaa", fontsize=10)
    ax.set_ylabel("Earthquake Count", color="#aaaaaa", fontsize=10)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333355",
              labelcolor="white", fontsize=9)
    ax.grid(axis="y", color="#333355", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 2. MAGNITUDE TREND FORECAST
# ─────────────────────────────────────────────
def forecast_magnitude_trend(df, forecast_years=5):
    """
    Plots average annual magnitude with polynomial trend + future projection.
    Args:
        df             : prepared dataframe
        forecast_years : how many years ahead to project (default 5)
    """
    yearly = df.groupby("year")["magnitude"].agg(["mean", "std", "count"]).reset_index()
    yearly.columns = ["year", "mean_mag", "std_mag", "count"]
    yearly["std_mag"] = yearly["std_mag"].fillna(0)

    x = yearly["year"].values
    y = yearly["mean_mag"].values

    # Polynomial fit (degree 2)
    coeffs = np.polyfit(x, y, 2)
    poly = np.poly1d(coeffs)

    future_years = np.arange(x.min(), x.max() + forecast_years + 1)
    trend_line = poly(future_years)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(13, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    # Error band (±std)
    ax.fill_between(x,
                    y - yearly["std_mag"].values,
                    y + yearly["std_mag"].values,
                    color="#a855f7", alpha=0.15, label="±1 Std Dev")

    # Actual points + line
    ax.plot(x, y, color="#a855f7", linewidth=1.8, label="Avg Annual Magnitude", zorder=3)
    ax.scatter(x, y, color="#a855f7", s=40, zorder=4)

    # Trend + forecast
    hist_mask = future_years <= x.max()
    fore_mask = future_years >= x.max()
    ax.plot(future_years[hist_mask], trend_line[hist_mask],
            color="#ffd700", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.plot(future_years[fore_mask], trend_line[fore_mask],
            color="#ffd700", linewidth=2.2, linestyle="--", label="Trend Projection")

    # Forecast zone shading
    ax.axvspan(x.max(), future_years[-1], color="#ffd700", alpha=0.05)
    ax.axvline(x.max(), color="white", linestyle=":", alpha=0.5, linewidth=1)
    ax.text(x.max() + 0.2, y.min(), " Projected →", color="white", fontsize=9, alpha=0.7)

    # Annotate projected value
    projected_val = trend_line[-1]
    ax.annotate(f"~{projected_val:.2f} in {future_years[-1]}",
                xy=(future_years[-1], projected_val),
                xytext=(-60, 15), textcoords="offset points",
                color="#ffd700", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#ffd700", lw=1.2))

    ax.set_title("Annual Average Magnitude Trend & Projection",
                 color="white", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Year", color="#aaaaaa", fontsize=10)
    ax.set_ylabel("Average Magnitude", color="#aaaaaa", fontsize=10)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333355",
              labelcolor="white", fontsize=9)
    ax.grid(axis="y", color="#333355", linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 3. RISK ZONE HEATMAP (FOLIUM)
# ─────────────────────────────────────────────
def forecast_risk_heatmap(df, grid_resolution=40):
    """
    Trains RandomForest on historical data and predicts risk probability
    across a lat/lon grid over India, displayed as a folium heatmap.
    Args:
        df               : prepared dataframe
        grid_resolution  : grid density (higher = finer, slower)
    """
    df = df.copy()
    df["strong_eq"] = (df["magnitude"] >= 4.0).astype(int)

    X = df[["lat", "lon", "depth_km"]].values
    y = df["strong_eq"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_scaled, y)

    # Build India grid
    lat_range = np.linspace(6, 38, grid_resolution)
    lon_range = np.linspace(68, 98, grid_resolution)
    avg_depth = df["depth_km"].median()

    grid_points = []
    heat_data = []

    for lat in lat_range:
        for lon in lon_range:
            grid_points.append([lat, lon, avg_depth])

    grid_arr = np.array(grid_points)
    grid_scaled = scaler.transform(grid_arr)
    probs = model.predict_proba(grid_scaled)[:, 1]

    for i, (lat, lon, _) in enumerate(grid_points):
        if probs[i] > 0.3:  # only show meaningful risk
            heat_data.append([lat, lon, float(probs[i])])

    # Build folium map
    m = folium.Map(location=[22.5, 80], zoom_start=5, tiles="CartoDB dark_matter")

    # Risk heatmap layer
    from folium.plugins import HeatMap
    HeatMap(
        heat_data,
        min_opacity=0.3,
        max_zoom=10,
        radius=18,
        blur=15,
        gradient={0.3: "blue", 0.5: "lime", 0.7: "orange", 1.0: "red"}
    ).add_to(m)

    # Actual earthquake markers (recent 5 years)
    recent = df[df["year"] >= 2020]
    for _, row in recent.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=row["magnitude"] * 1.5,
            color="white",
            fill=True,
            fill_color="white",
            fill_opacity=0.6,
            tooltip=f"Mag: {row['magnitude']} | {str(row['time'])[:10]}"
        ).add_to(m)

    # Legend
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:#1a1a2e; padding:12px 16px; border-radius:8px;
                border:1px solid #444; color:white; font-family:Arial; font-size:12px;">
        <b>🔴 Earthquake Risk Forecast</b><br><br>
        <span style="color:#ff4444">■</span> High Risk<br>
        <span style="color:#ff8800">■</span> Medium Risk<br>
        <span style="color:#00ff00">■</span> Low Risk<br>
        <span style="color:#0000ff">■</span> Minimal Risk<br><br>
        <span style="color:#ffffff">● Recent Earthquakes (2020+)</span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# ─────────────────────────────────────────────
# MAIN — run all three and save outputs
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv("earthquake_data.csv")
    df = prepare_data(df)

    print("Generating frequency forecast...")
    fig1 = forecast_frequency(df, periods=12)
    fig1.savefig("forecast_frequency.png", dpi=150, bbox_inches="tight")
    print("  ✓ Saved forecast_frequency.png")

    print("Generating magnitude trend...")
    fig2 = forecast_magnitude_trend(df, forecast_years=5)
    fig2.savefig("forecast_magnitude_trend.png", dpi=150, bbox_inches="tight")
    print("  ✓ Saved forecast_magnitude_trend.png")

    print("Generating risk heatmap (this takes ~20s)...")
    m = forecast_risk_heatmap(df, grid_resolution=40)
    m.save("forecast_risk_heatmap.html")
    print("  ✓ Saved forecast_risk_heatmap.html")

    print("\nAll forecast visualizations saved successfully!")