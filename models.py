import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils import resample


# ─────────────────────────────────────────────────────────────────────
# 1. CLASSIFICATION MODEL
# ─────────────────────────────────────────────────────────────────────
def train_classification_model(df):
    df_clean = df[["lat", "lon", "depth_km", "mag_category"]].dropna()

    X = df_clean[["lat", "lon", "depth_km"]]
    y = df_clean["mag_category"]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # accuracy for display
    acc = model.score(X_test, y_test)
    print(f"[Classification] Test accuracy: {acc:.3f}")

    return model, scaler                                    # 2 values


# ─────────────────────────────────────────────────────────────────────
# 2. WARNING / RISK MODEL
# ─────────────────────────────────────────────────────────────────────
def train_warning_model(df):
    df_clean = df[["lat", "lon", "depth_km", "magnitude", "time"]].dropna().copy()

    # ── Sort by time ──────────────────────────────────────────────────
    df_clean["time"] = pd.to_datetime(df_clean["time"])
    df_clean = df_clean.sort_values("time").reset_index(drop=True)

    # ── Zone grid (2°) ────────────────────────────────────────────────
    df_clean["lat_bin"] = (df_clean["lat"] // 2) * 2
    df_clean["lon_bin"] = (df_clean["lon"] // 2) * 2

    zone_counts = (
        df_clean.groupby(["lat_bin", "lon_bin"])["magnitude"]
        .agg(count="count", mean_mag="mean", max_mag="max", std_mag="std")
        .reset_index()
    )

    df_clean = df_clean.merge(zone_counts, on=["lat_bin", "lon_bin"], how="left")
    df_clean["std_mag"] = df_clean["std_mag"].fillna(0)

    # ── Depth features ────────────────────────────────────────────────
    df_clean["shallow"] = (df_clean["depth_km"] <  70).astype(int)
    df_clean["deep"]    = (df_clean["depth_km"] > 300).astype(int)

    # ── Time feature: days since last quake in same zone ──────────────
    df_clean["days_since_last"] = (
        df_clean
        .groupby(["lat_bin", "lon_bin"])["time"]
        .diff()
        .dt.days
        .fillna(999)
        .clip(upper=999)
    )

    # ── Feature list ──────────────────────────────────────────────────
    features = [
        "lat", "lon", "depth_km",
        "shallow", "deep",
        "count", "mean_mag", "max_mag", "std_mag",
        "days_since_last",
    ]

    # ── Risk label: top 40% magnitude = risk ──────────────────────────
    threshold        = df_clean["magnitude"].quantile(0.60)
    df_clean["risk"] = (df_clean["magnitude"] >= threshold).astype(int)

    # ── Balance classes 50/50 ─────────────────────────────────────────
    majority    = df_clean[df_clean["risk"] == 0]
    minority    = df_clean[df_clean["risk"] == 1]
    minority_up = resample(
        minority,
        replace=True,
        n_samples=len(majority),
        random_state=42,
    )
    df_balanced = pd.concat([majority, minority_up]).sample(
        frac=1, random_state=42
    )

    X = df_balanced[features]
    y = df_balanced["risk"]

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
    )
    model.fit(X_train, y_train)

    # accuracy for display
    acc = model.score(X_test, y_test)
    print(f"[Warning] Test accuracy: {acc:.3f}")

    return model, scaler, threshold, features, zone_counts  # 5 values


# ─────────────────────────────────────────────────────────────────────
# 3. HISTORICAL RISK SCORE
# ─────────────────────────────────────────────────────────────────────
def get_seismic_zone_risk(lat, lon, df):
    """
    Pure historical risk score 0-100 based on nearby events.
    Returns score, full details dict, and nearby_count for
    dynamic weighting.
    """
    results = {}

    for radius in [1, 2, 5]:
        nearby = df[
            (df["lat"].between(lat - radius, lat + radius)) &
            (df["lon"].between(lon - radius, lon + radius))
        ]
        if not nearby.empty:
            results[f"count_{radius}deg"]    = int(len(nearby))
            results[f"mean_mag_{radius}deg"] = round(float(nearby["magnitude"].mean()), 2)
            results[f"max_mag_{radius}deg"]  = round(float(nearby["magnitude"].max()),  2)
            results[f"pct_major_{radius}deg"]= round(float(
                (nearby["magnitude"] >= 6.0).mean() * 100), 2)
        else:
            results[f"count_{radius}deg"]    = 0
            results[f"mean_mag_{radius}deg"] = 0
            results[f"max_mag_{radius}deg"]  = 0
            results[f"pct_major_{radius}deg"]= 0

    score = min(100, (
        results["count_2deg"]     * 0.5  +
        results["mean_mag_2deg"]  * 8.0  +
        results["max_mag_2deg"]   * 5.0  +
        results["pct_major_2deg"] * 0.8
    ))

    return score, results, results["count_2deg"]            # 3 values


# ─────────────────────────────────────────────────────────────────────
# 4. DYNAMIC WEIGHT BLENDING
# ─────────────────────────────────────────────────────────────────────
def blend_scores(prob_ml, hist_score, nearby_count):
    """
    Dynamically weight ML vs Historical score based on
    how much historical data is available near the point.

    nearby_count > 100  →  trust history more  (75/25)
    nearby_count 20-100 →  balanced            (60/40)
    nearby_count < 20   →  trust ML more       (30/70)
    """
    if nearby_count > 100:
        w_hist = 0.75
        w_ml   = 0.25
    elif nearby_count > 20:
        w_hist = 0.60
        w_ml   = 0.40
    else:
        w_hist = 0.30
        w_ml   = 0.70

    prob_final = w_ml * prob_ml + w_hist * (hist_score / 100)

    return round(float(prob_final), 4), w_ml, w_hist        # 3 values