import numpy as np
import pandas as pd
from datetime import datetime

def omori_prob(M, t, k=0.05, c=0.1, p=1.1):
    rate = k * (M ** 2) / ((t + c) ** p)
    return float(1 - np.exp(-rate))


def compute_aftershock(df, horizon=7):

    df = df.copy()

    # ✅ FIX timezone
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df["time"] = df["time"].dt.tz_localize(None)

    df = df.dropna(subset=["time", "magnitude", "lat", "lon"])

    # find last major earthquake
    main = df[df["magnitude"] >= 5].sort_values("time")

    if main.empty:
        return None

    last = main.iloc[-1]

    t0 = (datetime.utcnow() - last["time"]).days
    M = last["magnitude"]

    # compute curve
    curve = []
    for d in range(1, horizon + 1):
        curve.append({
            "day": d,
            "prob": omori_prob(M, t0 + d)
        })

    return {
        "mag": float(M),
        "days_since": int(t0),
        "lat": float(last["lat"]),
        "lon": float(last["lon"]),
        "time": str(last["time"]),
        "curve": curve
    }