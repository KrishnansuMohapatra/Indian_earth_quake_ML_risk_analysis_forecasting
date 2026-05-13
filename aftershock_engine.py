import pandas as pd
import numpy as np
from datetime import datetime

def omori_aftershock_prob(main_mag, days_since, k=0.05, c=0.1, p=1.1):
    if days_since < 0:
        return 0.0
    rate = k * (main_mag ** 2) / ((days_since + c) ** p)
    return float(1 - np.exp(-rate))


def generate_forecast(df, horizon=7):

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # Find last major earthquake
    major = df[df["magnitude"] >= 5].sort_values("time")

    if major.empty:
        return {"aftershock": None}

    last = major.iloc[-1]

    days_since = (datetime.utcnow() - last["time"]).days
    main_mag = last["magnitude"]

    daily_probs = []
    for d in range(1, horizon + 1):
        prob = omori_aftershock_prob(main_mag, days_since + d)
        daily_probs.append({"day": d, "prob": prob})

    return {
        "aftershock": {
            "last_major_mag": main_mag,
            "days_since": days_since,
            "daily_probs": daily_probs
        }
    }