import pandas as pd
from forecast.realtime import fetch_usgs

def load_all_data(csv_df):

    df = csv_df.copy()

    required = ["time", "lat", "lon", "depth_km", "magnitude"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    # normalize time
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    df["time"] = df["time"].dt.tz_localize(None)

    df = df.dropna(subset=["time", "lat", "lon", "magnitude"])

    # fetch realtime
    try:
        rt = fetch_usgs()

        if rt is None or rt.empty:
            print("[INFO] Using only local data")
            return df

        rt["time"] = pd.to_datetime(rt["time"], errors="coerce", utc=True)
        rt["time"] = rt["time"].dt.tz_localize(None)

        rt = rt[required]

    except Exception as e:
        print("[WARNING] API failed:", e)
        return df

    combined = pd.concat([df, rt], ignore_index=True)
    combined = combined.drop_duplicates(subset=["time", "lat", "lon"])
    combined = combined.dropna(subset=["time", "lat", "lon", "magnitude"])

    return combined