import pandas as pd
import numpy as np

def mag_level(m):
    if m < 4:
        return "Minor"
    elif m < 5:
        return "Light"
    elif m < 6:
        return "Moderate"
    else:
        return "Strong"

# Omori's Law
def omori_law(t, K=50, c=1, p=1.2):
    return K / ((c + t) ** p)

def load_data():
    df = pd.read_csv("earthquake_data.csv")

    df = df.dropna(subset=["lat", "lon", "depth_km", "magnitude"])

    df["mag_category"] = df["magnitude"].apply(mag_level)

    df["time"] = pd.to_datetime(df['time'], errors='coerce')
    df["year"] = df["time"].dt.year

    df = df.sort_values("time")
    df["days_since_start"] = (df["time"] - df["time"].min()).dt.days

    # Omori feature
    df["aftershock_rate"] = df["days_since_start"].apply(omori_law)

    # Gutenberg-Richter feature
    df["magnitude_bin"] = df["magnitude"].round(1)
    freq = df["magnitude_bin"].value_counts().sort_index(ascending=False)
    cum_freq = freq.cumsum()

    gr_df = pd.DataFrame({
        "magnitude_bin": cum_freq.index,
        "cum_freq": cum_freq.values
    })

    gr_df["logN"] = np.log10(gr_df["cum_freq"])

    df = df.merge(gr_df, on="magnitude_bin", how="left")

    return df
df = load_data()
print(df.columns.tolist())
print(df.head(2))