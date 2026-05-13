import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_usgs(days=30):

    end = datetime.utcnow()
    start = end - timedelta(days=days)

    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    params = {
        "format": "geojson",
        "starttime": start.strftime("%Y-%m-%d"),
        "endtime": end.strftime("%Y-%m-%d"),
        "minmagnitude": 2.5
    }

    try:
        res = requests.get(url, params=params, timeout=10)

        if res.status_code != 200:
            print("API Error:", res.status_code)
            return pd.DataFrame()

        try:
            data = res.json()
        except Exception:
            print("Invalid JSON")
            return pd.DataFrame()

        rows = []
        for f in data.get("features", []):
            coords = f["geometry"]["coordinates"]
            props = f["properties"]

            rows.append({
                "time": pd.to_datetime(props["time"], unit="ms"),
                "lat": coords[1],
                "lon": coords[0],
                "depth_km": coords[2],
                "magnitude": props["mag"]
            })

        return pd.DataFrame(rows)

    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        return pd.DataFrame()