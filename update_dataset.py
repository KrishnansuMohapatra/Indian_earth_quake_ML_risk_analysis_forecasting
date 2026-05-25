import os
import requests
import pandas as pd
from datetime import datetime

CSV_PATH = "earthquake_data.csv"

def fetch_new_earthquakes(start_time_str):
    """
    Fetches earthquake data from the USGS API.
    Filters for the bounding box of India and surrounding regions.
    """
    print(f"Fetching new earthquakes from USGS since {start_time_str}...")
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    
    # Bounding box for India and surrounding region
    params = {
        "format": "geojson",
        "starttime": start_time_str,
        "minlatitude": 5.0,
        "maxlatitude": 39.0,
        "minlongitude": 67.0,
        "maxlongitude": 99.0,
        "minmagnitude": 3.0,
        "eventtype": "earthquake"
    }
    
    try:
        response = requests.get(url, params=params, timeout=20)
        if response.status_code != 200:
            print(f"Error fetching from USGS API: HTTP {response.status_code}")
            return pd.DataFrame()
            
        data = response.json()
    except Exception as e:
        print(f"Failed to query USGS API: {e}")
        return pd.DataFrame()
        
    features = data.get("features", [])
    print(f"Total features returned from API: {len(features)}")
    
    new_rows = []
    for f in features:
        props = f["properties"]
        geom = f["geometry"]
        place = props.get("place", "")
        
        # Check if the earthquake is associated with India (to match existing data style)
        if not place or "india" not in place.lower():
            continue
            
        coords = geom.get("coordinates", [0, 0, 0])
        lon, lat, depth = coords[0], coords[1], coords[2]
        
        # Format time to ISO 8601 UTC string to match CSV format
        # props["time"] is in milliseconds since epoch
        dt = pd.to_datetime(props["time"], unit="ms", utc=True)
        time_iso = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        
        new_rows.append({
            "time": time_iso,
            "lat": lat,
            "lon": lon,
            "depth_km": depth,
            "magnitude": props.get("mag"),
            "place": place,
            "Unnamed: 6": place  # To match the 7-column schema of the existing CSV
        })
        
    df_new = pd.DataFrame(new_rows)
    print(f"Filtered to {len(df_new)} India-related earthquakes.")
    return df_new

def main():
    # 1. Load existing data
    if os.path.exists(CSV_PATH):
        try:
            df_existing = pd.read_csv(CSV_PATH)
            print(f"Loaded existing dataset with {len(df_existing)} records.")
            
            # Find the latest time to start fetching from
            # Convert to datetime to find the max
            temp_time = pd.to_datetime(df_existing["time"], errors="coerce")
            max_idx = temp_time.idxmax()
            latest_time = df_existing.loc[max_idx, "time"] if not pd.isna(max_idx) else None
            
            if latest_time:
                # Parse and set start time slightly before the latest timestamp to prevent gaps
                dt_latest = pd.to_datetime(latest_time)
                # Convert back to string format required by USGS starttime parameter (YYYY-MM-DDTHH:MM:SS)
                start_time_str = dt_latest.strftime("%Y-%m-%dT%H:%M:%S")
            else:
                start_time_str = "2001-01-01T00:00:00"
        except Exception as e:
            print(f"Error reading existing CSV: {e}. Starting fresh.")
            df_existing = pd.DataFrame()
            start_time_str = "2001-01-01T00:00:00"
    else:
        print("CSV file not found. A new one will be created.")
        df_existing = pd.DataFrame()
        start_time_str = "2001-01-01T00:00:00"
        
    # 2. Fetch new earthquakes
    df_new = fetch_new_earthquakes(start_time_str)
    
    if df_new.empty:
        print("No new earthquakes found. Dataset is already up to date.")
        return
        
    # 3. Combine datasets
    if not df_existing.empty:
        df_combined = pd.concat([df_new, df_existing], ignore_index=True)
    else:
        df_combined = df_new
        
    # 4. Deduplicate based on time, lat, lon
    # Round lat and lon slightly to handle minor precision differences in API reporting
    df_combined["lat_round"] = df_combined["lat"].round(4)
    df_combined["lon_round"] = df_combined["lon"].round(4)
    
    initial_len = len(df_combined)
    df_combined = df_combined.drop_duplicates(subset=["time", "lat_round", "lon_round"])
    df_combined = df_combined.drop(columns=["lat_round", "lon_round"])
    
    new_added = len(df_combined) - len(df_existing)
    print(f"Removed {initial_len - len(df_combined)} duplicate records.")
    
    # 5. Sort by time descending (newest first)
    df_combined["time_dt"] = pd.to_datetime(df_combined["time"])
    df_combined = df_combined.sort_values(by="time_dt", ascending=False).reset_index(drop=True)
    df_combined = df_combined.drop(columns=["time_dt"])
    
    # Ensure columns order matches original
    cols = ["time", "lat", "lon", "depth_km", "magnitude", "place", "Unnamed: 6"]
    df_combined = df_combined[cols]
    
    # 6. Save back to CSV
    df_combined.to_csv(CSV_PATH, index=False)
    print(f"Successfully updated '{CSV_PATH}'!")
    print(f"Added {new_added} new earthquake(s). Total records now: {len(df_combined)}")
    print(f"Latest earthquake date in dataset: {df_combined.loc[0, 'time']}")

if __name__ == "__main__":
    main()
