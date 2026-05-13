import matplotlib.pyplot as plt
import folium
import pandas as pd
from folium.plugins import HeatMap
import requests


def magnitude_distribution(df):
    """Returns a matplotlib figure of magnitude category distribution."""
    fig, ax = plt.subplots()
    df["mag_category"].value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Magnitude Distribution")
    ax.set_xlabel("Category")
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig


def historical_heatmap(df):
    """Returns a folium heatmap of historical earthquake locations."""
    heat_data = df[["lat", "lon"]].values.tolist()
    m = folium.Map(location=[22.5, 80], zoom_start=5)
    HeatMap(heat_data).add_to(m)
    return m


def satellite_map(df):
    """Returns a folium satellite map with earthquake markers."""
    m = folium.Map(
        location=[22.5, 80],
        zoom_start=5,
        tiles="Esri.WorldImagery",
        attr="Esri"
    )
    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=row["magnitude"] * 2,
            color="red",
            fill=True
        ).add_to(m)
    return m


def depth_visualization(df):
    """Returns a folium map with depth-colored earthquake markers.

    Color coding:
        green  -> shallow  (depth < 70 km)
        orange -> mid      (70 <= depth < 300 km)
        red    -> deep     (depth >= 300 km)
    """
    m = folium.Map(location=[22.5, 80], zoom_start=5)
    for _, row in df.iterrows():
        depth = row["depth_km"]
        if depth < 70:
            color = "green"
        elif depth < 300:
            color = "orange"
        else:
            color = "red"

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=row["magnitude"] * 2,
            color=color,
            fill=True
        ).add_to(m)
    return m


def timeline_map(df, selected_year):
    """Returns a folium map filtered up to the selected year.

    Args:
        df: full dataframe
        selected_year: upper bound year (int)
    """
    dff = df[df["year"] <= selected_year]
    m = folium.Map(location=[22.5, 80], zoom_start=5)
    for _, row in dff.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=row["magnitude"] * 2,
            color="red",
            fill=True
        ).add_to(m)
    return m


def seismic_zone_map():
    """Returns a folium map of approximate India seismic zones."""
    m = folium.Map(location=[22.5, 80], zoom_start=5)
    zones = [
        {"location": [27, 92], "radius": 700000, "color": "red", "label": "Zone V"},
        {"location": [32, 78], "radius": 900000, "color": "orange", "label": "Zone IV"},
        {"location": [22, 85], "radius": 1200000, "color": "yellow", "label": "Zone III"},
        {"location": [18, 75], "radius": 1500000, "color": "green", "label": "Zone II"},
    ]
    for zone in zones:
        folium.Circle(
            location=zone["location"],
            radius=zone["radius"],
            color=zone["color"],
            fill=True,
            tooltip=zone["label"]
        ).add_to(m)
    return m


import folium
import requests


def realtime_earthquake_map():

    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"

    response = requests.get(url)
    data = response.json()

    m = folium.Map(location=[22.5, 80], zoom_start=2)

    for feature in data["features"]:

        coords = feature["geometry"]["coordinates"]
        mag = feature["properties"]["mag"]
        place = feature["properties"]["place"]

        if mag is None:
            continue

        folium.CircleMarker(
            location=[coords[1], coords[0]],
            radius=mag * 2,
            color="red",
            fill=True,
            fill_opacity=0.7,
            tooltip=f"{place} | Mag: {mag}"
        ).add_to(m)

    return m