# 🌍 SeismoIQ — Earthquake Intelligence System

SeismoIQ is a comprehensive, interactive data science dashboard built with Python and Streamlit. It leverages Machine Learning, historical data, and real-time feeds to analyze, predict, and forecast seismic activities in and around the Indian subcontinent.

## 🌟 Features

The application is split into 5 core modules:

1. **📊 Overview**
   Provides high-level statistics and data visualization of historical earthquakes. It includes:
   - Magnitude and depth distribution histograms.
   - Time-series area charts for events per year.
   - Gutenberg–Richter cumulative frequency analysis.

2. **🗺 Maps**
   Interactive geographical visualizations using Folium:
   - **Satellite Map:** Plots earthquakes over satellite imagery.
   - **Depth Visualization:** Color-coded markers based on earthquake depth (Shallow, Mid, Deep).
   - **Timeline Map:** A slider to view earthquakes up to a selected year.
   - **Seismic Zones:** Outlines the approximate seismic risk zones across India (Zone II to Zone V).

3. **🤖 ML Prediction**
   A Machine Learning-driven Seismic Risk Estimator:
   - Input custom coordinates (Latitude, Longitude) and Depth.
   - Uses a combination of a **Gradient Boosting Classifier** and localized historical scoring to predict the percentage risk of a significant earthquake.
   - Dynamically blends the ML score with the historical score based on data density.
   - Displays feature importance and retrieves nearby historical events.

4. **📈 Forecast**
   Predictive modeling of future seismic activity:
   - **Frequency Forecast:** Uses **Prophet** to predict the monthly frequency of earthquakes over the next 6-24 months.
   - **Magnitude Trend:** Polynomial regression forecasting the average annual magnitude trend.
   - **Risk Heatmap:** Trains a Random Forest on grid coordinates to map the highest risk areas.

5. **🌋 Aftershock (Real-Time)**
   Real-time aftershock analysis using Omori's Law:
   - Fetches the latest earthquake data using the **USGS API**.
   - Identifies the most recent major earthquake (Magnitude ≥ 5).
   - Calculates the decaying probability of aftershocks over the coming days and displays it on an interactive map and probability curve.

## 🛠️ Architecture

*   **`app2.py`**: The main Streamlit application script containing the UI layout and tab routing.
*   **`models.py`**: Contains the Machine Learning logic (Random Forest, Gradient Boosting) for classifying magnitude and evaluating risk.
*   **`data_processing.py`**: Cleans and prepares `earthquake_data.csv`, adding engineered features like Omori decay rates and Gutenberg-Richter bins.
*   **`visualization.py` & `forecast_visualization.py`**: Handles the generation of Plotly charts, Matplotlib graphs, Folium maps, and Prophet forecasting.
*   **`forecast/` & `core/`**: Utility modules that handle real-time API fetching from USGS and the mathematical implementation of aftershock probabilities.

## 🚀 How to Run the Application

### Prerequisites
Make sure you have Python installed (Python 3.8+ recommended). 

### 1. Install Dependencies
Open your terminal/command prompt, navigate to the project directory (`Indian_earth_quake`), and run the following command to install all required libraries:

```bash
pip install -r requirements.txt
```

### 2. Run the App
Launch the Streamlit dashboard by executing:

```bash
streamlit run app2.py
```

### 3. View the App
Once the command executes, a new browser tab should open automatically pointing to `http://localhost:8501`. If it doesn't, you can manually copy and paste that URL into your web browser.

## 📂 Data Sources
*   **Historical Data:** `earthquake_data.csv` (Must remain in the root directory for the app to function).
*   **Real-time Data:** Fetched dynamically via the public **USGS API**.
