import requests
import pandas as pd
from sqlalchemy import create_engine
import schedule
import time
from datetime import datetime

def extract_weather_data(latitude=40.7128, longitude=-74.0060):
    """Fetch hourly weather data from Open-Meteo API for given coordinates."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "timezone": "auto"
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def transform_weather_data(data):
    """Transform raw JSON into a clean DataFrame."""
    df = pd.DataFrame({
        "time": data["hourly"]["time"],
        "temperature_C": data["hourly"]["temperature_2m"],
        "precipitation_mm": data["hourly"]["precipitation"],
        "wind_speed_mps": data["hourly"]["wind_speed_10m"]
    })
    df["time"] = pd.to_datetime(df["time"])
    df["ingested_at"] = datetime.utcnow()
    return df

def load_to_csv(df, filename="weather_data.csv"):
    """Save data to CSV (append mode, prevent duplicates)."""
    try:
        existing = pd.read_csv(filename)
        df = pd.concat([existing, df]).drop_duplicates(subset=["time"])
    except FileNotFoundError:
        pass
    df.to_csv(filename, index=False)
    print(f"[CSV] Data saved -> {filename}")

def load_to_db(df):
    """Load data into local SQLite database."""
    engine = create_engine("sqlite:///weather_data.db")
    df.to_sql("weather", engine, if_exists="append", index=False)
    print(f"[DB] Data inserted into weather_data.db")

def run_pipeline():
    print("\n🚀 Running Weather ETL pipeline...")
    raw = extract_weather_data()
    clean = transform_weather_data(raw)
    load_to_csv(clean)
    load_to_db(clean)
    print("✅ Pipeline completed.\n")

# Schedule every 6 hours
schedule.every(6).hours.do(run_pipeline)

# Run once immediately
run_pipeline()

# Keep the scheduler alive
while True:
    schedule.run_pending()
    time.sleep(60)