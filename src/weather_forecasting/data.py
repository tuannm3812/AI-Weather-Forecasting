from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from weather_forecasting.config import (
    CURRENT_WEATHER_FILE,
    LATITUDE,
    LONGITUDE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    START_DATE,
    SYDNEY_TZ,
    TIMEZONE,
)


DAILY_WEATHER_VARIABLES = [
    "precipitation_sum",
    "rain_sum",
    "precipitation_hours",
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "vapour_pressure_deficit_max",
    "cloudcover_mean",
    "shortwave_radiation_sum",
    "sunshine_duration",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "pressure_msl_mean",
    "soil_moisture_0_to_7cm_mean",
    "soil_moisture_7_to_28cm_mean",
    "weathercode",
]

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def latest_complete_date() -> str:
    """Return yesterday in the project timezone as the latest stable daily observation."""
    return (datetime.now(SYDNEY_TZ).date() - timedelta(days=1)).isoformat()


def fetch_open_meteo_daily(start_date: str = START_DATE, end_date: str | None = None) -> pd.DataFrame:
    """Fetch daily weather history for Sydney from Open-Meteo Archive API."""
    end_date = end_date or latest_complete_date()
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "timezone": TIMEZONE,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_WEATHER_VARIABLES),
    }
    response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=90)
    response.raise_for_status()
    payload = response.json()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DATA_DIR / f"open_meteo_daily_{start_date}_{end_date}.json"
    raw_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    daily = pd.DataFrame(payload["daily"]).rename(columns={"time": "date"})
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    return daily


def save_current_weather(df: pd.DataFrame, processed_dir: Path = PROCESSED_DATA_DIR) -> Path:
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / CURRENT_WEATHER_FILE
    df.to_csv(out_path, index=False)
    return out_path


def refresh_current_weather(start_date: str = START_DATE, end_date: str | None = None) -> pd.DataFrame:
    df = fetch_open_meteo_daily(start_date=start_date, end_date=end_date)
    save_current_weather(df)
    return df


def load_current_weather(path: Path = PROCESSED_DATA_DIR / CURRENT_WEATHER_FILE) -> pd.DataFrame:
    if not path.exists():
        return refresh_current_weather()
    return pd.read_csv(path, parse_dates=["date"]).dropna(how="any")


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path)


def load_table(path: str) -> pd.DataFrame:
    """Load a CSV or Parquet table based on file extension."""
    if str(path).endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame, target_col: str):
    """Split dataframe into features and target."""
    x = df.drop(columns=[target_col])
    y = df[target_col]
    return x, y


def train_valid_split(x, y, test_size: float = 0.2, random_state: int = 42):
    """Train/validation split wrapper."""
    return train_test_split(x, y, test_size=test_size, random_state=random_state)
