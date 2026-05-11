import pandas as pd
import numpy as np

from weather_forecasting.config import CLASSIFICATION_TARGET, REGRESSION_TARGET


WEATHER_GROUP_MAP = {
    0: "clear",
    1: "clouds",
    2: "clouds",
    3: "clouds",
    51: "drizzle",
    53: "drizzle",
    55: "drizzle",
    61: "rain",
    63: "rain",
    65: "rain",
}

SEASON_COLUMNS = ["season_1", "season_2", "season_3", "season_4"]
WEATHER_COLUMNS = ["weather_clear", "weather_clouds", "weather_drizzle", "weather_rain"]

BASE_MODEL_FEATURES = [
    "precipitation_sum",
    "precipitation_hours",
    "temperature_2m_mean",
    "vapour_pressure_deficit_max",
    "cloudcover_mean",
    "shortwave_radiation_sum",
    "sunshine_duration",
    "wind_speed_10m_max",
    "wind_direction_10m_dominant",
    "pressure_msl_mean",
    "soil_moisture_0_to_7cm_mean",
    "soil_moisture_7_to_28cm_mean",
]

MODEL_FEATURES = [
    *BASE_MODEL_FEATURES,
    "precipitation_sum_lag1",
    "precipitation_sum_lag2",
    "precipitation_sum_lag3",
    "precipitation_hours_lag1",
    "precipitation_hours_lag2",
    "precipitation_hours_lag3",
    "temperature_2m_mean_lag1",
    "temperature_2m_mean_lag2",
    "temperature_2m_mean_lag3",
    "precipitation_sum_roll3_mean",
    "precipitation_sum_roll3_std",
    "precipitation_sum_roll7_mean",
    "precipitation_sum_roll7_std",
    "precipitation_hours_roll3_mean",
    "precipitation_hours_roll3_std",
    "precipitation_hours_roll7_mean",
    "precipitation_hours_roll7_std",
    "temperature_2m_mean_roll3_mean",
    "temperature_2m_mean_roll7_mean",
    "precip_wind_interaction",
    "rain_intensity",
    "sunshine_efficiency",
    "temp_vpd_interaction",
    "month_sin",
    "month_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    *SEASON_COLUMNS,
    *WEATHER_COLUMNS,
]


def add_datetime_parts(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """Add simple datetime-derived features."""
    out = df.copy()
    dt = pd.to_datetime(out[datetime_col])
    out["year"] = dt.dt.year
    out["month"] = dt.dt.month
    out["day"] = dt.dt.day
    out["hour"] = dt.dt.hour
    return out


def add_forecasting_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create the production feature set used by the saved weather models."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)

    out["month"] = out["date"].dt.month
    out["season"] = out["date"].dt.month.mod(12).floordiv(3).add(1)
    out["dayofweek"] = out["date"].dt.dayofweek
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    out["dayofweek_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7.0)
    out["dayofweek_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7.0)

    lag_cols = ["precipitation_sum", "precipitation_hours", "temperature_2m_mean"]
    for col in lag_cols:
        for lag in (1, 2, 3):
            out[f"{col}_lag{lag}"] = out[col].shift(lag)

    rolling_cols = ["precipitation_sum", "precipitation_hours"]
    for col in rolling_cols:
        for window in (3, 7):
            rolling = out[col].rolling(window=window, min_periods=1)
            out[f"{col}_roll{window}_mean"] = rolling.mean()
            out[f"{col}_roll{window}_std"] = rolling.std()

    for window in (3, 7):
        out[f"temperature_2m_mean_roll{window}_mean"] = (
            out["temperature_2m_mean"].rolling(window=window, min_periods=1).mean()
        )

    out["precip_wind_interaction"] = out["precipitation_sum"] * out["wind_speed_10m_max"]
    out["rain_intensity"] = out["precipitation_sum"] / out["precipitation_hours"].replace(0, np.nan)
    out["rain_intensity"] = out["rain_intensity"].fillna(0.0)
    out["sunshine_efficiency"] = out["sunshine_duration"] / out["shortwave_radiation_sum"].replace(0, np.nan)
    out["sunshine_efficiency"] = out["sunshine_efficiency"].fillna(0.0)
    out["temp_vpd_interaction"] = out["temperature_2m_mean"] * out["vapour_pressure_deficit_max"]

    for season_col in SEASON_COLUMNS:
        season = int(season_col.rsplit("_", 1)[1])
        out[season_col] = (out["season"] == season).astype(int)

    weather_group = out["weathercode"].map(WEATHER_GROUP_MAP).fillna("other")
    for weather_col in WEATHER_COLUMNS:
        group = weather_col.replace("weather_", "")
        out[weather_col] = (weather_group == group).astype(int)

    return out


def make_model_matrix(df: pd.DataFrame, feature_names: list[str] | None = None) -> pd.DataFrame:
    """Return model-ready rows with columns ordered for prediction."""
    feature_names = feature_names or MODEL_FEATURES
    featured = add_forecasting_features(df)
    matrix = featured.reindex(columns=feature_names)
    return matrix.replace([np.inf, -np.inf], np.nan)


def add_targets(
    df: pd.DataFrame,
    rain_threshold_mm: float = 0.1,
    rain_horizon_days: int = 7,
    precip_horizon_days: int = 3,
) -> pd.DataFrame:
    """Create prospective classification and regression targets."""
    out = df.sort_values("date").reset_index(drop=True).copy()
    future_rain = out["rain_sum"].shift(-rain_horizon_days)
    out[CLASSIFICATION_TARGET] = np.where(future_rain > rain_threshold_mm, 1, 0)
    out.loc[future_rain.isna(), CLASSIFICATION_TARGET] = np.nan

    shifted = [out["precipitation_sum"].shift(-step) for step in range(1, precip_horizon_days + 1)]
    out[REGRESSION_TARGET] = sum(shifted)
    return out


def build_supervised_dataset(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """Build a model matrix and target vector from daily weather rows."""
    with_targets = add_targets(df)
    matrix = make_model_matrix(with_targets, MODEL_FEATURES)
    dataset = pd.concat([with_targets[["date", target_col]], matrix], axis=1)
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()
    return dataset[MODEL_FEATURES], dataset[target_col], dataset["date"]
