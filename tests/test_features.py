import pandas as pd

from weather_forecasting.config import CLASSIFICATION_TARGET, REGRESSION_TARGET
from weather_forecasting.features import MODEL_FEATURES, add_datetime_parts, add_targets, make_model_matrix


def test_add_datetime_parts():
    df = pd.DataFrame({"timestamp": ["2026-01-01 00:00:00"]})
    out = add_datetime_parts(df, "timestamp")

    assert out.loc[0, "year"] == 2026
    assert out.loc[0, "month"] == 1
    assert out.loc[0, "day"] == 1
    assert out.loc[0, "hour"] == 0


def test_add_targets_uses_future_weather():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=8, freq="D"),
            "rain_sum": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.5],
            "precipitation_sum": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        }
    )

    out = add_targets(df)

    assert out.loc[0, CLASSIFICATION_TARGET] == 1
    assert out.loc[0, REGRESSION_TARGET] == 9.0
    assert pd.isna(out.loc[7, CLASSIFICATION_TARGET])
    assert pd.isna(out.loc[7, REGRESSION_TARGET])


def test_make_model_matrix_preserves_feature_order():
    rows = 10
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=rows, freq="D"),
            "precipitation_sum": [0.0] * rows,
            "rain_sum": [0.0] * rows,
            "precipitation_hours": [0.0] * rows,
            "temperature_2m_mean": [20.0] * rows,
            "temperature_2m_max": [25.0] * rows,
            "temperature_2m_min": [15.0] * rows,
            "vapour_pressure_deficit_max": [0.8] * rows,
            "cloudcover_mean": [50.0] * rows,
            "shortwave_radiation_sum": [18.0] * rows,
            "sunshine_duration": [30_000.0] * rows,
            "wind_speed_10m_max": [20.0] * rows,
            "wind_direction_10m_dominant": [180.0] * rows,
            "pressure_msl_mean": [1015.0] * rows,
            "soil_moisture_0_to_7cm_mean": [0.2] * rows,
            "soil_moisture_7_to_28cm_mean": [0.25] * rows,
            "weathercode": [61] * rows,
        }
    )

    matrix = make_model_matrix(df)

    assert list(matrix.columns) == MODEL_FEATURES
    assert matrix.loc[0, "weather_rain"] == 1
    assert matrix.loc[3:, MODEL_FEATURES].notna().all().all()
