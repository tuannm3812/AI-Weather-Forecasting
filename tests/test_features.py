import pandas as pd

from weather_forecasting.features import add_datetime_parts


def test_add_datetime_parts():
    df = pd.DataFrame({"timestamp": ["2026-01-01 00:00:00"]})
    out = add_datetime_parts(df, "timestamp")

    assert out.loc[0, "year"] == 2026
    assert out.loc[0, "month"] == 1
    assert out.loc[0, "day"] == 1
    assert out.loc[0, "hour"] == 0
