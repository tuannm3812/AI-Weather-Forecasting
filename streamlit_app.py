import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from weather_forecasting.config import CURRENT_WEATHER_FILE, PROCESSED_DATA_DIR
from weather_forecasting.data import load_current_weather, refresh_current_weather
from weather_forecasting.inference import predict_latest


st.set_page_config(page_title="Sydney Rainfall Forecasting", layout="wide")

st.title("Sydney Rainfall Forecasting")
st.caption("Live Open-Meteo refresh, local joblib model bundles, and production-style inference.")

try:
    if st.sidebar.button("Refresh weather data"):
        with st.spinner("Fetching current Open-Meteo archive data..."):
            weather = refresh_current_weather()
    else:
        weather = load_current_weather()
    result = predict_latest(weather)
except Exception as exc:
    st.error(f"Unable to load forecast artifacts: {exc}")
    st.stop()

latest_date = result["as_of_date"]
target_date = pd.Timestamp(latest_date) + pd.Timedelta(days=7)
threshold = result["threshold"]
weather_display = weather.copy()
weather_display["date"] = pd.to_datetime(weather_display["date"]).dt.date

col1, col2, col3 = st.columns(3)
col1.metric("Latest complete weather date", str(latest_date))
col2.metric("Rain probability in 7 days", f"{result['rain_probability_7d']:.1%}")
col3.metric("Next 3-day precipitation", f"{result['precipitation_3d_mm']:.1f} mm")

alert = "Rain alert" if result["rain_alert"] else "No rain alert"
st.subheader(f"{alert} for {target_date.date()}")
st.progress(min(1.0, result["rain_probability_7d"]))
st.caption(f"Alert threshold: {threshold:.0%}. Predictions use the latest complete daily weather observation.")

overview_tab, trends_tab, seasonality_tab, data_tab, model_tab = st.tabs(
    ["Overview", "Recent Trends", "Seasonality", "Source Data", "Model Details"]
)

with overview_tab:
    recent_30 = weather.tail(30).copy()
    recent_30["date"] = pd.to_datetime(recent_30["date"])
    rain_days_30 = int((recent_30["rain_sum"] > 0.1).sum())
    rain_total_30 = float(recent_30["rain_sum"].sum())
    avg_temp_30 = float(recent_30["temperature_2m_mean"].mean())

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Rain days, last 30 days", rain_days_30)
    kpi2.metric("Rainfall, last 30 days", f"{rain_total_30:.1f} mm")
    kpi3.metric("Average temperature, last 30 days", f"{avg_temp_30:.1f} C")

    st.bar_chart(
        recent_30.set_index("date")["rain_sum"],
        height=260,
        use_container_width=True,
    )

with trends_tab:
    recent_120 = weather.tail(120).copy()
    recent_120["date"] = pd.to_datetime(recent_120["date"])
    recent_120["rain_14d"] = recent_120["rain_sum"].rolling(14, min_periods=1).sum()
    recent_120["temp_14d"] = recent_120["temperature_2m_mean"].rolling(14, min_periods=1).mean()
    recent_120["pressure_14d"] = recent_120["pressure_msl_mean"].rolling(14, min_periods=1).mean()

    st.write("Rolling weather signals")
    st.line_chart(
        recent_120.set_index("date")[["rain_14d", "temp_14d"]],
        height=280,
        use_container_width=True,
    )
    st.line_chart(
        recent_120.set_index("date")[["pressure_14d", "cloudcover_mean"]],
        height=280,
        use_container_width=True,
    )

with seasonality_tab:
    seasonal = weather.copy()
    seasonal["date"] = pd.to_datetime(seasonal["date"])
    seasonal["month"] = seasonal["date"].dt.month_name().str.slice(stop=3)
    seasonal["month_num"] = seasonal["date"].dt.month
    monthly = (
        seasonal.groupby(["month_num", "month"], as_index=False)
        .agg(
            avg_rain_mm=("rain_sum", "mean"),
            rain_day_rate=("rain_sum", lambda s: (s > 0.1).mean()),
            avg_temperature_c=("temperature_2m_mean", "mean"),
        )
        .sort_values("month_num")
    )
    monthly = monthly.set_index("month")

    col_a, col_b = st.columns(2)
    with col_a:
        st.write("Average daily rainfall by month")
        st.bar_chart(monthly["avg_rain_mm"], height=280, use_container_width=True)
    with col_b:
        st.write("Historical rain-day rate by month")
        st.bar_chart(monthly["rain_day_rate"], height=280, use_container_width=True)

    st.write("Monthly climatology table")
    st.dataframe(monthly.drop(columns=["month_num"]).round(3), use_container_width=True)

with data_tab:
    st.subheader("Recent source observations")
    recent_cols = [
        "date",
        "precipitation_sum",
        "rain_sum",
        "temperature_2m_mean",
        "cloudcover_mean",
        "wind_speed_10m_max",
        "pressure_msl_mean",
    ]
    st.dataframe(weather_display[recent_cols].tail(21), use_container_width=True, hide_index=True)

with model_tab:
    st.write("Classification bundle")
    st.json(result["classification_metadata"])
    st.write("Regression bundle")
    st.json(result["regression_metadata"])

st.caption(f"Data source file: {PROCESSED_DATA_DIR / CURRENT_WEATHER_FILE}")
