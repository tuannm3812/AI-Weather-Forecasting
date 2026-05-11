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

col1, col2, col3 = st.columns(3)
col1.metric("Latest complete weather date", str(latest_date))
col2.metric("Rain probability in 7 days", f"{result['rain_probability_7d']:.1%}")
col3.metric("Next 3-day precipitation", f"{result['precipitation_3d_mm']:.1f} mm")

alert = "Rain alert" if result["rain_alert"] else "No rain alert"
st.subheader(f"{alert} for {target_date.date()}")
st.progress(min(1.0, result["rain_probability_7d"]))

with st.expander("Model details", expanded=False):
    st.write("Classification bundle")
    st.json(result["classification_metadata"])
    st.write("Regression bundle")
    st.json(result["regression_metadata"])

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
st.dataframe(weather[recent_cols].tail(14), use_container_width=True, hide_index=True)

st.line_chart(
    weather.tail(90).set_index("date")[["precipitation_sum", "temperature_2m_mean"]],
    height=260,
)

st.caption(f"Data source file: {PROCESSED_DATA_DIR / CURRENT_WEATHER_FILE}")
