# Sydney Rainfall Forecasting

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red)
![Open-Meteo](https://img.shields.io/badge/data-Open--Meteo-2ea44f)
![Status](https://img.shields.io/badge/status-active-brightgreen)

<p align="center">
  <img src="https://content.api.news/v3/images/bin/bf15daf56e540e05eef7ae59cb0d2d5a" alt="Sydney rainfall forecast cover" width="100%">
</p>

End-to-end machine learning project for Sydney rainfall forecasting using Open-Meteo weather history, scikit-learn models, reproducible `joblib` artifacts, and a Streamlit dashboard.

The pipeline refreshes daily Sydney weather observations through the latest complete day, builds supervised forecasting features, trains separate classification and regression models, stores versioned model bundles, and serves the latest local prediction through an interactive dashboard.

## Forecasting Tasks

| Task | Output | Horizon |
| --- | --- | --- |
| `rain_or_not` | Probability of measurable rain | 7 days after the latest complete observation |
| `precipitation_fall` | Estimated cumulative precipitation | Next 3 days |

## Project Structure

```text
.
|-- data/
|   |-- raw/                      # Open-Meteo JSON responses
|   |-- interim/                  # Optional intermediate datasets
|   `-- processed/                # Current weather data and train/validation/test matrices
|-- models/
|   |-- rain_or_not/              # Classification model bundle
|   `-- precipitation_fall/       # Regression model bundle
|-- notebooks/
|   `-- 01_weather_forecasting_experiment.ipynb
|-- reports/
|   `-- figures/                  # Generated analysis figures
|-- scripts/
|   `-- train_models.py           # Refresh data, train models, and save artifacts
|-- src/weather_forecasting/      # Reusable package code
|-- tests/                        # Unit tests
|-- pyproject.toml                # Package metadata and dependencies
`-- streamlit_app.py              # Local dashboard
```

## Quickstart

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

Refresh the Open-Meteo data and regenerate model bundles:

```powershell
python scripts\train_models.py
```

Run the dashboard:

```powershell
streamlit run streamlit_app.py
```

Run tests:

```powershell
pytest
```

## Data Pipeline

The training script fetches Open-Meteo archive data for Sydney from `2000-01-01` through yesterday in `Australia/Sydney` time. It writes the current daily dataset to:

- `data/processed/sydney_daily_current.csv`
- `data/processed/X_train_cls.parquet`, `X_val_cls.parquet`, `X_test_cls.parquet`
- `data/processed/X_train_reg.parquet`, `X_val_reg.parquet`, `X_test_reg.parquet`
- matching target files prefixed with `y_`

The latest generated dataset contains 9,627 daily rows from `2000-01-01` to `2026-05-10`.

## Modeling

Each model bundle contains:

- `model.joblib`: fitted scikit-learn estimator
- `features.txt`: model feature order used at inference time
- `metadata.json`: metrics, timestamp, threshold, split dates, and environment details

The current baseline uses histogram gradient boosting models. The classification threshold is selected on the validation split using F1, then reported on the held-out test split.

Latest local test metrics:

| Task | Metrics |
| --- | --- |
| Rain in 7 days | F1 `0.676`, Precision `0.512`, Recall `0.995`, ROC-AUC `0.565` |
| 3-day precipitation | MAE `9.056 mm`, RMSE `17.262 mm`, R2 `0.091` |

## Dashboard

The Streamlit app loads the saved model bundles and the latest processed weather file, then displays:

- latest complete observation date
- 7-day rain probability and alert threshold
- next 3-day precipitation estimate
- recent rainfall and temperature trends
- monthly seasonality summaries
- source observations and model metadata

## Notebook Workflow

[notebooks/01_weather_forecasting_experiment.ipynb](notebooks/01_weather_forecasting_experiment.ipynb) provides the experiment record:

1. Problem framing and decision horizon.
2. Current data inventory.
3. Feature schema review.
4. Model regeneration through reusable project code.
5. Live prediction smoke test.
6. Deployment notes and next experiments.

## Limitations

This is a portfolio-ready forecasting prototype, not an operational weather service. The live dashboard predicts from the latest observed daily weather row; a production-grade system should consume forecast-issued covariates, add probability calibration, monitor data drift, and schedule automated refreshes. Extreme rainfall events are rare, so regression error can be larger during heavy-rain periods.
