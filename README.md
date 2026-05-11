# Sydney Rainfall Forecasting

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red)
![Open-Meteo](https://img.shields.io/badge/data-Open--Meteo-2ea44f)
![Status](https://img.shields.io/badge/status-active-brightgreen)

<p align="center">
  <img src="https://content.api.news/v3/images/bin/bf15daf56e540e05eef7ae59cb0d2d5a" alt="Sydney rainfall forecast cover" width="100%">
</p>

End-to-end data science project for live Sydney rainfall forecasting with Open-Meteo, scikit-learn, and Streamlit.

The pipeline refreshes daily weather history through yesterday in Sydney time, rebuilds supervised features, trains two forecasting models, saves reproducible `joblib` bundles, and serves the latest prediction in a Streamlit dashboard.

## Forecasting Tasks

- `rain_or_not`: predicts the probability of measurable rain 7 days after the latest complete observation.
- `precipitation_fall`: predicts cumulative precipitation over the next 3 days.

## Repository Name And Description

Recommended GitHub repository name:

```text
sydney-rainfall-forecasting
```

Recommended GitHub description:

```text
Live Sydney rainfall forecasting pipeline with Open-Meteo data, scikit-learn models, joblib artifacts, and a Streamlit dashboard.
```

## Project Structure

```text
.
|-- assets/                       # README banner and project visuals
|-- data/
|   |-- raw/                      # Open-Meteo JSON responses
|   |-- interim/                  # Optional intermediate datasets
|   `-- processed/                # Current weather data and model matrices
|-- models/
|   |-- rain_or_not/              # Classification model bundle
|   `-- precipitation_fall/       # Regression model bundle
|-- notebooks/                    # Professional experiment notebook
|-- scripts/
|   `-- train_models.py           # Refresh data, train models, save artifacts
|-- src/weather_forecasting/      # Reusable package code
|-- tests/                        # Unit tests
`-- streamlit_app.py              # Local dashboard
```

## Quickstart

Create and activate an environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

Refresh current data and regenerate model bundles:

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

## Data Refresh

The training script fetches Open-Meteo archive data from `2000-01-01` through yesterday in `Australia/Sydney` time. It writes:

- `data/processed/sydney_daily_current.csv`
- `data/processed/X_train_cls.parquet`, `X_val_cls.parquet`, `X_test_cls.parquet`
- `data/processed/X_train_reg.parquet`, `X_val_reg.parquet`, `X_test_reg.parquet`
- matching target files prefixed with `y_`

The latest generated run used 9,627 daily rows from `2000-01-01` to `2026-05-10`.

## Model Artifacts

Each model bundle contains:

- `model.joblib`: fitted estimator
- `features.txt`: prediction column order
- `metadata.json`: metrics, timestamp, threshold, and time split metadata

The current baseline uses histogram gradient boosting models. The classifier threshold is selected on the validation split using F1.

Latest local run:

| Task | Primary Metrics |
| --- | --- |
| Rain in 7 days | F1 `0.676`, Precision `0.512`, Recall `0.995`, ROC-AUC `0.565` |
| 3-day precipitation | MAE `9.056 mm`, RMSE `17.262 mm`, R2 `0.091` |

## Notebook Workflow

Use [notebooks/01_weather_forecasting_experiment.ipynb](notebooks/01_weather_forecasting_experiment.ipynb) as the clean experiment record. It follows a professional data science flow:

1. Problem framing and decision horizon.
2. Current data inventory.
3. Feature schema review.
4. Model regeneration through reusable project code.
5. Live prediction smoke test.
6. Deployment notes and next experiments.

## Notes

This project is a portfolio-ready forecasting prototype. For production use, the next step is to replace same-day observed weather features with forecast-issued inputs, add model calibration, and schedule the data/model refresh.

## Limitations

- The current model is a baseline, not an operational weather service.
- The live dashboard predicts from the latest observed daily weather row; a stricter production system should consume forecast-issued covariates.
- Extreme rainfall events are rare and harder to model, so regression errors can be larger during heavy-rain periods.
