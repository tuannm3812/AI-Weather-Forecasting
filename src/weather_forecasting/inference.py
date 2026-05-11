import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from weather_forecasting.config import (
    DEFAULT_CLASSIFICATION_THRESHOLD,
    MODELS_DIR,
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
)
from weather_forecasting.data import load_current_weather
from weather_forecasting.features import MODEL_FEATURES, make_model_matrix


@dataclass
class ModelBundle:
    model: object
    features: list[str]
    metadata: dict


def load_bundle(task: str, models_dir: Path = MODELS_DIR) -> ModelBundle:
    bundle_dir = models_dir / task
    metadata_path = bundle_dir / "metadata.json"
    features_path = bundle_dir / "features.txt"

    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    if features_path.exists():
        features = [line.strip() for line in features_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        features = MODEL_FEATURES

    model = joblib.load(bundle_dir / "model.joblib")
    return ModelBundle(model=model, features=features, metadata=metadata)


def predict_latest(weather_df: pd.DataFrame | None = None) -> dict:
    weather_df = weather_df if weather_df is not None else load_current_weather()
    cls_bundle = load_bundle(TASK_CLASSIFICATION)
    reg_bundle = load_bundle(TASK_REGRESSION)

    cls_matrix = make_model_matrix(weather_df, cls_bundle.features).dropna()
    reg_matrix = make_model_matrix(weather_df, reg_bundle.features).dropna()
    if cls_matrix.empty or reg_matrix.empty:
        raise ValueError("Not enough complete weather history to create lag and rolling features.")

    latest_idx = min(cls_matrix.index.max(), reg_matrix.index.max())
    latest_date = pd.to_datetime(weather_df.loc[latest_idx, "date"]).date()

    x_cls = cls_matrix.loc[[latest_idx]]
    x_reg = reg_matrix.loc[[latest_idx]]
    rain_probability = float(cls_bundle.model.predict_proba(x_cls)[0, 1])
    threshold = float(cls_bundle.metadata.get("threshold") or DEFAULT_CLASSIFICATION_THRESHOLD)
    rain_alert = rain_probability >= threshold
    precipitation_3d = max(0.0, float(reg_bundle.model.predict(x_reg)[0]))

    return {
        "as_of_date": latest_date,
        "rain_probability_7d": rain_probability,
        "rain_alert": rain_alert,
        "threshold": threshold,
        "precipitation_3d_mm": precipitation_3d,
        "classification_metadata": cls_bundle.metadata,
        "regression_metadata": reg_bundle.metadata,
    }
