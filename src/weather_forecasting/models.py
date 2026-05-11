from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    root_mean_squared_error,
    roc_auc_score,
)

from weather_forecasting.config import (
    CLASSIFICATION_TARGET,
    DEFAULT_CLASSIFICATION_THRESHOLD,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REGRESSION_TARGET,
    TASK_CLASSIFICATION,
    TASK_REGRESSION,
)
from weather_forecasting.data import load_current_weather, refresh_current_weather, split_features_target, train_valid_split
from weather_forecasting.features import MODEL_FEATURES, build_supervised_dataset


def train_baseline(df: pd.DataFrame, target_col: str):
    x, y = split_features_target(df, target_col)
    x_train, x_valid, y_train, y_valid = train_valid_split(x, y)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(x_train, y_train)

    preds = model.predict(x_valid)
    mae = mean_absolute_error(y_valid, preds)
    return model, mae


def split_by_time(X: pd.DataFrame, y: pd.Series, dates: pd.Series) -> dict[str, pd.DataFrame | pd.Series]:
    """Use the most recent year as test and the previous year as validation."""
    dates = pd.to_datetime(dates)
    max_date = dates.max()
    test_start = max_date - pd.Timedelta(days=365)
    val_start = max_date - pd.Timedelta(days=730)

    train_mask = dates < val_start
    val_mask = (dates >= val_start) & (dates < test_start)
    test_mask = dates >= test_start

    return {
        "X_train": X.loc[train_mask].reset_index(drop=True),
        "y_train": y.loc[train_mask].reset_index(drop=True),
        "X_val": X.loc[val_mask].reset_index(drop=True),
        "y_val": y.loc[val_mask].reset_index(drop=True),
        "X_test": X.loc[test_mask].reset_index(drop=True),
        "y_test": y.loc[test_mask].reset_index(drop=True),
        "train_end": dates.loc[train_mask].max().date().isoformat(),
        "val_start": dates.loc[val_mask].min().date().isoformat(),
        "val_end": dates.loc[val_mask].max().date().isoformat(),
        "test_start": dates.loc[test_mask].min().date().isoformat(),
        "test_end": dates.loc[test_mask].max().date().isoformat(),
    }


def save_design_matrices(task_suffix: str, split_data: dict, processed_dir: Path = PROCESSED_DATA_DIR) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        split_data[f"X_{split}"].to_parquet(processed_dir / f"X_{split}_{task_suffix}.parquet", index=False)
        split_data[f"y_{split}"].to_frame().to_parquet(processed_dir / f"y_{split}_{task_suffix}.parquet", index=False)


def build_design_matrices(weather_df: pd.DataFrame) -> dict[str, dict]:
    X_cls, y_cls, dates_cls = build_supervised_dataset(weather_df, CLASSIFICATION_TARGET)
    X_reg, y_reg, dates_reg = build_supervised_dataset(weather_df, REGRESSION_TARGET)
    cls = split_by_time(X_cls, y_cls.astype(int), dates_cls)
    reg = split_by_time(X_reg, y_reg.astype(float), dates_reg)
    save_design_matrices("cls", cls)
    save_design_matrices("reg", reg)
    return {"cls": cls, "reg": reg}


def load_design_matrices(processed_dir: Path = PROCESSED_DATA_DIR):
    """Load train/validation/test matrices from processed data."""
    data = {}
    for task in ("cls", "reg"):
        for split in ("train", "val", "test"):
            data[f"X_{split}_{task}"] = pd.read_parquet(processed_dir / f"X_{split}_{task}.parquet")
            y = pd.read_parquet(processed_dir / f"y_{split}_{task}.parquet")
            data[f"y_{split}_{task}"] = y.iloc[:, 0]
    return data


def classification_metrics(y_true, y_proba, threshold: float) -> dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
        "threshold": threshold,
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def tune_threshold(y_true, y_proba) -> float:
    candidates = [round(x / 100, 2) for x in range(10, 91, 5)]
    scored = [
        (f1_score(y_true, (y_proba >= threshold).astype(int), zero_division=0), threshold)
        for threshold in candidates
    ]
    return max(scored)[1]


def save_model_bundle(
    task: str,
    model,
    features: list[str],
    metrics: dict,
    split_metadata: dict,
    threshold: float | None = None,
) -> Path:
    out_dir = MODELS_DIR / task
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.joblib"
    joblib.dump(model, model_path, compress=("xz", 3))
    (out_dir / "features.txt").write_text("\n".join(features), encoding="utf-8")

    metadata = {
        "task": task,
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "model_path": str(model_path),
        "features_count": len(features),
        "features_path": str(out_dir / "features.txt"),
        "threshold": threshold,
        "metrics": metrics,
        "splits": split_metadata,
        "code_version": "project-script-v2",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out_dir


def train_and_save_models(refresh_data: bool = True) -> dict[str, dict]:
    weather_df = refresh_current_weather() if refresh_data else load_current_weather()
    design = build_design_matrices(weather_df)

    cls_model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42,
    )
    cls = design["cls"]
    cls_model.fit(cls["X_train"], cls["y_train"].astype(int))
    val_proba = cls_model.predict_proba(cls["X_val"])[:, 1]
    threshold = tune_threshold(cls["y_val"].astype(int), val_proba)
    test_proba = cls_model.predict_proba(cls["X_test"])[:, 1]
    cls_metrics = classification_metrics(cls["y_test"].astype(int), test_proba, threshold)
    cls_dir = save_model_bundle(
        TASK_CLASSIFICATION,
        cls_model,
        MODEL_FEATURES,
        cls_metrics,
        {key: cls[key] for key in ("train_end", "val_start", "val_end", "test_start", "test_end")},
        threshold or DEFAULT_CLASSIFICATION_THRESHOLD,
    )

    reg_model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        random_state=42,
    )
    reg = design["reg"]
    reg_model.fit(reg["X_train"], reg["y_train"].astype(float))
    test_pred = reg_model.predict(reg["X_test"])
    reg_metrics = regression_metrics(reg["y_test"].astype(float), test_pred)
    reg_dir = save_model_bundle(
        TASK_REGRESSION,
        reg_model,
        MODEL_FEATURES,
        reg_metrics,
        {key: reg[key] for key in ("train_end", "val_start", "val_end", "test_start", "test_end")},
    )

    return {
        "data": {
            "rows": len(weather_df),
            "start_date": weather_df["date"].min().date().isoformat(),
            "end_date": weather_df["date"].max().date().isoformat(),
            "path": str(PROCESSED_DATA_DIR / "sydney_daily_current.csv"),
        },
        TASK_CLASSIFICATION: {"path": str(cls_dir), "metrics": cls_metrics},
        TASK_REGRESSION: {"path": str(reg_dir), "metrics": reg_metrics},
    }


def main():
    results = train_and_save_models(refresh_data=True)
    print("[OK] Refreshed weather data")
    print(results["data"])
    for task in (TASK_CLASSIFICATION, TASK_REGRESSION):
        print(f"[OK] Saved {task} bundle to {results[task]['path']}")
        print(results[task]["metrics"])


if __name__ == "__main__":
    main()
