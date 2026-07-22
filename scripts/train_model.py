"""Train and persist the demonstration price recommendation model."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynamic_pricing.features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    prepare_model_features,
)

SOURCE_DATA = ROOT / (
    "Dynamic-Pricing-Strategies-for-Retail-A-Data-Driven-Approach-main/adjusted_prices.csv"
)
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "dynamic_pricing_model.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
TARGET = "Adjusted_Price"
RANDOM_STATE = 42


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_pipeline, list(NUMERIC_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ]
    )
    regressor = RandomForestRegressor(
        n_estimators=180,
        max_depth=14,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("regressor", regressor)])


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _top_feature_importance(pipeline: Pipeline, limit: int = 15) -> list[dict[str, float | str]]:
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["regressor"].feature_importances_
    order = np.argsort(importances)[::-1][:limit]
    return [
        {
            "feature": str(names[index]).replace("numeric__", "").replace("categorical__", ""),
            "importance": round(float(importances[index]), 6),
        }
        for index in order
    ]


def train() -> tuple[Pipeline, dict[str, object]]:
    data = pd.read_csv(SOURCE_DATA)
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.loc[data[TARGET].notna() & data[TARGET].gt(0)].reset_index(drop=True)
    features = prepare_model_features(data)
    target = data[TARGET]

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE
    )
    evaluation_model = build_pipeline()
    evaluation_model.fit(x_train, y_train)
    prediction = evaluation_model.predict(x_test)

    metrics = {
        "mae": round(float(mean_absolute_error(y_test, prediction)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, prediction))), 4),
        "r2": round(float(r2_score(y_test, prediction)), 4),
        "within_5_percent": round(
            float(np.mean(np.abs((prediction - y_test.to_numpy()) / y_test.to_numpy()) <= 0.05)),
            4,
        ),
        "current_price_baseline_mae": round(
            float(mean_absolute_error(y_test, x_test["PRICE_CURRENT"])), 4
        ),
    }

    final_model = build_pipeline()
    final_model.fit(features, target)

    known_categories = {
        column: sorted(data[column].dropna().astype(str).unique().tolist())
        for column in ("DEPARTMENT", "CATEGORY", "PROMOTION")
    }
    categories_by_department = {
        str(department): sorted(group["CATEGORY"].dropna().astype(str).unique().tolist())
        for department, group in data.groupby("DEPARTMENT", dropna=True)
    }

    metadata: dict[str, object] = {
        "model_name": "Random Forest demonstration regressor",
        "model_version": "1.0.0",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "source_file": str(SOURCE_DATA.relative_to(ROOT)),
        "source_sha256": _file_sha256(SOURCE_DATA),
        "target": TARGET,
        "feature_columns": list(MODEL_FEATURES),
        "training_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "full_fit_rows": int(len(data)),
        "metrics": metrics,
        "known_categories": known_categories,
        "categories_by_department": categories_by_department,
        "top_feature_importance": _top_feature_importance(final_model),
        "data_scope": {
            "minimum_date": str(pd.to_datetime(data["RunDate"]).min().date()),
            "maximum_date": str(pd.to_datetime(data["RunDate"]).max().date()),
            "unique_products": int(data["PRODUCT_NAME"].nunique()),
            "unique_skus": int(data["SKU"].nunique()),
            "unique_locations": int(data["SHIPPING_LOCATION"].nunique()),
        },
        "limitations": [
            "The repository dataset is synthetic and represents a small, fixed product catalogue.",
            "Adjusted_Price is a project-generated target, not a realized market-clearing or profit-optimal price.",
            "The data contains no units sold, demand elasticity, product cost, inventory, or competitor price.",
            "Holdout metrics measure reproduction of the synthetic target and do not validate commercial uplift.",
            "Recommendations require human review and controlled experimentation before operational use.",
        ],
    }
    return final_model, metadata


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model, metadata = train()
    joblib.dump(model, MODEL_PATH, compress=3)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")
    print(json.dumps(metadata["metrics"], indent=2))


if __name__ == "__main__":
    main()
