"""Deterministic feature engineering used in training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import SEASON_BY_MONTH

NUMERIC_FEATURES = (
    "PRICE_RETAIL",
    "PRICE_CURRENT",
    "Month",
    "IsIceCream",
    "PRICE_GAP_PCT",
)

CATEGORICAL_FEATURES = (
    "DEPARTMENT",
    "CATEGORY",
    "Season",
    "PROMOTION",
)

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def prepare_model_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create the exact, ordered feature frame consumed by the model.

    The caller should validate rows first. Defensive coercion is retained so
    inference fails gracefully if the function is used independently.
    """

    frame = data.copy()
    frame["PRICE_RETAIL"] = pd.to_numeric(frame["PRICE_RETAIL"], errors="coerce")
    frame["PRICE_CURRENT"] = pd.to_numeric(frame["PRICE_CURRENT"], errors="coerce")

    run_date = pd.to_datetime(frame["RunDate"], errors="coerce")
    frame["Month"] = run_date.dt.month
    frame["Season"] = frame["Month"].map(SEASON_BY_MONTH).fillna("Unknown")

    category = frame["CATEGORY"].fillna("Unknown").astype(str).str.strip()
    frame["IsIceCream"] = category.str.casefold().eq("ice cream").astype(int)

    denominator = frame["PRICE_RETAIL"].replace(0, np.nan)
    frame["PRICE_GAP_PCT"] = (
        (frame["PRICE_CURRENT"] - frame["PRICE_RETAIL"]) / denominator
    ).clip(-10, 10)

    for column in ("DEPARTMENT", "CATEGORY"):
        frame[column] = frame[column].fillna("Unknown").astype(str).str.strip()

    if "PROMOTION" not in frame:
        frame["PROMOTION"] = "Regular"
    frame["PROMOTION"] = (
        frame["PROMOTION"]
        .fillna("Regular")
        .astype(str)
        .str.strip()
        .replace("", "Regular")
    )

    return frame.loc[:, MODEL_FEATURES]

