"""Model inference and business-safe post-processing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import prepare_model_features


def _round_prices(values: pd.Series, lower: pd.Series, upper: pd.Series, mode: str) -> pd.Series:
    rounded = values.round(2)
    if mode == "End in .99":
        candidate = np.floor(values) + 0.99
        within_guardrail = candidate.ge(lower) & candidate.le(upper) & candidate.gt(0)
        rounded = rounded.where(~within_guardrail, candidate.round(2))
    return rounded.clip(lower=0.01)


def score_dataframe(
    model: object,
    data: pd.DataFrame,
    max_change_pct: float = 15.0,
    rounding_mode: str = "Nearest cent",
) -> pd.DataFrame:
    """Predict recommended prices and apply an auditable change guardrail."""

    if not 0 <= max_change_pct <= 100:
        raise ValueError("max_change_pct must be between 0 and 100")
    if rounding_mode not in {"Nearest cent", "End in .99"}:
        raise ValueError("Unsupported rounding_mode")

    result = data.copy()
    if result.empty:
        for column in (
            "Model_Prediction",
            "Recommended_Price",
            "Recommended_Change",
            "Price_Change_Pct",
            "Pricing_Action",
            "Guardrail_Applied",
        ):
            result[column] = pd.Series(dtype="float64" if column != "Pricing_Action" else "object")
        return result

    features = prepare_model_features(result)
    raw_prediction = pd.Series(model.predict(features), index=result.index, dtype=float)
    current = pd.to_numeric(result["PRICE_CURRENT"], errors="coerce")
    fraction = max_change_pct / 100.0
    lower = (current * (1 - fraction)).clip(lower=0.01)
    upper = current * (1 + fraction)
    guarded = raw_prediction.clip(lower=lower, upper=upper)
    recommended = _round_prices(guarded, lower, upper, rounding_mode)

    result["Model_Prediction"] = raw_prediction.round(4)
    result["Recommended_Price"] = recommended
    # Keep this distinct from the source notebook's pre-existing Price_Change
    # feature so exported audit files preserve both meanings.
    result["Recommended_Change"] = (recommended - current).round(2)
    result["Price_Change_Pct"] = ((recommended / current - 1) * 100).round(2)
    result["Pricing_Action"] = np.select(
        [result["Price_Change_Pct"].gt(0.5), result["Price_Change_Pct"].lt(-0.5)],
        ["Increase", "Decrease"],
        default="Hold",
    )
    result["Guardrail_Applied"] = ~np.isclose(raw_prediction, guarded, rtol=0, atol=1e-9)
    return result
