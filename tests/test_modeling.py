from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from dynamic_pricing.modeling import score_dataframe


class FixedModel:
    def __init__(self, prediction: float) -> None:
        self.prediction = prediction

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.prediction)


def example_row(current: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "PRICE_RETAIL": 100.0,
                "PRICE_CURRENT": current,
                "RunDate": "2024-07-10",
                "DEPARTMENT": "Frozen Foods",
                "CATEGORY": "Ice Cream",
                "PROMOTION": "Summer Premium",
            }
        ]
    )


def test_guardrail_caps_model_prediction() -> None:
    result = score_dataframe(FixedModel(150), example_row(), max_change_pct=10)

    assert result.loc[0, "Recommended_Price"] == 110.0
    assert result.loc[0, "Price_Change_Pct"] == 10.0
    assert result.loc[0, "Pricing_Action"] == "Increase"
    assert bool(result.loc[0, "Guardrail_Applied"])


def test_psychological_rounding_respects_guardrail() -> None:
    result = score_dataframe(
        FixedModel(9.93), example_row(current=10.0), max_change_pct=10, rounding_mode="End in .99"
    )

    assert result.loc[0, "Recommended_Price"] == 9.99


def test_invalid_guardrail_is_rejected() -> None:
    with pytest.raises(ValueError):
        score_dataframe(FixedModel(10), example_row(), max_change_pct=101)


def test_committed_model_scores_a_valid_row() -> None:
    root = Path(__file__).resolve().parents[1]
    model = joblib.load(root / "artifacts/dynamic_pricing_model.joblib")

    result = score_dataframe(model, example_row(current=4.79), max_change_pct=15)

    assert result.loc[0, "Recommended_Price"] > 0
    assert result.loc[0, "Pricing_Action"] in {"Increase", "Hold", "Decrease"}

