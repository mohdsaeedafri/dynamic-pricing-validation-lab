import pandas as pd

from dynamic_pricing.features import MODEL_FEATURES, prepare_model_features


def test_prepare_model_features_derives_season_and_defaults() -> None:
    raw = pd.DataFrame(
        {
            "PRICE_RETAIL": [10, 20, 5, 4],
            "PRICE_CURRENT": [9, 22, 5, 4.4],
            "RunDate": ["2024-01-10", "2024-04-10", "2024-07-10", "2024-10-10"],
            "DEPARTMENT": ["A", "A", "Frozen Foods", "B"],
            "CATEGORY": ["One", "Two", "Ice Cream", "Three"],
        }
    )

    features = prepare_model_features(raw)

    assert tuple(features.columns) == MODEL_FEATURES
    assert features["Season"].tolist() == ["Winter", "Spring", "Summer", "Fall"]
    assert features["PROMOTION"].tolist() == ["Regular"] * 4
    assert features["IsIceCream"].tolist() == [0, 0, 1, 0]
    assert features["PRICE_GAP_PCT"].round(2).tolist() == [-0.1, 0.1, 0.0, 0.1]

