"""Dependency-light pricing model used when the reference model cannot load."""

from __future__ import annotations

import numpy as np
import pandas as pd


class PortablePricingModel:
    """Auditable fallback based on the source project's seasonal price rules.

    The committed scikit-learn model remains the preferred inference engine.
    This model keeps the public validation lab usable in constrained runtimes
    and mirrors the source notebook's 10% seasonal ice-cream adjustment.
    """

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        retail = pd.to_numeric(features["PRICE_RETAIL"], errors="coerce")
        current = pd.to_numeric(features["PRICE_CURRENT"], errors="coerce")

        # Anchor recommendations to the reference price while retaining a
        # small contribution from the observed selling price.
        prediction = retail.mul(0.92).add(current.mul(0.08))

        is_ice_cream = pd.to_numeric(features["IsIceCream"], errors="coerce").eq(1)
        season = features["Season"].fillna("Unknown").astype(str)
        winter = is_ice_cream & season.eq("Winter")
        summer = is_ice_cream & season.eq("Summer")
        prediction = prediction.mask(winter, prediction.mul(0.90))
        prediction = prediction.mask(summer, prediction.mul(1.10))

        return prediction.to_numpy(dtype=float)
