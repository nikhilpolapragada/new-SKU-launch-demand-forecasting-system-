"""
demand_transfer.py
==================
Stage 3b: Transfer demand curves from analogues to new SKUs.

Steps per new SKU:
1. Retrieve the analogue launch curves (first cold_start_weeks weeks).
2. Compute weighted average curve using AnalogueMatch weights.
3. Apply price-ratio scaling:
       scale = (mean_analogue_price / new_sku_price) ^ elasticity
   clamped to [0.1, 10.0] to prevent extreme extrapolation.

If a new SKU has no observed price, the scale defaults to 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from analogue_matcher import AnalogueMatch


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ELASTICITY: float = 1.0
SCALE_CLIP_MIN: float = 0.1
SCALE_CLIP_MAX: float = 10.0


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class TransferredForecast:
    """Demand forecast for a single new SKU after analogue transfer."""

    new_sku_id: str
    """The cold-start SKU."""

    forecast_curve: np.ndarray
    """Predicted weekly units_sold, shape (cold_start_weeks,)."""

    analogue_ids: list[str]
    """Analogues used."""

    weights: np.ndarray
    """Weights applied to analogue curves."""

    price_scale: float
    """Price-ratio scaling factor applied to the weighted average."""

    mean_analogue_price: float
    """Weighted mean price of the analogues."""

    new_sku_price: float
    """Observed or imputed price of the new SKU."""


# ---------------------------------------------------------------------------
# Demand transfer engine
# ---------------------------------------------------------------------------

class DemandTransfer:
    """
    Transfers launch demand curves from analogue SKUs to new SKUs.

    Parameters
    ----------
    cold_start_weeks : int
        Number of launch weeks to forecast (default 4).
    elasticity : float
        Price elasticity exponent (default 1.0).
    scale_clip : tuple[float, float]
        (min, max) for the price-ratio scale factor (default (0.1, 10.0)).
    """

    def __init__(
        self,
        cold_start_weeks: int = 4,
        elasticity: float = DEFAULT_ELASTICITY,
        scale_clip: tuple[float, float] = (SCALE_CLIP_MIN, SCALE_CLIP_MAX),
    ) -> None:
        self.cold_start_weeks = cold_start_weeks
        self.elasticity = elasticity
        self.scale_clip = scale_clip

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transfer(
        self,
        matches: list[AnalogueMatch],
        existing_sales: pd.DataFrame,
        new_sku_prices: pd.Series,
        existing_sku_prices: pd.Series,
    ) -> list[TransferredForecast]:
        """
        Produce demand forecasts for a list of new SKUs.

        Parameters
        ----------
        matches : list[AnalogueMatch]
            Output from AnalogueMatcher.match().
        existing_sales : pd.DataFrame
            Long-format weekly sales for existing SKUs (article_id, week_id, units_sold).
        new_sku_prices : pd.Series
            Mean price per new SKU (index = article_id).
        existing_sku_prices : pd.Series
            Mean price per existing SKU (index = article_id).

        Returns
        -------
        list[TransferredForecast]
        """
        # Pre-build launch curves for existing SKUs (first N weeks from launch)
        print("[Stage 3b] Building analogue launch curves ...")
        analogue_curves = self._build_launch_curves(existing_sales)

        print("[Stage 3b] Transferring demand to new SKUs ...")
        forecasts: list[TransferredForecast] = []

        for match in tqdm(matches, desc="  Demand transfer", unit="SKU"):
            forecast = self._transfer_single(
                match, analogue_curves, new_sku_prices, existing_sku_prices
            )
            forecasts.append(forecast)

        print(f"[Stage 3b] Generated {len(forecasts)} demand forecasts.")
        return forecasts

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_launch_curves(
        self, existing_sales: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        """
        Extract the first `cold_start_weeks` of sales for each existing SKU.

        The 'launch curve' is defined as the demand in the SKU's earliest
        observed weeks, sorted chronologically.

        Returns
        -------
        dict[str, np.ndarray]
            Maps article_id → launch curve of length cold_start_weeks.
            Missing weeks are filled with 0.
        """
        curves: dict[str, np.ndarray] = {}
        for article_id, grp in existing_sales.groupby("article_id"):
            grp_sorted = grp.sort_values("week_id")
            units = grp_sorted["units_sold"].values[: self.cold_start_weeks]
            # Pad with zeros if fewer than cold_start_weeks weeks exist
            if len(units) < self.cold_start_weeks:
                units = np.pad(
                    units, (0, self.cold_start_weeks - len(units)), constant_values=0.0
                )
            curves[str(article_id)] = units.astype(float)
        return curves

    def _transfer_single(
        self,
        match: AnalogueMatch,
        analogue_curves: dict[str, np.ndarray],
        new_sku_prices: pd.Series,
        existing_sku_prices: pd.Series,
    ) -> TransferredForecast:
        """Build one TransferredForecast."""
        # --- Weighted average of analogue launch curves ---
        weighted_curve = np.zeros(self.cold_start_weeks, dtype=float)
        valid_weights: list[float] = []
        valid_analogue_prices: list[float] = []

        for aid, w in zip(match.analogue_ids, match.weights):
            curve = analogue_curves.get(aid)
            if curve is None:
                continue
            weighted_curve += w * curve
            valid_weights.append(w)
            if aid in existing_sku_prices.index:
                valid_analogue_prices.append(existing_sku_prices[aid])

        # Renormalise if some analogues had no curves
        total_w = sum(valid_weights)
        if total_w > 0 and total_w < 1.0:
            weighted_curve /= total_w

        # --- Mean analogue price ---
        if valid_analogue_prices and total_w > 0:
            analogue_price_weights = []
            for aid, w in zip(match.analogue_ids, match.weights):
                if aid in existing_sku_prices.index:
                    analogue_price_weights.append((existing_sku_prices[aid], w))
            mean_analogue_price = float(
                sum(p * w for p, w in analogue_price_weights)
                / sum(w for _, w in analogue_price_weights)
            )
        else:
            mean_analogue_price = float(np.mean(valid_analogue_prices)) if valid_analogue_prices else 1.0

        # --- Price-ratio scaling ---
        new_price = (
            float(new_sku_prices[match.new_sku_id])
            if match.new_sku_id in new_sku_prices.index
            else mean_analogue_price
        )

        if new_price > 0 and mean_analogue_price > 0:
            raw_scale = (mean_analogue_price / new_price) ** self.elasticity
            price_scale = float(np.clip(raw_scale, self.scale_clip[0], self.scale_clip[1]))
        else:
            price_scale = 1.0

        forecast_curve = weighted_curve * price_scale

        return TransferredForecast(
            new_sku_id=match.new_sku_id,
            forecast_curve=forecast_curve,
            analogue_ids=match.analogue_ids,
            weights=match.weights,
            price_scale=price_scale,
            mean_analogue_price=mean_analogue_price,
            new_sku_price=new_price,
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def forecasts_to_dataframe(
    forecasts: list[TransferredForecast],
    week_labels: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Convert list of TransferredForecast objects to a long DataFrame.

    Columns: new_sku_id, week_offset (0-indexed), forecast, price_scale
    """
    rows = []
    for f in forecasts:
        for t, val in enumerate(f.forecast_curve):
            label = week_labels[t] if week_labels and t < len(week_labels) else t
            rows.append({
                "new_sku_id": f.new_sku_id,
                "week_offset": t,
                "week_label": label,
                "forecast": float(val),
                "price_scale": f.price_scale,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data_loader import HMDataLoader, get_sku_mean_price
    from feature_engineering import get_feature_matrix
    from analogue_matcher import AnalogueMatcher

    loader = HMDataLoader(data_dir="data")
    ds = loader.load()

    existing_ids = ds.existing_skus["article_id"].unique().tolist()
    feature_matrix, _ = get_feature_matrix(ds.articles, ds.weekly_sales, existing_ids)

    existing_features = feature_matrix.loc[feature_matrix.index.isin(existing_ids)]
    new_features = feature_matrix.loc[feature_matrix.index.isin(ds.new_sku_ids)]

    matcher = AnalogueMatcher(k=5)
    matcher.fit(existing_features)
    matches = matcher.match(new_features)

    prices = get_sku_mean_price(ds.weekly_sales)
    new_prices = prices.reindex(ds.new_sku_ids)
    existing_prices = prices.reindex(existing_ids)

    transfer = DemandTransfer(cold_start_weeks=4)
    forecasts = transfer.transfer(matches, ds.existing_skus, new_prices, existing_prices)

    df = forecasts_to_dataframe(forecasts)
    print("\n[Stage 3b complete]")
    print(df.head(20).to_string(index=False))
