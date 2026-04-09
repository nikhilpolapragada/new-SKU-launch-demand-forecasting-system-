"""
analogue_matcher.py
===================
Stage 3: Find the top-k most similar existing SKUs for each new SKU.

Method:
- L2-normalise feature vectors
- Cosine similarity matrix = normalised_new @ normalised_existing.T
- Top-k selection with ReLU-clipped softmax-style weights:
    w_i = max(0, sim_i) / sum_j max(0, sim_j)

Supports k ∈ {3, 5, 10} (or any positive integer).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class AnalogueMatch:
    """Stores analogue results for a single new SKU."""

    new_sku_id: str
    """The cold-start SKU being matched."""

    analogue_ids: list[str]
    """Top-k existing SKU article_ids, ordered by similarity (desc)."""

    similarities: np.ndarray
    """Cosine similarity scores for each analogue."""

    weights: np.ndarray
    """Non-negative weights summing to 1 (or 0 if all similarities ≤ 0)."""


# ---------------------------------------------------------------------------
# Matcher
# ---------------------------------------------------------------------------

class AnalogueMatcher:
    """
    Cosine-similarity-based analogue finder.

    Parameters
    ----------
    k : int
        Number of analogues to return per new SKU (default 5).
    """

    def __init__(self, k: int = 5) -> None:
        if k < 1:
            raise ValueError(f"k must be ≥ 1, got {k}.")
        self.k = k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, existing_features: pd.DataFrame) -> "AnalogueMatcher":
        """
        Store and L2-normalise existing SKU feature vectors.

        Parameters
        ----------
        existing_features : pd.DataFrame
            Feature matrix with article_id as index, numeric columns only.

        Returns
        -------
        self
        """
        self._existing_ids = np.array(existing_features.index.tolist())
        raw = existing_features.values.astype(np.float64)
        self._existing_normed = self._l2_normalise(raw)
        print(f"[Stage 3] AnalogueMatcher fitted on {len(self._existing_ids):,} "
              f"existing SKUs (k={self.k}).")
        return self

    def match(self, new_features: pd.DataFrame) -> list[AnalogueMatch]:
        """
        Find top-k analogues for each new SKU.

        Parameters
        ----------
        new_features : pd.DataFrame
            Feature matrix for new (cold-start) SKUs, same columns as fit().

        Returns
        -------
        list[AnalogueMatch]
            One entry per new SKU.
        """
        self._check_fitted()
        new_ids = np.array(new_features.index.tolist())
        raw_new = new_features.values.astype(np.float64)
        new_normed = self._l2_normalise(raw_new)

        # Full cosine similarity matrix: (n_new × n_existing)
        sim_matrix = new_normed @ self._existing_normed.T  # shape (n_new, n_existing)

        matches: list[AnalogueMatch] = []
        k_eff = min(self.k, len(self._existing_ids))

        for i, new_id in enumerate(tqdm(new_ids, desc="  Matching analogues", unit="SKU")):
            sims = sim_matrix[i]                            # (n_existing,)
            top_idx = np.argpartition(sims, -k_eff)[-k_eff:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]  # sort desc

            top_sims = sims[top_idx]
            top_ids = self._existing_ids[top_idx].tolist()
            weights = self._compute_weights(top_sims)

            matches.append(AnalogueMatch(
                new_sku_id=new_id,
                analogue_ids=top_ids,
                similarities=top_sims,
                weights=weights,
            ))

        return matches

    def match_multi_k(
        self,
        new_features: pd.DataFrame,
        k_values: list[int] = [3, 5, 10],
    ) -> dict[int, list[AnalogueMatch]]:
        """
        Run match() for multiple k values efficiently (single similarity pass).

        Parameters
        ----------
        new_features : pd.DataFrame
            Feature matrix for new SKUs.
        k_values : list[int]
            Values of k to evaluate (default [3, 5, 10]).

        Returns
        -------
        dict[int, list[AnalogueMatch]]
            Maps k → list of AnalogueMatch results.
        """
        self._check_fitted()
        new_ids = np.array(new_features.index.tolist())
        raw_new = new_features.values.astype(np.float64)
        new_normed = self._l2_normalise(raw_new)
        sim_matrix = new_normed @ self._existing_normed.T

        max_k = min(max(k_values), len(self._existing_ids))
        results: dict[int, list[AnalogueMatch]] = {k: [] for k in k_values}

        print(f"[Stage 3] Running multi-k matching: k ∈ {k_values} ...")
        for i, new_id in enumerate(tqdm(new_ids, desc="  Multi-k matching", unit="SKU")):
            sims = sim_matrix[i]
            top_idx_all = np.argpartition(sims, -max_k)[-max_k:]
            top_idx_all = top_idx_all[np.argsort(sims[top_idx_all])[::-1]]

            for k in k_values:
                k_eff = min(k, len(top_idx_all))
                top_idx = top_idx_all[:k_eff]
                top_sims = sims[top_idx]
                top_ids = self._existing_ids[top_idx].tolist()
                weights = self._compute_weights(top_sims)
                results[k].append(AnalogueMatch(
                    new_sku_id=new_id,
                    analogue_ids=top_ids,
                    similarities=top_sims,
                    weights=weights,
                ))

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _l2_normalise(X: np.ndarray) -> np.ndarray:
        """Row-wise L2 normalisation; zero vectors stay zero."""
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return X / norms

    @staticmethod
    def _compute_weights(similarities: np.ndarray) -> np.ndarray:
        """
        Compute ReLU-clipped weights from similarities.

        w_i = max(0, sim_i) / sum_j max(0, sim_j)

        If all similarities are non-positive, returns uniform weights.
        """
        clipped = np.maximum(0.0, similarities)
        total = clipped.sum()
        if total == 0.0:
            return np.ones(len(similarities)) / len(similarities)
        return clipped / total

    def _check_fitted(self) -> None:
        if not hasattr(self, "_existing_ids"):
            raise RuntimeError("Call fit() before match().")


# ---------------------------------------------------------------------------
# Utility: build similarity DataFrame
# ---------------------------------------------------------------------------

def matches_to_dataframe(matches: list[AnalogueMatch]) -> pd.DataFrame:
    """
    Flatten a list of AnalogueMatch objects into a long DataFrame.

    Columns: new_sku_id, analogue_id, similarity, weight, rank
    """
    rows = []
    for m in matches:
        for rank, (aid, sim, w) in enumerate(
            zip(m.analogue_ids, m.similarities, m.weights), start=1
        ):
            rows.append({
                "new_sku_id": m.new_sku_id,
                "analogue_id": aid,
                "similarity": float(sim),
                "weight": float(w),
                "rank": rank,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data_loader import HMDataLoader
    from feature_engineering import get_feature_matrix

    loader = HMDataLoader(data_dir="data")
    ds = loader.load()

    existing_ids = ds.existing_skus["article_id"].unique().tolist()
    feature_matrix, _ = get_feature_matrix(ds.articles, ds.weekly_sales, existing_ids)

    existing_features = feature_matrix.loc[feature_matrix.index.isin(existing_ids)]
    new_features = feature_matrix.loc[feature_matrix.index.isin(ds.new_sku_ids)]

    matcher = AnalogueMatcher(k=5)
    matcher.fit(existing_features)
    matches = matcher.match(new_features)

    df = matches_to_dataframe(matches)
    print("\n[Stage 3 complete]")
    print(df.head(15).to_string(index=False))
