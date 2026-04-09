"""
data_loader.py
==============
Stage 1: Load and preprocess the H&M Kaggle dataset.

Responsibilities:
- Load articles.csv, transactions_train.csv, customers.csv
- Aggregate transactions to weekly sales per article_id
- Simulate cold-start: hide first 4 weeks of 10% of SKUs as ground truth
- Split dataset into new SKUs (metadata only) and existing SKUs (full history)
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class HMDataset:
    """Container for all processed H&M data."""

    articles: pd.DataFrame
    """Article metadata (one row per article_id)."""

    customers: pd.DataFrame
    """Customer metadata (one row per customer_id)."""

    weekly_sales: pd.DataFrame
    """Weekly sales aggregated to (article_id, week_id) → units_sold."""

    existing_skus: pd.DataFrame
    """Weekly sales for SKUs with full history (training pool for analogues)."""

    new_skus: pd.DataFrame
    """Metadata-only rows for simulated new SKUs (cold-start)."""

    new_sku_ground_truth: pd.DataFrame
    """Hidden first-4-week sales for new SKUs (evaluation target)."""

    new_sku_ids: list[str] = field(default_factory=list)
    """article_ids selected as simulated new SKUs."""

    week_index: pd.Index = field(default_factory=pd.Index)
    """Sorted unique week labels across the full dataset."""


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

class HMDataLoader:
    """
    Loads and preprocesses the H&M Fashion Recommendations dataset.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing articles.csv, transactions_train.csv,
        customers.csv (and optionally sample_submission.csv).
    cold_start_fraction : float
        Fraction of SKUs to designate as 'new' (default 0.10).
    cold_start_weeks : int
        Number of launch weeks to hide as ground truth (default 4).
    min_history_weeks : int
        Minimum weeks of sales history an existing SKU must have (default 8).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        cold_start_fraction: float = 0.10,
        cold_start_weeks: int = 4,
        min_history_weeks: int = 8,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.cold_start_fraction = cold_start_fraction
        self.cold_start_weeks = cold_start_weeks
        self.min_history_weeks = min_history_weeks
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> HMDataset:
        """
        Full pipeline: read CSVs → aggregate → simulate cold-start → split.

        Returns
        -------
        HMDataset
            Populated data container ready for feature engineering.
        """
        print("[Stage 1] Loading H&M dataset ...")
        articles = self._load_articles()
        customers = self._load_customers()
        transactions = self._load_transactions()

        print("[Stage 1] Aggregating to weekly sales ...")
        weekly_sales = self._aggregate_weekly(transactions)

        week_index = pd.Index(sorted(weekly_sales["week_id"].unique()), name="week_id")
        print(f"[Stage 1] Dataset spans {len(week_index)} weeks "
              f"({week_index.min()} – {week_index.max()})")

        print("[Stage 1] Simulating cold-start split ...")
        new_sku_ids, ground_truth, existing_sales = self._cold_start_split(
            weekly_sales, week_index
        )

        new_skus = articles.loc[articles["article_id"].isin(new_sku_ids)].copy()
        existing_skus = existing_sales.copy()

        print(f"[Stage 1] New SKUs (cold-start): {len(new_sku_ids)}")
        print(f"[Stage 1] Existing SKUs (analogue pool): "
              f"{existing_skus['article_id'].nunique()}")

        return HMDataset(
            articles=articles,
            customers=customers,
            weekly_sales=weekly_sales,
            existing_skus=existing_skus,
            new_skus=new_skus,
            new_sku_ground_truth=ground_truth,
            new_sku_ids=new_sku_ids,
            week_index=week_index,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_articles(self) -> pd.DataFrame:
        path = self.data_dir / "articles.csv"
        self._check_file(path)
        print(f"  Reading {path} ...")
        df = pd.read_csv(path, dtype={"article_id": str})
        df["article_id"] = df["article_id"].str.zfill(10)
        print(f"  Loaded {len(df):,} articles with {len(df.columns)} columns.")
        return df

    def _load_customers(self) -> pd.DataFrame:
        path = self.data_dir / "customers.csv"
        self._check_file(path)
        print(f"  Reading {path} ...")
        df = pd.read_csv(path)
        print(f"  Loaded {len(df):,} customers.")
        return df

    def _load_transactions(self) -> pd.DataFrame:
        path = self.data_dir / "transactions_train.csv"
        self._check_file(path)
        print(f"  Reading {path} (this may take a moment) ...")
        df = pd.read_csv(
            path,
            dtype={"article_id": str, "customer_id": str},
            parse_dates=["t_dat"],
        )
        df["article_id"] = df["article_id"].str.zfill(10)
        print(f"  Loaded {len(df):,} transactions.")
        return df

    def _aggregate_weekly(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate raw transactions to weekly unit sales per article.

        Week IDs are ISO year-week strings (e.g. '2019-W36').
        """
        transactions = transactions.copy()
        transactions["week_id"] = (
            transactions["t_dat"].dt.isocalendar().year.astype(str)
            + "-W"
            + transactions["t_dat"].dt.isocalendar().week.astype(str).str.zfill(2)
        )

        weekly = (
            transactions.groupby(["article_id", "week_id"], as_index=False)
            .agg(units_sold=("price", "count"), mean_price=("price", "mean"))
        )
        print(f"  Weekly sales shape: {weekly.shape}")
        return weekly

    def _cold_start_split(
        self,
        weekly_sales: pd.DataFrame,
        week_index: pd.Index,
    ) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
        """
        Simulate cold-start by hiding first `cold_start_weeks` of selected SKUs.

        Selection criteria:
        - SKU must have at least `cold_start_weeks + min_history_weeks` total weeks
          so the hidden window has a real 'history' to learn from.

        Returns
        -------
        new_sku_ids : list[str]
        ground_truth : pd.DataFrame  — hidden weeks for new SKUs
        existing_sales : pd.DataFrame — full history for remaining SKUs
        """
        # Compute per-SKU stats
        sku_stats = (
            weekly_sales.groupby("article_id")["week_id"]
            .agg(["count", "min"])
            .rename(columns={"count": "n_weeks", "min": "first_week"})
            .reset_index()
        )

        # Eligible: enough history to survive hiding first N weeks
        min_weeks = self.cold_start_weeks + self.min_history_weeks
        eligible = sku_stats.loc[sku_stats["n_weeks"] >= min_weeks, "article_id"].tolist()

        n_new = max(1, int(len(eligible) * self.cold_start_fraction))
        new_sku_ids: list[str] = random.sample(eligible, n_new)
        new_sku_set = set(new_sku_ids)

        # Determine each new SKU's first N weeks (by calendar order)
        new_sku_sales = weekly_sales[weekly_sales["article_id"].isin(new_sku_set)].copy()

        hidden_rows = []
        remaining_rows = []

        for article_id, grp in tqdm(
            new_sku_sales.groupby("article_id"),
            desc="  Cold-start split",
            unit="SKU",
        ):
            grp_sorted = grp.sort_values("week_id")
            cutoff = grp_sorted["week_id"].iloc[self.cold_start_weeks]
            mask_hidden = grp_sorted["week_id"] < cutoff
            hidden_rows.append(grp_sorted[mask_hidden])
            # After the hidden window they join the visible pool
            remaining_rows.append(grp_sorted[~mask_hidden])

        ground_truth = pd.concat(hidden_rows, ignore_index=True)
        visible_new = pd.concat(remaining_rows, ignore_index=True)

        # Existing SKUs: all SKUs NOT in new_sku_set
        existing_other = weekly_sales[~weekly_sales["article_id"].isin(new_sku_set)]
        existing_sales = pd.concat([existing_other, visible_new], ignore_index=True)

        return new_sku_ids, ground_truth, existing_sales

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _check_file(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Required data file not found: {path}\n"
                "Download the H&M dataset from Kaggle and place CSVs in the data/ directory."
            )


# ---------------------------------------------------------------------------
# Convenience helpers used by downstream modules
# ---------------------------------------------------------------------------

def build_sku_weekly_pivot(
    weekly_sales: pd.DataFrame,
    article_ids: Optional[list[str]] = None,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    Pivot weekly_sales into a wide matrix: rows=article_id, cols=week_id.

    Parameters
    ----------
    weekly_sales : pd.DataFrame
        Long-format weekly sales (article_id, week_id, units_sold).
    article_ids : list[str] | None
        If provided, restrict to these SKUs.
    fill_value : float
        Value for missing (week, SKU) combinations (default 0).

    Returns
    -------
    pd.DataFrame
        Shape (n_skus, n_weeks) with week_ids as column names sorted chronologically.
    """
    if article_ids is not None:
        weekly_sales = weekly_sales[weekly_sales["article_id"].isin(article_ids)]

    pivot = weekly_sales.pivot_table(
        index="article_id",
        columns="week_id",
        values="units_sold",
        aggfunc="sum",
        fill_value=fill_value,
    )
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    return pivot


def get_sku_mean_price(weekly_sales: pd.DataFrame) -> pd.Series:
    """
    Compute the mean price per SKU across all observed weeks.

    Returns
    -------
    pd.Series
        Index = article_id, values = mean price.
    """
    return weekly_sales.groupby("article_id")["mean_price"].mean()


if __name__ == "__main__":
    loader = HMDataLoader(data_dir="data")
    dataset = loader.load()
    print("\n[Stage 1 complete]")
    print(f"  weekly_sales shape : {dataset.weekly_sales.shape}")
    print(f"  new_sku_ids sample : {dataset.new_sku_ids[:5]}")
    print(f"  ground_truth shape : {dataset.new_sku_ground_truth.shape}")
