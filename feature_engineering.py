"""
feature_engineering.py
=======================
Stage 2: Transform raw article metadata into numeric feature vectors.

Pipeline:
1. One-hot encode: product_type_name, colour_group_code, department_name,
                   index_group_name, garment_group_name
2. Min-max normalise: mean_price (computed from weekly_sales)
3. Seasonality: compute 52-week average unit sales index per product_type_name,
                then PCA-compress to 4 components per SKU

All transformers are fit on *existing* SKUs only, then applied to new SKUs
(no leakage from the cold-start test set).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORICAL_COLS = [
    "product_type_name",
    "colour_group_code",
    "department_name",
    "index_group_name",
    "garment_group_name",
]

N_SEASONALITY_PCA = 4   # PCA components for 52-week seasonality curve
N_SEASON_WEEKS = 52     # canonical number of weeks in a seasonal cycle


# ---------------------------------------------------------------------------
# Main engineer
# ---------------------------------------------------------------------------

class FeatureEngineer:
    """
    Builds a fixed-width numeric feature matrix for every SKU.

    Parameters
    ----------
    n_pca_components : int
        Number of PCA components for the seasonality block (default 4).
    season_weeks : int
        Length of the seasonal window (default 52).
    """

    def __init__(
        self,
        n_pca_components: int = N_SEASONALITY_PCA,
        season_weeks: int = N_SEASON_WEEKS,
    ) -> None:
        self.n_pca_components = n_pca_components
        self.season_weeks = season_weeks

        # Fit artefacts (populated during fit_transform)
        self._ohe_cols: list[str] = []
        self._price_scaler = MinMaxScaler()
        self._pca = PCA(n_components=n_pca_components, random_state=42)
        self._product_type_seasonality: dict[str, np.ndarray] = {}
        self._fitted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        articles: pd.DataFrame,
        weekly_sales: pd.DataFrame,
        existing_article_ids: list[str],
    ) -> pd.DataFrame:
        """
        Fit on existing SKUs and transform all SKUs.

        Parameters
        ----------
        articles : pd.DataFrame
            Full articles metadata.
        weekly_sales : pd.DataFrame
            Long-format weekly sales (existing SKUs only for fitting).
        existing_article_ids : list[str]
            article_ids belonging to the analogue pool (fit set).

        Returns
        -------
        pd.DataFrame
            Feature matrix indexed by article_id.
            Columns: ohe_* | price_normalised | season_pc_0..N-1
        """
        print("[Stage 2] Fitting feature engineer on existing SKUs ...")
        existing_mask = articles["article_id"].isin(existing_article_ids)
        existing_articles = articles[existing_mask].copy()
        existing_sales = weekly_sales[
            weekly_sales["article_id"].isin(existing_article_ids)
        ].copy()

        # 1. One-hot encode
        print("  One-hot encoding categorical columns ...")
        ohe_df = self._fit_ohe(existing_articles, articles)

        # 2. Normalise price
        print("  Normalising mean price ...")
        price_df = self._fit_price(existing_sales, weekly_sales, articles["article_id"])

        # 3. Seasonality → PCA
        print("  Computing 52-week seasonality profiles → PCA ...")
        season_df = self._fit_seasonality_pca(
            existing_articles, existing_sales, articles
        )

        # 4. Concatenate
        feature_matrix = ohe_df.join(price_df, how="left").join(season_df, how="left")
        feature_matrix = feature_matrix.fillna(0.0)

        self._fitted = True
        print(f"  Feature matrix: {feature_matrix.shape[0]} SKUs × "
              f"{feature_matrix.shape[1]} features")
        return feature_matrix

    def transform(
        self,
        articles: pd.DataFrame,
        weekly_sales: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform a new set of SKUs using already-fitted transformers.

        Parameters
        ----------
        articles : pd.DataFrame
            Article rows to transform.
        weekly_sales : pd.DataFrame
            Sales for these articles (may be empty for true cold-start).

        Returns
        -------
        pd.DataFrame
            Feature matrix aligned to the same columns as fit_transform output.
        """
        if not self._fitted:
            raise RuntimeError("Call fit_transform before transform.")

        ohe_df = self._apply_ohe(articles)
        price_df = self._apply_price(weekly_sales, articles["article_id"])
        season_df = self._apply_seasonality_pca(articles)

        feature_matrix = ohe_df.join(price_df, how="left").join(season_df, how="left")
        feature_matrix = feature_matrix.fillna(0.0)
        return feature_matrix

    # ------------------------------------------------------------------
    # One-hot encoding
    # ------------------------------------------------------------------

    def _fit_ohe(
        self, existing_articles: pd.DataFrame, all_articles: pd.DataFrame
    ) -> pd.DataFrame:
        """Fit OHE on existing SKUs; apply to all SKUs."""
        all_articles = all_articles.set_index("article_id")
        existing_articles = existing_articles.set_index("article_id")

        frames: list[pd.DataFrame] = []
        self._ohe_vocab: dict[str, list[str]] = {}

        for col in CATEGORICAL_COLS:
            if col not in existing_articles.columns:
                print(f"  Warning: column '{col}' not found, skipping.")
                continue
            vocab = sorted(existing_articles[col].dropna().unique().tolist())
            self._ohe_vocab[col] = vocab

            encoded = pd.get_dummies(
                all_articles[col].astype(str), prefix=f"ohe_{col}", dtype=float
            )
            # Align columns to fitted vocabulary only
            expected_cols = [f"ohe_{col}_{v}" for v in vocab]
            encoded = encoded.reindex(columns=expected_cols, fill_value=0.0)
            frames.append(encoded)

        self._ohe_cols = [c for f in frames for c in f.columns]
        return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=all_articles.index)

    def _apply_ohe(self, articles: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted OHE to new articles."""
        articles = articles.set_index("article_id")
        frames: list[pd.DataFrame] = []
        for col, vocab in self._ohe_vocab.items():
            if col not in articles.columns:
                zeros = pd.DataFrame(
                    0.0,
                    index=articles.index,
                    columns=[f"ohe_{col}_{v}" for v in vocab],
                )
                frames.append(zeros)
                continue
            encoded = pd.get_dummies(
                articles[col].astype(str), prefix=f"ohe_{col}", dtype=float
            )
            expected_cols = [f"ohe_{col}_{v}" for v in vocab]
            encoded = encoded.reindex(columns=expected_cols, fill_value=0.0)
            frames.append(encoded)
        return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=articles.index)

    # ------------------------------------------------------------------
    # Price normalisation
    # ------------------------------------------------------------------

    def _fit_price(
        self,
        existing_sales: pd.DataFrame,
        all_sales: pd.DataFrame,
        all_article_ids: pd.Series,
    ) -> pd.DataFrame:
        mean_price = all_sales.groupby("article_id")["mean_price"].mean()
        # Fit scaler on existing only
        existing_prices = existing_sales.groupby("article_id")["mean_price"].mean().values.reshape(-1, 1)
        self._price_scaler.fit(existing_prices)
        return self._price_series_to_df(mean_price)

    def _apply_price(
        self, sales: pd.DataFrame, article_ids: pd.Series
    ) -> pd.DataFrame:
        mean_price = sales.groupby("article_id")["mean_price"].mean()
        return self._price_series_to_df(mean_price)

    def _price_series_to_df(self, mean_price: pd.Series) -> pd.DataFrame:
        prices_arr = mean_price.values.reshape(-1, 1)
        scaled = self._price_scaler.transform(
            np.clip(prices_arr, self._price_scaler.data_min_, self._price_scaler.data_max_)
        )
        return pd.DataFrame(
            {"price_normalised": scaled.ravel()}, index=mean_price.index
        )

    # ------------------------------------------------------------------
    # Seasonality → PCA
    # ------------------------------------------------------------------

    def _fit_seasonality_pca(
        self,
        existing_articles: pd.DataFrame,
        existing_sales: pd.DataFrame,
        all_articles: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Build 52-week seasonality index per product_type_name from existing SKUs.
        Then represent each SKU by its product type's seasonality curve.
        PCA is fit on those curves and applied to all SKUs.
        """
        # Map week_id to ISO week number (1-52)
        existing_sales = existing_sales.copy()
        existing_sales["iso_week"] = existing_sales["week_id"].apply(
            lambda w: int(w.split("-W")[1])
        ).clip(1, 52)

        # Merge product_type_name
        article_type = existing_articles.set_index("article_id")["product_type_name"]
        existing_sales = existing_sales.join(article_type, on="article_id")

        # Build (product_type, iso_week) average sales
        type_week_avg = (
            existing_sales.groupby(["product_type_name", "iso_week"])["units_sold"]
            .mean()
            .reset_index()
        )

        # Pivot to (product_type × 52) matrix; fill missing weeks with 0
        season_pivot = type_week_avg.pivot_table(
            index="product_type_name",
            columns="iso_week",
            values="units_sold",
            fill_value=0.0,
        )
        season_pivot = season_pivot.reindex(columns=range(1, 53), fill_value=0.0)

        # L2-normalise each row (per-type curve)
        norms = np.linalg.norm(season_pivot.values, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        season_normalised = season_pivot.values / norms

        # Fit PCA
        self._pca.fit(season_normalised)
        pca_types = self._pca.transform(season_normalised)
        explained = self._pca.explained_variance_ratio_.cumsum()[-1]
        print(f"  Seasonality PCA: {self.n_pca_components} components explain "
              f"{explained:.1%} of variance.")

        # Store per product_type PCA embedding
        self._product_type_seasonality = {
            pt: pca_types[i]
            for i, pt in enumerate(season_pivot.index)
        }
        # Global mean for unknown types
        self._seasonality_fallback = pca_types.mean(axis=0)

        return self._build_season_df(all_articles)

    def _apply_seasonality_pca(self, articles: pd.DataFrame) -> pd.DataFrame:
        return self._build_season_df(articles)

    def _build_season_df(self, articles: pd.DataFrame) -> pd.DataFrame:
        cols = [f"season_pc_{i}" for i in range(self.n_pca_components)]
        rows = []
        ids = articles["article_id"].tolist() if "article_id" in articles.columns else articles.index.tolist()
        types = (
            articles["product_type_name"].tolist()
            if "product_type_name" in articles.columns
            else [None] * len(ids)
        )
        for aid, pt in zip(ids, types):
            vec = self._product_type_seasonality.get(pt, self._seasonality_fallback)
            rows.append(vec)
        return pd.DataFrame(rows, index=ids, columns=cols)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_feature_matrix(
    articles: pd.DataFrame,
    weekly_sales: pd.DataFrame,
    existing_article_ids: list[str],
    n_pca_components: int = N_SEASONALITY_PCA,
) -> tuple[pd.DataFrame, FeatureEngineer]:
    """
    Convenience wrapper: fit + transform in one call.

    Returns
    -------
    feature_matrix : pd.DataFrame
        Rows = all article_ids, cols = numeric features.
    engineer : FeatureEngineer
        Fitted engineer (for later transform of truly new SKUs).
    """
    engineer = FeatureEngineer(n_pca_components=n_pca_components)
    feature_matrix = engineer.fit_transform(articles, weekly_sales, existing_article_ids)
    return feature_matrix, engineer


if __name__ == "__main__":
    from data_loader import HMDataLoader

    loader = HMDataLoader(data_dir="data")
    ds = loader.load()

    feature_matrix, eng = get_feature_matrix(
        ds.articles, ds.weekly_sales, ds.existing_skus["article_id"].unique().tolist()
    )
    print("\n[Stage 2 complete]")
    print(f"  Feature matrix: {feature_matrix.shape}")
    print(f"  Columns: {list(feature_matrix.columns[:10])} ...")
