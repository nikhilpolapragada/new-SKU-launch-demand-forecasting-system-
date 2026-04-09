"""
evaluation.py
=============
Stage 5: Evaluate forecasts against baselines and produce plots.

Benchmarks:
- Naive: category average demand for the launch window.
- Random analogue: randomly selected existing SKU's launch curve.
- Best-possible (oracle k=1): the single analogue with highest cosine similarity.

Metrics computed:
- RMSE  (Root Mean Squared Error)
- MAE   (Mean Absolute Error)
- MASE  (Mean Absolute Scaled Error, scaled by in-sample naive MAE)
- Bias  (mean signed error: forecast − actual)

Plots (saved to results/):
1. actual_vs_predicted.png
2. bayesian_rmse_improvement.png
3. benchmark_comparison.png
4. category_rmse_heatmap.png
5. similarity_distribution.png
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for scripts
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from analogue_matcher import AnalogueMatch, AnalogueMatcher, matches_to_dataframe
from bayesian_updater import BayesianUpdate, updates_to_dataframe
from demand_transfer import TransferredForecast, forecasts_to_dataframe


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")
PLOT_STYLE = "seaborn-v0_8-whitegrid"
FIGSIZE = (10, 6)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    naive_scale: float,
) -> float:
    """
    Mean Absolute Scaled Error.

    MASE = MAE(model) / naive_scale
    where naive_scale is the MAE of the naive (category mean) forecast.
    """
    if naive_scale == 0:
        return float("nan")
    return mae(actual, predicted) / naive_scale


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean signed error (positive = over-forecast)."""
    return float(np.mean(predicted - actual))


def compute_all_metrics(
    actual: np.ndarray, predicted: np.ndarray, naive_scale: float
) -> dict[str, float]:
    return {
        "RMSE": rmse(actual, predicted),
        "MAE": mae(actual, predicted),
        "MASE": mase(actual, predicted, naive_scale),
        "Bias": bias(actual, predicted),
    }


# ---------------------------------------------------------------------------
# Baseline builders
# ---------------------------------------------------------------------------

class BaselineForecaster:
    """
    Generates three baseline forecasts.

    Parameters
    ----------
    cold_start_weeks : int
    seed : int
    """

    def __init__(self, cold_start_weeks: int = 4, seed: int = 42) -> None:
        self.cold_start_weeks = cold_start_weeks
        self.rng = np.random.default_rng(seed)

    def fit(
        self,
        existing_sales: pd.DataFrame,
        articles: pd.DataFrame,
    ) -> "BaselineForecaster":
        """
        Compute category average launch curves from existing SKUs.

        Parameters
        ----------
        existing_sales : pd.DataFrame
            Long-format weekly sales for existing SKUs.
        articles : pd.DataFrame
            Article metadata (must contain article_id, product_type_name).
        """
        # Map article_id → product_type_name
        type_map = articles.set_index("article_id")["product_type_name"].to_dict()

        # Build per-SKU launch curves
        launch_curves: dict[str, np.ndarray] = {}
        for aid, grp in existing_sales.groupby("article_id"):
            grp_sorted = grp.sort_values("week_id")
            units = grp_sorted["units_sold"].values[: self.cold_start_weeks].astype(float)
            if len(units) < self.cold_start_weeks:
                units = np.pad(units, (0, self.cold_start_weeks - len(units)), constant_values=0.0)
            launch_curves[str(aid)] = units

        # Category average = mean launch curve per product_type_name
        category_curves: dict[str, np.ndarray] = {}
        type_to_aids: dict[str, list[str]] = {}
        for aid, curve in launch_curves.items():
            pt = type_map.get(aid, "Unknown")
            type_to_aids.setdefault(pt, []).append(aid)

        for pt, aids in type_to_aids.items():
            stacked = np.vstack([launch_curves[a] for a in aids])
            category_curves[pt] = stacked.mean(axis=0)

        # Global fallback
        all_curves = np.vstack(list(launch_curves.values()))
        self._global_avg = all_curves.mean(axis=0)
        self._category_curves = category_curves
        self._launch_curves = launch_curves
        self._existing_ids = list(launch_curves.keys())
        self._type_map = type_map

        print(f"[Eval] BaselineForecaster fitted: {len(self._existing_ids)} existing SKUs, "
              f"{len(category_curves)} categories.")
        return self

    def naive_forecast(self, new_sku_id: str, product_type: Optional[str] = None) -> np.ndarray:
        """Category average launch curve (or global average if category unknown)."""
        curve = self._category_curves.get(product_type, self._global_avg)
        return curve.copy()

    def random_analogue_forecast(self) -> np.ndarray:
        """Randomly selected existing SKU's launch curve."""
        aid = str(self.rng.choice(self._existing_ids))
        return self._launch_curves[aid].copy()

    def best_possible_forecast(self, match: AnalogueMatch) -> np.ndarray:
        """Oracle k=1: use only the best-matching analogue (highest similarity)."""
        if not match.analogue_ids:
            return self._global_avg.copy()
        best_aid = match.analogue_ids[0]  # rank 1 = highest similarity
        return self._launch_curves.get(best_aid, self._global_avg).copy()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Comprehensive evaluation of the analogue demand forecasting system.

    Parameters
    ----------
    cold_start_weeks : int
    results_dir : str | Path
    """

    def __init__(
        self,
        cold_start_weeks: int = 4,
        results_dir: str | Path = RESULTS_DIR,
    ) -> None:
        self.cold_start_weeks = cold_start_weeks
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Full evaluation pipeline
    # ------------------------------------------------------------------

    def evaluate(
        self,
        forecasts: list[TransferredForecast],
        updates: list[BayesianUpdate],
        matches: list[AnalogueMatch],
        ground_truth_df: pd.DataFrame,
        baseline: BaselineForecaster,
        articles: pd.DataFrame,
        k_results: Optional[dict[int, list[AnalogueMatch]]] = None,
    ) -> pd.DataFrame:
        """
        Run full evaluation and produce all plots.

        Parameters
        ----------
        forecasts : list[TransferredForecast]
        updates : list[BayesianUpdate]
        matches : list[AnalogueMatch]
        ground_truth_df : pd.DataFrame
        baseline : BaselineForecaster
        articles : pd.DataFrame
        k_results : dict[int, list[AnalogueMatch]] | None
            Multi-k matching results for benchmark comparison.

        Returns
        -------
        pd.DataFrame
            Per-SKU metric summary table.
        """
        print("\n[Stage 5] Running evaluation ...")

        gt_lookup = self._build_gt_lookup(ground_truth_df)
        type_map = articles.set_index("article_id")["product_type_name"].to_dict()
        match_lookup = {m.new_sku_id: m for m in matches}

        rows = []
        for f, u in tqdm(
            zip(forecasts, updates), total=len(forecasts),
            desc="  Computing metrics", unit="SKU"
        ):
            gt = gt_lookup.get(f.new_sku_id, np.zeros(self.cold_start_weeks))
            pt = type_map.get(f.new_sku_id, "Unknown")
            match = match_lookup.get(f.new_sku_id)

            naive_fc = baseline.naive_forecast(f.new_sku_id, pt)
            naive_mae_scale = mae(gt, naive_fc)

            row = {
                "new_sku_id": f.new_sku_id,
                "product_type": pt,
                "price_scale": f.price_scale,
                "mean_analogue_price": f.mean_analogue_price,
                "new_sku_price": f.new_sku_price,
                "top1_similarity": float(match.similarities[0]) if match else np.nan,
            }

            # Analogue model (prior)
            for metric, val in compute_all_metrics(gt, f.forecast_curve, naive_mae_scale).items():
                row[f"analogue_{metric}"] = val

            # Bayesian posterior
            for metric, val in compute_all_metrics(gt, u.posterior_means, naive_mae_scale).items():
                row[f"bayesian_{metric}"] = val

            # Naive baseline
            for metric, val in compute_all_metrics(gt, naive_fc, naive_mae_scale).items():
                row[f"naive_{metric}"] = val

            # Random analogue baseline
            rand_fc = baseline.random_analogue_forecast()
            for metric, val in compute_all_metrics(gt, rand_fc, naive_mae_scale).items():
                row[f"random_{metric}"] = val

            # Best-possible (oracle k=1)
            if match:
                best_fc = baseline.best_possible_forecast(match)
                for metric, val in compute_all_metrics(gt, best_fc, naive_mae_scale).items():
                    row[f"oracle_{metric}"] = val

            rows.append(row)

        summary = pd.DataFrame(rows)
        summary.to_csv(self.results_dir / "metrics_per_sku.csv", index=False)
        print(f"  Saved metrics_per_sku.csv ({len(summary)} rows)")

        # Print aggregate table
        self._print_aggregate(summary)

        # Plots
        print("\n[Stage 5] Generating plots ...")
        self._plot_actual_vs_predicted(summary, updates, gt_lookup)
        self._plot_bayesian_rmse_improvement(updates)
        self._plot_benchmark_comparison(summary, k_results, gt_lookup, forecasts)
        self._plot_category_rmse_heatmap(summary)
        self._plot_similarity_distribution(matches)

        print(f"\n[Stage 5] All plots saved to {self.results_dir}/")
        return summary

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _plot_actual_vs_predicted(
        self,
        summary: pd.DataFrame,
        updates: list[BayesianUpdate],
        gt_lookup: dict[str, np.ndarray],
    ) -> None:
        """Plot 1: Actual vs. Predicted scatter for all SKU-weeks."""
        all_actual, all_prior, all_bayes = [], [], []
        for u in updates:
            gt = gt_lookup.get(u.new_sku_id, np.zeros(len(u.prior_mean)))
            all_actual.extend(gt.tolist())
            all_prior.extend(u.prior_mean.tolist())
            all_bayes.extend(u.posterior_means.tolist())

        all_actual = np.array(all_actual)
        all_prior = np.array(all_prior)
        all_bayes = np.array(all_bayes)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Actual vs. Predicted Weekly Demand (all SKU-weeks)", fontsize=14)

        lim = max(all_actual.max(), all_prior.max(), all_bayes.max()) * 1.05

        for ax, pred, label, colour in zip(
            axes,
            [all_prior, all_bayes],
            ["Analogue Prior", "Bayesian Posterior"],
            ["steelblue", "darkorange"],
        ):
            ax.scatter(all_actual, pred, alpha=0.3, s=12, color=colour, edgecolors="none")
            ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="Perfect forecast")
            ax.set_xlabel("Actual units sold")
            ax.set_ylabel("Predicted units sold")
            ax.set_title(label)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.legend()
            rmse_val = rmse(all_actual, pred)
            ax.text(0.05, 0.92, f"RMSE = {rmse_val:.3f}", transform=ax.transAxes,
                    fontsize=10, color=colour)

        plt.tight_layout()
        plt.savefig(self.results_dir / "actual_vs_predicted.png", dpi=150)
        plt.close()
        print("  actual_vs_predicted.png saved.")

    def _plot_bayesian_rmse_improvement(self, updates: list[BayesianUpdate]) -> None:
        """Plot 2: RMSE at each Bayesian update step (mean ± 1 std)."""
        n_weeks = max(len(u.rmse_by_week) for u in updates)
        all_rmse = np.vstack([
            np.pad(u.rmse_by_week, (0, n_weeks - len(u.rmse_by_week)), constant_values=np.nan)
            for u in updates
        ])
        prior_rmses = np.array([u.prior_rmse for u in updates])

        mean_rmse = np.nanmean(all_rmse, axis=0)
        std_rmse = np.nanstd(all_rmse, axis=0)

        weeks = np.arange(1, n_weeks + 1)

        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.axhline(np.mean(prior_rmses), color="red", linestyle="--",
                   label=f"Prior RMSE = {np.mean(prior_rmses):.3f}", linewidth=1.5)
        ax.plot(weeks, mean_rmse, "o-", color="steelblue", label="Posterior RMSE (mean)")
        ax.fill_between(
            weeks,
            mean_rmse - std_rmse,
            mean_rmse + std_rmse,
            alpha=0.2, color="steelblue", label="±1 std"
        )
        ax.set_xlabel("Week of sequential update")
        ax.set_ylabel("RMSE")
        ax.set_title("Bayesian Update: RMSE Improvement Per Week")
        ax.legend()
        ax.set_xticks(weeks)
        plt.tight_layout()
        plt.savefig(self.results_dir / "bayesian_rmse_improvement.png", dpi=150)
        plt.close()
        print("  bayesian_rmse_improvement.png saved.")

    def _plot_benchmark_comparison(
        self,
        summary: pd.DataFrame,
        k_results: Optional[dict[int, list[AnalogueMatch]]],
        gt_lookup: dict[str, np.ndarray],
        forecasts: list[TransferredForecast],
    ) -> None:
        """Plot 3: Bar chart comparing RMSE across methods."""
        methods = {
            "Naive\n(category avg)": summary["naive_RMSE"].mean(),
            "Random\nanalogue": summary["random_RMSE"].mean(),
            "Analogue\n(prior)": summary["analogue_RMSE"].mean(),
            "Bayesian\n(posterior)": summary["bayesian_RMSE"].mean(),
        }
        if "oracle_RMSE" in summary.columns:
            methods["Oracle\n(k=1 best)"] = summary["oracle_RMSE"].mean()

        labels = list(methods.keys())
        values = list(methods.values())
        colours = ["#d62728", "#ff7f0e", "#1f77b4", "#2ca02c", "#9467bd"][:len(labels)]

        fig, ax = plt.subplots(figsize=FIGSIZE)
        bars = ax.bar(labels, values, color=colours, edgecolor="black", linewidth=0.6)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
        ax.set_ylabel("Mean RMSE")
        ax.set_title("Benchmark Comparison: Mean RMSE Across Methods")
        ax.set_ylim(0, max(values) * 1.2)
        plt.tight_layout()
        plt.savefig(self.results_dir / "benchmark_comparison.png", dpi=150)
        plt.close()
        print("  benchmark_comparison.png saved.")

    def _plot_category_rmse_heatmap(self, summary: pd.DataFrame) -> None:
        """Plot 4: RMSE heatmap by product category and method."""
        metric_cols = {
            "Naive": "naive_RMSE",
            "Random": "random_RMSE",
            "Analogue": "analogue_RMSE",
            "Bayesian": "bayesian_RMSE",
        }
        available = {k: v for k, v in metric_cols.items() if v in summary.columns}

        heatmap_data = (
            summary.groupby("product_type")[[v for v in available.values()]]
            .mean()
            .rename(columns={v: k for k, v in available.items()})
        )
        # Keep top 20 categories by frequency
        top_cats = summary["product_type"].value_counts().head(20).index
        heatmap_data = heatmap_data.loc[heatmap_data.index.isin(top_cats)]

        if heatmap_data.empty:
            print("  Skipping category RMSE heatmap (no data).")
            return

        fig, ax = plt.subplots(figsize=(12, max(6, len(heatmap_data) * 0.35)))
        sns.heatmap(
            heatmap_data,
            annot=True, fmt=".2f",
            cmap="YlOrRd",
            linewidths=0.5,
            ax=ax,
        )
        ax.set_title("Mean RMSE by Product Category and Method (top 20 categories)")
        ax.set_xlabel("Method")
        ax.set_ylabel("Product Type")
        plt.tight_layout()
        plt.savefig(self.results_dir / "category_rmse_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  category_rmse_heatmap.png saved.")

    def _plot_similarity_distribution(self, matches: list[AnalogueMatch]) -> None:
        """Plot 5: Distribution of top-1 cosine similarity scores."""
        top1_sims = [m.similarities[0] for m in matches if len(m.similarities) > 0]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Cosine Similarity Distribution (Top-1 Analogue)", fontsize=13)

        # Histogram
        axes[0].hist(top1_sims, bins=40, color="steelblue", edgecolor="black", linewidth=0.4)
        axes[0].axvline(np.mean(top1_sims), color="red", linestyle="--",
                        label=f"Mean = {np.mean(top1_sims):.3f}")
        axes[0].axvline(np.median(top1_sims), color="orange", linestyle="--",
                        label=f"Median = {np.median(top1_sims):.3f}")
        axes[0].set_xlabel("Cosine Similarity")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Histogram")
        axes[0].legend()

        # All-rank violin plot
        all_sims_by_rank = {}
        max_k = max(len(m.similarities) for m in matches)
        for rank in range(min(max_k, 10)):
            sims_at_rank = [
                float(m.similarities[rank]) for m in matches if len(m.similarities) > rank
            ]
            all_sims_by_rank[f"k={rank+1}"] = sims_at_rank

        parts = axes[1].violinplot(
            [all_sims_by_rank[k] for k in all_sims_by_rank],
            positions=range(1, len(all_sims_by_rank) + 1),
            showmedians=True,
        )
        for pc in parts["bodies"]:
            pc.set_facecolor("steelblue")
            pc.set_alpha(0.6)
        axes[1].set_xticks(range(1, len(all_sims_by_rank) + 1))
        axes[1].set_xticklabels(list(all_sims_by_rank.keys()), fontsize=9)
        axes[1].set_xlabel("Analogue Rank")
        axes[1].set_ylabel("Cosine Similarity")
        axes[1].set_title("Similarity by Rank (violin)")

        plt.tight_layout()
        plt.savefig(self.results_dir / "similarity_distribution.png", dpi=150)
        plt.close()
        print("  similarity_distribution.png saved.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_gt_lookup(self, ground_truth_df: pd.DataFrame) -> dict[str, np.ndarray]:
        lookup: dict[str, np.ndarray] = {}
        for aid, grp in ground_truth_df.groupby("article_id"):
            grp_sorted = grp.sort_values("week_id")
            units = grp_sorted["units_sold"].values[: self.cold_start_weeks].astype(float)
            if len(units) < self.cold_start_weeks:
                units = np.pad(units, (0, self.cold_start_weeks - len(units)), constant_values=0.0)
            lookup[str(aid)] = units
        return lookup

    @staticmethod
    def _print_aggregate(summary: pd.DataFrame) -> None:
        """Print a formatted aggregate metric table."""
        methods = ["naive", "random", "analogue", "bayesian"]
        if "oracle_RMSE" in summary.columns:
            methods.append("oracle")
        metrics = ["RMSE", "MAE", "MASE", "Bias"]

        header = f"{'Method':<14}" + "".join(f"{m:>10}" for m in metrics)
        print("\n" + "=" * len(header))
        print("Aggregate Metrics (mean over all new SKUs)")
        print("=" * len(header))
        print(header)
        print("-" * len(header))
        for method in methods:
            row = f"{method.capitalize():<14}"
            for metric in metrics:
                col = f"{method}_{metric}"
                val = summary[col].mean() if col in summary.columns else float("nan")
                row += f"{val:>10.4f}"
            print(row)
        print("=" * len(header))


if __name__ == "__main__":
    from data_loader import HMDataLoader, get_sku_mean_price
    from feature_engineering import get_feature_matrix
    from analogue_matcher import AnalogueMatcher
    from demand_transfer import DemandTransfer
    from bayesian_updater import GaussianConjugateUpdater

    loader = HMDataLoader(data_dir="data")
    ds = loader.load()

    existing_ids = ds.existing_skus["article_id"].unique().tolist()
    feature_matrix, _ = get_feature_matrix(ds.articles, ds.weekly_sales, existing_ids)

    existing_features = feature_matrix.loc[feature_matrix.index.isin(existing_ids)]
    new_features = feature_matrix.loc[feature_matrix.index.isin(ds.new_sku_ids)]

    matcher = AnalogueMatcher(k=5)
    matcher.fit(existing_features)
    matches = matcher.match(new_features)
    k_results = matcher.match_multi_k(new_features, k_values=[3, 5, 10])

    prices = get_sku_mean_price(ds.weekly_sales)
    transfer = DemandTransfer(cold_start_weeks=4)
    forecasts = transfer.transfer(
        matches, ds.existing_skus, prices.reindex(ds.new_sku_ids), prices.reindex(existing_ids)
    )

    updater = GaussianConjugateUpdater()
    updates = updater.update_all(forecasts, ds.new_sku_ground_truth, cold_start_weeks=4)

    baseline = BaselineForecaster(cold_start_weeks=4)
    baseline.fit(ds.existing_skus, ds.articles)

    evaluator = Evaluator(cold_start_weeks=4)
    summary = evaluator.evaluate(
        forecasts, updates, matches, ds.new_sku_ground_truth, baseline, ds.articles, k_results
    )

    print("\n[Stage 5 complete]")
    print(summary.head(5).to_string(index=False))
