"""
bayesian_updater.py
===================
Stage 4: Sequential Bayesian updating of demand forecasts.

Two modes:
1. Gaussian conjugate update (fast, analytic):
   - Prior: μ₀ = analogue forecast, σ₀ estimated from analogue variance.
   - Observation model: x_t ~ N(μ, σ_obs²).
   - Conjugate update: closed-form posterior after each week's sales.
   - Tracks RMSE at each update step.

2. PyMC MCMC demo (optional, slow):
   - Full hierarchical Bayesian model on 5 randomly selected new SKUs.
   - Samples posterior of a per-week demand level.
   - Reports HDI and posterior predictive RMSE.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from demand_transfer import TransferredForecast


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class BayesianUpdate:
    """Per-SKU Bayesian update trajectory."""

    new_sku_id: str
    prior_mean: np.ndarray           # Shape (cold_start_weeks,)
    prior_std: float
    posterior_means: np.ndarray      # Shape (cold_start_weeks,)  – final posterior
    posterior_stds: np.ndarray       # Shape (cold_start_weeks,)
    actual: np.ndarray               # Ground truth
    rmse_by_week: np.ndarray         # RMSE after each week's update (length = cold_start_weeks)
    prior_rmse: float                # RMSE before any update (baseline)


# ---------------------------------------------------------------------------
# Conjugate Gaussian updater
# ---------------------------------------------------------------------------

class GaussianConjugateUpdater:
    """
    Sequential Gaussian conjugate update of the demand forecast.

    Observation model:
        x_t | μ_t, σ_obs ~ Normal(μ_t, σ_obs²)

    Prior (analogue forecast):
        μ_t ~ Normal(μ₀_t, σ₀²)

    Conjugate posterior after observing x_t:
        μ_t | x_t  ~ Normal(μ_post_t, σ_post_t²)
        1/σ_post² = 1/σ₀² + 1/σ_obs²
        μ_post  = σ_post² * (μ₀/σ₀² + x/σ_obs²)

    Parameters
    ----------
    obs_noise_fraction : float
        σ_obs = obs_noise_fraction × max(μ₀, 1).  Scales observation noise
        relative to the prior mean magnitude (default 0.5).
    prior_std_fraction : float
        σ₀ = prior_std_fraction × max(|μ₀|.mean(), 1).
        Controls how confident we are in the analogue prior (default 0.8).
    """

    def __init__(
        self,
        obs_noise_fraction: float = 0.5,
        prior_std_fraction: float = 0.8,
    ) -> None:
        self.obs_noise_fraction = obs_noise_fraction
        self.prior_std_fraction = prior_std_fraction

    def update(
        self,
        forecast: TransferredForecast,
        ground_truth: np.ndarray,
    ) -> BayesianUpdate:
        """
        Run sequential conjugate update for a single SKU.

        Parameters
        ----------
        forecast : TransferredForecast
            Prior forecast from demand_transfer.
        ground_truth : np.ndarray
            Actual weekly sales, shape (cold_start_weeks,).

        Returns
        -------
        BayesianUpdate
        """
        prior_mean = forecast.forecast_curve.copy()
        n_weeks = len(prior_mean)

        # Prior variance (shared across weeks for simplicity)
        scale = max(np.abs(prior_mean).mean(), 1.0)
        prior_std = self.prior_std_fraction * scale
        prior_var = prior_std ** 2

        # Observation noise variance
        obs_std = self.obs_noise_fraction * scale
        obs_var = obs_std ** 2

        # Sequential update
        posterior_means = np.zeros(n_weeks)
        posterior_stds = np.zeros(n_weeks)
        rmse_by_week = np.zeros(n_weeks)

        mu = prior_mean.copy()
        var = np.full(n_weeks, prior_var)

        # Prior RMSE (no data yet)
        prior_rmse = float(np.sqrt(np.mean((prior_mean - ground_truth) ** 2)))

        for t in range(n_weeks):
            # Conjugate update for week t using observed sales
            x_t = ground_truth[t]

            # Posterior precision
            post_prec = 1.0 / var[t] + 1.0 / obs_var
            post_var_t = 1.0 / post_prec
            post_mean_t = post_var_t * (mu[t] / var[t] + x_t / obs_var)

            mu[t] = post_mean_t
            var[t] = post_var_t

            posterior_means[t] = post_mean_t
            posterior_stds[t] = np.sqrt(post_var_t)

            # RMSE of full posterior mean vector vs ground truth (after t+1 updates)
            rmse_by_week[t] = float(np.sqrt(np.mean((mu - ground_truth) ** 2)))

        return BayesianUpdate(
            new_sku_id=forecast.new_sku_id,
            prior_mean=prior_mean,
            prior_std=prior_std,
            posterior_means=posterior_means,
            posterior_stds=posterior_stds,
            actual=ground_truth,
            rmse_by_week=rmse_by_week,
            prior_rmse=prior_rmse,
        )

    def update_all(
        self,
        forecasts: list[TransferredForecast],
        ground_truth_df: pd.DataFrame,
        cold_start_weeks: int = 4,
    ) -> list[BayesianUpdate]:
        """
        Run conjugate updates for all new SKUs.

        Parameters
        ----------
        forecasts : list[TransferredForecast]
        ground_truth_df : pd.DataFrame
            Long-format ground truth (article_id, week_offset, units_sold).
        cold_start_weeks : int

        Returns
        -------
        list[BayesianUpdate]
        """
        print("[Stage 4] Running Gaussian conjugate Bayesian updates ...")

        # Build ground truth lookup: sku_id → array of shape (cold_start_weeks,)
        gt_lookup = self._build_gt_lookup(ground_truth_df, cold_start_weeks)

        updates: list[BayesianUpdate] = []
        for f in tqdm(forecasts, desc="  Bayesian updates", unit="SKU"):
            gt = gt_lookup.get(f.new_sku_id, np.zeros(cold_start_weeks))
            update = self.update(f, gt)
            updates.append(update)

        # Summary
        avg_prior_rmse = np.mean([u.prior_rmse for u in updates])
        avg_final_rmse = np.mean([u.rmse_by_week[-1] for u in updates])
        improvement = (avg_prior_rmse - avg_final_rmse) / max(avg_prior_rmse, 1e-9) * 100
        print(f"  Prior RMSE (avg):   {avg_prior_rmse:.4f}")
        print(f"  Final RMSE (avg):   {avg_final_rmse:.4f}")
        print(f"  RMSE improvement:   {improvement:.1f}%")

        return updates

    @staticmethod
    def _build_gt_lookup(
        ground_truth_df: pd.DataFrame, cold_start_weeks: int
    ) -> dict[str, np.ndarray]:
        """Build {article_id: units_sold_array} from long-format ground truth."""
        lookup: dict[str, np.ndarray] = {}
        for article_id, grp in ground_truth_df.groupby("article_id"):
            grp_sorted = grp.sort_values("week_id")
            units = grp_sorted["units_sold"].values[:cold_start_weeks].astype(float)
            if len(units) < cold_start_weeks:
                units = np.pad(
                    units, (0, cold_start_weeks - len(units)), constant_values=0.0
                )
            lookup[str(article_id)] = units
        return lookup


# ---------------------------------------------------------------------------
# PyMC MCMC demo (optional)
# ---------------------------------------------------------------------------

def run_pymc_demo(
    forecasts: list[TransferredForecast],
    ground_truth_df: pd.DataFrame,
    n_skus: int = 5,
    cold_start_weeks: int = 4,
    samples: int = 1000,
    chains: int = 2,
    seed: int = 42,
) -> Optional[pd.DataFrame]:
    """
    Run a PyMC hierarchical demand model on a random sample of new SKUs.

    Model:
        μ_t ~ Normal(prior_t, σ_prior)   [per-week demand level]
        σ_prior ~ HalfNormal(scale=5)
        x_t ~ Normal(μ_t, σ_obs)
        σ_obs ~ HalfNormal(scale=5)

    Parameters
    ----------
    forecasts : list[TransferredForecast]
    ground_truth_df : pd.DataFrame
    n_skus : int
        Number of SKUs to include in the demo (default 5).
    cold_start_weeks : int
    samples : int
        Number of MCMC draws per chain (default 1000).
    chains : int
        Number of MCMC chains (default 2).
    seed : int

    Returns
    -------
    pd.DataFrame | None
        Summary DataFrame with posterior mean and HDI per (sku, week),
        or None if PyMC is not available.
    """
    try:
        import pymc as pm
        import arviz as az
    except ImportError:
        warnings.warn(
            "PyMC or ArviZ not installed. Skipping MCMC demo.\n"
            "Install with: pip install pymc arviz"
        )
        return None

    print(f"\n[Stage 4 – MCMC demo] Running PyMC on {n_skus} SKUs ...")

    gt_lookup = GaussianConjugateUpdater._build_gt_lookup(
        ground_truth_df, cold_start_weeks
    )
    rng = np.random.default_rng(seed)
    subset = rng.choice(forecasts, size=min(n_skus, len(forecasts)), replace=False)

    summary_rows = []

    for f in tqdm(subset, desc="  MCMC per SKU"):
        observed = gt_lookup.get(f.new_sku_id, np.zeros(cold_start_weeks))
        prior_curve = f.forecast_curve

        with pm.Model() as model:
            sigma_prior = pm.HalfNormal("sigma_prior", sigma=5.0)
            sigma_obs = pm.HalfNormal("sigma_obs", sigma=5.0)

            # Per-week demand level, informed by analogue prior
            mu = pm.Normal(
                "mu",
                mu=prior_curve,
                sigma=sigma_prior,
                shape=cold_start_weeks,
            )

            # Likelihood
            _ = pm.Normal(
                "x_obs",
                mu=mu,
                sigma=sigma_obs,
                observed=observed,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                idata = pm.sample(
                    draws=samples,
                    chains=chains,
                    random_seed=seed,
                    progressbar=False,
                    return_inferencedata=True,
                )

        post_mu = idata.posterior["mu"].values  # (chains, draws, weeks)
        post_mu_flat = post_mu.reshape(-1, cold_start_weeks)  # (chains*draws, weeks)

        hdi = az.hdi(idata, var_names=["mu"], hdi_prob=0.94)["mu"].values  # (weeks, 2)

        posterior_mean = post_mu_flat.mean(axis=0)
        rmse = float(np.sqrt(np.mean((posterior_mean - observed) ** 2)))

        for t in range(cold_start_weeks):
            summary_rows.append({
                "sku_id": f.new_sku_id,
                "week_offset": t,
                "prior": float(prior_curve[t]),
                "actual": float(observed[t]),
                "posterior_mean": float(posterior_mean[t]),
                "hdi_low": float(hdi[t, 0]),
                "hdi_high": float(hdi[t, 1]),
                "rmse": rmse,
            })

    summary_df = pd.DataFrame(summary_rows)
    print(f"[Stage 4 – MCMC demo] Done. Posterior RMSE (avg): "
          f"{summary_df.groupby('sku_id')['rmse'].first().mean():.4f}")
    return summary_df


# ---------------------------------------------------------------------------
# Utility: results → DataFrame
# ---------------------------------------------------------------------------

def updates_to_dataframe(updates: list[BayesianUpdate]) -> pd.DataFrame:
    """
    Flatten BayesianUpdate list to a long DataFrame.

    Columns: new_sku_id, week_offset, prior_mean, posterior_mean,
             posterior_std, actual, rmse_after_update
    """
    rows = []
    for u in updates:
        for t in range(len(u.prior_mean)):
            rows.append({
                "new_sku_id": u.new_sku_id,
                "week_offset": t,
                "prior_mean": float(u.prior_mean[t]),
                "posterior_mean": float(u.posterior_means[t]),
                "posterior_std": float(u.posterior_stds[t]),
                "actual": float(u.actual[t]),
                "rmse_after_update": float(u.rmse_by_week[t]),
                "prior_rmse": float(u.prior_rmse),
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data_loader import HMDataLoader, get_sku_mean_price
    from feature_engineering import get_feature_matrix
    from analogue_matcher import AnalogueMatcher
    from demand_transfer import DemandTransfer

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
    forecasts = DemandTransfer(cold_start_weeks=4).transfer(
        matches, ds.existing_skus, prices.reindex(ds.new_sku_ids), prices.reindex(existing_ids)
    )

    updater = GaussianConjugateUpdater()
    updates = updater.update_all(forecasts, ds.new_sku_ground_truth, cold_start_weeks=4)

    df = updates_to_dataframe(updates)
    print("\n[Stage 4 complete – conjugate]")
    print(df.head(16).to_string(index=False))

    # Optional MCMC demo
    mcmc_summary = run_pymc_demo(forecasts, ds.new_sku_ground_truth, n_skus=5)
    if mcmc_summary is not None:
        print("\n[Stage 4 complete – MCMC]")
        print(mcmc_summary.head(20).to_string(index=False))
