"""
main.py
=======
Full CLI entry-point for the H&M analogue-based new-SKU demand forecasting system.

Usage examples:
    # Run full pipeline with defaults
    python main.py

    # Custom data directory and parameters
    python main.py --data-dir /path/to/hm-data --k 10 --elasticity 0.8

    # Skip MCMC demo (faster)
    python main.py --no-mcmc

    # Only run feature engineering + matching (no evaluation)
    python main.py --stages load,features,match

    # Multi-k benchmark comparison
    python main.py --k-values 3 5 10

Available stages: load, features, match, transfer, bayes, eval
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analogue-based demand forecasting for new SKU launches (H&M dataset)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    data_grp = parser.add_argument_group("Data")
    data_grp.add_argument(
        "--data-dir", type=str, default="data",
        help="Directory containing articles.csv, transactions_train.csv, customers.csv"
    )
    data_grp.add_argument(
        "--results-dir", type=str, default="results",
        help="Directory to save plots and CSV outputs"
    )

    # Cold-start simulation
    cs_grp = parser.add_argument_group("Cold-start simulation")
    cs_grp.add_argument(
        "--cold-start-fraction", type=float, default=0.10,
        help="Fraction of SKUs to treat as newly launched (cold-start)"
    )
    cs_grp.add_argument(
        "--cold-start-weeks", type=int, default=4,
        help="Number of launch weeks to hide and forecast"
    )
    cs_grp.add_argument(
        "--min-history-weeks", type=int, default=8,
        help="Minimum history weeks required for a SKU to join the analogue pool"
    )

    # Feature engineering
    fe_grp = parser.add_argument_group("Feature engineering")
    fe_grp.add_argument(
        "--n-pca-components", type=int, default=4,
        help="Number of PCA components for the 52-week seasonality block"
    )

    # Analogue matching
    match_grp = parser.add_argument_group("Analogue matching")
    match_grp.add_argument(
        "--k", type=int, default=5,
        help="Number of analogues per new SKU (primary k)"
    )
    match_grp.add_argument(
        "--k-values", type=int, nargs="+", default=[3, 5, 10],
        help="Values of k to compare in multi-k benchmark"
    )

    # Demand transfer
    dt_grp = parser.add_argument_group("Demand transfer")
    dt_grp.add_argument(
        "--elasticity", type=float, default=1.0,
        help="Price-elasticity exponent for demand scaling"
    )
    dt_grp.add_argument(
        "--scale-min", type=float, default=0.1,
        help="Minimum price-ratio scale factor"
    )
    dt_grp.add_argument(
        "--scale-max", type=float, default=10.0,
        help="Maximum price-ratio scale factor"
    )

    # Bayesian
    bayes_grp = parser.add_argument_group("Bayesian updater")
    bayes_grp.add_argument(
        "--obs-noise", type=float, default=0.5,
        help="Observation noise fraction (σ_obs = frac × mean demand)"
    )
    bayes_grp.add_argument(
        "--prior-std", type=float, default=0.8,
        help="Prior std fraction (σ_prior = frac × mean demand)"
    )
    bayes_grp.add_argument(
        "--no-mcmc", action="store_true",
        help="Skip the optional PyMC MCMC demo"
    )
    bayes_grp.add_argument(
        "--mcmc-skus", type=int, default=5,
        help="Number of SKUs to include in the MCMC demo"
    )
    bayes_grp.add_argument(
        "--mcmc-samples", type=int, default=1000,
        help="MCMC draws per chain"
    )
    bayes_grp.add_argument(
        "--mcmc-chains", type=int, default=2,
        help="Number of MCMC chains"
    )

    # Pipeline control
    pipe_grp = parser.add_argument_group("Pipeline control")
    pipe_grp.add_argument(
        "--stages", type=str, default="load,features,match,transfer,bayes,eval",
        help="Comma-separated list of stages to run"
    )
    pipe_grp.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed"
    )

    return parser


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def stage_load(args: argparse.Namespace):
    from data_loader import HMDataLoader
    loader = HMDataLoader(
        data_dir=args.data_dir,
        cold_start_fraction=args.cold_start_fraction,
        cold_start_weeks=args.cold_start_weeks,
        min_history_weeks=args.min_history_weeks,
        seed=args.seed,
    )
    return loader.load()


def stage_features(args: argparse.Namespace, dataset):
    from data_loader import get_sku_mean_price
    from feature_engineering import get_feature_matrix

    existing_ids = dataset.existing_skus["article_id"].unique().tolist()
    feature_matrix, engineer = get_feature_matrix(
        dataset.articles, dataset.weekly_sales, existing_ids,
        n_pca_components=args.n_pca_components,
    )
    prices = get_sku_mean_price(dataset.weekly_sales)
    return feature_matrix, engineer, prices, existing_ids


def stage_match(
    args: argparse.Namespace,
    feature_matrix: pd.DataFrame,
    existing_ids: list[str],
    new_sku_ids: list[str],
):
    from analogue_matcher import AnalogueMatcher

    existing_features = feature_matrix.loc[feature_matrix.index.isin(existing_ids)]
    new_features = feature_matrix.loc[feature_matrix.index.isin(new_sku_ids)]

    matcher = AnalogueMatcher(k=max(args.k_values))  # fit with max k; sub-select later
    matcher.fit(existing_features)

    # Primary matches at chosen k
    matcher.k = args.k
    matches = matcher.match(new_features)

    # Multi-k matches for benchmarking
    k_results = matcher.match_multi_k(new_features, k_values=args.k_values)

    return matcher, matches, k_results, new_features


def stage_transfer(
    args: argparse.Namespace,
    matches,
    dataset,
    prices: pd.Series,
    existing_ids: list[str],
):
    from demand_transfer import DemandTransfer

    transfer = DemandTransfer(
        cold_start_weeks=args.cold_start_weeks,
        elasticity=args.elasticity,
        scale_clip=(args.scale_min, args.scale_max),
    )
    forecasts = transfer.transfer(
        matches,
        dataset.existing_skus,
        prices.reindex(dataset.new_sku_ids),
        prices.reindex(existing_ids),
    )
    return forecasts


def stage_bayes(
    args: argparse.Namespace,
    forecasts,
    dataset,
):
    from bayesian_updater import GaussianConjugateUpdater, run_pymc_demo

    updater = GaussianConjugateUpdater(
        obs_noise_fraction=args.obs_noise,
        prior_std_fraction=args.prior_std,
    )
    updates = updater.update_all(
        forecasts, dataset.new_sku_ground_truth, cold_start_weeks=args.cold_start_weeks
    )

    mcmc_summary = None
    if not args.no_mcmc:
        mcmc_summary = run_pymc_demo(
            forecasts,
            dataset.new_sku_ground_truth,
            n_skus=args.mcmc_skus,
            cold_start_weeks=args.cold_start_weeks,
            samples=args.mcmc_samples,
            chains=args.mcmc_chains,
            seed=args.seed,
        )
        if mcmc_summary is not None:
            out_path = Path(args.results_dir) / "mcmc_summary.csv"
            mcmc_summary.to_csv(out_path, index=False)
            print(f"  MCMC summary saved to {out_path}")

    return updates, mcmc_summary


def stage_eval(
    args: argparse.Namespace,
    forecasts,
    updates,
    matches,
    k_results,
    dataset,
):
    from evaluation import BaselineForecaster, Evaluator

    baseline = BaselineForecaster(cold_start_weeks=args.cold_start_weeks, seed=args.seed)
    baseline.fit(dataset.existing_skus, dataset.articles)

    evaluator = Evaluator(
        cold_start_weeks=args.cold_start_weeks,
        results_dir=args.results_dir,
    )
    summary = evaluator.evaluate(
        forecasts=forecasts,
        updates=updates,
        matches=matches,
        ground_truth_df=dataset.new_sku_ground_truth,
        baseline=baseline,
        articles=dataset.articles,
        k_results=k_results,
    )
    return summary


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    stages = [s.strip().lower() for s in args.stages.split(",")]

    np.random.seed(args.seed)

    t_total = time.time()
    print("=" * 60)
    print("  H&M New-SKU Analogue Demand Forecasting System")
    print("=" * 60)
    print(f"  Stages : {stages}")
    print(f"  Data   : {args.data_dir}")
    print(f"  k      : {args.k}  |  k_values: {args.k_values}")
    print(f"  Seed   : {args.seed}")
    print("=" * 60)

    dataset = feature_matrix = engineer = prices = existing_ids = None
    matches = k_results = forecasts = updates = None

    # ---- Stage: load ----
    if "load" in stages:
        t = time.time()
        dataset = stage_load(args)
        print(f"  [load] done in {time.time()-t:.1f}s\n")

    # ---- Stage: features ----
    if "features" in stages:
        assert dataset is not None, "--stages must include 'load' before 'features'"
        t = time.time()
        feature_matrix, engineer, prices, existing_ids = stage_features(args, dataset)
        print(f"  [features] done in {time.time()-t:.1f}s\n")

    # ---- Stage: match ----
    if "match" in stages:
        assert feature_matrix is not None, "--stages must include 'features' before 'match'"
        t = time.time()
        matcher, matches, k_results, new_features = stage_match(
            args, feature_matrix, existing_ids, dataset.new_sku_ids
        )
        print(f"  [match] done in {time.time()-t:.1f}s\n")

    # ---- Stage: transfer ----
    if "transfer" in stages:
        assert matches is not None, "--stages must include 'match' before 'transfer'"
        t = time.time()
        forecasts = stage_transfer(args, matches, dataset, prices, existing_ids)
        print(f"  [transfer] done in {time.time()-t:.1f}s\n")

    # ---- Stage: bayes ----
    if "bayes" in stages:
        assert forecasts is not None, "--stages must include 'transfer' before 'bayes'"
        t = time.time()
        updates, mcmc_summary = stage_bayes(args, forecasts, dataset)
        print(f"  [bayes] done in {time.time()-t:.1f}s\n")

    # ---- Stage: eval ----
    if "eval" in stages:
        assert updates is not None, "--stages must include 'bayes' before 'eval'"
        t = time.time()
        summary = stage_eval(args, forecasts, updates, matches, k_results, dataset)
        print(f"  [eval] done in {time.time()-t:.1f}s\n")

    elapsed = time.time() - t_total
    print("=" * 60)
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Results saved to: {Path(args.results_dir).resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
