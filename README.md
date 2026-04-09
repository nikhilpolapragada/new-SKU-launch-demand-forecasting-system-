# New-SKU Launch Demand Forecasting System

Analogue-based demand forecasting for cold-start SKUs using the H&M Fashion Recommendations dataset. Dissertation project.

---

## Overview

When a brand launches a new SKU, there is no sales history to feed into traditional time-series models. This system solves the cold-start problem by:

1. **Finding analogues** – existing SKUs that are most similar to the new one based on product attributes and seasonality.
2. **Transferring demand curves** – using a weighted average of analogue launch curves, scaled by a price-ratio factor.
3. **Bayesian updating** – refining the forecast week-by-week as early sales data arrives, using both fast Gaussian conjugate updates and an optional PyMC MCMC model.

---

## Dataset

Download the [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data) dataset from Kaggle and place the following files in a `data/` directory:

```
data/
├── articles.csv
├── transactions_train.csv
└── customers.csv
```

---

## Installation

```bash
pip install -r requirements.txt
```

> PyMC requires a C compiler. On macOS: `xcode-select --install`. On Ubuntu: `apt install build-essential`.

---

## Quick Start

```bash
# Run the full pipeline with default settings
python main.py

# Custom data directory, k=10 analogues, no MCMC demo
python main.py --data-dir /path/to/hm-data --k 10 --no-mcmc

# Multi-k benchmark (k=3, 5, 10) with higher elasticity
python main.py --k-values 3 5 10 --elasticity 1.5

# Run only load + features + match stages
python main.py --stages load,features,match
```

---

## Architecture

```
main.py                   ← CLI orchestrator (argparse)
│
├── data_loader.py         ← Stage 1: Load CSV, aggregate weekly sales,
│                                      cold-start simulation
│
├── feature_engineering.py ← Stage 2: One-hot encoding, price normalisation,
│                                      PCA seasonality (52w → 4 PCs)
│
├── analogue_matcher.py    ← Stage 3: Cosine similarity, top-k selection,
│                                      ReLU-clipped weights
│
├── demand_transfer.py     ← Stage 3b: Weighted average launch curve,
│                                       price-ratio scaling
│
├── bayesian_updater.py    ← Stage 4: Gaussian conjugate update (per-week),
│                                      optional PyMC MCMC demo (5 SKUs)
│
└── evaluation.py          ← Stage 5: RMSE/MAE/MASE/Bias metrics,
                                       benchmarks, 5 plots → results/
```

---

## Pipeline Stages

### Stage 1 – Data Loading (`data_loader.py`)

| Step | Description |
|------|-------------|
| Load | `articles.csv`, `transactions_train.csv`, `customers.csv` |
| Aggregate | Weekly units sold per `article_id` (ISO year-week IDs) |
| Cold-start sim | Hide first 4 weeks of 10% of eligible SKUs as ground truth |
| Split | New SKUs (metadata only) vs existing SKUs (full history) |

### Stage 2 – Feature Engineering (`feature_engineering.py`)

| Feature block | Method |
|---------------|--------|
| Categorical | One-hot encode: `product_type_name`, `colour_group_code`, `department_name`, `index_group_name`, `garment_group_name` |
| Price | Min-max normalise mean price (fit on existing SKUs only) |
| Seasonality | 52-week average unit sales index per product type → PCA (4 components) |

### Stage 3 – Analogue Matching (`analogue_matcher.py`)

- **Similarity**: cosine similarity = L2-normalised dot product
- **Top-k**: selects k=3, 5, or 10 analogues (tested via `--k-values`)
- **Weights**: `w_i = max(0, sim_i) / Σ max(0, sim_j)`

### Stage 3b – Demand Transfer (`demand_transfer.py`)

- Weighted average of analogue launch curves (first N weeks)
- Price-ratio scaling: `scale = (mean_analogue_price / new_sku_price)^elasticity`
- Scale clamped to [0.1, 10.0]

### Stage 4 – Bayesian Updating (`bayesian_updater.py`)

**Gaussian conjugate (fast, analytic)**

```
Prior:        μ_t ~ N(analogue_forecast_t, σ_prior²)
Observation:  x_t ~ N(μ_t, σ_obs²)
Posterior:    1/σ_post² = 1/σ_prior² + 1/σ_obs²
              μ_post = σ_post² × (μ_prior/σ_prior² + x/σ_obs²)
```

Tracks RMSE after each weekly update step.

**PyMC MCMC demo (optional, 5 SKUs)**

Full hierarchical model with `HalfNormal` priors on noise. Outputs posterior mean, 94% HDI, and RMSE.

### Stage 5 – Evaluation (`evaluation.py`)

**Benchmarks**

| Name | Description |
|------|-------------|
| Naive | Category-average launch curve |
| Random analogue | Random existing SKU's launch curve |
| Oracle (k=1) | Best possible single analogue |

**Metrics**: RMSE, MAE, MASE, Bias

**Plots** (saved to `results/`)

| File | Description |
|------|-------------|
| `actual_vs_predicted.png` | Scatter: actual vs prior/posterior demand |
| `bayesian_rmse_improvement.png` | RMSE trajectory across Bayesian update weeks |
| `benchmark_comparison.png` | Bar chart: RMSE by method |
| `category_rmse_heatmap.png` | Heatmap: RMSE by product type and method |
| `similarity_distribution.png` | Histogram + violin of cosine similarities |

---

## CLI Reference

```
python main.py [OPTIONS]

Data:
  --data-dir DIR            Path to H&M CSV files (default: data)
  --results-dir DIR         Output directory (default: results)

Cold-start:
  --cold-start-fraction F   Fraction of SKUs as new (default: 0.10)
  --cold-start-weeks N      Launch weeks to forecast (default: 4)
  --min-history-weeks N     Min weeks for analogue pool (default: 8)

Feature engineering:
  --n-pca-components N      Seasonality PCA components (default: 4)

Matching:
  --k N                     Primary k for analogues (default: 5)
  --k-values N [N ...]      k values for benchmark comparison (default: 3 5 10)

Demand transfer:
  --elasticity E            Price elasticity exponent (default: 1.0)
  --scale-min F             Min price scale (default: 0.1)
  --scale-max F             Max price scale (default: 10.0)

Bayesian:
  --obs-noise F             Observation noise fraction (default: 0.5)
  --prior-std F             Prior std fraction (default: 0.8)
  --no-mcmc                 Skip PyMC MCMC demo
  --mcmc-skus N             SKUs in MCMC demo (default: 5)
  --mcmc-samples N          MCMC draws per chain (default: 1000)
  --mcmc-chains N           MCMC chains (default: 2)

Pipeline:
  --stages STAGES           Comma-separated: load,features,match,transfer,bayes,eval
  --seed N                  Random seed (default: 42)
```

---

## Output Files

```
results/
├── metrics_per_sku.csv          All metrics for every new SKU
├── mcmc_summary.csv             PyMC posterior summary (if MCMC ran)
├── actual_vs_predicted.png
├── bayesian_rmse_improvement.png
├── benchmark_comparison.png
├── category_rmse_heatmap.png
└── similarity_distribution.png
```

---

## Design Decisions

- **No data leakage**: all transformers (OHE, price scaler, seasonality PCA) are fit exclusively on existing SKUs before being applied to new SKUs.
- **ReLU-clipped weights**: negative cosine similarities receive zero weight rather than negative contribution.
- **Price-ratio clamping**: prevents extreme scaling when price data is missing or anomalous.
- **Modular stages**: each module is independently importable and runnable for iterative development.

---

## Project Structure

```
new-SKU-launch-demand-forecasting-system/
├── data/                    ← H&M CSVs (not tracked in git)
├── results/                 ← Generated plots and metrics
├── data_loader.py
├── feature_engineering.py
├── analogue_matcher.py
├── demand_transfer.py
├── bayesian_updater.py
├── evaluation.py
├── main.py
├── requirements.txt
└── README.md
```
