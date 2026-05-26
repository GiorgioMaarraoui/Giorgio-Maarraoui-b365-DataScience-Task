"""
utils.py
Quant Analyst: Giorgio Maarraoui
=================================
Shared utilities used across the modelling pipeline. I centralise these here to avoid repeating the same definitions in 03_compare_models, 03b_tune,
03c_retrain_v8, 03d_xgboost, 04_zinb_calibrate, and 05_final_eval.

Contents
--------
FEATURES_NUM   the 37-feature numeric list fed to every LightGBM / XGBoost model
mae            mean absolute error
rmse           root mean squared error
nb_params      convert (mu, var) to scipy NegBin(n, p) parameterisation
nb_cdf         P(Y <= y) under NB2; falls back to Poisson when under-dispersed
crps           discrete CRPS via NB CDF (closed-form sum over k=0..k_max)
crps_dist      discrete CRPS with a caller-supplied CDF function
fit_dispersion empirical NB dispersion table: mu-bin -> alpha
lookup_alpha   nearest-bin lookup into the dispersion table
"""

import numpy as np
import pandas as pd
from scipy.stats import nbinom, poisson

# ---------------------------------------------------------------------------
# Feature list — identical across all gradient-boosting scripts.
# ---------------------------------------------------------------------------
FEATURES_NUM = [
    "pts_ewma_5", "pts_ewma_10", "pts_ewma_20", "pts_mean_season", "pts_std_10",
    "min_ewma_10", "min_std_10", "fga_ewma_10", "fta_ewma_10", "tpa_ewma_10",
    "usage_ewma_10", "pts_per36_ewma_10", "pts_per_min_ewma_10", "ts_pct_ewma_20",
    "team_attempt_share_ewma_10", "starts_proxy_ewma_10", "games_played_so_far",
    "is_home", "days_rest", "is_back_to_back", "is_three_in_four",
    "games_last_5d", "games_last_14d",
    "opp_def_rating_roll20", "opp_off_rating_roll20", "opp_pace_roll20",
    "opp_fga_allowed_per100_roll20", "opp_3pa_allowed_per100_roll20",
    "opp_fta_allowed_per100_roll20", "opp_ot_rate_roll10",
    "opp_pts_allowed_to_pos_roll20",
    "team_pace_roll20", "team_off_rating_roll20", "expected_pace",
    "team_game_idx_in_season", "opp_game_idx_in_season", "month",
]

# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------
def mae(y, yhat):
    return float(np.mean(np.abs(y - yhat)))

def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))

# ---------------------------------------------------------------------------
# Distributional helpers (Negative Binomial)
# ---------------------------------------------------------------------------
def nb_params(mu, var):
    """Convert (mu, var) to scipy.stats.nbinom (n, p). Returns (n, p, use_pois_mask)."""
    mu = np.asarray(mu, dtype=float)
    var = np.asarray(var, dtype=float)
    use_pois = (var <= mu * 1.0001) | (mu <= 1e-6)
    safe_var = np.where(use_pois, np.maximum(mu, 1e-3) + 1e-3, var)
    p = np.clip(mu / safe_var, 1e-6, 1 - 1e-6)
    n_ = np.clip(mu * p / (1 - p), 1e-6, None)
    return n_, p, use_pois

def nb_cdf(y, mu, var):
    """P(Y <= y) under NB2; falls back to Poisson when under-dispersed."""
    n_, p, up = nb_params(mu, var)
    return np.where(up, poisson.cdf(y, np.maximum(mu, 1e-6)), nbinom.cdf(y, n_, p))

def crps(y, mu, var, k_max=80):
    """Discrete CRPS via closed-form sum, CDF from NB2.

    CRPS(F, y) = sum_k (F(k) - 1{y <= k})^2,  k = 0 .. k_max
    """
    y = np.asarray(y)
    out = np.zeros(len(y))
    for k in range(k_max + 1):
        out += (nb_cdf(k, mu, var) - (y <= k).astype(float)) ** 2
    return out

def crps_dist(y, cdf_fn, k_max=80):
    """Discrete CRPS with a caller-supplied CDF function.

    cdf_fn must accept a scalar k and return a vector of probabilities P(Y <= k).
    """
    out = np.zeros(len(y))
    for k in range(k_max + 1):
        out += (cdf_fn(k) - (y <= k).astype(float)) ** 2
    return out

# ---------------------------------------------------------------------------
# Empirical dispersion
# ---------------------------------------------------------------------------
def fit_dispersion(mu_val, y_val, n_bins=12):
    """Fit empirical NB alpha per mu-bin on a validation fold.

    Returns a DataFrame with columns [mu_mean, var, alpha], sorted by mu_mean.
    alpha is the NB2 dispersion parameter: var = mu + alpha * mu^2.
    """
    bins = pd.qcut(mu_val, q=n_bins, duplicates="drop")
    tab = pd.DataFrame({"mu": mu_val, "sq": (y_val - mu_val) ** 2, "b": bins})
    bt = tab.groupby("b", observed=True).agg(
        mu_mean=("mu", "mean"), var=("sq", "mean")
    ).reset_index(drop=True)
    bt["alpha"] = ((bt["var"] - bt["mu_mean"]) / bt["mu_mean"] ** 2).clip(lower=0)
    return bt.sort_values("mu_mean").reset_index(drop=True)

def lookup_alpha(mu, tab):
    """Nearest-mu-bin lookup of empirical NB shape parameter alpha."""
    mus = tab["mu_mean"].values
    alphas = tab["alpha"].values
    idx = np.clip(np.searchsorted(mus, mu), 0, len(mus) - 1)
    ld = np.abs(mu - mus[np.maximum(idx - 1, 0)])
    rd = np.abs(mus[idx] - mu)
    use_left = (ld < rd) & (idx > 0)
    return alphas[np.where(use_left, idx - 1, idx)]
