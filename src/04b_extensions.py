"""
04b_extensions.py
Quant Analyst: Giorgio Maarraoui
================================

Curated extension experiments after the original V0-V10 model comparison previously sent.
    V11  Direct LightGBM mean blended with minutes * points-per-minute.
    V12  Hurdle distribution with temperature-calibrated CDF.
    V13  Direct LightGBM mean plus a second LightGBM variance model.

Outputs:
    artifacts/eval/extensions_comparison.csv
    artifacts/eval/extensions_details.json
    artifacts/data/predictions_v11.parquet
    artifacts/models/extension_*.lgb / *.json

The original baseline remains V3 LightGBM Tweedie + validation-binned NB.
"""

from pathlib import Path
import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import nbinom, poisson

from utils import FEATURES_NUM, crps, fit_dispersion, lookup_alpha, mae, nb_params, rmse

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
DATA = ART / "data"
EVAL = ART / "eval"
MODELS = ART / "models"
EVAL.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

TARGET = "points"
K_MAX = 80
KS = np.arange(K_MAX + 1)


def split_data():
    df = pl.read_parquet(DATA / "features.parquet").sort("game_date_time").to_pandas()
    n = len(df)
    val_start = int(n * 0.70)
    test_start = int(n * 0.80)
    train = df.iloc[:val_start].copy()
    val = df.iloc[val_start:test_start].copy()
    test = df.iloc[test_start:].copy()
    test = test[test["pts_ewma_10"].notna()].copy()
    return train, val, test


def train_lgb(train, val, features, target, objective="regression", extra=None, rounds=2000):
    params = dict(
        objective=objective,
        metric="binary_logloss" if objective == "binary" else "rmse",
        learning_rate=0.05,
        num_leaves=63,
        min_data_in_leaf=200,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        verbose=-1,
        seed=42,
    )
    if extra:
        params.update(extra)
    dtr = lgb.Dataset(train[features], label=train[target], free_raw_data=False)
    dva = lgb.Dataset(val[features], label=val[target], reference=dtr, free_raw_data=False)
    return lgb.train(
        params,
        dtr,
        num_boost_round=rounds,
        valid_sets=[dtr, dva],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )


def nb_metrics(y_val, y_test, mu_val, mu_test):
    tab = fit_dispersion(mu_val, y_val)
    var_test = mu_test + lookup_alpha(mu_test, tab) * mu_test**2
    return {
        "mae": mae(y_test, mu_test),
        "rmse": rmse(y_test, mu_test),
        "crps": float(crps(y_test, mu_test, var_test).mean()),
    }, tab


def crps_from_cdf(y, cdf):
    indicators = (np.asarray(y)[:, None] <= KS[None, :]).astype(float)
    return float(np.mean(np.sum((cdf - indicators) ** 2, axis=1)))


def mean_from_cdf(cdf):
    return np.sum(1 - cdf[:, :-1], axis=1)


def hurdle_cdf(p_pos, mu_pos, var_pos):
    p_pos = np.clip(np.asarray(p_pos, dtype=float), 1e-6, 1 - 1e-6)
    mu_pos = np.clip(np.asarray(mu_pos, dtype=float), 1e-6, None)
    var_pos = np.maximum(np.asarray(var_pos, dtype=float), mu_pos + 1e-6)
    n_, p, use_pois = nb_params(mu_pos, var_pos)
    f0_nb = np.where(use_pois, poisson.cdf(0, mu_pos), nbinom.cdf(0, n_, p))
    denom = np.clip(1 - f0_nb, 1e-8, None)
    cols = []
    for k in KS:
        if k == 0:
            cols.append(1 - p_pos)
        else:
            f_nb = np.where(use_pois, poisson.cdf(k, mu_pos), nbinom.cdf(k, n_, p))
            f_pos = np.clip((f_nb - f0_nb) / denom, 0, 1)
            cols.append((1 - p_pos) + p_pos * f_pos)
    return np.column_stack(cols)


def fit_temperature(cdf_val, y_val):
    x = np.clip(cdf_val.reshape(-1), 1e-6, 1 - 1e-6)
    z = logit(x)
    target = (np.asarray(y_val)[:, None] <= KS[None, :]).astype(float).reshape(-1)

    def objective(theta):
        a, log_b = theta
        b = np.exp(log_b)
        p = np.clip(expit(a + b * z), 1e-6, 1 - 1e-6)
        return -np.mean(target * np.log(p) + (1 - target) * np.log(1 - p))

    res = minimize(objective, x0=np.array([0.0, 0.0]), method="Nelder-Mead")
    a, log_b = res.x
    return float(a), float(np.exp(log_b))


def apply_temperature(cdf, a, b):
    raw = np.clip(cdf, 1e-6, 1 - 1e-6)
    out = expit(a + b * logit(raw))
    return np.maximum.accumulate(np.clip(out, 0, 1), axis=1)


def run_v11(train, val, test, direct_model, mu_direct_val, mu_direct_test):
    print("[04b] V11: direct mean blended with minutes * points/minute...")
    y_val = val[TARGET].values
    y_test = test[TARGET].values

    train_rate = train.copy()
    val_rate = val.copy()
    test_rate = test.copy()
    for frame in [train_rate, val_rate, test_rate]:
        frame["ppm_target"] = np.where(frame["minutes"] > 1.0, frame["points"] / frame["minutes"], 0.0)
        frame["ppm_target"] = frame["ppm_target"].clip(0, 2.5)

    minutes = train_lgb(train_rate, val_rate, FEATURES_NUM, "minutes")
    ppm = train_lgb(train_rate, val_rate, FEATURES_NUM, "ppm_target")
    mu_stage_val = (
        np.clip(minutes.predict(val_rate[FEATURES_NUM]), 0, 60)
        * np.clip(ppm.predict(val_rate[FEATURES_NUM]), 0, 2.5)
    )
    mu_stage_test = (
        np.clip(minutes.predict(test_rate[FEATURES_NUM]), 0, 60)
        * np.clip(ppm.predict(test_rate[FEATURES_NUM]), 0, 2.5)
    )

    weights = np.linspace(0, 1, 21)
    val_mae = [mae(y_val, w * mu_direct_val + (1 - w) * mu_stage_val) for w in weights]
    best_w = float(weights[int(np.argmin(val_mae))])
    mu_blend_val = best_w * mu_direct_val + (1 - best_w) * mu_stage_val
    mu_blend_test = best_w * mu_direct_test + (1 - best_w) * mu_stage_test
    metrics, tab = nb_metrics(y_val, y_test, mu_blend_val, mu_blend_test)
    var_blend_test = mu_blend_test + lookup_alpha(mu_blend_test, tab) * mu_blend_test**2

    minutes.save_model(str(MODELS / "extension_v11_minutes.lgb"))
    ppm.save_model(str(MODELS / "extension_v11_points_per_minute.lgb"))
    tab.to_json(MODELS / "extension_v11_dispersion.json", orient="records", indent=2)
    pd.DataFrame(
        {
            "points": y_test,
            "mu_hat": mu_blend_test,
            "var": var_blend_test,
            "model_variant": f"V11 direct/two-stage blend (w={best_w:.2f})",
        }
    ).to_parquet(DATA / "predictions_v11.parquet", index=False)

    return {
        "variant": f"V11 direct/two-stage blend (w={best_w:.2f})",
        **metrics,
        "kind": "extension",
    }, {
        "blend_weight_direct": best_w,
        "blend_weight_two_stage": 1 - best_w,
        "selection_metric": "validation MAE",
        "direct_loss": "LightGBM Tweedie; early stopping on validation RMSE",
        "minutes_loss": "LightGBM regression/RMSE",
        "points_per_minute_loss": "LightGBM regression/RMSE on clipped points/minute",
        "distribution": "Negative Binomial with validation-binned alpha",
    }


def run_v12(train, val, test):
    print("[04b] V12: hurdle distribution with temperature calibration...")
    y_val = val[TARGET].values
    y_test = test[TARGET].values

    train_gate = train.copy()
    val_gate = val.copy()
    train_gate["is_positive"] = (train_gate[TARGET] > 0).astype(int)
    val_gate["is_positive"] = (val_gate[TARGET] > 0).astype(int)

    gate = train_lgb(train_gate, val_gate, FEATURES_NUM, "is_positive", "binary", rounds=1500)
    p_pos_val = np.clip(gate.predict(val[FEATURES_NUM]), 1e-5, 1 - 1e-5)
    p_pos_test = np.clip(gate.predict(test[FEATURES_NUM]), 1e-5, 1 - 1e-5)

    train_pos = train[train[TARGET] > 0].copy()
    val_pos = val[val[TARGET] > 0].copy()
    pos_mean = train_lgb(
        train_pos,
        val_pos,
        FEATURES_NUM,
        TARGET,
        "tweedie",
        {"tweedie_variance_power": 1.5},
        rounds=1500,
    )

    mu_pos_val_all = np.clip(pos_mean.predict(val[FEATURES_NUM]), 1e-6, None)
    mu_pos_test = np.clip(pos_mean.predict(test[FEATURES_NUM]), 1e-6, None)
    mu_pos_val_pos = np.clip(pos_mean.predict(val_pos[FEATURES_NUM]), 1e-6, None)
    tab = fit_dispersion(mu_pos_val_pos, val_pos[TARGET].values)
    var_pos_val_all = mu_pos_val_all + lookup_alpha(mu_pos_val_all, tab) * mu_pos_val_all**2
    var_pos_test = mu_pos_test + lookup_alpha(mu_pos_test, tab) * mu_pos_test**2

    cdf_val = hurdle_cdf(p_pos_val, mu_pos_val_all, var_pos_val_all)
    cdf_test = hurdle_cdf(p_pos_test, mu_pos_test, var_pos_test)
    a, b = fit_temperature(cdf_val, y_val)
    cdf_test_temp = apply_temperature(cdf_test, a, b)
    mu_test_temp = mean_from_cdf(cdf_test_temp)

    gate.save_model(str(MODELS / "extension_v12_positive_gate.lgb"))
    pos_mean.save_model(str(MODELS / "extension_v12_positive_mean.lgb"))
    tab.to_json(MODELS / "extension_v12_positive_dispersion.json", orient="records", indent=2)

    return {
        "variant": f"V12 hurdle + temperature CDF (a={a:.2f}, b={b:.2f})",
        "mae": mae(y_test, mu_test_temp),
        "rmse": rmse(y_test, mu_test_temp),
        "crps": crps_from_cdf(y_test, cdf_test_temp),
        "kind": "extension",
    }, {
        "gate_loss": "LightGBM binary log-loss for P(points > 0)",
        "positive_mean_loss": "LightGBM Tweedie on validation-positive points",
        "positive_distribution": "Positive-truncated Negative Binomial",
        "temperature_loss": "validation CDF binary log-loss over thresholds 0..80",
        "temperature_intercept": a,
        "temperature_slope": b,
    }


def chronological_oof_mu(train, n_folds=5):
    n = len(train)
    edges = np.linspace(0, n, n_folds + 1, dtype=int)
    oof = np.full(n, float(train[TARGET].mean()))
    for fold in range(1, n_folds):
        start, end = edges[fold], edges[fold + 1]
        model = train_lgb(
            train.iloc[:start].copy(),
            train.iloc[start:end].copy(),
            FEATURES_NUM,
            TARGET,
            "tweedie",
            {"tweedie_variance_power": 1.5},
        )
        oof[start:end] = np.clip(model.predict(train.iloc[start:end][FEATURES_NUM]), 0, None)
    return oof


def train_variance_model(train_var, val_var):
    params = dict(
        objective="regression",
        metric="rmse",
        learning_rate=0.04,
        num_leaves=31,
        min_data_in_leaf=300,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        lambda_l2=1.0,
        verbose=-1,
        seed=123,
    )
    dtr = lgb.Dataset(train_var[FEATURES_NUM], label=train_var["log_sq_resid_oof"], free_raw_data=False)
    dva = lgb.Dataset(val_var[FEATURES_NUM], label=val_var["log_sq_resid"], reference=dtr, free_raw_data=False)
    return lgb.train(
        params,
        dtr,
        num_boost_round=1500,
        valid_sets=[dtr, dva],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )


def run_v13(train, val, test, direct_model, mu_direct_val, mu_direct_test):
    print("[04b] V13: second LightGBM variance model...")
    y_val = val[TARGET].values
    y_test = test[TARGET].values

    train_var = train.copy()
    mu_train_oof = chronological_oof_mu(train_var)
    train_var["log_sq_resid_oof"] = np.log1p((train_var[TARGET].values - mu_train_oof) ** 2)

    val_var = val.copy()
    val_var["log_sq_resid"] = np.log1p((val_var[TARGET].values - mu_direct_val) ** 2)
    var_model = train_variance_model(train_var, val_var)

    var_val_raw = np.maximum(np.expm1(var_model.predict(val[FEATURES_NUM])), mu_direct_val + 1e-3)
    var_test_raw = np.maximum(np.expm1(var_model.predict(test[FEATURES_NUM])), mu_direct_test + 1e-3)

    scales = np.array([0.50, 0.65, 0.80, 0.90, 1.00, 1.10, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00])
    scale_rows = []
    best = None
    for s in scales:
        var_val = mu_direct_val + s * np.maximum(var_val_raw - mu_direct_val, 1e-3)
        val_crps = float(crps(y_val, mu_direct_val, var_val).mean())
        row = {"scale": float(s), "val_crps": val_crps}
        scale_rows.append(row)
        if best is None or val_crps < best["val_crps"]:
            best = row

    s = best["scale"]
    var_test_scaled = mu_direct_test + s * np.maximum(var_test_raw - mu_direct_test, 1e-3)
    var_model.save_model(str(MODELS / "extension_v13_variance.lgb"))
    pd.DataFrame(scale_rows).to_csv(EVAL / "extensions_v13_variance_scale_grid.csv", index=False)

    return {
        "variant": f"V13 second LightGBM variance + validation scale {s:.2f}",
        "mae": mae(y_test, mu_direct_test),
        "rmse": rmse(y_test, mu_direct_test),
        "crps": float(crps(y_test, mu_direct_test, var_test_scaled).mean()),
        "kind": "extension",
    }, {
        "mean_loss": "LightGBM Tweedie; early stopping on validation RMSE",
        "variance_loss": "LightGBM regression/RMSE on log1p(out-of-fold squared residual)",
        "variance_floor": "var >= mu + 1e-3",
        "best_validation_scale": s,
        "distribution": "Negative Binomial from predicted mean and predicted variance",
    }


def main():
    train, val, test = split_data()
    y_val = val[TARGET].values
    y_test = test[TARGET].values

    print("[04b] Training shared direct LightGBM Tweedie mean model...")
    direct_model = train_lgb(
        train,
        val,
        FEATURES_NUM,
        TARGET,
        "tweedie",
        {"tweedie_variance_power": 1.5},
    )
    mu_direct_val = np.clip(direct_model.predict(val[FEATURES_NUM]), 0, None)
    mu_direct_test = np.clip(direct_model.predict(test[FEATURES_NUM]), 0, None)
    baseline_metrics, baseline_tab = nb_metrics(y_val, y_test, mu_direct_val, mu_direct_test)
    direct_model.save_model(str(MODELS / "extension_shared_direct_mean.lgb"))
    baseline_tab.to_json(MODELS / "extension_v3_baseline_dispersion.json", orient="records", indent=2)

    rows = [{"variant": "V3 baseline LightGBM Tweedie + NB", **baseline_metrics, "kind": "baseline"}]
    details = {"V3_baseline": {"distribution": "Negative Binomial with validation-binned alpha"}}

    v11, d11 = run_v11(train, val, test, direct_model, mu_direct_val, mu_direct_test)
    rows.append(v11)
    details["V11"] = d11

    v12, d12 = run_v12(train, val, test)
    rows.append(v12)
    details["V12"] = d12

    v13, d13 = run_v13(train, val, test, direct_model, mu_direct_val, mu_direct_test)
    rows.append(v13)
    details["V13"] = d13

    out = pd.DataFrame(rows).sort_values(["crps", "mae"])
    out.to_csv(EVAL / "extensions_comparison.csv", index=False)


    out.to_csv(EVAL / "curated_experiment_comparison.csv", index=False)
    with open(EVAL / "extensions_details.json", "w") as f:
        json.dump(details, f, indent=2)

    print("\n[04b] Extension comparison:")
    print(out.to_string(index=False))
    print("\n[04b] Wrote artifacts/eval/extensions_comparison.csv")


if __name__ == "__main__":
    main()
