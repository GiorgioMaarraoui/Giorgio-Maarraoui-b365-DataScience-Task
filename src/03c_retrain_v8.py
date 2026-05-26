"""
03c_retrain_v8.py
Quant Analyst: Giorgio Maarraoui
==================
Retrain the LightGBM Tweedie model with the Optuna-best parameters and a larger boost budget (early-stopped on validation). Appended to the model_comparison_partial.csv (V8).
"""
import polars as pl, pandas as pd, numpy as np, lightgbm as lgb, json
from pathlib import Path
from utils import (
    FEATURES_NUM, mae, rmse,
    nb_params, nb_cdf, crps,
    fit_dispersion, lookup_alpha,
)

ROOT   = Path(__file__).resolve().parent.parent
ART    = ROOT / "artifacts"
DATA   = ART / "data"
MODELS = ART / "models"; MODELS.mkdir(exist_ok=True, parents=True)
EVAL   = ART / "eval";   EVAL.mkdir(exist_ok=True, parents=True)

with open(MODELS / "optuna_best_params.json") as f:
    best = json.load(f)["best_params"]
print("Loaded best params:", best)

df = pl.read_parquet(DATA / "features.parquet").sort("game_date_time").to_pandas()
n = len(df); vs = int(n*0.70); te = int(n*0.80)
train = df.iloc[:vs].copy(); val = df.iloc[vs:te].copy()
test  = df.iloc[te:].copy()
test  = test[test["pts_ewma_10"].notna()].copy()

TARGET = "points"

# Re-training and Evaluation following 03b
print("Training V8 (LGBM Tweedie tuned)…")
params = dict(
    objective="tweedie",
    tweedie_variance_power=best["tvp"],
    learning_rate=best["lr"],
    num_leaves=best["num_leaves"],
    min_data_in_leaf=best["mdil"],
    feature_fraction=best["ff"],
    bagging_fraction=best["bf"],
    bagging_freq=best["bfreq"],
    lambda_l2=best["l2"],
    verbose=-1, seed=42, metric="rmse",
)
dtr = lgb.Dataset(train[FEATURES_NUM], label=train[TARGET], free_raw_data=False)
dv  = lgb.Dataset(val[FEATURES_NUM], label=val[TARGET], reference=dtr, free_raw_data=False)
m = lgb.train(params, dtr, num_boost_round=2500,
              valid_sets=[dtr,dv], valid_names=["tr","va"],
              callbacks=[lgb.early_stopping(60), lgb.log_evaluation(0)])
mu_val  = np.clip(m.predict(val[FEATURES_NUM]),  0, None)
mu_test = np.clip(m.predict(test[FEATURES_NUM]), 0, None)
tab = fit_dispersion(mu_val, val[TARGET].values)
alpha = lookup_alpha(mu_test, tab)
var = mu_test + alpha*mu_test**2
cs = float(crps(test[TARGET].values, mu_test, var).mean())
v8 = {"mae":mae(test[TARGET], mu_test), "rmse":rmse(test[TARGET], mu_test),
      "crps":cs, "best_iter":m.best_iteration}
print(f"V8 LGBM tuned: MAE={v8['mae']:.4f}  RMSE={v8['rmse']:.4f}  CRPS={v8['crps']:.4f}  (iter {v8['best_iter']})")

m.save_model(str(MODELS / "model_v8.lgb"))
tab.to_json(MODELS / "dispersion_table_v8.json", orient="records", indent=2)
np.save(EVAL / "v8_mu_test.npy", mu_test)

# Append to partial comparison.
partial = pd.read_csv(EVAL / "model_comparison_partial.csv")
# Remove any prior V8 row we have.
partial = partial[~partial["variant"].str.startswith("V8")]
row = pd.DataFrame([{"variant":"V8 LGBM Tweedie (Optuna-tuned)",
                     "mae":v8["mae"],"rmse":v8["rmse"],"crps":v8["crps"],"kind":"lgb"}])
pd.concat([partial, row], ignore_index=True).to_csv(EVAL / "model_comparison_partial.csv", index=False)
print(f"Saved.")
