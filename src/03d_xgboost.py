"""
03d_xgboost.py
Quant Analyst: Giorgio Maarraoui
==============
Here, I benchmark the LightGBM against 2 XGBoost models: V9 using reg:tweedie and V10 using count:poisson
I use the ame feature set as the LightGBM numeric-only variants (V3/V4), same temporal split, same empirical-NB dispersion convention.
This robustness is useful to test whether results hold against similar models, or whether they are framework-bound
"""
import polars as pl, pandas as pd, numpy as np, xgboost as xgb, json
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
df = pl.read_parquet(DATA / "features.parquet").sort("game_date_time").to_pandas()
n = len(df); vs = int(n*0.70); te = int(n*0.80)
train = df.iloc[:vs].copy(); val = df.iloc[vs:te].copy()
test  = df.iloc[te:].copy()
test  = test[test["pts_ewma_10"].notna()].copy()

TARGET = "points"

def train_xgb(objective, name, **extra):
    dtr = xgb.DMatrix(train[FEATURES_NUM].values, label=train[TARGET].values)
    dv  = xgb.DMatrix(val[FEATURES_NUM].values,   label=val[TARGET].values)
    dts = xgb.DMatrix(test[FEATURES_NUM].values,  label=test[TARGET].values)
    params = dict(objective=objective, eval_metric="rmse",
                  learning_rate=0.05, max_depth=6, min_child_weight=10,
                  subsample=0.8, colsample_bytree=0.9,
                  reg_lambda=1.0, seed=42, verbosity=0, **extra)
    m = xgb.train(params, dtr, num_boost_round=1500,
                  evals=[(dtr,"tr"),(dv,"va")],
                  early_stopping_rounds=50, verbose_eval=False)
    mu_val  = np.clip(m.predict(dv),  0, None)
    mu_test = np.clip(m.predict(dts), 0, None)
    tab = fit_dispersion(mu_val, val[TARGET].values)
    alpha = lookup_alpha(mu_test, tab)
    var = mu_test + alpha*mu_test**2
    cs = float(crps(test[TARGET].values, mu_test, var).mean())
    print(f"{name}: MAE={mae(test[TARGET],mu_test):.4f}  "
          f"RMSE={rmse(test[TARGET],mu_test):.4f}  CRPS={cs:.4f}")
    return {"variant":name,"mae":mae(test[TARGET],mu_test),
            "rmse":rmse(test[TARGET],mu_test),"crps":cs,"kind":"xgb",
            "mu_test":mu_test,"model":m,"disp_tab":tab}

v9  = train_xgb("reg:tweedie",   "V9 XGBoost Tweedie (num)",  tweedie_variance_power=1.5)
v10 = train_xgb("count:poisson", "V10 XGBoost Poisson (num)")

# Save outputs.
v9["model"].save_model(str(MODELS / "model_v9_xgb.json"))
v10["model"].save_model(str(MODELS / "model_v10_xgb.json"))
np.save(EVAL / "v9_mu_test.npy",  v9["mu_test"])
np.save(EVAL / "v10_mu_test.npy", v10["mu_test"])

# Append XGBoost rows to comparison, saved in model_comparison.csv rather than model_comparison_partial.csv
partial = pd.read_csv(EVAL / "model_comparison_partial.csv")
partial = partial[~partial["variant"].str.startswith(("V9","V10"))]
rows = pd.DataFrame([
    {"variant":v9["variant"],  "mae":v9["mae"],  "rmse":v9["rmse"],  "crps":v9["crps"],  "kind":"xgb"},
    {"variant":v10["variant"], "mae":v10["mae"], "rmse":v10["rmse"], "crps":v10["crps"], "kind":"xgb"},
])
combined = pd.concat([partial, rows], ignore_index=True)
combined.to_csv(EVAL / "model_comparison.csv", index=False)
print("\n=== FULL MODEL COMPARISON ===")
print(combined.to_string(index=False))
