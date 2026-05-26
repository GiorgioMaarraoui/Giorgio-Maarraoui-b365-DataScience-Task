"""
03_compare_models.py
Quant Analyst: Giorgio Maarraoui
===========================================================================================================================
Comparison of different model variants (on the same temporal test set).

  NAIVE BASELINES
    V0  global mean
    V1  10-game EWMA PPG
    V2  EWMA(minutes) * EWMA(pts/min)             (naive two-stage)
  LIGHTGBM
    V3  Tweedie, numeric features only
    V4  Poisson, numeric features only
    V5  RMSE, numeric features only
    V6  Tweedie + raw categoricals (position, team, opp)
    V7  Tweedie + player_id (memorisation test)
    V8  Tweedie hyperparameter-tuned via Optuna (temporal validation, MAE objective)
  XGBOOST
    V9  Tweedie, numeric features only
    V10 Poisson, numeric features only
Note V0 to V7 are run in this script, V8 in 03b_tune.py, V9/V10 in 03d_xgboost.py.

CRPS: closed-form discrete sum truncated at K=80, with the CDF coming from a Negative Binomial whose variance
is an empirical mean-binned dispersion fit on the validation fold. All variants use the same dispersion-binning
convention to ensure comparability.
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
from pathlib import Path
from utils import (
    FEATURES_NUM, mae, rmse,
    nb_params, nb_cdf, crps,
    fit_dispersion, lookup_alpha,
)

warnings.filterwarnings("ignore")

ROOT   = Path(__file__).resolve().parent.parent
ART    = ROOT / "artifacts"
DATA   = ART / "data";   DATA.mkdir(exist_ok=True, parents=True)
MODELS = ART / "models"; MODELS.mkdir(exist_ok=True, parents=True)
EVAL   = ART / "eval";   EVAL.mkdir(exist_ok=True, parents=True)
df = pl.read_parquet(DATA / "features.parquet").sort("game_date_time").to_pandas()

n = len(df); val_start = int(n*0.70); train_end = int(n*0.80)
train = df.iloc[:val_start].copy()
val   = df.iloc[val_start:train_end].copy()
test  = df.iloc[train_end:].copy()
test  = test[test["pts_ewma_10"].notna()].copy()

CATS_BASIC = ["position_code","team_id","opp_id"]
CATS_FULL  = CATS_BASIC + ["player_id"]
TARGET = "points"

# Naive baselines.
results = []
test["v0"] = train[TARGET].mean()
test["v1"] = test["pts_ewma_10"]
test["v2"] = (test["min_ewma_10"] * test["pts_per_min_ewma_10"]).fillna(test["pts_ewma_10"])
for name, col in [("V0 global_mean","v0"),("V1 ewma_ppg","v1"),("V2 min*ppm","v2")]:
    results.append({"variant":name,"mae":mae(test[TARGET],test[col]),
                    "rmse":rmse(test[TARGET],test[col]),"crps":np.nan,"kind":"baseline"})

# LightGBM
def train_lgb(objective, feature_cols, cat_cols, name, **extra):
    dtr = lgb.Dataset(train[feature_cols], label=train[TARGET],
                      categorical_feature=cat_cols, free_raw_data=False)
    dv  = lgb.Dataset(val[feature_cols], label=val[TARGET],
                      categorical_feature=cat_cols, reference=dtr, free_raw_data=False)
    params = dict(objective=objective, metric="rmse",
                  learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=5,
                  verbose=-1, seed=42, **extra)
    m = lgb.train(params, dtr, num_boost_round=2000,
                  valid_sets=[dtr,dv], valid_names=["tr","va"],
                  callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    mu_val  = np.clip(m.predict(val[feature_cols]), 0, None)
    mu_test = np.clip(m.predict(test[feature_cols]), 0, None)
    tab = fit_dispersion(mu_val, val[TARGET].values)
    alpha = lookup_alpha(mu_test, tab)
    var = mu_test + alpha*mu_test**2
    cs = crps(test[TARGET].values, mu_test, var).mean()
    return {"variant":name,"mae":mae(test[TARGET],mu_test),
            "rmse":rmse(test[TARGET],mu_test),"crps":float(cs),
            "best_iter":m.best_iteration,"mu_test":mu_test,"model":m,
            "disp_tab":tab,"kind":"lgb"}

print("V3 LGBM Tweedie (numeric)…")
v3 = train_lgb("tweedie", FEATURES_NUM, [], "V3 LGBM Tweedie (num)", tweedie_variance_power=1.5)
print("V4 LGBM Poisson…")
v4 = train_lgb("poisson", FEATURES_NUM, [], "V4 LGBM Poisson (num)")
print("V5 LGBM RMSE…")
v5 = train_lgb("regression", FEATURES_NUM, [], "V5 LGBM RMSE (num)")
print("V6 LGBM Tweedie + cat…")
v6 = train_lgb("tweedie", FEATURES_NUM + CATS_BASIC, CATS_BASIC,
               "V6 LGBM Tweedie (num+cat)", tweedie_variance_power=1.5)
print("V7 LGBM Tweedie + player_id…")
v7 = train_lgb("tweedie", FEATURES_NUM + CATS_FULL, CATS_FULL,
               "V7 LGBM Tweedie (+player_id)", tweedie_variance_power=1.5)

for v in [v3,v4,v5,v6,v7]:
    results.append({k:v[k] for k in ["variant","mae","rmse","crps","kind"]})

# Persist V3's mu_test, model, dispersion table as the current "best LGBM untuned" reference; V8 (tuned, in 03b) may replace this.
v3["model"].save_model(str(MODELS / "model.lgb"))
v3["disp_tab"].to_json(MODELS / "dispersion_table.json", orient="records", indent=2)
np.save(EVAL / "v3_mu_test.npy", v3["mu_test"])

# Save partial comparison; subsequent scripts append V8, V9, V10.
pd.DataFrame(results).to_csv(EVAL / "model_comparison_partial.csv", index=False)
print("\nPartial comparison (V0..V7):")
print(pd.DataFrame(results).to_string(index=False))
