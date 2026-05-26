"""
03b_tune.py
Quant Analyst: Giorgio Maarraoui
==============================================================================================================================================
LightGBM hyperparameter search via Optuna with persistent storage.

I utilise here an 8-dim search space (Tweedie power, learning rate, leaves, min-data, feature and bagging fractions, bagging frequency, L2).

As an objective, I look to minimise MAE due to interpretability. But, CRPS moves similarly anyway.

I use SQLite to save results so that older searches are quickly invoked.

I aim for around 25 trials total, not an exhaustive tuning exercise for the purpose of this task.

Output: artifacts/optuna_best_params.json

The final tuned model is retrained in 03c_retrain_v8.py
"""
import polars as pl, pandas as pd, numpy as np, lightgbm as lgb, json, warnings, optuna, time
from pathlib import Path
from utils import FEATURES_NUM

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT   = Path(__file__).resolve().parent.parent
ART    = ROOT / "artifacts"
DATA   = ART / "data"
MODELS = ART / "models"; MODELS.mkdir(exist_ok=True, parents=True)

df = pl.read_parquet(DATA / "features.parquet").sort("game_date_time").to_pandas()
n = len(df); vs = int(n*0.70); te = int(n*0.80)
train = df.iloc[:vs].copy(); val = df.iloc[vs:te].copy()

TARGET = "points"

# Persistent LGBM datasets (built once and then re-used for every trial).
dtr = lgb.Dataset(train[FEATURES_NUM], label=train[TARGET], free_raw_data=False)
dv  = lgb.Dataset(val[FEATURES_NUM], label=val[TARGET], reference=dtr, free_raw_data=False)
y_val = val[TARGET].values

# Optuna objective function, returning validation MAE.
def obj(trial):
    params = dict(
        objective="tweedie",
        tweedie_variance_power=trial.suggest_float("tvp", 1.1, 1.9),
        learning_rate=trial.suggest_float("lr", 0.02, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 255),
        min_data_in_leaf=trial.suggest_int("mdil", 50, 500),
        feature_fraction=trial.suggest_float("ff", 0.6, 1.0),
        bagging_fraction=trial.suggest_float("bf", 0.6, 1.0),
        bagging_freq=trial.suggest_int("bfreq", 1, 10),
        lambda_l2=trial.suggest_float("l2", 1e-6, 1.0, log=True),
        verbose=-1, seed=42, metric="rmse",
    )
    m = lgb.train(params, dtr, num_boost_round=300,
                  valid_sets=[dv],
                  callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
    mu_val = np.clip(m.predict(val[FEATURES_NUM]), 0, None)
    return float(np.mean(np.abs(y_val - mu_val)))

# Save the optuna search in a database (I use /tmp so SQLite write locks work regardless of mount permissions.)
storage = "sqlite:////tmp/optuna_study.db"
study = optuna.create_study(
    study_name="lgbm_tweedie_mae",
    storage=storage,
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    load_if_exists=True,
)

# 25 trials in total
TARGET_TRIALS = 25
done = len(study.trials)
remaining = max(0, TARGET_TRIALS - done)
print(f"Resuming Optuna study: {done}/{TARGET_TRIALS} trials done, running {remaining}…")

# Add a deadline guard: stop with time left (here, 45 seconds) to save params + persist to avoid crashing:
DEADLINE = time.time() + 35  # leave 10secs headroom under 45secs timeout
def safe_obj(trial):
    if time.time() >= DEADLINE:
        raise optuna.exceptions.TrialPruned("deadline reached")
    return obj(trial)

if remaining > 0:
    try:
        study.optimize(safe_obj, n_trials=remaining, show_progress_bar=False)
    except KeyboardInterrupt:
        pass

print(f"Total trials done: {len(study.trials)}; best MAE so far: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Save best optuna parameters in JSON file
best = {k: (float(v) if isinstance(v, np.floating)
           else (int(v) if isinstance(v, np.integer) else v))
        for k, v in study.best_params.items()}
with open(MODELS / "optuna_best_params.json", "w") as f:
    json.dump({"best_val_mae": float(study.best_value),
               "best_params": best,
               "n_trials": len(study.trials)}, f, indent=2)
print(f"Saved → {MODELS / 'optuna_best_params.json'}")
