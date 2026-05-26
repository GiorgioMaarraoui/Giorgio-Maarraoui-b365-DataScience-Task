# NBA Player Points Prediction

Prepared by Giorgio Maarraoui

---

## What this is

I built a model that predicts how many points an NBA player will score in a given game, before the game tips off. The output is a full probability distribution over possible point totals, not just a single number. That means I can quote P(player scores more than X points) at any line, which is directly useful in a sportsbook context.

The data is possession-level play-by-play covering three seasons (October 2022 through November 2024), about 5.5 million rows. I collapse this to one row per player-game and engineer pre-game features from it.

## Why this approach

The core insight is that a player's scoring in a single game is noisy, but their recent form, playing time, role on the team, and the quality of the opponent's defense are all predictable ahead of time. I encode these as rolling and exponentially weighted features, making sure none of them peek at the current game's outcome.

LightGBM with a Tweedie loss handles the non-negative, zero-inflated count-like structure of points well, and it learns nonlinear interactions between features that simpler models miss. For the distributional layer, I fit an empirical Negative Binomial variance on the validation fold so I can generate calibrated over/under probabilities.

I also benchmarked XGBoost variants to make sure the result is not framework-specific.

## How to reproduce

Run the scripts in order from the project root:

```bash
python3 src/01_aggregate.py
python3 src/02_features.py
python3 src/leakage_check.py
python3 src/03a_compare_models.py
python3 src/03b_tune.py
python3 src/03c_retrain_v8.py
python3 src/03d_xgboost.py
python3 src/04_zinb_calibrate.py
python3 src/05_final_eval.py
```

Dependencies: `polars`, `lightgbm`, `xgboost`, `scipy`, `scikit-learn`, `optuna`, `matplotlib`, `pandas`, `numpy`.

## Key design decisions

**Leakage protection.** Every feature is computed from games strictly before the current one. I sort by date, compute rolling stats within each season, and shift by one row so the current game never feeds its own feature. `leakage_check.py` verifies this automatically.

**Temporal split.** I use the first 70% of player-games for training, the next 10% for validation, and the last 20% for testing. Random splits would leak future information.

**No raw player identity.** Adding a raw player ID as a feature hurts out-of-sample performance because player roles change across seasons and the model memorises rather than generalises.

**Plain NB over ZINB.** I tested a Zero-Inflated Negative Binomial alternative but the plain NB with empirical dispersion wins on CRPS (the proper distributional score), so that is what the final pipeline uses.

**Shared utilities.** Common functions (metrics, NB helpers, dispersion fitting) live in `src/utils.py` and are imported by the relevant scripts rather than redefined in each one.

## File structure

```
src/
  utils.py                shared metrics, NB helpers, FEATURES_NUM, dispersion
  01_aggregate.py         possession rows to player-game (Polars)
  02_features.py          40 leakage-protected pre-game features
  03a_compare_models.py   baselines and LGBM variants V0 through V7
  03b_tune.py             Optuna hyperparameter search (V8)
  03c_retrain_v8.py       V8 retrain with best Optuna params
  03d_xgboost.py          XGBoost benchmarks V9 and V10
  04_zinb_calibrate.py    ZINB alternative (tested, not adopted)
  05_final_eval.py        consolidated metrics and plots
  leakage_check.py        unit test for leakage discipline

artifacts/
  data/
    player_game.parquet     aggregated player-game rows
    features.parquet        engineered feature set (40 features)
    predictions.parquet     test-set predictions with distributional layer

  models/
    model.lgb               V3 LightGBM Tweedie (final model)
    model_v8.lgb            Optuna-tuned variant
    model_v9_xgb.json       XGBoost Tweedie
    model_v10_xgb.json      XGBoost Poisson
    zero_model.lgb          binary P(Y=0) head (ZINB alternative)
    dispersion_table.json   empirical NB dispersion by mu-bin (V3)
    dispersion_table_v8.json empirical NB dispersion (V8)
    optuna_best_params.json

  eval/
    model_comparison.csv    V0 through V10 head-to-head
    overunder_comparison.csv M1 vs M2 over/under at 6 lines
    final_headline.csv      summary table
    final_overunder.csv     over/under probabilities at 6 lines
    final_segment.csv       MAE by predicted-mean bucket
    metrics.json            consolidated evaluation metrics
    v3_mu_test.npy          test-set mean predictions (V3)
    v8_mu_test.npy          test-set mean predictions (V8)
    v9_mu_test.npy          test-set mean predictions (V9)
    v10_mu_test.npy         test-set mean predictions (V10)

  plots/
    overunder_lines_final.png
    reliability_lines.png
    pit_final.png
    pit_compare.png
    reliability_14_5_compare.png

methodology_summary.pdf  required 2-page methodology submission
technical_report.pdf     full extended analysis and results
```

## Honest limitations

The model is conditional on a player appearing in the game. It does not predict whether a player will play at all, which would require an upstream availability model fed by injury reports and lineup data. It also has no knowledge of specific defender matchups, market signals, or misdeason trades. Calibration is reasonable but not perfect, and player roles can shift faster than rolling averages adapt!
