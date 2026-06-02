# NBA Player Points Prediction: Baseline + Curated Extensions

Prepared by Giorgio Maarraoui for the bet365 Data Science task.

This project predicts NBA player points before tip-off and produces a full probability distribution over point totals, allowing over/under probabilities to be quoted at different lines.

## Contents

- `src/`: reproducible modelling pipeline and leakage checks
- `artifacts/`: processed data, trained models, evaluation outputs, and plots
- `Documentation/README.md`: detailed project notes and reproduction instructions
- `Documentation/methodology_summary.pdf`: concise methodology submission
- `Documentation/technical_report.pdf`: extended technical report
- `Documentation/extension_experiments_summary.pdf`: one-page summary of the curated extension experiments
- `nba_possessions.parquet`: source possession-level dataset

## Reproduce Baseline And Extensions

Run from the project root:

```bash
python3 src/01_aggregate.py
python3 src/02_features.py
python3 src/leakage_check.py
python3 src/03a_compare_models.py
python3 src/03b_tune.py
python3 src/03c_retrain_v8.py
python3 src/03d_xgboost.py
python3 src/04_zinb_calibrate.py
python3 src/04b_extensions.py
python3 src/05_final_eval.py
```

## Extension Workflow

The original model comparison stops at V10. The curated extension script continues the numbering:

- `V11`: direct LightGBM mean blended with a minutes x points-per-minute structural estimate
- `V12`: hurdle distribution with temperature-calibrated CDF
- `V13`: direct LightGBM mean with a second LightGBM variance model

Run the extensions with:

```bash
python3 src/04b_extensions.py
```

Then run:

```bash
python3 src/05_final_eval.py
```

This produces:

- `artifacts/eval/extensions_comparison.csv`
- `artifacts/eval/extensions_details.json`
- `artifacts/eval/final_headline_with_extensions.csv`
- `artifacts/data/predictions_v11.parquet`

When `predictions_v11.parquet` exists, `05_final_eval.py` uses V11 for the final over/under, reliability, PIT and segment diagnostics. The comparison table still reports all baseline and extension variants.

## Headline Results

- V11 direct/two-stage blend is the practical recommended extension: MAE `4.3681`, RMSE `5.7532`, CRPS `3.0639`.
- V12 hurdle + temperature CDF is best by CRPS: CRPS `3.0621`, but with worse MAE.
- V13 second LightGBM variance model was tested and rejected because it worsened CRPS.

Dependencies: `polars`, `lightgbm`, `xgboost`, `scipy`, `scikit-learn`, `optuna`, `matplotlib`, `pandas`, `pyarrow`, and `numpy`.
