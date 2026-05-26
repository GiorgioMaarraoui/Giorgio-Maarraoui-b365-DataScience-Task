# NBA Player Points Prediction

Prepared by Giorgio Maarraoui for the bet365 Data Science task.

This project predicts NBA player points before tip-off and produces a full probability distribution over point totals, allowing over/under probabilities to be quoted at different lines.

## Contents

- `src/`: reproducible modelling pipeline and leakage checks
- `artifacts/`: processed data, trained models, evaluation outputs, and plots
- `Documentation/README.md`: detailed project notes and reproduction instructions
- `Documentation/methodology_summary.pdf`: concise methodology submission
- `Documentation/technical_report.pdf`: extended technical report
- `nba_possessions.parquet`: source possession-level dataset

## Reproduce

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
python3 src/05_final_eval.py
```

Dependencies: `polars`, `lightgbm`, `xgboost`, `scipy`, `scikit-learn`, `optuna`, `matplotlib`, `pandas`, and `numpy`.
