"""
05_final_eval.py
Quant Analyst: Giorgio Maarraoui
============================================================================================================================
In this script, I produce the following 7 artifacts:

  * artifacts/eval/final_headline.csv         original baseline variants on the test set
  * artifacts/eval/final_headline_with_extensions.csv   original variants with 04b extension rows (added in this version)
  * artifacts/eval/final_overunder.csv        OU at 6 lines for the final model
  * artifacts/eval/final_segment.csv          MAE by predicted-mean bucket
  * artifacts/eval/metrics.json               consolidated
  * artifacts/plots/overunder_lines_final.png
  * artifacts/plots/reliability_lines.png
  * artifacts/plots/pit_final.png

The headline comparison reports the original baseline variants and, when available, the 04b extensions.
If 04b_extensions.py has been run, final diagnostic plots use V11 because it is the practical recommended
extension: it improves MAE, RMSE and CRPS relative to the original V3 baseline.
"""
import polars as pl, pandas as pd, numpy as np, json, lightgbm as lgb
from pathlib import Path
from utils import nb_params, nb_cdf, crps_dist, mae, rmse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ART  = ROOT / "artifacts"
DATA = ART / "data"
EVAL = ART / "eval";  EVAL.mkdir(exist_ok=True, parents=True)
PLOT = ART / "plots"; PLOT.mkdir(exist_ok=True)

try:
    pred = pl.read_parquet(DATA / "predictions.parquet").to_pandas()
except Exception:
    pred = pd.read_parquet(DATA / "predictions.parquet")
diagnostic_model = "V3 baseline final predictions"
v11_pred_path = DATA / "predictions_v11.parquet"
if v11_pred_path.exists():
    pred = pd.read_parquet(v11_pred_path)
    diagnostic_model = str(pred.get("model_variant", pd.Series(["V11 recommended extension"])).iloc[0])
comp = pd.read_csv(EVAL / "model_comparison.csv")
ext_path = EVAL / "extensions_comparison.csv"
if ext_path.exists():
    ext = pd.read_csv(ext_path)
    comp_with_ext = pd.concat([comp, ext], ignore_index=True, sort=False)
    comp_with_ext = comp_with_ext.drop_duplicates("variant", keep="last")
else:
    ext = pd.DataFrame()
    comp_with_ext = comp.copy()

# Original baseline best by CRPS (lower better, NaN dropped).
crps_sorted = comp[comp["crps"].notna()].sort_values("crps")
best = crps_sorted.iloc[0]
print(f"Best original baseline by CRPS: {best['variant']} (CRPS={best['crps']:.4f})")
print(f"Diagnostic predictions: {diagnostic_model}")

y  = pred["points"].values
mu = pred["mu_hat"].values
var = pred["var"].values if "var" in pred.columns else pred["var_hat"].values

# ----- final headline -----
print("\n=== HEADLINE TABLE ===")
print(comp.to_string(index=False))
comp.to_csv(EVAL / "final_headline.csv", index=False)

if not ext.empty:
    comp_with_ext = comp_with_ext.sort_values(["crps", "mae"], na_position="last")
    print("\n=== HEADLINE TABLE WITH EXTENSIONS ===")
    print(comp_with_ext.to_string(index=False))
    comp_with_ext.to_csv(EVAL / "final_headline_with_extensions.csv", index=False)

# ----- over/under at lines using the selected diagnostic prediction set -----
LINES = [4.5, 9.5, 14.5, 19.5, 24.5, 29.5]
ou_rows = []
for L in LINES:
    y_bin = (y > L).astype(int)
    p_final = 1 - nb_cdf(L, mu, var)
    brier = float(np.mean((p_final - y_bin)**2))
    ll = float(-np.mean(y_bin*np.log(np.clip(p_final,1e-6,1-1e-6))
                        + (1-y_bin)*np.log(np.clip(1-p_final,1e-6,1-1e-6))))
    ou_rows.append({"line": L, "empirical_rate": float(y_bin.mean()),
                    "avg_predicted": float(p_final.mean()),
                    "brier": brier, "logloss": ll})
ou = pd.DataFrame(ou_rows)
print("\n=== OVER/UNDER (selected diagnostic model) ===")
print(ou.to_string(index=False))
ou.to_csv(EVAL / "final_overunder.csv", index=False)

fig, ax = plt.subplots(figsize=(7,4.5))
xp = np.arange(len(ou))
ax.bar(xp-0.2, ou["empirical_rate"], width=0.4, label="Empirical")
ax.bar(xp+0.2, ou["avg_predicted"], width=0.4, label="Predicted")
ax.set_xticks(xp); ax.set_xticklabels([f"{L}" for L in ou["line"]])
ax.set_xlabel("Line"); ax.set_ylabel("P(points > line)")
ax.set_title(f"{diagnostic_model}: predicted vs empirical over-rate")
ax.legend()
plt.tight_layout(); fig.savefig(PLOT/"overunder_lines_final.png", dpi=120); plt.close(fig)

# ----- 6-panel reliability -----
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, L in zip(axes.flat, LINES):
    y_bin = (y > L).astype(int)
    p = 1 - nb_cdf(L, mu, var)
    n_bins = 10
    bins = np.linspace(0,1,n_bins+1)
    idx = np.clip(np.digitize(p, bins)-1, 0, n_bins-1)
    xs, ys = [], []
    for b in range(n_bins):
        m = idx==b
        if m.sum() < 20: continue
        xs.append(p[m].mean()); ys.append(y_bin[m].mean())
    ax.plot([0,1],[0,1],"k--", alpha=0.5)
    ax.plot(xs, ys, "o-")
    ax.set_xlabel(f"Pred P(>{L})"); ax.set_ylabel("Empirical")
    ax.set_title(f"line {L}"); ax.grid(alpha=0.3)
plt.tight_layout(); fig.savefig(PLOT/"reliability_lines.png", dpi=120); plt.close(fig)

# ----- PIT -----
rng = np.random.default_rng(0)
Fy   = nb_cdf(y,     mu, var)
Fym1 = nb_cdf(y - 1, mu, var)
Fym1 = np.where(y == 0, 0.0, Fym1)
U = rng.uniform(size=len(y))
pit = Fym1 + U*(Fy - Fym1)
fig, ax = plt.subplots(figsize=(7,4))
ax.hist(pit, bins=20, edgecolor="k")
ax.axhline(len(y)/20, color="red", linestyle="--", label="uniform")
ax.set_xlabel("PIT"); ax.set_title(f"{diagnostic_model}: randomised PIT"); ax.legend()
plt.tight_layout(); fig.savefig(PLOT/"pit_final.png", dpi=120); plt.close(fig)

# ----- segment metrics -----
buckets = pd.cut(mu, bins=[-1,5,10,15,20,100], labels=["<=5","5 to 10","10 to 15","15 to 20",">20"])
seg = pd.DataFrame({"mu":mu,"y":y,"b":buckets}).groupby("b", observed=True).agg(
    n=("y","size"),
    mae=("y", lambda v: float(np.mean(np.abs(v - mu[v.index])))),
    rmse=("y", lambda v: float(np.sqrt(np.mean((v - mu[v.index])**2)))),
    emp_mean=("y","mean"), pred_mean=("mu","mean"),
)
print("\n=== SEGMENT METRICS ===")
print(seg.to_string())
seg.to_csv(EVAL / "final_segment.csv")

# ----- consolidated metrics.json -----
final = {
    "n_test": int(len(pred)),
    "diagnostic_prediction_source": diagnostic_model,
    "comparison": comp.to_dict(orient="records"),
    "extensions": ext.to_dict(orient="records") if not ext.empty else [],
    "comparison_with_extensions": comp_with_ext.to_dict(orient="records") if not ext.empty else [],
    "best_by_crps": best["variant"],
    "over_under": ou_rows,
    "segments":  seg.reset_index().rename(columns={"b":"bucket"}).to_dict(orient="records"),
}
with open(EVAL / "metrics.json", "w") as f:
    json.dump(final, f, indent=2, default=str)
print("\nWrote artifacts/eval/metrics.json")
