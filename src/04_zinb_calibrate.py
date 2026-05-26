"""
04_zinb_calibrate.py
Quant Analyst: Giorgio Maarraoui
===============================================================================================================================================
In this script, I switch to a zero-inflated negative binomial (ZINB) layer as an alternative test to plain NB

Rationale: The plain-NB layer under-predicts the marginal zero mass (12.3% of player-games have Y=0). The natural extension is a ZINB mixture:

    Y = 0 with probability `gate`,
    Y ~ NB(mu, var) with probability (1 - gate).

I train a binary LightGBM to predict p0 = P(Y = 0 | x), then convert this marginal probability into the mixture's zero-inflation gate via

    NB0  = NB(0 ; mu_hat, var)
    gate = clip( (p0 - NB0) / (1 - NB0), 0, 1 )

This conversion guarantees the mixture's actual P(Y = 0) equals the trained p0:

    gate + (1 - gate) * NB0
    = (p0 - NB0)/(1 - NB0) + (1 - (p0 - NB0)/(1 - NB0)) * NB0
    = p0

The NB component uses the marginal mu directly as its mean (no rescaling). The mixture's unconditional mean is therefore (1 - gate) * mu_hat,
slightly less than mu_hat. Point predictions (MAE/RMSE) are evaluated against mu_hat directly.

I then evaluate M1 (plain NB) vs M2 (ZINB) on CRPS, PIT-deviation, Brier, and log-loss for over/under lines. The downstream final-eval script
picks whichever wins on CRPS.

Outputs
-------
artifacts/data/predictions.parquet     mu_hat + p0 + gate + var + nb0
artifacts/models/zero_model.lgb        binary LightGBM for P(Y=0)
artifacts/eval/overunder_comparison.csv
artifacts/eval/metrics.json
artifacts/plots/pit_compare.png        plain NB vs ZINB
artifacts/plots/reliability_14_5_compare.png
"""
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from pathlib import Path
from utils import FEATURES_NUM, nb_params, nb_cdf, crps_dist, lookup_alpha
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT   = Path(__file__).resolve().parent.parent
ART    = ROOT / "artifacts"
DATA   = ART / "data";   DATA.mkdir(exist_ok=True, parents=True)
MODELS = ART / "models"; MODELS.mkdir(exist_ok=True, parents=True)
EVAL   = ART / "eval";   EVAL.mkdir(exist_ok=True, parents=True)
PLOT   = ART / "plots";  PLOT.mkdir(exist_ok=True)

df = pl.read_parquet(DATA / "features.parquet").sort("game_date_time").to_pandas()
n = len(df); vs = int(n*0.70); te = int(n*0.80)
train = df.iloc[:vs].copy()
val   = df.iloc[vs:te].copy()
test  = df.iloc[te:].copy()
test  = test[test["pts_ewma_10"].notna()].copy()

# Load the mean model from step 3.
mean_model = lgb.Booster(model_file=str(MODELS / "model.lgb"))
disp_tab = pd.read_json(MODELS / "dispersion_table.json")

def compute_gate(p0, mu, var):
    """Convert trained p0 = P(Y=0) into a zero-inflation gate.

    gate = clip((p0 - NB0) / (1 - NB0), 0, 1)

    After this, the mixture gate + (1-gate) * NB(0; mu, var)
    equals p0 wherever p0 > NB0 (i.e.. the trained zero probability
    exceeds what NB alone produces). If p0 <= NB0 then gate clamps to 0
    and the mixture reduces to plain NB (the NB already produces enough
    zero mass on its own; no inflation needed).
    """
    NB0 = nb_cdf(0, mu, var)
    raw = (p0 - NB0) / np.clip(1 - NB0, 1e-9, None)
    return np.clip(raw, 0.0, 1.0)

def zinb_cdf(y, gate, mu, var):
    """Corrected ZINB CDF using the gate (not p0 directly).
       CDF(y) = gate * 1{y >= 0}  +  (1 - gate) * NB.cdf(y; mu, var).
       1{.} here is there indicator function
    """
    base = nb_cdf(y, mu, var)
    y_arr = np.asarray(y)
    ind = (y_arr >= 0).astype(float)
    return gate * ind + (1 - gate) * base


# ---------- Mean predictions on val + test ----------------------------
val["mu_hat"]  = np.clip(mean_model.predict(val[FEATURES_NUM]),  0, None)
test["mu_hat"] = np.clip(mean_model.predict(test[FEATURES_NUM]), 0, None)
for d in (val, test):
    d["alpha"] = lookup_alpha(d["mu_hat"].values, disp_tab)
    d["var"]   = d["mu_hat"] + d["alpha"] * d["mu_hat"]**2

# ---------- Zero-probability classifier (P(Y=0)) ----------------------
print("[04] Training binary LightGBM for P(Y=0)…")
train["is_zero"] = (train["points"] == 0).astype(int)
val["is_zero"]   = (val["points"]   == 0).astype(int)
dz_tr = lgb.Dataset(train[FEATURES_NUM], label=train["is_zero"], free_raw_data=False)
dz_va = lgb.Dataset(val[FEATURES_NUM],   label=val["is_zero"],   reference=dz_tr, free_raw_data=False)
zparams = dict(objective="binary", metric="binary_logloss",
               learning_rate=0.05, num_leaves=31, min_data_in_leaf=200,
               feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=5,
               verbose=-1, seed=42)
zm = lgb.train(zparams, dz_tr, num_boost_round=1000,
               valid_sets=[dz_tr, dz_va],
               callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
zm.save_model(str(MODELS / "zero_model.lgb"))
val["p0"]  = zm.predict(val[FEATURES_NUM])
test["p0"] = zm.predict(test[FEATURES_NUM])
print(f"  train zero rate = {train['is_zero'].mean():.3f}")
print(f"  val   pred mean = {val['p0'].mean():.3f}   empirical = {val['is_zero'].mean():.3f}")
print(f"  test  pred mean = {test['p0'].mean():.3f}   empirical = {(test['points']==0).mean():.3f}")

# ---------- Convert p0 -> gate ----------------------------------------
val["nb0"]  = nb_cdf(0, val["mu_hat"].values,  val["var"].values)
test["nb0"] = nb_cdf(0, test["mu_hat"].values, test["var"].values)
val["gate"]  = compute_gate(val["p0"].values,  val["mu_hat"].values,  val["var"].values)
test["gate"] = compute_gate(test["p0"].values, test["mu_hat"].values, test["var"].values)
print(f"  test  NB0 mean  = {test['nb0'].mean():.3f}  (NB-implied zero mass)")
print(f"  test  gate mean = {test['gate'].mean():.3f}  (zero-inflation gate after conversion)")
print(f"  test  effective ZINB P(Y=0) mean = "
      f"{(test['gate'] + (1-test['gate'])*test['nb0']).mean():.3f}  "
      f"(should ≈ trained p0 = {test['p0'].mean():.3f})")

# ---------- M1 vs M2 CRPS ---------------------------------------------
y_test = test["points"].values
m1_cdf = lambda k: nb_cdf(k, test["mu_hat"].values, test["var"].values)
m2_cdf = lambda k: zinb_cdf(k, test["gate"].values, test["mu_hat"].values, test["var"].values)
crps_m1 = float(crps_dist(y_test, m1_cdf).mean())
crps_m2 = float(crps_dist(y_test, m2_cdf).mean())
print(f"\n=== Corrected ZINB vs plain NB ===")
print(f"  M1 plain NB         CRPS = {crps_m1:.4f}")
print(f"  M2 corrected ZINB   CRPS = {crps_m2:.4f}  (Δ = {crps_m2-crps_m1:+.4f})")

if crps_m2 < crps_m1:
    print(f"  → Corrected ZINB improves CRPS. Recommend M2 as final distribution.")
    FINAL_MODEL = "M2"
else:
    print(f"  → Corrected ZINB does NOT improve CRPS. Recommend reverting to M1.")
    FINAL_MODEL = "M1"

# ---------- PIT histograms --------------------------------------------
rng = np.random.default_rng(0)
def pit_hist(cdf_fn, y_arr, ax, title):
    Fy   = cdf_fn(y_arr)
    Fym1 = cdf_fn(y_arr - 1)
    Fym1 = np.where(y_arr == 0, 0.0, Fym1)
    U = rng.uniform(size=len(y_arr))
    pit = Fym1 + U*(Fy - Fym1)
    ax.hist(pit, bins=20, edgecolor="k")
    ax.axhline(len(y_arr)/20, color="red", linestyle="--", label="uniform")
    ax.set_xlabel("PIT"); ax.set_title(title); ax.legend()
    return pit

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
pit1 = pit_hist(m1_cdf, y_test, axes[0], "PIT plain NB (M1)")
pit2 = pit_hist(m2_cdf, y_test, axes[1], "PIT corrected ZINB (M2)")
fig.savefig(PLOT / "pit_compare.png", dpi=120, bbox_inches="tight")
plt.close(fig)

ks_m1 = float(np.max(np.abs(np.sort(pit1) - np.linspace(1/len(pit1), 1, len(pit1)))))
ks_m2 = float(np.max(np.abs(np.sort(pit2) - np.linspace(1/len(pit2), 1, len(pit2)))))
print(f"  PIT KS-dev:  M1 = {ks_m1:.3f}   M2 = {ks_m2:.3f}")

# ---------- Over/under @ lines: M1 vs M2 ------------------------------
LINES = [4.5, 9.5, 14.5, 19.5, 24.5, 29.5]
print(f"\n{'line':>6} {'emp':>6} | {'M1_p':>6} {'M1_brier':>9} {'M1_ll':>8} | "
      f"{'M2_p':>6} {'M2_brier':>9} {'M2_ll':>8}")
ou_rows = []
for L in LINES:
    y_bin = (y_test > L).astype(int)
    p1 = 1 - m1_cdf(L)
    p2 = 1 - m2_cdf(L)
    def stats(p):
        b = float(np.mean((p-y_bin)**2))
        ll = float(-np.mean(y_bin*np.log(np.clip(p,1e-6,1-1e-6))
                            + (1-y_bin)*np.log(np.clip(1-p,1e-6,1-1e-6))))
        return p.mean(), b, ll
    p1m,b1,l1 = stats(p1); p2m,b2,l2 = stats(p2)
    print(f"{L:>6} {y_bin.mean():.3f} | {p1m:.3f}    {b1:.4f}   {l1:.4f} | "
          f"{p2m:.3f}    {b2:.4f}   {l2:.4f}")
    ou_rows.append({"line": L, "emp": float(y_bin.mean()),
                    "M1_avgp": float(p1m), "M1_brier": b1, "M1_ll": l1,
                    "M2_avgp": float(p2m), "M2_brier": b2, "M2_ll": l2})
pd.DataFrame(ou_rows).to_csv(EVAL / "overunder_comparison.csv", index=False)

# ---------- Save predictions (always include both layers) -------------
test_out = test[[
    "game_id","game_date","player_id","player_name","team","opp_name",
    "is_home","position_mode","points","minutes","mu_hat",
    "p0", "gate", "alpha", "var", "nb0",
]].copy()
test_out["baseline_global_mean"] = train["points"].mean()
test_out["baseline_season_ppg"]  = test["pts_mean_season"].fillna(train["points"].mean()).values
test_out["baseline_ewma_ppg"]    = test["pts_ewma_10"].fillna(train["points"].mean()).values
test_out["final_model_tag"]      = FINAL_MODEL
pl.from_pandas(test_out).write_parquet(DATA / "predictions.parquet")

# ---------- Reliability for line 14.5: M1 vs M2 -----------------------
L = 14.5
y_bin = (y_test > L).astype(int)
p1 = 1 - m1_cdf(L)
p2 = 1 - m2_cdf(L)
def reliability(p, y, n_bins=10):
    bins = np.linspace(0, 1, n_bins+1)
    idx = np.clip(np.digitize(p, bins)-1, 0, n_bins-1)
    xs, ys = [], []
    for b in range(n_bins):
        m = idx == b
        if m.sum() < 30: continue
        xs.append(p[m].mean()); ys.append(y[m].mean())
    return xs, ys

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot([0,1],[0,1],"k--", label="perfect")
for p, lab in [(p1,"M1 plain NB"), (p2,"M2 corrected ZINB")]:
    xs, ys = reliability(p, y_bin)
    ax.plot(xs, ys, "o-", label=lab)
ax.set_xlabel(f"Predicted P(points > {L})"); ax.set_ylabel("Empirical rate")
ax.set_title(f"Reliability — line {L} (plain NB vs corrected ZINB)")
ax.legend(); ax.grid(alpha=0.3)
fig.savefig(PLOT / "reliability_14_5_compare.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# ---------- Persist summary metrics ----------------------------------
summary = {
    "n_test": int(len(test)),
    "M1_plain_NB": {"CRPS": crps_m1, "PIT_KS_dev": ks_m1},
    "M2_corrected_ZINB": {"CRPS": crps_m2, "PIT_KS_dev": ks_m2,
                          "trained_p0_mean": float(test["p0"].mean()),
                          "computed_gate_mean": float(test["gate"].mean()),
                          "implied_P0_mean": float((test["gate"] + (1-test["gate"])*test["nb0"]).mean())},
    "final_choice": FINAL_MODEL,
    "over_under": ou_rows,
}
with open(EVAL / "metrics.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\n[04] Final layer choice: {FINAL_MODEL}")
print(f"[04] Wrote metrics.json")
