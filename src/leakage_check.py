"""
leakage_check.py
Quant Analyst: Giorgio Maarraoui
=========================================================================================================================================================================
Sanity test that the engineered features contain no information from the target game or later games.

Method
------
1. Pick a random sample of (player, game) test rows.
2. For each, recompute a key feature (`pts_mean_season`) by hand from the player-game frame restricted to strictly earlier games in the same season.
3. Assert that the recomputed value matches the feature value in features.parquet.

I picked pts_mean_season because it's the most likely place for an off-by-one (forgot to `.shift(1)`) bug. If this passes, the same shift logic applies to all EWMAs.
"""

import polars as pl
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
pg   = pl.read_parquet(ROOT / "artifacts" / "data" / "player_game.parquet").sort(["player_id", "game_date_time"])
feat = pl.read_parquet(ROOT / "artifacts" / "data" / "features.parquet").sort(["player_id", "game_date_time"])

# Re-derive season exactly as features.py does; season starts around October.
pg = pg.with_columns(
    pl.when(pl.col("game_date").dt.month() >= 10)
      .then(pl.col("game_date").dt.year())
      .otherwise(pl.col("game_date").dt.year() - 1)
      .alias("season")
)

# Pick 300 random test rows that have a non-null feature.
np.random.seed(42)
candidates = feat.filter(pl.col("pts_mean_season").is_not_null())
idx = np.random.choice(len(candidates), size=300, replace=False)
sample = candidates[idx]

errs = 0
for row in sample.iter_rows(named=True):
    pid = row["player_id"]
    season = row["season"]
    game_dt = row["game_date_time"]
    expected = row["pts_mean_season"]
    # Manual recomputation: mean of player's prior games this season (I use less than to only include previous seasons)
    prior = pg.filter(
        (pl.col("player_id") == pid)
        & (pl.col("season") == season)
        & (pl.col("game_date_time") < game_dt)
    )
    if len(prior) == 0:
        manual = None
    else:
        manual = float(prior["points"].mean())
    # Compare, allowing for small tolerance
    if manual is None and expected is not None:
        print(f"FAIL: row {pid}/{game_dt} expected {expected} but no prior games")
        errs += 1
        continue
    if manual is not None and abs(manual - expected) > 1e-6:
        print(f"FAIL: row {pid}/{game_dt} got {expected}, manual {manual}")
        errs += 1
        continue

print(f"[leakage_check] checked 300 random rows, errors = {errs}")
assert errs == 0, "LEAKAGE DETECTED, need to investigate before training!!"
print("[leakage_check] OK; no leakage in pts_mean_season; trust the same shift logic for EWMAs.")
