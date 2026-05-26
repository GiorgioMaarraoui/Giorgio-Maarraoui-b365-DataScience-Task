"""
01_aggregate.py
Quant Analyst: Giorgio Maarraoui
====================================================================================================================
The objective of this script is to collapse the possession-level dataset (5.49M rows, 10 per possession) down to
ONE ROW PER PLAYER-GAME, which is the level required for the model predictions.

Why Polars instead of pandas
----------------------------
Both libraries do similar tasks, but Polars is columnar and approx 3-5x cheaper in memory; pandas expands to ~1.5 GB.

Output schema
-------------
game_id, game_date, player_id, player_name, team_id, team, opp_id, opp_name, is_home, position_mode, points, fgm,
fga, tpm, tpa, ftm, fta, oreb, dreb, trb, ast, stl, blk, tov, pf, minutes, n_possessions_on_floor, team_poss_off,
team_pts_off, opp_poss_off, opp_pts_off

Key decisions
-------------------------
1. Minutes ≈ max(time_played) / 60.

2. The schema says `time_played` is "cumulative seconds played by this player up to this possession". However, in the
   data, possession #1 always shows time_played=0, so it's the cumulative time AT THE START of each possession. This means
   max() systematically under-estimates true minutes by the duration of the player's last possession on the floor (~0-25 s).
   For our purposes that error is small, non-systematic across players, and cancels out in rate features.

2. n_possessions_on_floor = number of distinct possessions in which the player appears. I use this for the player's true
   exposure for usage/pace calculations.

3. Modal position per game: A player's position can show as "SG" in one row and "G" in another; I take the mode across
   the player's rows in this game. Then I collapse the 7-way position column to 5 canonical buckets in the feature step.

4. Team possessions for the game (offensive possessions) is needed for opponent pace and for the denominator of
   pace-adjusted opponent defensive ratings.

5. opp_pts_off is the opponent's points scored = points allowed by the player's team. Used to build opponent's
   def-rating history.
"""

import polars as pl
from pathlib import Path

# 01_1 paths -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
RAW_PARQUET = ROOT / "nba_possessions.parquet"
OUT_DIR = ROOT / "artifacts" / "data"
OUT_DIR.mkdir(exist_ok=True, parents=True)
OUT_PATH = OUT_DIR / "player_game.parquet"

print(f"[01] Reading raw parquet: {RAW_PARQUET}")
# scan_parquet is lazy; Polars only loads the columns we need
lf = pl.scan_parquet(RAW_PARQUET).select([
    "game_id", "game_date_time",
    "home_id", "home_name", "away_id", "away_name",
    "possession_number", "period",
    "offense_team", "possession_points",
    "player_id", "player_name", "team_id", "team", "position",
    "time_played",
    "field_goals_made", "field_goals_attempted",
    "three_point_field_goals_made", "three_point_field_goals_attempted",
    "free_throws_made", "free_throws_attempted",
    "offensive_rebounds", "defensive_rebounds", "rebounds",
    "assists", "steals", "blocks", "turnovers", "fouls",
    "points",
])

# 01_2 player-game aggregation ------------------------------------------------------------
sum_cols = [
    "field_goals_made", "field_goals_attempted",
    "three_point_field_goals_made", "three_point_field_goals_attempted",
    "free_throws_made", "free_throws_attempted",
    "offensive_rebounds", "defensive_rebounds", "rebounds",
    "assists", "steals", "blocks", "turnovers", "fouls",
    "points",
]

# By game_id, player_id
pg_lf = lf.group_by(["game_id", "player_id"]).agg(
    # Sum each of the sum_cols above
    *[pl.col(c).sum().alias(c) for c in sum_cols],
    # Compute max time and number of possessions on floor
    pl.col("time_played").max().alias("minutes_sec"),
    pl.col("possession_number").n_unique().alias("n_possessions_on_floor"),
    # Take first instance of constant metadata
    pl.col("player_name").first(),
    pl.col("team_id").first(),
    pl.col("team").first(),
    pl.col("home_id").first(),
    pl.col("home_name").first(),
    pl.col("away_id").first(),
    pl.col("away_name").first(),
    pl.col("game_date_time").first(),
    # Take Modal position.
    pl.col("position").mode().first().alias("position_mode"),
)

pg = pg_lf.collect()
print(f"[01] Aggregated to {len(pg):,} player-games")

# 01_3 team offensive possessions for opponent feature later -----------------------
# Deduplicate to one row per posession, then group by offense_team.
poss_lf = lf.unique(subset=["game_id", "possession_number"]).select([
    "game_id", "possession_number", "offense_team", "possession_points",
])
team_off = (
    poss_lf.group_by(["game_id", "offense_team"])
    .agg(
        pl.col("possession_number").n_unique().alias("team_poss_off"),
        pl.col("possession_points").sum().alias("team_pts_off"),
    )
    .rename({"offense_team": "team_id"})
    .collect()
)
print(f"[01] Computed team-off rows: {len(team_off):,}")

# Build the opponent variant by renaming.
opp_off = team_off.rename({
    "team_id": "opp_id",
    "team_poss_off": "opp_poss_off",
    "team_pts_off": "opp_pts_off",
})

# 01_4 derive home/away, opponent identity -----------------------------------------
pg = pg.with_columns(
    (pl.col("team_id") == pl.col("home_id")).cast(pl.Int8).alias("is_home"),
    pl.when(pl.col("team_id") == pl.col("home_id"))
      .then(pl.col("away_id"))
      .otherwise(pl.col("home_id")).alias("opp_id"),
    pl.when(pl.col("team_id") == pl.col("home_id"))
      .then(pl.col("away_name"))
      .otherwise(pl.col("home_name")).alias("opp_name"),
    (pl.col("minutes_sec") / 60.0).alias("minutes"),
    pl.col("game_date_time").dt.date().alias("game_date"),
)

# Join team and opponent per-game stats.
pg = pg.join(team_off, on=["game_id", "team_id"], how="left")
pg = pg.join(opp_off, on=["game_id", "opp_id"], how="left")

# 01_5 final tidy and quick sanity pritns -------------------------------------------
pg = pg.sort(["player_id", "game_date_time"])

print(
    f"[01] Date range: {pg['game_date'].min()} → {pg['game_date'].max()}"
)
print(
    f"[01] Points: mean={pg['points'].mean():.2f}, "
    f"median={pg['points'].median():.0f}, "
    f"max={pg['points'].max()}, "
    f"frac_zero={(pg['points']==0).mean():.3f}"
)
print(
    f"[01] Avg team off possessions / game: "
    f"{team_off['team_poss_off'].mean():.1f}"
)

pg.write_parquet(OUT_PATH)
print(f"[01] Wrote {OUT_PATH}  ({OUT_PATH.stat().st_size/1e6:.1f} MB)")
