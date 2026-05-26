"""
02_features.py
Quant Analyst: Giorgio Maarraoui
===================================================================================================
the objective of this script is to engineer pre-game features for each (player, game) pair.

Output: one row per (game, player) with 40 features grouped into:

  * PLAYER FORM (rolling, per-player, season-scoped): EWMA of points (spans 5/10/20); season-to-date PPG;
    rolling 10-game std of points and minutes; EWMA-10 of minutes, FGA, FTA, 3PA; usage proxy;
    points-per-36 and points-per-minute; true-shooting%EWMA-20;
    team-attempt-share EWMA-10 (player FGA / team FGA);
    starts-proxy EWMA-10 (frac of recent games on possession 1); games-played-so-far.

  * CONTEXT (per-row, pre-game): is_home; days_rest; is_back_to_back; is_three_in_four (today is the
    3rd game in a 4-day window); games_last_5d / 14d; position code (7-way and 3-way buckets).

  * TEAM / OPPONENT (rolling, per-team, season-scoped): Def-rating, off-rating, pace;
    FGA/3PA/FTA allowed per 100; OT-rate; opp points allowed to player's 3-way position bucket;
    team's own pace and off-rating; expected pace (harmonic mean).

  * SCHEDULE: team_game_idx_in_season (at unique team-game level); opp_game_idx_in_season; month.

Note: For every feature value at row r, the value is computable from rows whose game_date_time < r.game_date_time
(per the relevant grain — player, team, or opponent). This is enforced by sort + shift(1), grouped by
(entity, season) so windows reset across season boundaries. I also verify this by leakage_check.py.

Pre-game interpretations:
* is_three_in_four uses today's date and the date of 2 games ago. Both are known at tip-off (since the schedule is public).
* days_rest uses today's date − previous game's date.
* team_game_idx_in_season is the team's 0-indexed game number in the season, computable from the schedule pre-game.
"""

import polars as pl
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IN_PATH  = ROOT / "artifacts" / "data" / "player_game.parquet"
OUT_PATH = ROOT / "artifacts" / "data" / "features.parquet"

print(f"[02] Reading player-game: {IN_PATH}")
pg = pl.read_parquet(IN_PATH)
print(f"[02] Rows: {len(pg):,}")

# --- season assignment ---------------------------------------------------
# Usually starts from the October
pg = pg.with_columns(
    pl.when(pl.col("game_date").dt.month() >= 10)
      .then(pl.col("game_date").dt.year())
      .otherwise(pl.col("game_date").dt.year() - 1)
      .alias("season")
)

# --- starts proxy at player-game level ----------------------------------
# Did the player appear on possession #1 of the game? Computed once from
# the raw parquet (I re-open it just for this). It's a strong proxy for
# "starter" but not perfect (as some starters miss the opening tip).
raw = pl.scan_parquet(ROOT / "nba_possessions.parquet").select(
    ["game_id", "player_id", "possession_number"]
)
poss1 = (
    raw.filter(pl.col("possession_number") == 1)
       .select(["game_id", "player_id"])
       .unique()
       .with_columns(pl.lit(1, dtype=pl.Int8).alias("on_poss1"))
       .collect()
)
pg = pg.join(poss1, on=["game_id", "player_id"], how="left").with_columns(
    pl.col("on_poss1").fill_null(0).cast(pl.Int8)
)

# --- team_game_idx_in_season at unique team-game grain ------------------
# Assign 0,1,2... by date at the (team_id, season, game_id) level so
# every player on a team in a given game gets the same index. I join
# this back to the player-game frame to give the schedule
# position feature its meaning.
tgi = (
    pg.select(["team_id", "season", "game_id", "game_date_time"])
      .unique(["team_id", "season", "game_id"])
      .sort(["team_id", "season", "game_date_time"])
      .with_columns(
          pl.int_range(0, pl.len())
            .over(["team_id", "season"])
            .alias("team_game_idx_in_season")
      )
      .select(["team_id", "season", "game_id", "team_game_idx_in_season"])
)
pg = pg.join(tgi, on=["team_id", "season", "game_id"], how="left")

# Same for opp's game number. This is useful so model can spot e.g. opp first
# game of a season.
opp_gi = tgi.rename({
    "team_id": "opp_id",
    "team_game_idx_in_season": "opp_game_idx_in_season",
})
pg = pg.join(opp_gi, on=["opp_id", "season", "game_id"], how="left")

# --- per-player rolling features  -----------------------------------------
pg = pg.sort(["player_id", "season", "game_date_time"])

def per_player_features(df: pl.DataFrame) -> pl.DataFrame:
    """Apply rolling/EWMA features per (player_id, season). I make sure to season reset
    so a player's season opener has no carry over from last year.

    Every rolling/expanding feature ends with `.shift(1)` so the current
    row never contributes to its own feature value (leakage protection).
    """
    # Game-level derived signals.
    df = df.with_columns(
        # Usage proxy.
        (
            (pl.col("field_goals_attempted")
             + 0.44 * pl.col("free_throws_attempted")
             + pl.col("turnovers"))
            / pl.when(pl.col("n_possessions_on_floor") > 0)
                .then(pl.col("n_possessions_on_floor"))
                .otherwise(1)
        ).alias("usage_game"),
        # True shooting %.
        (
            pl.col("points")
            / (2.0 * (pl.col("field_goals_attempted")
                      + 0.44 * pl.col("free_throws_attempted")).clip(lower_bound=1e-9))
        ).alias("ts_pct_game"),
        # Per-36 minutes.
        (pl.col("points") / pl.col("minutes").clip(lower_bound=1.0) * 36.0)
            .alias("pts_per36_game"),
        # Direct points per minute (cleaner unit; equivalent info to per-36
        # but the model may use it differently when interacted with min_ewma).
        (pl.col("points") / pl.col("minutes").clip(lower_bound=1.0))
            .alias("pts_per_min_game"),
        # Player's share of team FGA (compute team-FGA-per-game first below).
    )

    def ewma(col: str, span: int) -> pl.Expr:
        return pl.col(col).ewm_mean(span=span, adjust=False, ignore_nulls=True)

    def roll_std(col: str, n: int) -> pl.Expr:
        return pl.col(col).rolling_std(window_size=n, min_samples=2)

    df = df.with_columns([
        # Recency PPG
        ewma("points", 5).alias("pts_ewma_5"),
        ewma("points", 10).alias("pts_ewma_10"),
        ewma("points", 20).alias("pts_ewma_20"),
        # Volatility for dispersion signal.
        roll_std("points", 10).alias("pts_std_10"),
        # Minutes form and volatility.
        ewma("minutes", 10).alias("min_ewma_10"),
        roll_std("minutes", 10).alias("min_std_10"),
        # Volume.
        ewma("field_goals_attempted", 10).alias("fga_ewma_10"),
        ewma("free_throws_attempted", 10).alias("fta_ewma_10"),
        ewma("three_point_field_goals_attempted", 10).alias("tpa_ewma_10"),
        # Usage proxy & efficiency.
        ewma("usage_game", 10).alias("usage_ewma_10"),
        ewma("pts_per36_game", 10).alias("pts_per36_ewma_10"),
        ewma("pts_per_min_game", 10).alias("pts_per_min_ewma_10"),
        ewma("ts_pct_game", 20).alias("ts_pct_ewma_20"),
        # Starts proxy.
        ewma("on_poss1", 10).alias("starts_proxy_ewma_10"),
        # Season-to-date accumulators (will be shifted below).
        pl.col("points").cum_sum().alias("__cum_pts"),
        pl.col("points").cum_count().alias("__cum_n"),
    ])

    rolling_cols = [
        "pts_ewma_5", "pts_ewma_10", "pts_ewma_20",
        "pts_std_10",
        "min_ewma_10", "min_std_10",
        "fga_ewma_10", "fta_ewma_10", "tpa_ewma_10",
        "usage_ewma_10", "pts_per36_ewma_10", "pts_per_min_ewma_10",
        "ts_pct_ewma_20", "starts_proxy_ewma_10",
        "__cum_pts", "__cum_n",
    ]
    df = df.with_columns([pl.col(c).shift(1).alias(c) for c in rolling_cols])

    # Season-to-date mean = cum_pts / cum_n (both already shifted).
    df = df.with_columns(
        (pl.col("__cum_pts") / pl.col("__cum_n").clip(lower_bound=1))
        .alias("pts_mean_season")
    ).drop(["__cum_pts", "__cum_n"])

    # Games played so far (pre-game): integer position within season.
    df = df.with_columns(
        pl.int_range(0, pl.len()).over(["player_id", "season"]).alias("games_played_so_far")
    )
    return df

pg = pg.group_by(["player_id", "season"], maintain_order=True).map_groups(per_player_features)

# --- days rest / back-to-back / 3-in-4 -----------------------------------
pg = pg.sort(["player_id", "game_date_time"])
pg = pg.with_columns(
    pl.col("game_date").shift(1).over("player_id").alias("__prev_date")
)
pg = pg.with_columns(
    ((pl.col("game_date") - pl.col("__prev_date")).dt.total_days())
        .clip(lower_bound=0, upper_bound=14)
        .fill_null(7)
        .alias("days_rest")
).drop("__prev_date")
pg = pg.with_columns(
    (pl.col("days_rest") == 1).cast(pl.Int8).alias("is_back_to_back"),
)
# "3 games in 4 days" = today's date minus date-2-games-ago <= 3 days.
# Interpretation: today is the third game in a 4-day window. Pre-game
# (schedule known).
pg = pg.with_columns(
    pl.col("game_date").shift(2).over("player_id").alias("__date_lag2")
)
pg = pg.with_columns(
    ((pl.col("game_date") - pl.col("__date_lag2")).dt.total_days() <= 3)
        .fill_null(False).cast(pl.Int8).alias("is_three_in_four")
).drop("__date_lag2")

# Games-played-by-player counts in trailing windows (pre-game): count of
# games in the last 5 / 14 days (excluding today). Implemented by
# iterating per player.
def games_in_trailing(df: pl.DataFrame, days: int, out_col: str) -> pl.DataFrame:
    """For each row, count this player's games with date in [today-days, today-1]."""
    # Polars rolling on dates: use group_by_rolling-style.
    out = []
    for (pid,), g in df.group_by(["player_id"], maintain_order=True):
        dates = g["game_date"].to_numpy().astype("datetime64[D]")
        counts = np.zeros(len(dates), dtype=np.int32)
        for i, d in enumerate(dates):
            lo = d - np.timedelta64(days, "D")
            hi = d - np.timedelta64(1, "D")
            counts[i] = int(np.sum((dates[:i] >= lo) & (dates[:i] <= hi)))
        out.append(g.with_columns(pl.Series(out_col, counts)))
    return pl.concat(out)

pg = games_in_trailing(pg, 5, "games_last_5d")
pg = games_in_trailing(pg, 14, "games_last_14d")

# --- position bucket -----------------------------------------------------
position_map = {"PG": 1, "SG": 2, "SF": 3, "PF": 4, "C": 5, "G": 6, "F": 7}
pg = pg.with_columns(
    pl.col("position_mode").replace_strict(position_map, default=0).alias("position_code")
)
# Simpler 3-way bucket for opp-by-position defensive features (G/F/C).
position_3way_map = {
    "PG": "G", "SG": "G", "G": "G",
    "SF": "F", "PF": "F", "F": "F",
    "C": "C",
}
pg = pg.with_columns(
    pl.col("position_mode").replace_strict(position_3way_map, default="F").alias("position_3way")
)

# --- TEAM / OPPONENT rolling; season-scoped ----------------------------
# Partition by (team_id, season) so the rolling window resets across
# season boundaries (teams potentually change rosters year to year).
team_game = (
    pg.select([
        "game_id", "season", "game_date_time", "game_date",
        "team_id", "opp_id",
        "team_poss_off", "team_pts_off",
        "opp_poss_off", "opp_pts_off",
    ])
    .unique(["game_id", "team_id"])
    .sort(["team_id", "season", "game_date_time"])
)

# Per-game derived signals at team-game grain.
team_game = team_game.with_columns(
    # Defensive rating: opp pts allowed per 100 poss.
    (pl.col("opp_pts_off") / pl.col("opp_poss_off").clip(lower_bound=1) * 100.0)
        .alias("def_rating_game"),
    # Offensive rating: team's own points per 100 of its own possessions.
    (pl.col("team_pts_off") / pl.col("team_poss_off").clip(lower_bound=1) * 100.0)
        .alias("off_rating_game"),
    # Pace ~ team possessions.
    pl.col("team_poss_off").alias("pace_game"),
)

# Did this team's game go to OT? Get max(period) per game from raw.
ot_raw = (
    pl.scan_parquet(ROOT / "nba_possessions.parquet")
      .group_by("game_id")
      .agg(pl.col("period").max().alias("max_period"))
      .collect()
)
team_game = team_game.join(ot_raw, on="game_id", how="left").with_columns(
    (pl.col("max_period") >= 5).cast(pl.Int8).alias("went_to_ot_game")
).drop("max_period")

# Opp shot-volume allowed per 100 possessions, computed at team-game levrl
# from the player-game aggregates summed up.
# First: per team-game, total FGA / 3PA / FTA AGAINST this team (i.e.,
# opponent's offensive volume in that game).
opp_vol = (
    pg.group_by(["game_id", "opp_id"])  # group by player's opponent — that team's offense
      .agg(
          pl.col("field_goals_attempted").sum().alias("opp_fga_in_game"),
          pl.col("three_point_field_goals_attempted").sum().alias("opp_3pa_in_game"),
          pl.col("free_throws_attempted").sum().alias("opp_fta_in_game"),
      )
      .rename({"opp_id": "team_id"})  # this is the offensive team's volume
)
# Now from the perspective of the defending team it's "volume allowed".
team_game = team_game.join(
    opp_vol.rename({"team_id": "_def_team_id"}),
    left_on=["game_id", "team_id"], right_on=["game_id", "_def_team_id"], how="left",
)
# Per 100 (using opp_poss_off).
team_game = team_game.with_columns(
    (pl.col("opp_fga_in_game") / pl.col("opp_poss_off").clip(lower_bound=1) * 100.0)
        .alias("fga_allowed_per100_game"),
    (pl.col("opp_3pa_in_game") / pl.col("opp_poss_off").clip(lower_bound=1) * 100.0)
        .alias("threepa_allowed_per100_game"),
    (pl.col("opp_fta_in_game") / pl.col("opp_poss_off").clip(lower_bound=1) * 100.0)
        .alias("fta_allowed_per100_game"),
)

# Rolling 20-game team stats, season scoped.
team_game = team_game.with_columns([
    pl.col("def_rating_game").rolling_mean(window_size=20, min_samples=3)
        .shift(1).over(["team_id", "season"]).alias("team_def_rating_roll20"),
    pl.col("off_rating_game").rolling_mean(window_size=20, min_samples=3)
        .shift(1).over(["team_id", "season"]).alias("team_off_rating_roll20"),
    pl.col("pace_game").rolling_mean(window_size=20, min_samples=3)
        .shift(1).over(["team_id", "season"]).alias("team_pace_roll20"),
    pl.col("fga_allowed_per100_game").rolling_mean(window_size=20, min_samples=3)
        .shift(1).over(["team_id", "season"]).alias("team_fga_allowed_per100_roll20"),
    pl.col("threepa_allowed_per100_game").rolling_mean(window_size=20, min_samples=3)
        .shift(1).over(["team_id", "season"]).alias("team_3pa_allowed_per100_roll20"),
    pl.col("fta_allowed_per100_game").rolling_mean(window_size=20, min_samples=3)
        .shift(1).over(["team_id", "season"]).alias("team_fta_allowed_per100_roll20"),
    pl.col("went_to_ot_game").rolling_mean(window_size=10, min_samples=3)
        .shift(1).over(["team_id", "season"]).alias("team_ot_rate_roll10"),
])

# Opponent view of all those.
opp_view = team_game.select([
    "game_id", "team_id",
    "team_def_rating_roll20", "team_off_rating_roll20", "team_pace_roll20",
    "team_fga_allowed_per100_roll20", "team_3pa_allowed_per100_roll20",
    "team_fta_allowed_per100_roll20",
    "team_ot_rate_roll10",
]).rename({
    "team_id": "opp_id",
    "team_def_rating_roll20": "opp_def_rating_roll20",
    "team_off_rating_roll20": "opp_off_rating_roll20",  # actually opp's offence
    "team_pace_roll20": "opp_pace_roll20",
    "team_fga_allowed_per100_roll20": "opp_fga_allowed_per100_roll20",
    "team_3pa_allowed_per100_roll20": "opp_3pa_allowed_per100_roll20",
    "team_fta_allowed_per100_roll20": "opp_fta_allowed_per100_roll20",
    "team_ot_rate_roll10": "opp_ot_rate_roll10",
})

# Player's own team rolling.
team_view = team_game.select([
    "game_id", "team_id",
    "team_pace_roll20", "team_off_rating_roll20",
])

pg = pg.join(team_view, on=["game_id", "team_id"], how="left")
pg = pg.join(opp_view, on=["game_id", "opp_id"], how="left")

# Expected pace is harmonic mean.
pg = pg.with_columns(
    (2.0 * pl.col("team_pace_roll20") * pl.col("opp_pace_roll20")
     / (pl.col("team_pace_roll20") + pl.col("opp_pace_roll20")).clip(lower_bound=1e-9))
        .alias("expected_pace")
)

# --- OPPONENT DEFENCE BY POSITION ----------------------------------------
# For each (opp_id, position_3way, season), compute the opponent's
# rolling-20-game mean of points allowed per game to that position.
# At level (opp_id, season, game_id, position_3way) sum players' points,
# then roll per (opp_id, season, position_3way).
pos_pts = (
    pg.group_by(["opp_id", "season", "game_id", "game_date_time", "position_3way"])
      .agg(pl.col("points").sum().alias("pts_allowed_to_position_game"))
      .sort(["opp_id", "season", "position_3way", "game_date_time"])
      .with_columns(
          pl.col("pts_allowed_to_position_game")
            .rolling_mean(window_size=20, min_samples=3)
            .shift(1)
            .over(["opp_id", "season", "position_3way"])
            .alias("opp_pts_allowed_to_pos_roll20")
      )
      .select(["opp_id", "season", "game_id", "position_3way",
               "opp_pts_allowed_to_pos_roll20"])
)
pg = pg.join(pos_pts, on=["opp_id", "season", "game_id", "position_3way"], how="left")

# --- team_attempt_share rolling -----------------------------------------
# Player's share of team FGA in games they played, rolled.
# Step 1: team FGA per team-game.
team_fga = (
    pg.group_by(["game_id", "team_id"])
      .agg(pl.col("field_goals_attempted").sum().alias("team_fga_in_game"))
)
pg = pg.join(team_fga, on=["game_id", "team_id"], how="left")
pg = pg.with_columns(
    (pl.col("field_goals_attempted") / pl.col("team_fga_in_game").clip(lower_bound=1))
        .alias("fga_share_game")
)
# EWMA per (player, season), shifted to exclude current.
pg = pg.sort(["player_id", "season", "game_date_time"])
pg = pg.with_columns(
    pl.col("fga_share_game")
      .ewm_mean(span=10, adjust=False, ignore_nulls=True)
      .shift(1)
      .over(["player_id", "season"])
      .alias("team_attempt_share_ewma_10")
)

# --- calendar feature ----------------------------------------------------
pg = pg.with_columns(pl.col("game_date").dt.month().alias("month"))

# --- final tidy ------------------------------------------------------------
pg = pg.sort(["game_date_time", "player_id"])

# We keep team_id, opp_id, player_id as raw columns; the model script
# decides whether to use them as categorical.
FEATURES_NUMERIC = [
    # Player form
    "pts_ewma_5", "pts_ewma_10", "pts_ewma_20", "pts_mean_season",
    "pts_std_10",
    "min_ewma_10", "min_std_10",
    "fga_ewma_10", "fta_ewma_10", "tpa_ewma_10",
    "usage_ewma_10", "pts_per36_ewma_10", "pts_per_min_ewma_10",
    "ts_pct_ewma_20",
    "team_attempt_share_ewma_10",
    "starts_proxy_ewma_10",
    "games_played_so_far",
    # Context
    "is_home", "days_rest", "is_back_to_back", "is_three_in_four",
    "games_last_5d", "games_last_14d",
    # Opponent + team
    "opp_def_rating_roll20", "opp_off_rating_roll20", "opp_pace_roll20",
    "opp_fga_allowed_per100_roll20", "opp_3pa_allowed_per100_roll20",
    "opp_fta_allowed_per100_roll20", "opp_ot_rate_roll10",
    "opp_pts_allowed_to_pos_roll20",
    "team_pace_roll20", "team_off_rating_roll20",
    "expected_pace",
    # Season context
    "team_game_idx_in_season", "opp_game_idx_in_season",
    "month",
]
CATEGORICAL = ["position_code", "team_id", "opp_id"]  # player_id optional

ALL_FEATURES = FEATURES_NUMERIC + CATEGORICAL

print(f"[02] Final features ({len(ALL_FEATURES)}):")
for f in ALL_FEATURES:
    nn = pg[f].is_not_null().sum()
    print(f"     {f:<36} non-null: {nn:>6} / {len(pg):>6} ({100*nn/len(pg):.1f}%)")

keep_meta = [
    "game_id", "game_date", "game_date_time", "season",
    "player_id", "player_name", "team", "opp_name",
    "position_mode", "position_3way",
    "minutes", "points",
]
# Dedup against ALL_FEATURES (team_id, opp_id, position_code are in features).
keep = keep_meta + [c for c in ALL_FEATURES if c not in keep_meta]
pg.select(keep).write_parquet(OUT_PATH)
print(f"[02] Wrote {OUT_PATH}  ({OUT_PATH.stat().st_size/1e6:.1f} MB)")
