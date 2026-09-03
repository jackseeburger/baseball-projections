"""Team-season walk-forward: cut a season at a date, project the rest of it.

The team analogue of `src/eval/intraseason.py`. That module cuts a season at a
date, hands a rate model everything strictly before it, and scores the rest of
the season against what the players actually did. This one does the same thing
one level up the rollup: it cuts a *season* at a date, hands the whole station
E/F/G chain everything strictly before it, simulates the remainder, and returns
each club's projected final wins and its playoff, division, pennant and World
Series probabilities — the four numbers `public/data/playoff_odds/` publishes
every night and that nothing has ever scored.

The split at an as-of date produces two game frames:

    played     regular-season games with `date <  as_of`, results and all.
               Everything the projection is allowed to see: the standings, the
               run environment, the rate tables, the pen and rotation windows.
    future     regular-season games with `date >= as_of`. The schedule the
               Monte Carlo draws, and — once the season is over — the outcome
               every arm is scored against.

`assert_team_split_clean` is the leakage guard, and it is stricter than a
comment: it walks every dated frame the chain reads and raises if any of them
carries a game on or after the cutoff. The cut itself is applied twice on
purpose — once here, when `chain_inputs_before` builds `ChainInputs` from
game logs already truncated at the cutoff, and once inside
`game_model.build_slate`, which cuts every frame again on its way to a slate.
Either cut alone would be enough; both together mean a bug in one of them
fails the guard instead of quietly improving the score.

**The projection is the production one.** `project_chain` calls
`scripts/run_playoff_odds.chain_terms` with the fetch injected, which is the
same function the nightly job calls, and then `sim.odds.run_playoff_odds`,
which is the same Monte Carlo. Nothing about the model is re-implemented here;
what this module owns is the *cut*, the baselines and the outcomes.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date as date_cls
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sim import game_model as gm
from src.sim.bracket import DEFAULT_ROTATION_SIZE, PlayoffFormat, format_for_season
from src.sim.odds import run_playoff_odds
from src.sim.season import SeasonState
from src.sim.strength import estimate_hfa, regressed_strength

REGULAR_SEASON = "R"
# Wild card, division series, league championship, World Series.
POSTSEASON_TYPES = ("F", "D", "L", "W")
# The window the nightly job looks ahead for posted probables, and therefore
# the window this harness lets the starter term reach. See
# `docs/team-projection-backtest.md` for why that is slightly generous.
STARTER_WINDOW_DAYS = 7
DEFAULT_REGRESS_GAMES = 60.0
# How hard the preseason arm regresses last season's run rates toward the
# league. A full season of ballast against a full season of evidence is a
# half-and-half shrink, which is roughly where a club's next-year win% sits
# relative to its last one.
PRESEASON_REGRESS_GAMES = 162.0
# A club that has won two of three in April is not a .667 club; the
# extrapolate-your-own-rate baseline is capped so the Monte Carlo cannot be
# handed a 1.000 or a .000 talent level in the season's first fortnight.
WPCT_FLOOR, WPCT_CEIL = 0.25, 0.75

PROB_COLUMNS = ("p_playoffs", "p_division", "p_pennant", "p_ws")


# ─── the split ───

def regular_season_games(schedule: pd.DataFrame,
                         teams: pd.DataFrame) -> pd.DataFrame:
    """Regular-season games between two major-league clubs, one row per game.

    Three shapes on the schedule are not one played game each, and all three
    have to go before anything is counted:

    * a **postponement** — status Final with no score, whose makeup appears as
      its own row — never happened and is dropped;
    * a **tie** (a suspended game called level) counts in neither the standings
      nor this frame;
    * a **suspended game finished on a later date appears twice under one
      `game_pk`**, once on the night it was stopped and once on the night it
      was completed, both marked Final and both carrying the final score. Four
      of them in 2025 alone. Left in, a club plays 165 games, the standings
      double-count four results and the Monte Carlo draws four games twice.
      The row kept is the **later** one, because that is the date the result
      became known and this harness is walk-forward.
    """
    valid = set(teams["team_id"].astype(int))
    reg = schedule[schedule["game_type"] == REGULAR_SEASON].copy()
    reg = reg[reg["home_id"].astype(int).isin(valid)
              & reg["away_id"].astype(int).isin(valid)]
    final = reg["status"] == "Final"
    scored = reg["home_score"].notna() & reg["away_score"].notna()
    tie = scored & (reg["home_score"] == reg["away_score"])
    reg = reg[~(final & (~scored | tie))]
    if "game_pk" in reg.columns:
        reg = (reg.sort_values("date")
               .drop_duplicates("game_pk", keep="last"))
    return reg.reset_index(drop=True)


def standings_from_games(played: pd.DataFrame,
                         teams: pd.DataFrame) -> pd.DataFrame:
    """Standings as of a cutoff, summed from the games before it.

    `mlb_stats_api.fetch_standings` serves *today's* table, which for a season
    already in the books is the final one — using it at an as-of date inside
    that season would hand the projection the answer. The whole harness turns
    on this function being the only source of a club's record and run
    environment. Columns are the ones `strength.regressed_run_rates` and
    `game_model.league_run_rates` read: team_id, wins, losses, runs_scored,
    runs_allowed, plus league and division for convenience.
    """
    ids = teams["team_id"].astype(int).to_numpy()
    zero = pd.Series(0.0, index=pd.Index(ids, name="team_id"))
    wins, losses = zero.copy(), zero.copy()
    rs, ra = zero.copy(), zero.copy()
    if len(played):
        h = played["home_id"].astype(int).to_numpy()
        a = played["away_id"].astype(int).to_numpy()
        hs = played["home_score"].astype(float).to_numpy()
        as_ = played["away_score"].astype(float).to_numpy()
        for i in range(len(h)):
            rs[h[i]] += hs[i]; ra[h[i]] += as_[i]
            rs[a[i]] += as_[i]; ra[a[i]] += hs[i]
            if hs[i] > as_[i]:
                wins[h[i]] += 1; losses[a[i]] += 1
            else:
                wins[a[i]] += 1; losses[h[i]] += 1
    out = pd.DataFrame({
        "team_id": ids,
        "wins": wins.reindex(ids).to_numpy().astype(int),
        "losses": losses.reindex(ids).to_numpy().astype(int),
        "runs_scored": rs.reindex(ids).to_numpy(),
        "runs_allowed": ra.reindex(ids).to_numpy(),
    })
    meta = teams.set_index("team_id")
    out["league_id"] = out["team_id"].map(meta["league_id"]).to_numpy()
    out["division_id"] = out["team_id"].map(meta["division_id"]).to_numpy()
    return out


@dataclass(frozen=True)
class TeamSplit:
    """One season cut at one date: what is known, what is left, and the state.

    `state.completed` and `state.remaining` are the two halves in the shape the
    Monte Carlo wants; `played` and `future` are the same games with their
    scores still attached, because the outcome side of the harness needs them.
    """
    season: int
    as_of: str
    teams: pd.DataFrame
    played: pd.DataFrame
    future: pd.DataFrame
    state: SeasonState
    standings: pd.DataFrame
    hfa: float

    @property
    def games_played(self) -> int:
        return int(len(self.played))

    @property
    def games_remaining(self) -> int:
        return int(len(self.future))

    def club_games_remaining(self) -> pd.Series:
        """team_id → games this club still has to play.

        Not `games_remaining / 15`: clubs run a few games apart all season and
        by September the gap is large enough to matter to a rate. The
        through-season curve divides a projected-wins error by this, which is
        the only way to compare an April error to a September one.
        """
        ids = self.teams["team_id"].astype(int)
        counts = pd.Series(0, index=pd.Index(ids, name="team_id"))
        for col in ("home_id", "away_id"):
            hit = self.future[col].astype(int).value_counts()
            counts = counts.add(hit.reindex(counts.index).fillna(0), fill_value=0)
        return counts.astype(int)

    @property
    def fmt(self) -> PlayoffFormat:
        return format_for_season(self.season)


def split_season_at(schedule: pd.DataFrame, teams: pd.DataFrame,
                    as_of: str, season: int) -> TeamSplit:
    """Everything a projection made on `as_of` is allowed to know, and no more.

    A game *on* the as-of date counts as remaining, matching the live job: the
    nightly runs before first pitch, so today's games are still to be drawn.
    """
    as_of = str(as_of)
    reg = regular_season_games(schedule, teams)
    dates = reg["date"].astype(str)
    # `played` is the games that were played *and* have a result, which for a
    # season in the books is every game before the cutoff and for a season
    # still running (2026) excludes anything the schedule has not settled yet.
    done = reg["home_score"].notna() & reg["away_score"].notna()
    played = reg[(dates < as_of) & done].reset_index(drop=True)
    future = reg[dates >= as_of].reset_index(drop=True)

    cols = ["game_pk", "date", "home_id", "away_id"]
    completed = played[cols].copy()
    completed["home_win"] = (played["home_score"].astype(float)
                             > played["away_score"].astype(float)).to_numpy()
    state = SeasonState(
        teams=teams.sort_values("team_id").reset_index(drop=True),
        completed=completed.sort_values("date").reset_index(drop=True),
        remaining=future[cols].sort_values("date").reset_index(drop=True),
    )
    standings = standings_from_games(played, teams)
    return TeamSplit(season=int(season), as_of=as_of, teams=state.teams,
                     played=played, future=future, state=state,
                     standings=standings, hfa=estimate_hfa(state.completed))


def weekly_cutoffs(schedule: pd.DataFrame, teams: pd.DataFrame,
                   *, step_days: int = 7, skip_days: int = 14,
                   min_remaining: int = 30) -> list[str]:
    """Weekly as-of dates through one season's regular schedule.

    Starts `skip_days` after opening day — before that a club has played a
    handful of games and every record-extrapolation baseline is noise — and
    stops once fewer than `min_remaining` games are left, because past that
    point the standings have decided the season and there is nothing for any
    model to project.

    For a season still in progress the walk also stops at the last game with a
    result on file: a cutoff past today would hand the projection a standings
    table built from games nobody has played.
    """
    reg = regular_season_games(schedule, teams)
    if reg.empty:
        return []
    dates = pd.to_datetime(reg["date"].astype(str))
    settled = reg["home_score"].notna() & reg["away_score"].notna()
    first, last = dates.min(), dates.max()
    if settled.any():
        last = min(last, dates[settled].max())
    out, cursor = [], first + pd.Timedelta(days=skip_days)
    while cursor <= last:
        iso = str(cursor.date())
        if int((dates >= cursor).sum()) < min_remaining:
            break
        out.append(iso)
        cursor = cursor + pd.Timedelta(days=step_days)
    return out


# ─── the leakage guard ───

def _max_date(frame: pd.DataFrame, col: str = "date"):
    if frame is None or not len(frame) or col not in frame.columns:
        return None
    s = frame[col].dropna().astype(str)
    return s.max() if len(s) else None


def _min_date(frame: pd.DataFrame, col: str = "date"):
    if frame is None or not len(frame) or col not in frame.columns:
        return None
    s = frame[col].dropna().astype(str)
    return s.min() if len(s) else None


def assert_team_split_clean(
    split: TeamSplit,
    *,
    inputs: "gm.ChainInputs | None" = None,
    probables: pd.DataFrame | None = None,
    window_days: int = STARTER_WINDOW_DAYS,
) -> None:
    """Raise if any game on or after the as-of date reached the inputs.

    The team analogue of `intraseason.assert_split_clean`, and the same
    contract: every frame the projection reads is checked against the cutoff,
    not trusted to have been cut. Six things are asserted.

    1. No game in `played` — and therefore none in `state.completed` or in the
       standings summed from it — falls on or after the cutoff.
    2. No game in `future` falls before it.
    3. The standings frame reconciles, game for game, with `played`: a club's
       wins plus losses equal the games it appears in. A standings table
       fetched from the API (which serves the *final* table for a season in
       the books) fails this immediately, which is the mistake this guard
       exists to catch.
    4. Every dated frame inside `ChainInputs` — pitching appearances, relief
       outings, pitch counts, starts, start innings, hitting lines, plate
       appearances — ends strictly before the cutoff.
    5. `ChainInputs`' prior-season frames carry no row from the season being
       projected.
    6. The probables frame reaches no further than the starter window the live
       job sees. For a season already played, the API serves the pitcher who
       actually started on any past date, so an untruncated probables feed
       would hand the chain every remaining game's starter — the one leak in
       this harness that would look like skill.
    """
    as_of = str(split.as_of)
    late = _max_date(split.played)
    if late is not None and late >= as_of:
        n = int((split.played["date"].astype(str) >= as_of).sum())
        raise ValueError(
            f"leakage: {n} played game(s) fall on or after the cutoff {as_of} "
            f"(latest {late})")
    early = _min_date(split.future)
    if early is not None and early < as_of:
        n = int((split.future["date"].astype(str) < as_of).sum())
        raise ValueError(
            f"leakage: {n} remaining game(s) fall before the cutoff {as_of} "
            f"(earliest {early})")
    if not len(split.future):
        raise ValueError(f"leakage guard: nothing left to project at {as_of}")

    counted = int((split.standings["wins"] + split.standings["losses"]).sum())
    if counted != 2 * split.games_played:
        raise ValueError(
            f"leakage: the standings account for {counted} club-games but "
            f"only {2 * split.games_played} were played before {as_of}; the "
            f"standings were not summed from the pre-cutoff games")

    if inputs is not None:
        dated = {
            "pitcher_counts": inputs.pitcher_counts, "relief": inputs.relief,
            "usage": inputs.usage, "starts": inputs.starts,
            "start_ip": inputs.start_ip, "hitter_counts": inputs.hitter_counts,
            "hitter_pa": inputs.hitter_pa,
        }
        for name, frame in dated.items():
            latest = _max_date(frame)
            if latest is not None and latest >= as_of:
                bad = int((frame["date"].astype(str) >= as_of).sum())
                raise ValueError(
                    f"leakage: ChainInputs.{name} carries {bad} row(s) on or "
                    f"after the cutoff {as_of} (latest {latest})")
        for name, frame in (("pitcher_prior_counts", inputs.pitcher_prior_counts),
                            ("hitter_prior_counts", inputs.hitter_prior_counts)):
            if frame is not None and "season" in frame.columns and len(frame):
                worst = int(pd.to_numeric(frame["season"]).max())
                if worst >= split.season:
                    raise ValueError(
                        f"leakage: ChainInputs.{name} contains season {worst}, "
                        f"which is not a completed season before "
                        f"{split.season}")

    if probables is not None and len(probables):
        horizon = str((pd.Timestamp(as_of)
                       + pd.Timedelta(days=int(window_days))).date())
        latest = _max_date(probables)
        if latest is not None and latest > horizon:
            bad = int((probables["date"].astype(str) > horizon).sum())
            raise ValueError(
                f"leakage: {bad} probable-starter row(s) fall past the "
                f"{window_days}-day starter window ending {horizon} "
                f"(latest {latest}); a past season's feed serves the pitcher "
                f"who actually started, so this is every remaining game's "
                f"starter, not an announcement")


def chain_inputs_before(season: int, pitching_logs: pd.DataFrame,
                        hitting_logs: pd.DataFrame, prior_pitching: pd.DataFrame,
                        prior_hitting: pd.DataFrame, as_of: str) -> gm.ChainInputs:
    """`ChainInputs` built from game logs already truncated at the cutoff.

    `build_slate` cuts again; this cut is what makes the guard above able to
    see a leak at all, since `ChainInputs` as the nightly job builds it holds
    the whole season and is only sliced on its way to a slate.
    """
    as_of = str(as_of)

    def cut(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or not len(frame):
            return frame
        return frame[frame["date"].astype(str) < as_of].reset_index(drop=True)

    return gm.ChainInputs.from_logs(season, cut(pitching_logs), cut(hitting_logs),
                                    prior_pitching, prior_hitting)


def probables_to_window(probables: pd.DataFrame, as_of: str,
                        window_days: int = STARTER_WINDOW_DAYS) -> pd.DataFrame:
    """Trim a season's probables feed to what the live job would have seen.

    Rows before the cutoff stay (they are the start log the postseason rotation
    pool is built from, and they are settled facts); rows after it survive only
    inside the `window_days` the nightly job looks ahead for announcements.
    """
    horizon = str((pd.Timestamp(str(as_of))
                   + pd.Timedelta(days=int(window_days))).date())
    return probables[probables["date"].astype(str) <= horizon].reset_index(drop=True)


# ─── the arms ───

def strength_even(split: TeamSplit) -> pd.Series:
    """Every club a .500 coin flip: the season is pure standings arithmetic.

    The control `docs/playoff-odds-validation.md` has scored against since the
    first run, and — read as a projection of final wins — the "current record,
    .500 the rest of the way" baseline.
    """
    return pd.Series(0.5, index=pd.Index(split.state.team_ids, name="team_id"),
                     name="strength")


def strength_own_rate(split: TeamSplit) -> pd.Series:
    """Each club plays out the string at its own season-to-date win rate.

    The naive baseline a fan uses, and the hardest of the three to beat in
    September, when a club's record *is* most of what is knowable about it.
    """
    st = split.standings.set_index("team_id")
    games = (st["wins"] + st["losses"]).clip(lower=1)
    pct = (st["wins"] / games).clip(WPCT_FLOOR, WPCT_CEIL)
    return pd.Series(
        {int(t): float(pct.get(int(t), 0.5)) for t in split.state.team_ids},
        name="strength")


def strength_preseason(prior_standings: pd.DataFrame,
                       team_ids,
                       regress_games: float = PRESEASON_REGRESS_GAMES) -> pd.Series:
    """Last completed season's run rates, regressed — the preseason arm.

    Held fixed for the whole season by construction: it reads nothing dated
    inside the season being projected, so the same vector serves every cutoff.
    A roster-based preseason system would be better, and this repository has no
    archive of one for 2015; the doc says so rather than dressing this up.
    """
    s = regressed_strength(prior_standings, regress_games=regress_games)
    return pd.Series({int(t): float(s.get(int(t), 0.5)) for t in team_ids},
                     name="strength")


# ─── projecting ───

def _projection_frame(split: TeamSplit, odds: pd.DataFrame,
                      arm: str) -> pd.DataFrame:
    """The odds table reshaped into the harness's long row-per-club frame."""
    st = split.standings.set_index("team_id")
    out = pd.DataFrame({
        "season": split.season,
        "as_of": split.as_of,
        "arm": arm,
        "team_id": odds["team_id"].astype(int).to_numpy(),
        "abbrev": odds["abbrev"].to_numpy(),
        "league_id": odds["league_id"].astype(int).to_numpy(),
        "division_id": odds["division_id"].astype(int).to_numpy(),
        "strength": odds["strength"].astype(float).to_numpy(),
        "proj_final_wins": odds["mean_wins"].astype(float).to_numpy(),
    })
    out["wins_to_date"] = out["team_id"].map(st["wins"]).astype(int)
    out["losses_to_date"] = out["team_id"].map(st["losses"]).astype(int)
    out["proj_rest_wins"] = out["proj_final_wins"] - out["wins_to_date"]
    for c in PROB_COLUMNS:
        out[c] = odds[c].astype(float).to_numpy()
    out["games_played"] = split.games_played
    out["games_remaining"] = split.games_remaining
    out["club_games_remaining"] = out["team_id"].map(
        split.club_games_remaining()).astype(int)
    return out


def project(split: TeamSplit, strength: pd.Series, arm: str, *,
            n_sims: int = 5_000, seed: int = 0,
            p_home_overrides: dict[int, float] | None = None,
            rotations=None) -> pd.DataFrame:
    """Simulate the rest of the season from one strength vector.

    The Monte Carlo is `sim.odds.run_playoff_odds`, unchanged and unwrapped,
    with the season's own postseason format — five clubs a league before 2022,
    six from 2022 — because the outcome being scored changed shape then.
    """
    odds = run_playoff_odds(split.state, strength, split.hfa, n_sims=n_sims,
                            seed=seed, p_home_overrides=p_home_overrides,
                            rotations=rotations, fmt=split.fmt)
    return _projection_frame(split, odds, arm)


def project_chain(split: TeamSplit, inputs: gm.ChainInputs,
                  probables: pd.DataFrame, schedule: pd.DataFrame, *,
                  n_sims: int = 5_000, seed: int = 0,
                  window_days: int = STARTER_WINDOW_DAYS,
                  rotation_size: int = DEFAULT_ROTATION_SIZE,
                  regress_games: float = DEFAULT_REGRESS_GAMES,
                  ) -> tuple[pd.DataFrame, dict]:
    """The production projection, at a past date.

    Calls `scripts/run_playoff_odds.chain_terms` with the fetch injected, so
    the strength every unannounced game is drawn with, the per-game
    probabilities on the announced ones and the postseason rotations are built
    by the same code the nightly job runs — not by a second implementation
    that agrees today.

    The posted-lineup branch is switched off (`cards` and `card_history` are
    empty). That is what the nightly job does in practice too: it runs at 09:15
    UTC and no club has published a card, so `n_games_with_lineups` reads 0 and
    the served model is exactly `pythag_C_sp_bpa_ip_lvl`. Feeding a backtest the
    cards of games that had not been played would be leakage of a different
    kind, since lineups go up two to four hours before first pitch.
    """
    from scripts import run_playoff_odds as rp   # noqa: PLC0415 — see docstring

    as_of = date_cls.fromisoformat(split.as_of)
    data = {"probables": probables, "inputs": inputs,
            "cards": {}, "card_history": {}}
    overrides, rotations, strength, diag = rp.chain_terms(
        split.season, split.state, split.standings, schedule, split.hfa,
        regress_games, as_of, window_days=window_days,
        rotation_size=rotation_size, use_lineups=False, data=data)
    frame = project(split, strength, "chain", n_sims=n_sims, seed=seed,
                    p_home_overrides=overrides or None,
                    rotations=rotations if rotations.by_team else None)
    return frame, diag


# ─── outcomes ───

def final_records(schedule: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """Each club's actual final regular-season wins and losses."""
    reg = regular_season_games(schedule, teams)
    done = reg[reg["status"] == "Final"]
    st = standings_from_games(done, teams)
    return st.rename(columns={"wins": "final_wins", "losses": "final_losses"})[
        ["team_id", "final_wins", "final_losses"]]


def postseason_outcomes(schedule: pd.DataFrame,
                        teams: pd.DataFrame) -> pd.DataFrame:
    """Who played in October, who reached the World Series and who won it.

    Read off the postseason schedule rather than a standings flag, because it
    is the outcome itself: a club that appears in any postseason game made the
    playoffs, one that appears in a World Series game won a pennant, and the
    winner of the last World Series game is the champion.
    """
    ids = teams["team_id"].astype(int).to_numpy()
    out = pd.DataFrame({"team_id": ids, "made_playoffs": 0, "won_pennant": 0,
                        "won_ws": 0})
    post = schedule[schedule["game_type"].isin(POSTSEASON_TYPES)].copy()
    post = post[post["home_score"].notna() & post["away_score"].notna()]
    if post.empty:
        return out
    played = set(post["home_id"].astype(int)) | set(post["away_id"].astype(int))
    ws = post[post["game_type"] == "W"].sort_values("date")
    pennants = set(ws["home_id"].astype(int)) | set(ws["away_id"].astype(int))
    champ = None
    if len(ws):
        last = ws.iloc[-1]
        champ = int(last["home_id"] if last["home_score"] > last["away_score"]
                    else last["away_id"])
    out["made_playoffs"] = [int(t in played) for t in ids]
    out["won_pennant"] = [int(t in pennants) for t in ids]
    out["won_ws"] = [int(t == champ) for t in ids]
    return out


def season_outcomes(schedule: pd.DataFrame, teams: pd.DataFrame,
                    standings: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    """Every outcome one season's projections are scored against.

    `standings` is the API's *final* table for the season — the one place it is
    safe to read, because this frame is the answer, not an input.

    With `strict`, the wins this harness counts off the schedule must equal the
    wins the API's own table reports, club for club. That reconciliation is the
    end-to-end check on the whole game-accounting path — the same summation
    that builds every as-of standings table, where no independent answer
    exists to check it against. It is what caught suspended games being counted
    twice (see `regular_season_games`), which had four clubs finishing 2025
    with 163 to 165 games.
    """
    out = final_records(schedule, teams).merge(
        postseason_outcomes(schedule, teams), on="team_id")
    st = standings.set_index("team_id")
    out["won_division"] = out["team_id"].map(
        st["division_champ"].astype(int)).fillna(0).astype(int)
    if strict and "wins" in st.columns:
        check = out.assign(api_wins=out["team_id"].map(st["wins"]),
                           api_losses=out["team_id"].map(st["losses"]))
        bad = check[(check["final_wins"] != check["api_wins"])
                    | (check["final_losses"] != check["api_losses"])]
        if len(bad):
            worst = bad.iloc[0]
            raise ValueError(
                f"game accounting: {len(bad)} club(s) disagree with the API's "
                f"final standings — team {int(worst['team_id'])} counted "
                f"{int(worst['final_wins'])}-{int(worst['final_losses'])} off "
                f"the schedule against {int(worst['api_wins'])}-"
                f"{int(worst['api_losses'])} reported")
    return out


OUTCOME_OF = {"p_playoffs": "made_playoffs", "p_division": "won_division",
              "p_pennant": "won_pennant", "p_ws": "won_ws"}


__all__ = [
    "TeamSplit", "split_season_at", "standings_from_games",
    "regular_season_games", "weekly_cutoffs", "assert_team_split_clean",
    "chain_inputs_before", "probables_to_window", "strength_even",
    "strength_own_rate", "strength_preseason", "project", "project_chain",
    "final_records", "postseason_outcomes", "season_outcomes",
    "PROB_COLUMNS", "OUTCOME_OF", "STARTER_WINDOW_DAYS",
]
