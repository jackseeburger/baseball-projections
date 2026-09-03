"""The live odds job and the backtest price the same game the same way.

Station E is a stack of terms — the starter over his own expected innings, the
availability-weighted pen over the rest, the posted card, all on top of station
C's bottom-up run environment — and until `src/sim/game_model.py` existed the
two callers assembled that stack separately: `scripts/backtest_game_odds.py`
walking a season to score it, `scripts/run_playoff_odds.py` pricing tonight to
serve it. Two assemblies of one model is how a scoreboard starts describing
something the site is not doing, and nothing fails when it happens.

So this fixes one game and asserts the two paths return the same P(home) to the
last bit.

The fixture is a whole synthetic season — 4 clubs, 45 dates, every pitching
appearance, every hitting line, every posted card — served to *both* scripts
through their own fetch functions, so each one runs its real assembly:

  * the backtest sees the target date as played and scores it walk-forward,
    from games strictly before it;
  * the odds job sees the same date as scheduled and prices it from the same
    history, with standings built from exactly those completed games.

Anything that made the two disagree — a different ballast, a different window,
a league rate computed off a different population, a leak — moves one number
and not the other.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.sim import bullpen as bp_model            # noqa: E402
from src.sim import lineups as lu_model            # noqa: E402
from src.sim import run_environment as rn_model    # noqa: E402
from src.sim import starters as sp_model           # noqa: E402
from src.sim.season import from_schedule           # noqa: E402
from src.sim.strength import estimate_hfa          # noqa: E402

SEASON = 2026
START = date(2026, 4, 1)
N_DATES = 45
REGRESS_GAMES = 60.0
# team_id, abbrev, league, division
CLUBS = [(1, "AAA", 103, 201), (2, "BBB", 103, 201),
         (3, "CCC", 104, 204), (4, "DDD", 104, 204)]
# One park per club, numbered apart from the club ids so a fallback to the home
# team's id would be visible rather than silently right.
VENUE_OF = {1: 7001, 2: 7002, 3: 7003, 4: 7004}
# Ballasts that leave both new terms visibly on. The shipped constants are not
# used here: what this file tests is that the two callers agree *when a term is
# on*, which is a question about the assembly and not about the constant.
PARK_BALLAST = 100.0
DEF_BALLAST = 1000.0
STARTERS_PER_CLUB = 5
RELIEVERS_PER_CLUB = 5
HITTERS_PER_CLUB = 9


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ─── the synthetic season ───

def _pitcher_ids(team_index: int) -> tuple[list[int], list[int]]:
    base = 1000 * (team_index + 1)
    starters = [base + i for i in range(STARTERS_PER_CLUB)]
    relievers = [base + 50 + i for i in range(RELIEVERS_PER_CLUB)]
    return starters, relievers


def _batter_ids(team_index: int) -> list[int]:
    return [9000 + 100 * (team_index + 1) + i for i in range(HITTERS_PER_CLUB)]


def build_season() -> dict:
    """One deterministic season: schedule, results, appearances, cards.

    Every club plays every date, so the last 15 dates are the last 15 games and
    the two card histories (the backtest's banked cards, the odds job's last
    completed games) cover exactly the same games.
    """
    rng = np.random.default_rng(20260903)
    dates = [(START + timedelta(days=d)).isoformat() for d in range(N_DATES)]
    target_date = dates[-1]

    games, pk = [], 700_000
    for di, day in enumerate(dates):
        for a, b in ((0, 1), (2, 3)):
            home, away = (a, b) if di % 2 == 0 else (b, a)
            pk += 1
            games.append({"game_pk": pk, "date": day, "day_index": di,
                          "home_i": home, "away_i": away})
    sched = pd.DataFrame(games)
    played = sched["date"] < target_date

    hs, as_ = [], []
    for _ in range(len(sched)):
        h, a = int(rng.integers(0, 11)), int(rng.integers(0, 11))
        while h == a:
            a = int(rng.integers(0, 11))
        hs.append(h), as_.append(a)
    # Every game has a result on file; `schedule_frame` decides which of them
    # have been played yet, and that is the only difference between the season
    # the backtest scores and the season the odds job is looking forward from.
    sched["home_score"] = hs
    sched["away_score"] = as_

    # ── pitching appearances: one start and three relief outings per club-game
    p_rows, ip_rows = [], []
    for g in sched[played].itertuples(index=False):
        for side, ti in (("home", g.home_i), ("away", g.away_i)):
            team_id = CLUBS[ti][0]
            starters, relievers = _pitcher_ids(ti)
            sp = starters[g.day_index % STARTERS_PER_CLUB]
            outs = int(rng.integers(12, 22))
            bf = outs + int(rng.integers(3, 9))
            p_rows.append({
                "pitcher": sp, "season": SEASON, "date": g.date,
                "game_pk": g.game_pk, "game_type": "R", "team": team_id,
                "outs": float(outs), "bf": float(bf),
                "k": float(rng.integers(2, 9)), "bb": float(rng.integers(0, 4)),
                "hbp": 0.0, "hr": float(rng.integers(0, 3)),
                "er": 3.0, "gs": 1, "pitches": float(outs * 5 + 15),
                # Hits, at-bats and sacrifice flies: the balls-in-play half of
                # the line, which is what the defence term reads. Club 1's
                # pitchers give up fewer hits on the same contact, so the term
                # has something to find.
                "h": float(rng.integers(3, 8) - (2 if team_id == 1 else 0)),
                "ab": float(bf - 2), "sf": 0.0})
            for j in range(3):
                rp = relievers[(g.day_index + j) % RELIEVERS_PER_CLUB]
                r_outs = int(rng.integers(1, 5))
                p_rows.append({
                    "pitcher": rp, "season": SEASON, "date": g.date,
                    "game_pk": g.game_pk, "game_type": "R", "team": team_id,
                    "outs": float(r_outs), "bf": float(r_outs + 1),
                    "k": float(rng.integers(0, 3)), "bb": float(rng.integers(0, 2)),
                    "hbp": 0.0, "hr": float(rng.integers(0, 2)),
                    "er": 1.0, "gs": 0,
                    "pitches": float(r_outs * 6 + int(rng.integers(2, 20))),
                    "h": float(rng.integers(0, 3)), "ab": float(r_outs + 1),
                    "sf": 0.0})
    pitching = pd.DataFrame(p_rows)

    # ── hitting lines: the nine who started, and the cards they started in
    h_rows, cards = [], {}
    for g in sched[played].itertuples(index=False):
        for side, ti in (("home", g.home_i), ("away", g.away_i)):
            ids = _batter_ids(ti)
            # A rotating order, so a club's cards are not all worth the same
            # and the baseline is a real average rather than a constant.
            order = ids[g.day_index % HITTERS_PER_CLUB:] + \
                ids[:g.day_index % HITTERS_PER_CLUB]
            cards.setdefault(int(g.game_pk), {})[side] = order
            for b in ids:
                ab = int(rng.integers(3, 5))
                k = int(rng.integers(0, 2))
                hr = int(rng.integers(0, 2))
                hits = min(ab, hr + int(rng.integers(0, 3)))
                h_rows.append({
                    "batter": b, "season": SEASON, "date": g.date,
                    "game_pk": g.game_pk, "game_type": "R",
                    "team_id": CLUBS[ti][0], "pa": ab + 1, "ab": ab,
                    "h": hits, "doubles": int(rng.integers(0, 2)), "triples": 0,
                    "hr": hr, "k": k, "bb": 1, "hbp": 0, "sf": 0})
    hitting = pd.DataFrame(h_rows)

    # ── probables: every game, past and future, has both starters announced
    prob_rows = []
    for g in sched.itertuples(index=False):
        home_sp = _pitcher_ids(g.home_i)[0][g.day_index % STARTERS_PER_CLUB]
        away_sp = _pitcher_ids(g.away_i)[0][g.day_index % STARTERS_PER_CLUB]
        prob_rows.append({"game_pk": g.game_pk, "date": g.date, "game_type": "R",
                          "home_sp_id": home_sp, "away_sp_id": away_sp})
    probables = pd.DataFrame(prob_rows)
    for c in ("home_sp_id", "away_sp_id"):
        probables[c] = probables[c].astype("Int64")

    # ── the completed prior seasons, on both sides of the ball
    prior_p, prior_h = [], []
    for ti in range(len(CLUBS)):
        starters, relievers = _pitcher_ids(ti)
        for season in (SEASON - 2, SEASON - 1):
            for i, p in enumerate(starters + relievers):
                bf = 400.0 + 30 * i
                prior_p.append({"pitcher": p, "season": season, "bf": bf,
                                "k": bf * (0.18 + 0.01 * (i % 5)),
                                "bb": bf * 0.08, "hbp": bf * 0.01,
                                "hr": bf * (0.02 + 0.003 * (i % 4)),
                                "er": 40.0, "gs": 20, "pitches": bf * 3.9,
                                "outs": bf * 2.1})
            for i, b in enumerate(_batter_ids(ti)):
                pa = 500.0 + 20 * i
                ab = pa * 0.9
                prior_h.append({"batter": b, "season": season, "pa": pa,
                                "ab": ab, "h": ab * (0.24 + 0.005 * i),
                                "doubles": ab * 0.05, "triples": ab * 0.004,
                                "hr": ab * (0.03 + 0.002 * i),
                                "k": pa * 0.22, "bb": pa * 0.08, "hbp": pa * 0.01,
                                "sf": pa * 0.005, "age": 28})
    return {
        "dates": dates, "target_date": target_date, "schedule": sched,
        "pitching": pitching, "hitting": hitting, "probables": probables,
        "cards": cards,
        "prior_pitching": pd.DataFrame(prior_p),
        "prior_hitting": pd.DataFrame(prior_h),
    }


def schedule_frame(season: dict, target_final: bool) -> pd.DataFrame:
    """The Stats API schedule shape, with the target date played or scheduled.

    `venue_id` is each home club's own park (`VENUE_OF`), which is what the
    park term keys its run multipliers on and what the defence term reads to
    tell a road game from a home one.
    """
    s = season["schedule"]
    final = (s["date"] < season["target_date"]) | target_final
    home_ids = [CLUBS[i][0] for i in s["home_i"]]
    return pd.DataFrame({
        "game_pk": s["game_pk"], "date": s["date"],
        "game_datetime": s["date"] + "T23:00:00Z",
        "venue_id": [VENUE_OF[t] for t in home_ids],
        "venue_name": [f"Park {VENUE_OF[t]}" for t in home_ids],
        "status": np.where(final, "Final", "Preview"),
        "game_type": "R",
        "home_id": home_ids,
        "away_id": [CLUBS[i][0] for i in s["away_i"]],
        "home_score": np.where(final, s["home_score"], np.nan),
        "away_score": np.where(final, s["away_score"], np.nan),
    })


def prior_season_schedule(season: dict, year: str) -> pd.DataFrame:
    """A completed prior season for the park factors to be built from.

    The same four clubs and the same four parks, with one of them (club 1's)
    a hitters' park, so the factors that come out are not all 1.0 and the two
    callers have something to disagree about.
    """
    rows, pk = [], 500_000
    for i, (home, away) in enumerate([(a, b) for a in CLUBS for b in CLUBS
                                      if a[0] != b[0]] * 6):
        pk += 1
        venue = VENUE_OF[home[0]]
        runs = 12.0 if venue == VENUE_OF[CLUBS[0][0]] else 8.0
        rows.append({"game_pk": pk, "date": f"{year}-05-{i % 28 + 1:02d}",
                     "venue_id": venue, "status": "Final", "game_type": "R",
                     "home_id": home[0], "away_id": away[0],
                     "home_score": runs / 2, "away_score": runs / 2})
    return pd.DataFrame(rows)


def teams_frame() -> pd.DataFrame:
    return pd.DataFrame([{"team_id": t, "abbrev": a, "name": f"Team {a}",
                          "league_id": lg, "division_id": dv}
                         for t, a, lg, dv in CLUBS])


def standings_frame(season: dict) -> pd.DataFrame:
    """Standings built from exactly the games completed before the target date.

    The odds job reads runs scored / allowed and win-loss from here; the
    backtest sums the same games itself. If these two disagree the whole
    comparison is meaningless, so they are built from one source.
    """
    sched = schedule_frame(season, target_final=False)
    done = sched[sched["status"] == "Final"]
    rows = []
    for team_id, _abbrev, _lg, _dv in CLUBS:
        home = done[done["home_id"] == team_id]
        away = done[done["away_id"] == team_id]
        rs = home["home_score"].sum() + away["away_score"].sum()
        ra = home["away_score"].sum() + away["home_score"].sum()
        wins = int((home["home_score"] > home["away_score"]).sum()
                   + (away["away_score"] > away["home_score"]).sum())
        rows.append({"team_id": team_id, "wins": wins,
                     "losses": len(home) + len(away) - wins,
                     "runs_scored": float(rs), "runs_allowed": float(ra)})
    return pd.DataFrame(rows)


# ─── the two paths ───

def _fake_fetchers(season: dict, cards_for_target: bool):
    """The fetch surface both scripts call, served from the synthetic season."""
    target = season["target_date"]
    pitching = season["pitching"]
    hitting = season["hitting"]

    def probables(start_date, end_date, refresh=False):
        return season["probables"].copy()

    def schedule(start_date, end_date):
        """This season's games, or a completed prior one for the park factors.

        The park term asks for the seasons *before* the one being priced, and
        this is where that is served: anything but `SEASON` comes back as a
        finished season at the same four parks.
        """
        year = str(start_date)[:4]
        if year != str(SEASON):
            return prior_season_schedule(season, year)
        return schedule_frame(season, target_final=False)

    def season_pitching(year, page_size=1000, refresh=False):
        if year == SEASON:
            return (pitching.groupby(["pitcher", "season"], as_index=False)
                    .sum(numeric_only=True))
        return season["prior_pitching"][
            season["prior_pitching"]["season"] == year].copy()

    def pitcher_logs(ids, year, refresh=False, workers=1):
        keep = {int(p) for p in ids}
        return pitching[pitching["pitcher"].isin(keep)].copy()

    def season_hitting(year, page_size=500):
        return (hitting.groupby(["batter", "season"], as_index=False)
                .sum(numeric_only=True))

    def hitter_logs(ids, year, refresh=False, workers=1):
        keep = {int(b) for b in ids}
        return hitting[hitting["batter"].isin(keep)].copy()

    def batter_logs(ids, year, refresh=False, pace=0.0):
        return hitter_logs(ids, year).drop(columns=["team_id"])

    def seasons_table(start, end, cache_path=None, refresh=False):
        h = season["prior_hitting"]
        return h[h["season"].between(start, end)].copy()

    def lineups(game_pks, refresh=False, pace=0.0, workers=1):
        rows = []
        for pk in sorted({int(p) for p in game_pks}):
            card = season["cards"].get(pk)
            if card is None:
                continue
            for side, ids in card.items():
                for slot, b in enumerate(ids, start=1):
                    rows.append({"game_pk": pk, "side": side, "slot": slot,
                                 "batter": int(b)})
        return pd.DataFrame(rows, columns=["game_pk", "side", "slot", "batter"])

    if cards_for_target:
        # The one branch the nightly cannot exercise at 5am: a club that has
        # already posted its card for a game still to be played.
        sched = season["schedule"]
        for g in sched[sched["date"] == target].itertuples(index=False):
            season["cards"].setdefault(int(g.game_pk), {})
            for side, ti in (("home", g.home_i), ("away", g.away_i)):
                season["cards"][int(g.game_pk)][side] = _batter_ids(ti)

    return {"fetch_probables": probables, "fetch_schedule": schedule,
            "fetch_season_pitching": season_pitching,
            "fetch_pitcher_game_logs": pitcher_logs,
            "fetch_season_hitting": season_hitting,
            "fetch_hitter_game_logs": hitter_logs,
            "fetch_batter_game_logs": batter_logs,
            "build_seasons_table": seasons_table, "fetch_lineups": lineups}


def _patch(mp, module, fakes: dict) -> None:
    """Serve the synthetic season wherever a fetch is reached from.

    Both scripts import the fetch functions by name, and `starters.rate_inputs`
    imports two of them again inside the function body, so the source module
    has to be patched as well or half the chain quietly runs on the live API.
    """
    import src.data.mlb_stats_api as api
    for name, fn in fakes.items():
        if hasattr(api, name):
            mp.setattr(api, name, fn)
        if hasattr(module, name):
            mp.setattr(module, name, fn)


def backtest_probabilities(season: dict, fakes: dict, mp, *,
                           park_ballast: float = float("inf"),
                           def_ballast: float = float("inf")) -> pd.DataFrame:
    """The harness's own walk-forward frame for the target date.

    The two ballasts default to infinite, which is the park and defence terms
    switched off — what the nightly serves, and therefore what the columns
    above them have to keep matching.
    """
    bt = _load("backtest_game_odds_chain", ROOT / "scripts/backtest_game_odds.py")
    _patch(mp, bt, fakes)

    sched = schedule_frame(season, target_final=True)
    scored = sched[sched["status"] == "Final"].dropna(
        subset=["home_score", "away_score"]).copy()
    scored["home_win"] = scored["home_score"] > scored["away_score"]

    sp_ctx = bt.build_sp_context(SEASON, scored, sp_model.BALLAST_BF,
                                 sp_model.STARTER_IP)
    lu_ctx = bt.build_lu_context(
        SEASON, scored, lu_model.BALLAST, bt.LU_WEIGHT, bt.LU_BASELINE,
        lu_model.BASELINE_BALLAST_GAMES,
        pa_per_game=sp_ctx["league"]["bf_per_ip"] * 9.0)
    bp_ctx = bt.build_bp_context(
        SEASON, sp_model.BALLAST_BF, bp_model.BASELINE,
        bp_model.ROSTER_WINDOW_DAYS, bp_model.REST_DAYS, bp_model.REST_MIN_DAYS,
        bp_model.RELIEF_IP, sp_ctx["league"], sp_ctx["prior_counts"],
        home_by_game={int(pk): int(t) for pk, t
                      in zip(sched["game_pk"], sched["home_id"])})
    c_ctx = bt.build_c_context(
        SEASON, lu_ctx, bp_ctx, rn_model.BLEND_WEIGHT,
        rn_model.SHARE_WINDOW_DAYS, rn_model.ROTATION_WINDOW_DAYS,
        rn_model.ROTATION_TOP_N, lu_ctx["pa_per_game"], schedule=sched,
        park_ballast=park_ballast, def_ballast=def_ballast)
    preds = bt.walk_forward(scored, teams_frame()["team_id"].to_numpy(),
                            0, [REGRESS_GAMES], sp_ctx, lu_ctx, bp_ctx, c_ctx)
    return preds[preds["date"] == season["target_date"]].set_index("game_pk")


def nightly_probabilities(season: dict, fakes: dict, use_lineups: bool, mp, *,
                          park_ballast: float = float("inf"),
                          def_ballast: float = float("inf")) -> dict:
    """The odds job's overrides for the target date, from the same season."""
    rp = _load("run_playoff_odds_chain", ROOT / "scripts/run_playoff_odds.py")
    _patch(mp, rp, fakes)

    sched = schedule_frame(season, target_final=False)
    state = from_schedule(sched, teams_frame())
    standings = standings_frame(season)
    hfa = estimate_hfa(state.completed)
    as_of = date.fromisoformat(season["target_date"])
    from src.sim import game_model as gm
    overrides, rotations, strength, diag = rp.chain_terms(
        SEASON, state, standings, sched, hfa, REGRESS_GAMES, as_of,
        refresh=False, workers=1, use_lineups=use_lineups,
        config=gm.ChainConfig(regress_games=REGRESS_GAMES,
                              park_ballast=park_ballast,
                              def_ballast=def_ballast))
    return {"overrides": overrides, "rotations": rotations,
            "strength": strength, "diagnostics": diag, "hfa": hfa,
            "state": state}


@pytest.fixture(scope="module")
def season():
    return build_season()


class TestOneGameTwoPaths:
    def test_the_nightly_prices_the_game_the_backtest_scores(self, season, monkeypatch):
        """The gated chain, both ways, on one game: identical to the last bit."""
        fakes = _fake_fetchers(season, cards_for_target=False)
        scored = backtest_probabilities(season, fakes, monkeypatch)
        live = nightly_probabilities(season, fakes, True, monkeypatch)

        assert len(scored) and live["overrides"], "the fixture priced nothing"
        for game_pk, row in scored.iterrows():
            assert int(game_pk) in live["overrides"], game_pk
            assert live["overrides"][int(game_pk)] == pytest.approx(
                row["pythag_C_sp_bpa_ip"], abs=1e-12)

    def test_a_posted_card_is_the_same_on_both_sides(self, season, monkeypatch):
        """...and so is the branch that fires when a club has posted its card."""
        fakes = _fake_fetchers(season, cards_for_target=True)
        scored = backtest_probabilities(season, fakes, monkeypatch)
        live = nightly_probabilities(season, fakes, True, monkeypatch)

        assert (scored["c_lineup_slots"] > 0).all(), "no card reached the model"
        for game_pk, row in scored.iterrows():
            assert live["overrides"][int(game_pk)] == pytest.approx(
                row["pythag_C_sp_bpa_ip_lu"], abs=1e-12)
        # The card moved something, or the test above would be vacuous.
        assert (scored["pythag_C_sp_bpa_ip_lu"]
                != scored["pythag_C_sp_bpa_ip"]).any()

    def test_a_game_with_no_starter_named_is_the_strength_the_sim_draws(self, season, monkeypatch):
        """The horizon is a boundary in what is known, not in how it is priced.

        A remaining game nobody has announced is drawn from team strength by
        the Monte Carlo; the chain asked about that same game with no starter
        returns exactly that probability, which is why the overrides can stop
        at the probables horizon without a seam.
        """
        from src.sim import game_model as gm
        from src.sim.strength import home_win_prob

        fakes = _fake_fetchers(season, cards_for_target=False)
        live = nightly_probabilities(season, fakes, False, monkeypatch)
        state, hfa = live["state"], live["hfa"]
        strength = live["strength"]

        g = state.remaining.iloc[0]
        drawn = float(home_win_prob(strength[int(g.home_id)],
                                    strength[int(g.away_id)], hfa))
        # Rebuild the same slate the job used and ask it with nobody named.
        rp = _load("run_playoff_odds_chain2", ROOT / "scripts/run_playoff_odds.py")
        for name, fn in fakes.items():
            if hasattr(rp, name):
                setattr(rp, name, fn)
        data = rp.fetch_chain_data(SEASON, date.fromisoformat(season["target_date"]),
                                   7, refresh=False, workers=1)
        standings = standings_frame(season)
        games = float((standings["wins"] + standings["losses"]).sum())
        lg_rs9, lg_ra9 = gm.league_run_rates(float(standings["runs_scored"].sum()),
                                             float(standings["runs_allowed"].sum()),
                                             games)
        from src.sim.strength import regressed_run_rates
        slate = gm.build_slate(season["target_date"], data["inputs"],
                               regressed_run_rates(standings, REGRESS_GAMES),
                               lg_rs9, lg_ra9,
                               # The config the job serves: park and defence
                               # switched off, because neither cleared its gate.
                               config=gm.ChainConfig(
                                   regress_games=REGRESS_GAMES,
                                   park_ballast=rp.NOT_SERVED,
                                   def_ballast=rp.NOT_SERVED))
        priced, _ = gm.home_win_probability(slate, int(g.home_id), int(g.away_id),
                                            None, hfa)
        assert priced == pytest.approx(drawn, abs=1e-12)


class TestParkAndDefenceAgreeToo:
    """The two new terms, through both callers, on the same game.

    They are the first terms that change the *top-down* half (the park is
    neutralised out of a club's own totals before they are regressed) and the
    first that need a fact about the game other than who is playing (which
    park). Both are places the two assemblies could drift apart while every
    older column stayed identical, so both are pinned here.
    """

    def test_the_nightly_and_the_backtest_agree_with_both_terms_on(self, season,
                                                                   monkeypatch):
        fakes = _fake_fetchers(season, cards_for_target=False)
        scored = backtest_probabilities(season, fakes, monkeypatch,
                                        park_ballast=PARK_BALLAST,
                                        def_ballast=DEF_BALLAST)
        live = nightly_probabilities(season, fakes, False, monkeypatch,
                                     park_ballast=PARK_BALLAST,
                                     def_ballast=DEF_BALLAST)
        assert len(scored) and live["overrides"], "the fixture priced nothing"
        for game_pk, row in scored.iterrows():
            assert int(game_pk) in live["overrides"], game_pk
            assert live["overrides"][int(game_pk)] == pytest.approx(
                row["pythag_C_sp_bpa_ip_pk_def"], abs=1e-12)

    def test_the_terms_actually_moved_the_price(self, season, monkeypatch):
        """...or the test above would pass on two identical models."""
        fakes = _fake_fetchers(season, cards_for_target=False)
        scored = backtest_probabilities(season, fakes, monkeypatch,
                                        park_ballast=PARK_BALLAST,
                                        def_ballast=DEF_BALLAST)
        assert (scored["pythag_C_sp_bpa_ip_pk"]
                != scored["pythag_C_sp_bpa_ip"]).any()
        assert (scored["pythag_C_sp_bpa_ip_def"]
                != scored["pythag_C_sp_bpa_ip"]).any()
        # The fixture's hitters' park is priced above one and the others below.
        assert (scored["c_park_factor"] != 1.0).any()

    def test_switching_both_terms_off_is_the_gated_chain_to_the_last_bit(
            self, season, monkeypatch):
        """The nesting the gate comparison rests on, end to end.

        An infinite ballast on either term has to give back exactly the model
        without it — including through the top-down half, where the park term
        rebuilds a regression the caller would otherwise have done itself.
        """
        fakes = _fake_fetchers(season, cards_for_target=False)
        scored = backtest_probabilities(season, fakes, monkeypatch)
        assert (scored["pythag_C_sp_bpa_ip_pk_def"]
                == scored["pythag_C_sp_bpa_ip"]).all()
        assert (scored["pythag_C_sp_bpa_ip_pk"]
                == scored["pythag_C_sp_bpa_ip"]).all()

    def test_the_nightly_serves_the_gated_chain_and_not_the_new_terms(
            self, season, monkeypatch):
        """What the gate rule means in code: the baseline runs until it loses.

        Neither term cleared, so the odds job asks for them switched off, and
        its answer is the scored `pythag_C_sp_bpa_ip` — not the park-and-
        defence column beside it.
        """
        fakes = _fake_fetchers(season, cards_for_target=False)
        scored = backtest_probabilities(season, fakes, monkeypatch)
        live = nightly_probabilities(season, fakes, False, monkeypatch)
        with_terms = backtest_probabilities(season, fakes, monkeypatch,
                                            park_ballast=PARK_BALLAST,
                                            def_ballast=DEF_BALLAST)
        for game_pk, row in scored.iterrows():
            assert live["overrides"][int(game_pk)] == pytest.approx(
                row["pythag_C_sp_bpa_ip"], abs=1e-12)
            assert live["overrides"][int(game_pk)] != pytest.approx(
                with_terms.loc[game_pk, "pythag_C_sp_bpa_ip_pk_def"], abs=1e-12)


class TestTheStrengthTheSimDraws:
    def test_station_c_moves_the_clubs_off_their_run_differential(self, season, monkeypatch):
        """The served strength is the blend, not the regressed rates."""
        from src.sim.strength import regressed_strength

        fakes = _fake_fetchers(season, cards_for_target=False)
        live = nightly_probabilities(season, fakes, False, monkeypatch)
        top_down = regressed_strength(standings_frame(season), REGRESS_GAMES)
        served = live["strength"]
        assert set(served.index) == set(top_down.index)
        assert (served - top_down).abs().max() > 0
        # ...and it is a talent win%, not a run rate.
        assert served.between(0.2, 0.8).all()

    def test_every_club_carries_a_rotation_and_a_run_environment(self, season, monkeypatch):
        fakes = _fake_fetchers(season, cards_for_target=False)
        live = nightly_probabilities(season, fakes, False, monkeypatch)
        rot = live["rotations"]
        assert len(rot.by_team) == len(CLUBS)
        assert rot.run_env is not None and (rot.run_env > 0).all()
        assert live["diagnostics"]["n_games_with_starters"] > 0
