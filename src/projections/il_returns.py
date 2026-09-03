"""Station B — how long a hitter who is out today stays out.

Station B's roster gate zeroes every hitter the 40-man roster says is
unavailable at the cutoff: injured list, optioned to the minors, everything
else. `docs/playing-time.md` §4 measures what that gate is worth, and the
answer depends entirely on the horizon — it gains 2.0–2.8 plate appearances a
hitter over one month and *costs* 3.6–4.8 over two, because over two months
most of the injured come back and play. A binary gate is the wrong shape for
that; a hitter who has been on the 10-day list for a week is a different
projection from one who went on the 60-day list in May.

This module builds the missing piece: **the distribution of days-until-return,
per list type, conditioned on the days already elapsed.**

    transactions  →  events  →  spells  →  survival table  →  P(back by day d)

1. `parse_events` turns the Stats API transaction feed into dated events —
   `il_placement`, `il_transfer`, `il_activation`, `option`, `recall`, and the
   handful of transactions (release, free agency, retirement) that end a spell
   without a return.
2. `build_spells` walks each player's events in date order and pairs them:
   an injured-list spell runs placement → activation, an option spell runs
   option → recall, and a spell still open at the end of the season is
   **censored**, not a non-return. A transfer from the 15- to the 60-day list
   splits one stint into two rows — the 15-day part censored at the transfer,
   the 60-day part *left-truncated* at it, entering the risk set only from the
   day it becomes a 60-day case — so a player is only ever counted against the
   list he is actually on.
3. `survival_table` is Kaplan-Meier over those spells, one table per type:
   at each day with a return, how many spells were still out (`at_risk`), how
   many came back (`returns`), and the running survival `S(t)`.
4. `return_probability` reads the table the way a projection needs it:

       P(back by day e + d | still out at day e) = 1 − S(e + d) / S(e)

   and `expected_active_fraction` averages that over the days remaining in
   the horizon, which is the fraction of the rest of the season a hitter who
   is out today is expected to be available for.

Nothing here is fitted in the machine-learning sense: the "model" is an
empirical survival curve per list type, printable in twenty rows
(`scripts/build_playing_time.py --il-table`). It is estimated **walk-forward**
— the seasons it is estimated from never include the season being projected —
and every function that reads transactions takes a cutoff and keeps only rows
strictly before it.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# --- event vocabulary ---

IL_PLACEMENT = "il_placement"
IL_TRANSFER = "il_transfer"
IL_ACTIVATION = "il_activation"
OPTION = "option"
RECALL = "recall"
DEPARTURE = "departure"

EVENTS = (IL_PLACEMENT, IL_TRANSFER, IL_ACTIVATION, OPTION, RECALL, DEPARTURE)

# Spell kinds, and the type label carried on an option spell (the injured-list
# types are IL7 / IL10 / IL15 / IL60, read off the transaction text).
IL = "il"
OPTION_KIND = "option"
OPTION_TYPE = "OPT"

# Roster status code → the spell type whose curve applies to that player.
# `D7`/`D10`/`D15`/`D60` are the day-count injured lists; `RM` is optioned to
# the minors. Everything else a 40-man can say (`PL` paternity, `BRV`
# bereavement, `RL` restricted, `SU` suspended) is left out on purpose: the
# short ones are a day or two and the indefinite ones have no return-time
# distribution worth estimating, so they keep the old hard zero.
def status_spell_type(status_code) -> str | None:
    """`D15` → `IL15`, `RM` → `OPT`, anything else → None."""
    code = str(status_code).upper()
    if code.startswith("D") and code[1:].isdigit():
        return f"IL{int(code[1:])}"
    if code == "RM":
        return OPTION_TYPE
    return None


# The Stats API files every injured-list move under one type code, `SC`
# ("status change"), and says which move it was only in English. These are the
# three sentences it uses; `\d+` is the list's day count.
_PLACED = re.compile(r"\bplaced\b.*?\bon the (\d+)-day injured list", re.I)
_ACTIVATED_FROM = re.compile(
    r"\b(?:activated|reinstated)\b.*?\bfrom the (\d+)-day injured list", re.I)
_TRANSFERRED = re.compile(
    r"\btransferred\b.*?\bfrom the (\d+)-day injured list to the (\d+)-day injured list",
    re.I)
_ACTIVATED_BARE = re.compile(r"\b(?:activated|reinstated)\b", re.I)

# Type codes that take a player out of the majors or off the roster entirely.
# They end an open spell without a return: he did not come back, and we stop
# watching rather than counting him as still out.
DEPARTURE_CODES = frozenset({"REL", "DFA", "RET", "OUT"})
# Optioned / recalled / selected. A selection (`SE`) is a contract selected
# from the minors, which is how a player on a *minor-league* option comes back
# when his 40-man spot lapsed; for our purpose it is a recall.
OPTION_CODES = frozenset({"OPT"})
RECALL_CODES = frozenset({"CU", "SE"})

EVENT_COLUMNS = ["player_id", "event", "date", "type"]
SPELL_COLUMNS = ["player_id", "kind", "type", "start", "entry_day", "exit_day",
                 "returned"]
SURVIVAL_COLUMNS = ["type", "day", "at_risk", "returns", "survival"]


def _as_date(value) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def classify(type_code, description) -> tuple[str, str | None] | None:
    """One transaction → `(event, type)`, or None when it is not a spell event.

    `type` is the injured list the event refers to (`IL10`, `IL60`, …), `OPT`
    for an option or recall, and None for a departure or a bare activation
    whose list the feed does not name.
    """
    code = str(type_code).upper()
    if code in DEPARTURE_CODES:
        return DEPARTURE, None
    if code in OPTION_CODES:
        return OPTION, OPTION_TYPE
    if code in RECALL_CODES:
        return RECALL, OPTION_TYPE
    text = str(description or "")
    m = _TRANSFERRED.search(text)
    if m:
        return IL_TRANSFER, f"IL{int(m.group(2))}"
    m = _ACTIVATED_FROM.search(text)
    if m:
        return IL_ACTIVATION, f"IL{int(m.group(1))}"
    m = _PLACED.search(text)
    if m:
        return IL_PLACEMENT, f"IL{int(m.group(1))}"
    if _ACTIVATED_BARE.search(text):
        # "Boston Red Sox activated LHP Steven Matz." — the feed drops the
        # list often enough (roughly one activation in three) that ignoring
        # these would leave a third of all stints looking like they never
        # ended. It closes an open injured-list spell and nothing else.
        return IL_ACTIVATION, None
    return None


def parse_events(transactions: pd.DataFrame) -> pd.DataFrame:
    """Transaction rows → `player_id, event, date, type`, sorted by date.

    Rows the vocabulary does not recognise (trades, signings, number changes,
    rehab assignments — a rehab assignment does *not* end an injured-list
    stint) are dropped.
    """
    if transactions is None or not len(transactions):
        return pd.DataFrame(columns=EVENT_COLUMNS)
    rows = []
    for player_id, date, code, desc in zip(
            transactions["player_id"], transactions["date"],
            transactions["type_code"], transactions["description"]):
        hit = classify(code, desc)
        if hit is None:
            continue
        event, spell_type = hit
        rows.append({"player_id": int(player_id), "event": event,
                     "date": _as_date(date), "type": spell_type})
    out = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    if len(out):
        out = out.sort_values(["player_id", "date"], kind="stable").reset_index(drop=True)
    return out


def events_before(events: pd.DataFrame, cutoff) -> pd.DataFrame:
    """The events a projection made on the morning of `cutoff` may see.

    Strictly before the cutoff, the same rule the game logs use: a transaction
    filed *on* the cutoff date is that day's news, not yesterday's.
    """
    if not len(events):
        return events
    return events[pd.to_datetime(events["date"]) < _as_date(cutoff)]


def _close(open_spell: dict, day: int, returned: bool, out: list) -> None:
    """Emit a spell segment if it spent any time at risk."""
    if open_spell is None or day <= open_spell["entry_day"]:
        return
    out.append({
        "player_id": open_spell["player_id"], "kind": open_spell["kind"],
        "type": open_spell["type"], "start": open_spell["start"],
        "entry_day": int(open_spell["entry_day"]), "exit_day": int(day),
        "returned": bool(returned),
    })


def build_spells(events: pd.DataFrame, season_end) -> pd.DataFrame:
    """Pair the events into spells: `player_id, kind, type, start, entry_day,
    exit_day, returned`.

    `entry_day` and `exit_day` are days since the spell *started* — the
    original placement, even for the 60-day half of a transferred stint, so
    that a projection which knows only "on the 60-day list, placed 40 days
    ago" can look the row up. `entry_day` is normally 0 and is positive only
    for a segment the player was left-truncated into (see the module
    docstring). `returned=False` is censoring, not a non-return: the spell was
    still open when the season ended or when we stopped watching the player.

    An injured-list spell and an option spell can be open at the same time
    (a player is optioned, then goes on the minor-league injured list), so
    each kind is tracked independently.
    """
    season_end = _as_date(season_end)
    out: list[dict] = []
    if not len(events):
        return pd.DataFrame(columns=SPELL_COLUMNS)
    for player_id, group in events.groupby("player_id", sort=True):
        open_spells: dict[str, dict | None] = {IL: None, OPTION_KIND: None}
        for event, date, spell_type in zip(group["event"], group["date"], group["type"]):
            date = _as_date(date)
            il_open, opt_open = open_spells[IL], open_spells[OPTION_KIND]
            if event == IL_PLACEMENT:
                # A second placement with no activation between means we
                # missed the activation; stop watching the first stint rather
                # than pretending it ran on.
                if il_open is not None:
                    _close(il_open, (date - il_open["start"]).days, False, out)
                open_spells[IL] = {"player_id": int(player_id), "kind": IL,
                                   "type": spell_type, "start": date, "entry_day": 0}
            elif event == IL_TRANSFER:
                if il_open is not None:
                    day = (date - il_open["start"]).days
                    _close(il_open, day, False, out)
                    open_spells[IL] = {**il_open, "type": spell_type,
                                       "entry_day": max(day, 0)}
                else:
                    open_spells[IL] = {"player_id": int(player_id), "kind": IL,
                                       "type": spell_type, "start": date,
                                       "entry_day": 0}
            elif event == IL_ACTIVATION:
                if il_open is not None:
                    _close(il_open, (date - il_open["start"]).days, True, out)
                    open_spells[IL] = None
            elif event == OPTION:
                if opt_open is not None:
                    _close(opt_open, (date - opt_open["start"]).days, False, out)
                open_spells[OPTION_KIND] = {
                    "player_id": int(player_id), "kind": OPTION_KIND,
                    "type": OPTION_TYPE, "start": date, "entry_day": 0}
            elif event == RECALL:
                if opt_open is not None:
                    _close(opt_open, (date - opt_open["start"]).days, True, out)
                    open_spells[OPTION_KIND] = None
            elif event == DEPARTURE:
                for kind, spell in list(open_spells.items()):
                    if spell is not None:
                        _close(spell, (date - spell["start"]).days, False, out)
                    open_spells[kind] = None
        for spell in open_spells.values():
            if spell is not None:
                _close(spell, (season_end - spell["start"]).days, False, out)
    return pd.DataFrame(out, columns=SPELL_COLUMNS)


def survival_table(spells: pd.DataFrame) -> pd.DataFrame:
    """Kaplan-Meier `S(t)` per spell type, with left truncation and censoring.

    One row per (type, day) at which somebody came back:

        type  day  at_risk  returns  survival

    `at_risk` is the number of spells of that type that had entered the risk
    set (`entry_day < day`) and had not yet left it (`exit_day >= day`);
    `survival` is the running product of `1 − returns / at_risk`, the
    probability a spell of that type is still out after `day` days.

    Censored spells (the season ended first, the player was released) count in
    `at_risk` up to the day they were censored and never in `returns`, which
    is the whole point of doing this Kaplan-Meier rather than by averaging
    completed stints: at any cutoff the long stints are the ones most likely
    to be unfinished, and averaging only the finished ones would say every
    list is shorter than it is.
    """
    cols = SURVIVAL_COLUMNS
    if spells is None or not len(spells):
        return pd.DataFrame(columns=cols)
    rows = []
    for spell_type, group in spells.groupby("type", sort=True):
        entry = group["entry_day"].to_numpy(float)
        exit_ = group["exit_day"].to_numpy(float)
        returned = group["returned"].to_numpy(bool)
        days = np.unique(exit_[returned])
        survival = 1.0
        for day in days:
            at_risk = int(((entry < day) & (exit_ >= day)).sum())
            n_return = int(((exit_ == day) & returned).sum())
            if at_risk <= 0:
                continue
            survival *= 1.0 - n_return / at_risk
            rows.append({"type": spell_type, "day": int(day), "at_risk": at_risk,
                         "returns": n_return, "survival": float(survival)})
    return pd.DataFrame(rows, columns=cols)


def survival_at(table: pd.DataFrame, spell_type: str, day) -> np.ndarray | float:
    """`S(day)` for one type — the step function, read at (a vector of) days.

    `S(t) = 1` before the first return and flat between returns. An unknown
    type has no curve at all and returns 1 (nobody comes back), which is the
    old hard gate.
    """
    d = np.asarray(day, dtype=float)
    rows = table[table["type"] == spell_type] if len(table) else table
    if rows is None or not len(rows):
        out = np.ones_like(d)
        return out if d.ndim else float(out)
    days = rows["day"].to_numpy(float)
    surv = rows["survival"].to_numpy(float)
    idx = np.searchsorted(days, d, side="right") - 1
    out = np.where(idx >= 0, surv[np.clip(idx, 0, len(surv) - 1)], 1.0)
    return out if d.ndim else float(out)


def return_probability(table: pd.DataFrame, spell_type: str, elapsed: float,
                       horizon_days: int) -> np.ndarray:
    """`P(back by day elapsed + d | still out at day elapsed)` for d = 1…D.

    The conditional form of the survival curve, which is the only form a
    projection can use: the roster tells us he is still out *now*, so the
    denominator is the population that made it this far.

    If the curve has already reached zero at `elapsed` — every spell of that
    type in the fitting seasons had ended by then — the conditional
    probability is undefined and this returns zeros, i.e. the old hard gate.
    In practice that never fires, because the longest spells are censored at
    the end of the season and censoring keeps `S` positive.
    """
    horizon_days = int(max(horizon_days, 0))
    if horizon_days <= 0:
        return np.zeros(0)
    s_now = float(survival_at(table, spell_type, float(elapsed)))
    if s_now <= 0.0:
        return np.zeros(horizon_days)
    days = float(elapsed) + np.arange(1, horizon_days + 1, dtype=float)
    s_then = np.asarray(survival_at(table, spell_type, days), dtype=float)
    return np.clip(1.0 - s_then / s_now, 0.0, 1.0)


def expected_active_fraction(table: pd.DataFrame, spell_type: str, elapsed: float,
                             horizon_days: int) -> float:
    """The share of the remaining horizon a player out today is available for.

    The mean over the remaining days of `P(back by that day)`. Clubs play
    close to every day, so a mean over days is a mean over games to within the
    off-day pattern, and that is the number the projection multiplies a
    pre-injury share by: a hitter expected to be back for the last two of six
    remaining weeks gets a third of the plate appearances he would have taken
    healthy.
    """
    probs = return_probability(table, spell_type, elapsed, horizon_days)
    return float(probs.mean()) if len(probs) else 0.0


def open_spells_at(events: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Who is out at `cutoff`, on what, and since when.

    Returns `player_id, kind, type, start, elapsed_days` for every spell still
    open on the cutoff morning, built from events **strictly before** the
    cutoff. This is the state a projection is allowed to know; the roster
    status at the cutoff says who is out, and this says how long he has been.
    """
    cutoff = _as_date(cutoff)
    cols = ["player_id", "kind", "type", "start", "elapsed_days"]
    events = events_before(events, cutoff)
    if not len(events):
        return pd.DataFrame(columns=cols)
    rows = []
    for player_id, group in events.groupby("player_id", sort=True):
        open_spells: dict[str, dict | None] = {IL: None, OPTION_KIND: None}
        for event, date, spell_type in zip(group["event"], group["date"], group["type"]):
            date = _as_date(date)
            if event == IL_PLACEMENT:
                open_spells[IL] = {"kind": IL, "type": spell_type, "start": date}
            elif event == IL_TRANSFER:
                start = open_spells[IL]["start"] if open_spells[IL] else date
                open_spells[IL] = {"kind": IL, "type": spell_type, "start": start}
            elif event == IL_ACTIVATION:
                open_spells[IL] = None
            elif event == OPTION:
                open_spells[OPTION_KIND] = {"kind": OPTION_KIND,
                                            "type": OPTION_TYPE, "start": date}
            elif event == RECALL:
                open_spells[OPTION_KIND] = None
            elif event == DEPARTURE:
                open_spells = {IL: None, OPTION_KIND: None}
        for spell in open_spells.values():
            if spell is not None:
                rows.append({"player_id": int(player_id), **spell,
                             "elapsed_days": int((cutoff - spell["start"]).days)})
    return pd.DataFrame(rows, columns=cols)


def fit(transactions: pd.DataFrame, season_ends: dict) -> pd.DataFrame:
    """The survival table, estimated over one or more whole seasons.

    `transactions` is the concatenation of `fetch_transactions(season)` over
    the fitting seasons; `season_ends` maps season → the last day of its
    regular season, which is where an unfinished spell is censored.
    """
    frames = []
    for season, group in transactions.groupby("season", sort=True):
        frames.append(build_spells(parse_events(group), season_ends[int(season)]))
    spells = (pd.concat(frames, ignore_index=True) if frames
              else pd.DataFrame(columns=SPELL_COLUMNS))
    return survival_table(spells)


ACTIVE_FRACTION_COLUMNS = ["batter", "spell_type", "elapsed_days", "active_fraction"]


def expected_active_fractions(roster: pd.DataFrame, events: pd.DataFrame,
                              table: pd.DataFrame, cutoff, horizon_days: int,
                              ) -> pd.DataFrame:
    """One `active_fraction` per unavailable hitter on the roster.

    `batter, spell_type, elapsed_days, active_fraction`, covering only the
    hitters whose 40-man status maps to a spell type (`status_spell_type`) and
    whose spell start is known from the transactions before the cutoff.
    Everyone left out keeps the old behaviour — an active hitter is projected
    at his full share, an unavailable one the projection cannot date is zero.

    Walk-forward: `events` is filtered to strictly before `cutoff` here, so a
    transaction filed on or after the cutoff cannot move the projection.
    """
    if "status_code" not in roster.columns or not len(roster):
        return pd.DataFrame(columns=ACTIVE_FRACTION_COLUMNS)
    spell_types = roster["status_code"].map(status_spell_type)
    out_now = roster.loc[spell_types.notna(), ["batter"]].copy()
    out_now["spell_type"] = spell_types[spell_types.notna()].to_numpy()
    if not len(out_now):
        return pd.DataFrame(columns=ACTIVE_FRACTION_COLUMNS)

    open_now = open_spells_at(events, cutoff)
    # An injured-list status is dated by the injured-list spell and an option
    # by the option spell; a player can have both open at once.
    kind = np.where(out_now["spell_type"] == OPTION_TYPE, OPTION_KIND, IL)
    out_now["kind"] = kind
    if len(open_now):
        elapsed = (open_now.set_index(["player_id", "kind"])["elapsed_days"])
        keys = pd.MultiIndex.from_arrays([out_now["batter"].astype(int), out_now["kind"]])
        out_now["elapsed_days"] = elapsed.reindex(keys).to_numpy()
    else:
        out_now["elapsed_days"] = np.nan

    out_now = out_now.dropna(subset=["elapsed_days"]).copy()
    out_now["elapsed_days"] = out_now["elapsed_days"].astype(int)
    out_now["active_fraction"] = [
        expected_active_fraction(table, t, e, horizon_days)
        for t, e in zip(out_now["spell_type"], out_now["elapsed_days"])
    ]
    return out_now.loc[:, ACTIVE_FRACTION_COLUMNS].reset_index(drop=True)
