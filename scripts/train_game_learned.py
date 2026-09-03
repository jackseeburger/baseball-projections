"""Train station E's learned challenger walk-forward and score it honestly.

    python scripts/train_game_learned.py --score-season 2026 \
        --market data/parquet/market_closes_2026.parquet
    python scripts/train_game_learned.py --score-season 2025

The protocol, which is the whole point of the script:

  * **train on seasons strictly before the scored one.** Scoring 2026 trains
    on everything up to 2025; scoring 2025 trains on everything up to 2024.
    Nothing from the scored season reaches a fit, a hyperparameter or a
    calibration curve.
  * **hyperparameters on an inner split of the training seasons.** The last
    training season is the inner validation set; the grid is scored there and
    the tree count comes from early stopping there. The scored season is never
    consulted.
  * **calibration on held-out training-season games.** Each training season is
    predicted by a model fitted on the *other* training seasons (season-blocked
    out-of-fold), and the isotonic curve is fitted on those out-of-fold
    predictions. The same out-of-fold frame fits the chain/learned blend
    weights, so the blend is out of sample too.
  * **a permuted-label control** trained through the identical protocol on
    shuffled training labels. It has to land at ~0.25 Brier. If it does not,
    the harness is leaking and every other number here is worthless.

Scored against the hand-built chain (`pythag_C_sp_bpa_ip`, the model the
nightly serves), the production `pythag_60`, and — where the season has them —
the exchange closes, on exactly the games every venue priced.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.sim import game_features as gf
from src.sim import learned_game as lg

FEATURES_DIR = Path(__file__).resolve().parent.parent / "data" / "features"
CHAIN = "chain_p"          # pythag_C_sp_bpa_ip — the gate baseline
PRODUCTION = "pythag_60"

# The inner grid. Three knobs and nothing else: how fast the model learns, how
# much shape a tree is allowed, and how large a leaf has to be. Everything else
# is `learned_game.DEFAULT_PARAMS`, which is regularisation the problem's shape
# dictates rather than a fitted choice.
GRID = [
    {"learning_rate": 0.02, "num_leaves": 3, "max_depth": 2, "min_child_samples": 500},
    {"learning_rate": 0.02, "num_leaves": 7, "max_depth": 3, "min_child_samples": 300},
    {"learning_rate": 0.02, "num_leaves": 15, "max_depth": 4, "min_child_samples": 200},
    {"learning_rate": 0.05, "num_leaves": 7, "max_depth": 3, "min_child_samples": 300},
    {"learning_rate": 0.05, "num_leaves": 31, "max_depth": -1, "min_child_samples": 100},
]
MAX_TREES = 2000


# ─── the table ───

def load_features(directory: Path, seasons=None) -> pd.DataFrame:
    files = sorted(directory.glob("game_features_*.parquet"))
    if not files:
        raise SystemExit(f"no feature tables in {directory}; run "
                         f"scripts/build_game_features.py first")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    if seasons is not None:
        df = df[df["season"].isin(list(seasons))]
    return df.sort_values(["season", "date", "game_pk"]).reset_index(drop=True)


def brier(p, y) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return float(np.mean((p - np.asarray(y, dtype=float)) ** 2))


def log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def paired_t(p_model, p_base, y) -> dict:
    """Paired t on the per-game Brier difference (model - base)."""
    y = np.asarray(y, dtype=float)
    d = (np.asarray(p_model, float) - y) ** 2 - (np.asarray(p_base, float) - y) ** 2
    n = len(d)
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {"diff": float(d.mean()), "se": se,
            "t": float(d.mean() / se) if se else float("nan"), "n": int(n)}


def reliability(p, y, edges=(0, .4, .45, .5, .55, .6, .65, 1.0)) -> pd.DataFrame:
    df = pd.DataFrame({"p": np.asarray(p, dtype=float),
                       "y": np.asarray(y, dtype=float)})
    buckets = pd.cut(df["p"], list(edges))
    out = df.groupby(buckets, observed=True).agg(n=("y", "size"),
                                                 predicted=("p", "mean"),
                                                 realized=("y", "mean"))
    return out.round(3)


def calibration_fit(p, y) -> dict:
    """Slope and intercept of the logistic recalibration of `p` on `y`.

    Slope 1 / intercept 0 is a perfectly calibrated set of probabilities;
    slope below 1 means the model is too confident.
    """
    from sklearn.linear_model import LogisticRegression
    z = lg._logit(p).reshape(-1, 1)
    fit = LogisticRegression(C=1e6, solver="lbfgs").fit(z, np.asarray(y).astype(int))
    return {"slope": float(fit.coef_[0][0]), "intercept": float(fit.intercept_[0])}


# ─── the walk-forward protocol ───

def choose_params(train: pd.DataFrame, inner_season: int, seed: int,
                  verbose: bool = True) -> tuple[dict, int, list]:
    """Pick a grid point and a tree count on the last training season only."""
    import lightgbm as lgb_lib

    inner_tr = train[train["season"] < inner_season]
    inner_va = train[train["season"] == inner_season]
    X_tr, y_tr = gf.feature_matrix(inner_tr), inner_tr[gf.LABEL].astype(int)
    X_va, y_va = gf.feature_matrix(inner_va), inner_va[gf.LABEL].astype(int)
    rows, best = [], None
    for params in GRID:
        p = {**lg.DEFAULT_PARAMS, **params}
        p.pop("n_estimators")
        model = lgb_lib.LGBMClassifier(n_estimators=MAX_TREES, random_state=seed, **p)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="binary_logloss",
                  callbacks=[lgb_lib.early_stopping(100, verbose=False)])
        n_best = int(model.best_iteration_ or MAX_TREES)
        pv = model.predict_proba(X_va)[:, 1]
        row = {**params, "trees": n_best, "inner_log_loss": log_loss(pv, y_va),
               "inner_brier": brier(pv, y_va)}
        rows.append(row)
        if best is None or row["inner_log_loss"] < best["inner_log_loss"]:
            best = row
        if verbose:
            print(f"  grid lr={params['learning_rate']} leaves={params['num_leaves']} "
                  f"min_leaf={params['min_child_samples']}: {n_best} trees, "
                  f"inner log loss {row['inner_log_loss']:.5f}, "
                  f"Brier {row['inner_brier']:.5f}", flush=True)
    chosen = {k: best[k] for k in ("learning_rate", "num_leaves", "max_depth",
                                   "min_child_samples")}
    # The chosen tree count came from a fit on one season fewer, so scale it by
    # how much more data the final fit sees. Rounding up is the conservative
    # direction only because the learning rate is small.
    scale = len(train) / max(len(inner_tr), 1)
    trees = max(int(round(best["trees"] * scale)), 10)
    return chosen, trees, rows


def out_of_fold(train: pd.DataFrame, params: dict, trees: int,
                seed: int) -> pd.DataFrame:
    """Season-blocked out-of-fold predictions over the training seasons.

    Each training season is predicted by a model fitted on the others. This is
    the only frame the calibrator and the blend weights ever see, and it
    contains no game from the scored season.
    """
    preds = []
    for season in sorted(train["season"].unique()):
        fit_rows = train[train["season"] != season]
        held = train[train["season"] == season]
        model = lg.fit_booster(gf.feature_matrix(fit_rows),
                               fit_rows[gf.LABEL].astype(int),
                               {**params, "n_estimators": trees}, seed=seed)
        out = held.loc[:, ["season", "date", "game_pk", gf.LABEL, CHAIN]].copy()
        out["raw"] = model.predict_proba(gf.feature_matrix(held))[:, 1]
        preds.append(out)
    return pd.concat(preds, ignore_index=True)


def train_learned(train: pd.DataFrame, inner_season: int, *, seed: int,
                  calibration: str = "isotonic", verbose: bool = True) -> dict:
    """The whole protocol on one training set. Returns the model and its scaffolding."""
    params, trees, grid_rows = choose_params(train, inner_season, seed, verbose)
    if verbose:
        print(f"  chosen: {params}, {trees} trees "
              f"(inner validation season {inner_season})", flush=True)
    oof = out_of_fold(train, params, trees, seed)
    calibrator = lg.Calibrator.fit(oof["raw"], oof[gf.LABEL].astype(int),
                                   kind=calibration)
    oof["learned"] = calibrator(oof["raw"])
    weights = lg.blend_weights(oof[CHAIN], oof["learned"], oof[gf.LABEL].astype(int))
    model = lg.fit_booster(gf.feature_matrix(train), train[gf.LABEL].astype(int),
                           {**params, "n_estimators": trees}, seed=seed)
    learned = lg.LearnedModel.from_fitted(
        model, gf.FEATURE_COLUMNS, calibrator,
        meta={"train_seasons": [int(s) for s in sorted(train["season"].unique())],
              "inner_validation_season": int(inner_season),
              "params": params, "trees": int(trees), "n_train": int(len(train)),
              "calibration": calibration, "blend": weights, "seed": int(seed)})
    return {"model": learned, "oof": oof, "params": params, "trees": trees,
            "grid": grid_rows, "blend": weights}


def logistic_baseline(train: pd.DataFrame, score: pd.DataFrame, seed: int):
    """The same features through a plain logistic regression.

    A linear model on these inputs is the honest middle case between the chain
    (a fixed nonlinear form somebody wrote down) and the booster (any form it
    likes). If the booster cannot beat it, the answer is that the extra
    flexibility was never the binding constraint.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=0.05, max_iter=2000,
                                            random_state=seed))
    pipe.fit(gf.feature_matrix(train), train[gf.LABEL].astype(int))
    return pipe.predict_proba(gf.feature_matrix(score))[:, 1]


# ─── scoring ───

def market_columns(score: pd.DataFrame, market: Path | None):
    if market is None:
        return score, []
    closes = pd.read_parquet(market)
    wide = closes.pivot_table(index="game_pk", columns="venue", values="p_home_close")
    wide.columns = [f"{v}_close" for v in wide.columns]
    joined = score.merge(wide, left_on="game_pk", right_index=True, how="inner")
    return joined.dropna(subset=list(wide.columns)), list(wide.columns)


def score_table(df: pd.DataFrame, models: list) -> pd.DataFrame:
    y = df[gf.LABEL].astype(float).to_numpy()
    rows = [{"model": m, "brier": brier(df[m], y), "log_loss": log_loss(df[m], y),
             "mean_p_home": float(np.clip(df[m], 1e-6, 1 - 1e-6).mean())}
            for m in models]
    return pd.DataFrame(rows).sort_values("brier").reset_index(drop=True)


def report(df: pd.DataFrame, models: list, label: str, pairs: list) -> dict:
    y = df[gf.LABEL].astype(float).to_numpy()
    table = score_table(df, models)
    print(f"\n{label}: {len(df)} games "
          f"({df['date'].min()} → {df['date'].max()}), "
          f"realized home win rate {y.mean():.4f}\n")
    print(table.round(5).to_string(index=False))
    tests = {}
    for model, base in pairs:
        if model not in df.columns or base not in df.columns:
            continue
        t = paired_t(df[model], df[base], y)
        tests[f"{model} - {base}"] = t
        print(f"\npaired Brier {model} - {base}: {t['diff']:+.5f} "
              f"(se {t['se']:.5f}, t = {t['t']:+.2f}, n = {t['n']})")
    return {"n_games": int(len(df)), "first_date": str(df["date"].min()),
            "last_date": str(df["date"].max()),
            "realized_home_win_rate": float(y.mean()),
            "scores": json.loads(table.to_json(orient="records")),
            "paired_t": tests}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES_DIR)
    parser.add_argument("--score-season", type=int, default=2026)
    parser.add_argument("--min-train-season", type=int, default=2015)
    parser.add_argument("--market", type=Path, default=None,
                        help="market_closes parquet; scores each venue's close "
                             "as a model on the games every venue priced")
    parser.add_argument("--calibration", choices=("isotonic", "platt", "identity"),
                        default="isotonic")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--no-permuted", action="store_true",
                        help="skip the permuted-label control (it doubles the "
                             "run time and is the reason to trust the rest)")
    parser.add_argument("--save-model", type=Path, default=None,
                        help="write the fitted artifact here (JSON)")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--preds-out", type=Path, default=None,
                        help="write the scored season's per-game frame here")
    args = parser.parse_args()

    df = load_features(args.features)
    df = df[df["season"] >= args.min_train_season]
    train = df[df["season"] < args.score_season]
    score = df[df["season"] == args.score_season].copy()
    if not len(score):
        raise SystemExit(f"no feature rows for {args.score_season}")
    if train["season"].nunique() < 3:
        raise SystemExit("need at least three training seasons for an inner split")
    seasons = sorted(train["season"].unique())
    print(f"train {seasons[0]}-{seasons[-1]} ({len(train)} games), "
          f"score {args.score_season} ({len(score)} games)\n")
    print("choosing hyperparameters on the last training season:", flush=True)

    inner_season = int(seasons[-1])
    fitted = train_learned(train, inner_season, seed=args.seed,
                           calibration=args.calibration)
    model = fitted["model"]
    score["learned_raw"] = model.raw(gf.feature_matrix(score))
    score["learned"] = model.calibrator(score["learned_raw"])
    score["blend"] = lg.blend_predict(fitted["blend"], score[CHAIN], score["learned"])
    score["logistic"] = logistic_baseline(train, score, args.seed)

    payload = {"score_season": int(args.score_season),
               "train_seasons": [int(s) for s in seasons],
               "n_train_games": int(len(train)),
               "params": fitted["params"], "trees": int(fitted["trees"]),
               "grid": fitted["grid"], "blend_weights": fitted["blend"],
               "calibration": args.calibration}

    if not args.no_permuted:
        rng = np.random.default_rng(args.seed)
        shuffled = train.copy()
        shuffled[gf.LABEL] = rng.permutation(shuffled[gf.LABEL].to_numpy())
        print("\npermuted-label control (same protocol, shuffled training labels):",
              flush=True)
        control = train_learned(shuffled, inner_season, seed=args.seed,
                                calibration=args.calibration, verbose=False)
        score["permuted"] = control["model"].predict(gf.feature_matrix(score))
        print(f"  {control['params']}, {control['trees']} trees", flush=True)

    models = [m for m in ["learned", "blend", "logistic", CHAIN, "chain_p_lu",
                          PRODUCTION, "permuted"] if m in score.columns]
    pairs = [("learned", CHAIN), ("blend", CHAIN), ("blend", "learned"),
             ("logistic", CHAIN), ("learned", PRODUCTION)]

    payload["full_season"] = report(score, models, f"{args.score_season}, every game",
                                    pairs)

    joined, venues = market_columns(score, args.market)
    if venues:
        payload["market_models"] = venues
        payload["common_games"] = report(
            joined, models + venues,
            f"{args.score_season}, games priced by every venue in "
            f"{args.market.name}",
            pairs + [(CHAIN, venues[0]), ("learned", venues[0])])

    # Reliability, before and after the calibrator, on the scored season.
    y = score[gf.LABEL].astype(float).to_numpy()
    print("\nReliability, learned model BEFORE calibration:")
    print(reliability(score["learned_raw"], y).to_string())
    print(f"  logistic recalibration: {calibration_fit(score['learned_raw'], y)}")
    print("\nReliability, learned model AFTER calibration:")
    print(reliability(score["learned"], y).to_string())
    print(f"  logistic recalibration: {calibration_fit(score['learned'], y)}")
    print(f"\nReliability, the hand-built chain ({CHAIN}):")
    print(reliability(score[CHAIN], y).to_string())
    print(f"  logistic recalibration: {calibration_fit(score[CHAIN], y)}")
    payload["calibration_curves"] = {
        "learned_raw": json.loads(reliability(score["learned_raw"], y)
                                  .reset_index().astype(str).to_json(orient="records")),
        "learned": json.loads(reliability(score["learned"], y)
                              .reset_index().astype(str).to_json(orient="records")),
        "chain": json.loads(reliability(score[CHAIN], y)
                            .reset_index().astype(str).to_json(orient="records")),
        "fit_raw": calibration_fit(score["learned_raw"], y),
        "fit_calibrated": calibration_fit(score["learned"], y),
        "fit_chain": calibration_fit(score[CHAIN], y),
    }

    imp = model.importances("gain")
    total = imp.sum() or 1.0
    print("\nFeature importance (gain, top 20):")
    for name, value in imp.head(20).items():
        print(f"  {name:18s} {100 * value / total:5.1f}%")
    payload["importances"] = {k: float(v) for k, v in imp.items()}

    if args.save_model is not None:
        model.save(args.save_model)
        print(f"\nwrote {args.save_model}")
    if args.preds_out is not None:
        args.preds_out.parent.mkdir(parents=True, exist_ok=True)
        score.to_parquet(args.preds_out, index=False)
        print(f"wrote {args.preds_out}")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
