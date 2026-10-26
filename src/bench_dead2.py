"""Detecteur de ballon mort v2 : fenetres gauche/droite + signaux du ballon avec indicateur de presence,
entraine sur donnees DFL degradees (autre match, autre graine). Meme degradation de test que bench_dfl.py.
  python src/bench_dead2.py J03WOY J03WPY"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from inspect_open import EXT  # noqa: E402
from eval_learned import fit_logistic, predict_probability  # noqa: E402
from dead_transfer import features as features_v1  # noqa: E402
from predict_v2 import track_evidence  # noqa: E402
from predict_v3 import shape_evidence  # noqa: E402
import predict_v4b as v4b  # noqa: E402
from bench_dfl import markov_mask, drop_players, boundary_errors  # noqa: E402

FPS = 25
H = 12
VMAX = 12.0
L, W = 105.0, 68.0


def roll(a, w, kind):
    s = pd.Series(a)
    mp = max(1, w // 5)
    if kind == "c":
        return s.rolling(w, center=True, min_periods=mp).mean().to_numpy()
    if kind == "l":
        return s.rolling(w, min_periods=mp).mean().to_numpy()
    return s[::-1].rolling(w, min_periods=mp).mean()[::-1].to_numpy()


def features2(ball, players):
    n = len(ball)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disp = np.full(players.shape[:2], np.nan, np.float32)
        disp[H:n - H] = np.linalg.norm(players[2 * H:] - players[:n - 2 * H], axis=2)
        v = disp / (2 * H / FPS)
        v[v > VMAX] = np.nan
        valid = ~np.isnan(v)
        med = np.nanmedian(v, axis=1)
        p75 = np.nanpercentile(v, 75, axis=1)
        slow = ((v < 1.0) & valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
        slow[~valid.any(axis=1)] = np.nan
        bd = np.full(n, np.nan, np.float32)
        bd[H:n - H] = np.linalg.norm(ball[2 * H:] - ball[:n - 2 * H], axis=1)
        bspeed = bd / (2 * H / FPS)
        bspeed[bspeed > 40] = np.nan
        cols = []
        for b in (med, slow, p75):
            cols.append(roll(b, 375, "c"))
            for w in (50, 150):
                cols.append(roll(b, w, "l"))
                cols.append(roll(b, w, "r"))
        vis = (~np.isnan(ball[:, 0])).astype(float)
        out = ((ball[:, 0] < 0) | (ball[:, 0] > L) | (ball[:, 1] < 0) | (ball[:, 1] > W)).astype(float)
        sp, m = np.nan_to_num(bspeed), (~np.isnan(bspeed)).astype(float)
        num, den = roll(sp, 50, "c"), roll(m, 50, "c")
        cols.append(roll(vis, 125, "c"))
        cols.append(np.where(den > 0.02, num / np.maximum(den, 1e-9), 0.0))
        cols.append(roll(out, 125, "c"))
    return np.column_stack(cols)


def fit(match, seed):
    z = np.load(os.path.join(EXT, "cache", match + ".npz"))
    rng = np.random.default_rng(seed)
    ball = z["ball"].copy()
    ball[markov_mask(z["alive"], rng)] = np.nan
    players = drop_players(z["players"], rng)
    X = features2(ball, players)
    ok = np.flatnonzero(~np.isnan(X).any(axis=1))[::10]
    mu, sd = X[ok].mean(axis=0), X[ok].std(axis=0) + 1e-9
    w = fit_logistic((X[ok] - mu) / sd, (~z["alive"][ok]).astype(int))
    return w, mu, sd


def main():
    test, train = sys.argv[1], sys.argv[2]
    z = np.load(os.path.join(EXT, "cache", test + ".npz"))
    n, alive, owner, period, team = len(z["alive"]), z["alive"], z["owner"], z["period"], z["team"]
    blocks = []
    for p in sorted(set(period.tolist())):
        idx = np.flatnonzero(period == p)
        blocks.append((int(p), int(idx[0]), int(idx[-1])))
    gt = np.where(alive, owner, pm.D).astype(np.int8)
    gt[alive & (owner < 0)] = pm.X
    for _, a, _b in blocks[1:]:
        gt[a] = pm.X

    rng = np.random.default_rng(0)                         # meme degradation de test que bench_dfl.py
    ball = z["ball"].copy()
    ball[markov_mask(alive, rng)] = np.nan
    players = drop_players(z["players"], rng)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zt = np.load(os.path.join(EXT, "cache", train + ".npz"))
        Xtr = features_v1(zt["players"])
        ok = np.flatnonzero(~np.isnan(Xtr).any(axis=1))[::10]
        mu, sd = Xtr[ok].mean(axis=0), Xtr[ok].std(axis=0) + 1e-9
        w1 = fit_logistic((Xtr[ok] - mu) / sd, (~zt["alive"][ok]).astype(int))
        X1 = features_v1(players)
        ok1 = ~np.isnan(X1).any(axis=1)
        p1 = np.full(n, np.nan)
        p1[ok1] = predict_probability(w1, (X1[ok1] - mu) / sd)
        ps1 = pd.Series(p1).rolling(125, center=True, min_periods=25).mean().to_numpy()

        w2, mu2, sd2 = fit(train, seed=1)
        X2 = features2(ball, players)
        ok2 = ~np.isnan(X2).any(axis=1)
        p2 = np.full(n, np.nan)
        p2[ok2] = predict_probability(w2, (X2[ok2] - mu2) / sd2)
        ps2_1s = pd.Series(p2).rolling(25, center=True, min_periods=5).mean().to_numpy()
        ps2_5s = pd.Series(p2).rolling(125, center=True, min_periods=25).mean().to_numpy()
        e_track = track_evidence(ball, players, team)
        e_shape = shape_evidence(ball, players, team, blocks)
    e_tag = np.zeros((n, 2))

    truth = ~alive
    print("{} : ballon masque {:.0f} %".format(test, 100 * np.isnan(ball[:, 0]).mean()))
    print("\n{:<40} {:>8} {:>9} {:>8} {:>8} {:>9}".format(
        "detecteur de ballon mort", "bal.acc", "macroF1", "spellF1", "poss.", "arrets"))
    dead_by = {}
    for name, dead in (("v1 (propre, lissage 5 s)", v4b.dead_mask(ps1, blocks, n)),
                       ("v2 (degrade, lissage 1 s)", v4b.dead_mask(ps2_1s, blocks, n)),
                       ("v2 (degrade, lissage 5 s)", v4b.dead_mask(ps2_5s, blocks, n)),
                       ("oracle", ~alive)):
        pred = np.full(n, pm.X, np.int8)
        for _, a, b, lab in v4b.decode(e_track, e_shape, e_tag, dead, 12, 12, blocks):
            pred[a:b + 1] = lab
        r = pm.evaluate(gt, pred, bootstrap_n=0)
        bal = 0.5 * ((dead & truth).sum() / truth.sum() + (~dead & ~truth).sum() / (~truth).sum())
        d = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
        print("{:<40} {:>8.3f} {:>9.3f} {:>8.3f} {:>8} {:>9}".format(
            name, bal, r["macro_f1"], r["spell_f1_mean"], r["n_pred_spells"], int((d == 1).sum())))
        dead_by[name] = dead
    for name in ("v1 (propre, lissage 5 s)", "v2 (degrade, lissage 1 s)"):
        print("\nPlacement des frontieres, {} :".format(name))
        boundary_errors(truth, dead_by[name])


if __name__ == "__main__":
    main()