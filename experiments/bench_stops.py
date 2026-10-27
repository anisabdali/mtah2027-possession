"""Ablation du detecteur de ballon mort : v1, v2 complet, v2 sans signaux du ballon.
Deux scenarios de degradation du ballon + rappel par duree d'arret. Aucune verite terrain du hackathon.
  python src/bench_stops.py J03WOY J03WPY"""
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
import bench_dfl  # noqa: E402
from bench_dfl import markov_mask, drop_players, coarsen  # noqa: E402
from bench_dead2 import features2  # noqa: E402

FPS = 25
NOBALL = 15            # les 15 premieres colonnes de features2 sont les colonnes de mouvement
BINS = [(0, 10), (10, 20), (20, 40), (40, 1e9)]
DEFAULT = dict(bench_dfl.BALL_PARAMS)
SCENARIOS = {
    "A : ballon absent surtout pendant les arrets (hypothese d'entrainement)": {True: (3.0, 20.0), False: (15.0, 7.4)},
    "B : ballon moins lie aux arrets (absent ~33 % pendant un arret)": {True: (3.0, 20.0), False: (5.0, 10.0)},
}


def fit_cols(match, seed, ncols):
    z = np.load(os.path.join(EXT, "cache", match + ".npz"))
    bench_dfl.BALL_PARAMS = DEFAULT
    rng = np.random.default_rng(seed)
    ball = z["ball"].copy()
    ball[markov_mask(z["alive"], rng)] = np.nan
    players = drop_players(z["players"], rng)
    X = features2(ball, players)[:, :ncols]
    ok = np.flatnonzero(~np.isnan(X).any(axis=1))[::10]
    mu, sd = X[ok].mean(axis=0), X[ok].std(axis=0) + 1e-9
    w = fit_logistic((X[ok] - mu) / sd, (~z["alive"][ok]).astype(int))
    return w, mu, sd


def prob(model, X, smooth):
    w, mu, sd = model
    ok = ~np.isnan(X).any(axis=1)
    p = np.full(len(X), np.nan)
    p[ok] = predict_probability(w, (X[ok] - mu) / sd)
    return pd.Series(p).rolling(smooth, center=True, min_periods=max(1, smooth // 5)).mean().to_numpy()


def stop_segments(truth, blocks):
    segs = []
    for _, a, b in blocks:
        d = np.diff(np.concatenate(([0], truth[a:b + 1].astype(np.int8), [0])))
        segs += [(a + x, a + y) for x, y in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]
    return segs


def near(gt_dead, pred_dead, sign):
    t = np.flatnonzero(np.diff(gt_dead.astype(int)) == sign)
    q = np.flatnonzero(np.diff(pred_dead.astype(int)) == sign)
    if len(t) == 0 or len(q) == 0:
        return float("nan")
    idx = np.searchsorted(q, t)
    cand = np.stack([q[np.clip(idx - 1, 0, len(q) - 1)], q[np.clip(idx, 0, len(q) - 1)]]) - t
    return float((np.abs(cand).min(axis=0) / FPS < 1).mean())


def main():
    test, train = sys.argv[1], sys.argv[2]
    v4b.MIN_STOP_S = 3.0
    z = np.load(os.path.join(EXT, "cache", test + ".npz"))
    n, alive, owner, period, team = len(z["alive"]), z["alive"], z["owner"], z["period"], z["team"]
    blocks = []
    for p in sorted(set(period.tolist())):
        idx = np.flatnonzero(period == p)
        blocks.append((int(p), int(idx[0]), int(idx[-1])))
    gt = np.where(alive, owner, pm.D).astype(np.int8)
    gt[alive & (owner < 0)] = pm.X
    gt = coarsen(gt, int(6 * FPS))
    for _, a, _b in blocks[1:]:
        gt[a] = pm.X

    zt = np.load(os.path.join(EXT, "cache", train + ".npz"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xtr = features_v1(zt["players"])
        ok = np.flatnonzero(~np.isnan(Xtr).any(axis=1))[::10]
        mu1, sd1 = Xtr[ok].mean(axis=0), Xtr[ok].std(axis=0) + 1e-9
        m1 = (fit_logistic((Xtr[ok] - mu1) / sd1, (~zt["alive"][ok]).astype(int)), mu1, sd1)
        m_full, m_nb = fit_cols(train, 1, 18), fit_cols(train, 1, NOBALL)

    truth = ~alive
    segs = stop_segments(truth, blocks)
    dur = np.array([(b - a) / FPS for a, b in segs])
    print("{} : {} arrets reels ; par duree : {}".format(test, len(segs), ", ".join(
        "{}-{} s : {}".format(lo, "inf" if hi > 1e8 else int(hi), int(((dur >= lo) & (dur < hi)).sum())) for lo, hi in BINS)))
    e_tag = np.zeros((n, 2))
    e_static = {}

    for name, params in SCENARIOS.items():
        bench_dfl.BALL_PARAMS = params
        rng = np.random.default_rng(0)
        ball = z["ball"].copy()
        ball[markov_mask(alive, rng)] = np.nan
        players = drop_players(z["players"], rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X1, X2 = features_v1(players), features2(ball, players)
            probs = {"v1": prob(m1, X1, 125), "v2 complet": prob(m_full, X2, 25),
                     "v2 sans ballon": prob(m_nb, X2[:, :NOBALL], 25)}
            e_track = track_evidence(ball, players, team)
            e_shape = shape_evidence(ball, players, team, blocks)
        print("\n=== Scenario {} ; ballon masque {:.0f} % ===".format(name, 100 * np.isnan(ball[:, 0]).mean()))
        print("{:<16} {:>7} {:>8} {:>7} | {:>6} {:>6} {:>6} {:>6} | {:>7} | {:>7} {:>7}".format(
            "detecteur", "bal.acc", "spellF1", "arrets", "<10 s", "10-20", "20-40", ">40 s",
            "fausses", "debut<1s", "fin<1s"))
        masks = {k: v4b.dead_mask(p, blocks, n) for k, p in probs.items()}
        masks["oracle"] = truth
        for k, dead in masks.items():
            pred = np.full(n, pm.X, np.int8)
            for _, a, b, lab in v4b.decode(e_track, e_shape, e_tag, dead, 12, 12, blocks):
                pred[a:b + 1] = lab
            r = pm.evaluate(gt, pred, bootstrap_n=0)
            bal = 0.5 * ((dead & truth).sum() / truth.sum() + (~dead & ~truth).sum() / (~truth).sum())
            rec = []
            for lo, hi in BINS:
                sel = [(a, b) for (a, b), d in zip(segs, dur) if lo <= d < hi]
                rec.append(np.mean([dead[a:b].mean() for a, b in sel]) if sel else float("nan"))
            d = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
            print("{:<16} {:>7.3f} {:>8.3f} {:>7} | {:>5.0f}% {:>5.0f}% {:>5.0f}% {:>5.0f}% | {:>6.1f}% | {:>7.0f}% {:>6.0f}%".format(
                k, bal, r["spell_f1_mean"], int((d == 1).sum()), *[100 * x for x in rec],
                100 * (dead & ~truth).sum() / (~truth).sum(), 100 * near(truth, dead, 1), 100 * near(truth, dead, -1)))


if __name__ == "__main__":
    main()