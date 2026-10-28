"""Robustesse du detecteur de ballon mort a la visibilite du ballon pendant les arrets.
Scenarios A (actuel), B, C (ballon toujours visible pendant les arrets). Sans verite terrain du hackathon.
  python experiments/bench_robust.py J03WOY J03WPY      (test, entrainement)"""
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from inspect_open import EXT  # noqa: E402
from eval_learned import fit_logistic  # noqa: E402
from predict_v2 import track_evidence  # noqa: E402
from predict_v3 import shape_evidence  # noqa: E402
import predict_v4b as v4b  # noqa: E402
import bench_dfl  # noqa: E402
from bench_dfl import markov_mask, drop_players, coarsen  # noqa: E402
from bench_dead2 import features2  # noqa: E402
from bench_stops import stop_segments, near, prob  # noqa: E402

FPS = 25
NOBALL = 15
SCEN = {
    "A": {True: (3.0, 20.0), False: (15.0, 7.4)},
    "B": {True: (3.0, 20.0), False: (5.0, 10.0)},
    "C": {True: (3.0, 20.0), False: (0.04, 1.0e6)},      # ballon visible en permanence pendant les arrets
}
BINS = [(0, 10), (10, 20), (20, 1e9)]


def degrade(z, params, seed):
    bench_dfl.BALL_PARAMS = params
    rng = np.random.default_rng(seed)
    ball = z["ball"].copy()
    ball[markov_mask(z["alive"], rng)] = np.nan
    return ball, drop_players(z["players"], rng)


def train_sets(match):
    z = np.load(os.path.join(EXT, "cache", match + ".npz"))
    out = {}
    for i, name in enumerate("ABC"):
        ball, players = degrade(z, SCEN[name], 10 + i)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X = features2(ball, players)
        ok = np.flatnonzero(~np.isnan(X).any(axis=1))[::10]
        out[name] = (X[ok], (~z["alive"][ok]).astype(int))
    return out


def fit(X, y):
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    return fit_logistic((X - mu) / sd, y), mu, sd


def main():
    test, train = sys.argv[1], sys.argv[2]
    v4b.MIN_STOP_S = 3.0
    z = np.load(os.path.join(EXT, "cache", test + ".npz"))
    n, alive, owner, period, team = len(z["alive"]), z["alive"], z["owner"], z["period"], z["team"]
    blocks = []
    for p in sorted(set(period.tolist())):
        idx = np.flatnonzero(period == p)
        blocks.append((int(p), int(idx[0]), int(idx[-1])))
    base = np.where(alive, owner, pm.D).astype(np.int8)
    base[alive & (owner < 0)] = pm.X
    gts = {}
    for label, merge in (("fine", 0), ("6 s", 6)):
        g = coarsen(base, int(merge * FPS)) if merge else base.copy()
        for _, a, _b in blocks[1:]:
            g[a] = pm.X
        gts[label] = g

    sets = train_sets(train)
    XA, yA = sets["A"]
    models = {
        "v2 entraine sur A (actuel)": (fit(XA, yA), 18),
        "v2 entraine sur A+B+C": (fit(np.vstack([sets[k][0] for k in "ABC"]), np.concatenate([sets[k][1] for k in "ABC"])), 18),
        "v2 sans ballon": (fit(XA[:, :NOBALL], yA), NOBALL),
    }
    truth = ~alive
    segs = stop_segments(truth, blocks)
    dur = np.array([(b - a) / FPS for a, b in segs])
    e_tag = np.zeros((n, 2))
    print("{} (entrainement : {}) : {} arrets reels".format(test, train, len(segs)))

    for sname in "ABC":
        ball, players = degrade(z, SCEN[sname], 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X2 = features2(ball, players)
            e_track = track_evidence(ball, players, team)
            e_shape = shape_evidence(ball, players, team, blocks)
        print("\n=== Scenario {} : ballon visible pendant les arrets {:.0f} %, pendant le jeu {:.0f} % ===".format(
            sname, 100 * (~np.isnan(ball[truth, 0])).mean(), 100 * (~np.isnan(ball[alive, 0])).mean()))
        print("{:<28} {:>7} {:>8} {:>8} | {:>6} {:>6} {:>6} | {:>7} {:>6}".format(
            "detecteur", "bal.acc", "sF1 fin", "sF1 6s", "<10 s", "10-20", ">20 s", "debut<1s", "fin<1s"))
        masks = {k: v4b.dead_mask(prob(m, X2[:, :nc], 25), blocks, n) for k, (m, nc) in models.items()}
        masks["oracle"] = truth
        for k, dead in masks.items():
            pred = np.full(n, pm.X, np.int8)
            for _, a, b, lab in v4b.decode(e_track, e_shape, e_tag, dead, 12, 12, blocks):
                pred[a:b + 1] = lab
            sf = {g: pm.evaluate(gts[g], pred, bootstrap_n=0)["spell_f1_mean"] for g in gts}
            bal = 0.5 * ((dead & truth).sum() / truth.sum() + (~dead & ~truth).sum() / (~truth).sum())
            rec = []
            for lo, hi in BINS:
                sel = [(a, b) for (a, b), d in zip(segs, dur) if lo <= d < hi]
                rec.append(100 * np.mean([dead[a:b].mean() for a, b in sel]))
            print("{:<28} {:>7.3f} {:>8.3f} {:>8.3f} | {:>5.0f}% {:>5.0f}% {:>5.0f}% | {:>6.0f}% {:>5.0f}%".format(
                k, bal, sf["fine"], sf["6 s"], *rec, 100 * near(truth, dead, 1), 100 * near(truth, dead, -1)))


if __name__ == "__main__":
    main()