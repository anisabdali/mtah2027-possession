"""Experience : poids de preuve du joueur le plus proche calibres sur la DFL (au lieu des tags).
Table apprise sur un match DFL degrade, test sur l'autre, D oracle. Ne modifie rien de ce qui est fige.
  python src/bench_weights.py J03WOY J03WPY      (test, calibration)"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from inspect_open import EXT  # noqa: E402
import predict_v2  # noqa: E402
from predict_v2 import track_evidence, WEIGHT as WEIGHT_TAGS  # noqa: E402
from predict_v3 import shape_evidence  # noqa: E402
import predict_v4b as v4b  # noqa: E402
from bench_dfl import markov_mask, drop_players, coarsen  # noqa: E402

FPS = 25


def load_degraded(match, seed):
    z = np.load(os.path.join(EXT, "cache", match + ".npz"))
    rng = np.random.default_rng(seed)
    ball = z["ball"].copy()
    ball[markov_mask(z["alive"], rng)] = np.nan
    return z, ball, drop_players(z["players"], rng)


def calibrate(match, seed=5):
    z, ball, players = load_degraded(match, seed)
    team, alive, owner = z["team"], z["alive"], z["owner"]
    d = np.linalg.norm(players - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    dH, dA = d[:, team == 0].min(axis=1), d[:, team == 1].min(axis=1)
    ok = alive & (owner >= 0) & np.isfinite(dH) & np.isfinite(dA)
    dmin, margin = np.minimum(dH, dA)[ok], np.abs(dA - dH)[ok]
    agree = (dA[ok] < dH[ok]).astype(int) == owner[ok]
    r = np.digitize(dmin, predict_v2.DIST_EDGES)
    c = np.digitize(margin, predict_v2.MARGIN_EDGES)
    table = np.full((5, 5), 0.5)
    for i in range(5):
        for j in range(5):
            m = (r == i) & (c == j)
            if m.sum() >= 200:
                table[i, j] = agree[m].mean()
    return table


def main():
    test, calib = sys.argv[1], sys.argv[2]
    table = calibrate(calib)
    weight_dfl = np.where(table > 0.5, np.log(table / (1 - table)), 0.0)
    print("accord appris sur {} (lignes : distance 0-1, 1-2, 2-4, 4-8, 8+ m ; colonnes : marge 0-0.5, 0.5-1, 1-2, 2-4, 4+ m)".format(calib))
    print(np.round(table, 2))
    print("table utilisee jusqu'ici (tags) :")
    print(np.round(predict_v2.AGREEMENT, 2))

    z, ball, players = load_degraded(test, 0)           # meme degradation de test que bench_dfl.py
    n, alive, owner, period, team = len(z["alive"]), z["alive"], z["owner"], z["period"], z["team"]
    blocks = []
    for p in sorted(set(period.tolist())):
        idx = np.flatnonzero(period == p)
        blocks.append((int(p), int(idx[0]), int(idx[-1])))
    e_shape = shape_evidence(ball, players, team, blocks)
    e_tag = np.zeros((n, 2))
    base = np.where(alive, owner, pm.D).astype(np.int8)
    base[alive & (owner < 0)] = pm.X

    print("\n{} : D oracle, ks=12\n".format(test))
    print("{:<14} {:<8} {:>4} | {:>8} {:>10}".format("possession DFL", "poids", "k", "spellF1", "equipe ok"))
    for gt_name, min_s in (("fine", 0), ("regroupee 6 s", 6)):
        gt = coarsen(base, int(min_s * FPS)) if min_s else base.copy()
        for _, a, _b in blocks[1:]:
            gt[a] = pm.X
        for name, weight in (("tags", WEIGHT_TAGS), ("DFL", weight_dfl)):
            predict_v2.WEIGHT = weight
            e_track = track_evidence(ball, players, team)
            for k in (6, 12):
                pred = np.full(n, pm.X, np.int8)
                for _, a, b, lab in v4b.decode(e_track, e_shape, e_tag, ~alive, k, 12, blocks):
                    pred[a:b + 1] = lab
                r = pm.evaluate(gt, pred, bootstrap_n=0)
                print("{:<14} {:<8} {:>4} | {:>8.3f} {:>10.3f}".format(
                    gt_name, name, k, r["spell_f1_mean"], r["team_acc_both_alive"]))
    predict_v2.WEIGHT = WEIGHT_TAGS


if __name__ == "__main__":
    main()