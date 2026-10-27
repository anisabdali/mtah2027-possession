"""Banc d'essai DFL : pipeline v4b sur un match DFL degrade, score avec l'outil officiel.
Detecteur de ballon mort entraine sur l'AUTRE match. Aucune verite terrain du hackathon.
  python src/bench_dfl.py J03WOY J03WPY [duree_min_possession_s]"""
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
from dead_transfer import features  # noqa: E402
from predict_v2 import track_evidence  # noqa: E402
from predict_v3 import shape_evidence  # noqa: E402
import predict_v4b as v4b  # noqa: E402

FPS = 25
BALL_PARAMS = {True: (3.0, 20.0), False: (15.0, 7.4)}   # (trou moyen s, visible moyen s) : jeu / arret
DROP_FRAC, DROP_BLOCK = 0.07, 50


def markov_mask(alive, rng):
    n = len(alive)
    u = rng.random(n)
    out = np.zeros(n, bool)
    missing = False
    for i in range(n):
        gap_s, vis_s = BALL_PARAMS[bool(alive[i])]
        if u[i] < 1.0 / ((gap_s if missing else vis_s) * FPS):
            missing = not missing
        out[i] = missing
    return out


def drop_players(players, rng):
    n, K = players.shape[:2]
    nb = -(-n // DROP_BLOCK)
    drop = np.repeat(rng.random((nb, K)) < DROP_FRAC, DROP_BLOCK, axis=0)[:n]
    out = players.copy()
    out[drop] = np.nan
    return out


def coarsen(gt, min_len):
    out = gt.copy()
    cuts = np.concatenate(([0], np.flatnonzero(np.diff(out)) + 1, [len(out)]))
    prev = None
    for a, b in zip(cuts[:-1], cuts[1:]):
        c = out[a]
        if c in (pm.D, pm.X):
            prev = None
            continue
        if b - a < min_len and prev is not None:
            out[a:b] = prev
        else:
            prev = c
    return out


def baseline_like(ball, players, team, window):
    d = np.linalg.norm(players - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    dH, dA = d[:, team == 0].min(axis=1), d[:, team == 1].min(axis=1)
    seen = np.isfinite(np.minimum(dH, dA))
    h = (seen & (dH < dA)).astype(np.int64)
    a = (seen & ~(dH < dA)).astype(np.int64)
    box = np.ones(2 * window + 1, np.int64)
    ch, ca = np.convolve(h, box, "same"), np.convolve(a, box, "same")
    return np.where(ch + ca == 0, pm.D, np.where(ch >= ca, pm.H, pm.A)).astype(np.int8)


def boundary_errors(gt_dead, pred_dead):
    for name, sign in (("debut d'arret (vivant -> mort)", 1), ("fin d'arret (mort -> vivant)", -1)):
        t = np.flatnonzero(np.diff(gt_dead.astype(int)) == sign)
        q = np.flatnonzero(np.diff(pred_dead.astype(int)) == sign)
        if len(t) == 0 or len(q) == 0:
            continue
        idx = np.searchsorted(q, t)
        cand = np.stack([q[np.clip(idx - 1, 0, len(q) - 1)], q[np.clip(idx, 0, len(q) - 1)]]) - t
        off = cand[np.abs(cand).argmin(axis=0), np.arange(len(t))] / FPS
        print("  {:<32} n={:>3}  decalage signe (s) p25/med/p75 : {:+5.1f} / {:+5.1f} / {:+5.1f}"
              "  | a moins de 1 s : {:.0f} %  moins de 3 s : {:.0f} %".format(
                  name, len(t), *np.percentile(off, [25, 50, 75]),
                  100 * (np.abs(off) < 1).mean(), 100 * (np.abs(off) < 3).mean()))


def main():
    test, train = sys.argv[1], sys.argv[2]
    min_spell = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    rng = np.random.default_rng(0)
    z = np.load(os.path.join(EXT, "cache", test + ".npz"))
    zt = np.load(os.path.join(EXT, "cache", train + ".npz"))
    n, alive, owner, period, team = len(z["alive"]), z["alive"], z["owner"], z["period"], z["team"]
    blocks = []
    for p in sorted(set(period.tolist())):
        idx = np.flatnonzero(period == p)
        blocks.append((int(p), int(idx[0]), int(idx[-1])))

    gt = np.where(alive, owner, pm.D).astype(np.int8)
    gt[alive & (owner < 0)] = pm.X
    if min_spell > 0:
        gt = coarsen(gt, int(min_spell * FPS))
    for _, a, _b in blocks[1:]:
        gt[a] = pm.X                               # separe les deux periodes pour valid_runs

    ball = z["ball"].copy()
    ball[markov_mask(alive, rng)] = np.nan
    players = drop_players(z["players"], rng)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xtr = features(zt["players"])
        ok = np.flatnonzero(~np.isnan(Xtr).any(axis=1))[::10]
        mu, sd = Xtr[ok].mean(axis=0), Xtr[ok].std(axis=0) + 1e-9
        w = fit_logistic((Xtr[ok] - mu) / sd, (~zt["alive"][ok]).astype(int))
        X = features(players)
        okm = ~np.isnan(X).any(axis=1)
        p = np.full(n, np.nan)
        p[okm] = predict_probability(w, (X[okm] - mu) / sd)
        ps = pd.Series(p).rolling(125, center=True, min_periods=25).mean().to_numpy()
        e_track = track_evidence(ball, players, team)
        e_shape = shape_evidence(ball, players, team, blocks)
    e_tag = np.zeros((n, 2))

    def to_codes(paths):
        pred = np.full(n, pm.X, np.int8)
        for _, a, b, lab in paths:
            pred[a:b + 1] = lab
        return pred

    print("{} : {} images, ballon masque {:.0f} %, joueurs suivis {:.1f} / image vivante".format(
        test, n, 100 * np.isnan(ball[:, 0]).mean(),
        (~np.isnan(players[alive][:, :, 0])).sum(axis=1).mean()))
    first = [True]

    def score(name, pred):
        r = pm.evaluate(gt, pred, bootstrap_n=100)
        if first[0]:
            print("verite DFL : {} possessions, duree moyenne {:.1f} s (hackathon public : 17, 14.2 s)\n".format(
                r["n_gt_spells"], r["gt_mean_spell_duration_s"]))
            print("{:<34} {:>7} {:>8} {:>13} {:>9} {:>9} {:>9} {:>7}".format(
                "variante", "macroF1", "spellF1", "[5%-95%]", "poss.", "moy (s)", "alive/D", "equipe"))
            first[0] = False
        print("{:<34} {:>7.3f} {:>8.3f} {:>13} {:>9} {:>9.1f} {:>9.3f} {:>7.3f}".format(
            name, r["macro_f1"], r["spell_f1_mean"],
            "[{:.2f}-{:.2f}]".format(r["spell_bootstrap_low"], r["spell_bootstrap_high"]),
            r["n_pred_spells"], r["mean_spell_duration_s"], r["alive_dead_f1"], r["team_acc_both_alive"]))

    score("baseline-like fenetre +/-50", baseline_like(ball, players, team, 50))
    score("baseline-like fenetre +/-125", baseline_like(ball, players, team, 125))
    v4b.MIN_STOP_S = 3.0
    dead = v4b.dead_mask(ps, blocks, n)
    for k, ks in [(12, 12), (12, 0), (25, 12), (6, 12)]:
        score("v4b k={} ks={}".format(k, ks), to_codes(v4b.decode(e_track, e_shape, e_tag, dead, k, ks, blocks)))
    for ms in (6.0, 10.0):
        v4b.MIN_STOP_S = ms
        d2 = v4b.dead_mask(ps, blocks, n)
        score("v4b k=12 ks=12 arret min {:g} s".format(ms),
              to_codes(v4b.decode(e_track, e_shape, e_tag, d2, 12, 12, blocks)))
    score("v4b k=12 ks=12, D oracle", to_codes(v4b.decode(e_track, e_shape, e_tag, ~alive, 12, 12, blocks)))

    print("\nPlacement des frontieres du detecteur (arret min 3 s) :")
    boundary_errors(~alive, dead)


if __name__ == "__main__":
    main()