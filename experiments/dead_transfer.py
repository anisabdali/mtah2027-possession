"""Transfert du detecteur de ballon mort : entraine sur les deux matchs DFL, applique au match du hackathon.
Une seule execution de controle ; aucun reglage d'apres la verite terrain publique."""
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import GAME, half_video_blocks  # noqa: E402
from diagnostics import load_tracking  # noqa: E402
from inspect_open import EXT  # noqa: E402
from eval_learned import fit_logistic, predict_probability  # noqa: E402

FPS = 25
H = 12
VMAX = 12.0                      # m/s : au-dela, saut d'identite et non une vitesse reelle
WINDOWS = [25, 125, 375]         # 1 s, 5 s, 15 s


def rolling(a, w):
    return pd.Series(a).rolling(w, center=True, min_periods=max(1, w // 5)).mean().to_numpy()


def features(players):
    n = len(players)
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
    return np.column_stack([rolling(b, w) for b in (med, slow, p75) for w in WINDOWS])


def clock(frame):
    s = frame / FPS
    return "{:d}:{:02d}".format(int(s // 60), int(s % 60))


def runs(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def main():
    Xs, ys = [], []
    for m in ("J03WPY", "J03WOY"):
        z = np.load(os.path.join(EXT, "cache", m + ".npz"))
        X = features(z["players"])
        ok = np.flatnonzero(~np.isnan(X).any(axis=1))[::10]
        Xs.append(X[ok])
        ys.append((~z["alive"][ok]).astype(int))
    Xtr, ytr = np.vstack(Xs), np.concatenate(ys)
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
    w = fit_logistic((Xtr - mu) / sd, ytr)
    print("entrainement : {} images DFL, {:.0f} % de ballon mort".format(len(ytr), 100 * ytr.mean()))

    ball, players, team = load_tracking()
    n = len(ball)
    X = features(players)
    ok = ~np.isnan(X).any(axis=1)
    p = np.full(n, np.nan)
    p[ok] = predict_probability(w, (X[ok] - mu) / sd)
    ps = pd.Series(p).rolling(125, center=True, min_periods=25).mean().to_numpy()
    dead = np.nan_to_num(ps, nan=0.0) >= 0.5
    np.save(os.path.join(ROOT, "data", "cache", "dead_prob.npy"), ps)

    visible = ~np.isnan(ball[:, 0])
    npl = (~np.isnan(players[:, :, 0])).sum(axis=1)
    graded = np.zeros(n, bool)
    for _, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True
    print("\n== Match du hackathon, images notees ==")
    print("ballon mort predit : {:.1f} %".format(100 * dead[graded].mean()))
    print("ballon visible : {:.0f} % si D predit, {:.0f} % si vivant predit".format(
        100 * visible[graded & dead].mean(), 100 * visible[graded & ~dead].mean()))

    allstops = []
    for half, a0, b0 in half_video_blocks(GAME):
        s, e = runs(dead[a0:b0 + 1])
        allstops += [(a0 + x, a0 + y) for x, y in zip(s, e)]
    L = np.array([(y - x) / FPS for x, y in allstops])
    print("arrets predits : {}  duree (s) : mediane {:.1f}  p90 {:.1f}  max {:.0f}".format(
        len(L), np.median(L), np.percentile(L, 90), L.max()))
    print("\nplus longs arrets predits (heure video) :")
    for x, y in sorted(allstops, key=lambda t: t[0] - t[1])[:8]:
        print("  {} -> {}  {:5.0f} s | ballon visible {:.0f} % | joueurs suivis {:.1f}".format(
            clock(x), clock(y), (y - x) / FPS, 100 * visible[x:y].mean(), npl[x:y].mean()))

    gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))
    lab = gt["label"].astype(str).values
    win = np.flatnonzero(lab != "X")
    truth, pred = lab[win] == "D", dead[win]
    tp = int((truth & pred).sum())
    prec, rec = tp / max(1, int(pred.sum())), tp / max(1, int(truth.sum()))
    ar = (~truth & ~pred).sum() / max(1, int((~truth).sum()))
    print("\n== Controle unique sur les 5 minutes publiques ({} images) ==".format(len(win)))
    print("D reel {:.1f} %  D predit {:.1f} %".format(100 * truth.mean(), 100 * pred.mean()))
    print("precision D {:.1f} %  rappel D {:.1f} %  rappel 'vivant' {:.1f} %".format(100 * prec, 100 * rec, 100 * ar))


if __name__ == "__main__":
    main()