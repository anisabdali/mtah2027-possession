"""Detecter le ballon mort a partir du seul mouvement collectif des joueurs (sans le ballon).
Entraine sur un match DFL, teste sur l'autre, dans les deux sens."""
import os
import warnings

import numpy as np
import pandas as pd

from inspect_open import EXT
from eval_learned import fit_logistic, predict_probability, balanced_accuracy

FPS = 25
H = 12
WINDOWS = [25, 125, 375]                     # 1 s, 5 s, 15 s
BASES = ["v_med", "slow", "v_p75"]
NAMES = ["{}_{}s".format(b, w // FPS) for b in BASES for w in WINDOWS]


def rolling(a, w):
    return pd.Series(a).rolling(w, center=True, min_periods=max(1, w // 5)).mean().to_numpy()


def build(match):
    z = np.load(os.path.join(EXT, "cache", match + ".npz"))
    players, alive = z["players"], z["alive"]
    n = len(players)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disp = np.full(players.shape[:2], np.nan, np.float32)
        disp[H:n - H] = np.linalg.norm(players[2 * H:] - players[:n - 2 * H], axis=2)
        v = disp / (2 * H / FPS)
        valid = ~np.isnan(v)
        med = np.nanmedian(v, axis=1)
        p75 = np.nanpercentile(v, 75, axis=1)
    slow = ((v < 1.0) & valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    slow[~valid.any(axis=1)] = np.nan
    X = np.column_stack([rolling(b, w) for b in (med, slow, p75) for w in WINDOWS])
    return X, (~alive).astype(int), ~np.isnan(X).any(axis=1)


def seg_stats(y):
    d = np.diff(np.concatenate(([0], y.astype(np.int8), [0])))
    lens = (np.flatnonzero(d == -1) - np.flatnonzero(d == 1)) / FPS
    return len(lens), (np.median(lens) if len(lens) else 0.0)


def line(name, y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    prec = tp / max(1, int((pred == 1).sum()))
    rec = tp / max(1, int((y == 1).sum()))
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    k, med = seg_stats(pred)
    print("{:<34} bal.acc {:5.1f}%  F1(D) {:5.1f}%  prec {:5.1f}%  rappel {:5.1f}%  arrets {:>4}  med {:4.1f}s".format(
        name, 100 * balanced_accuracy(y, pred), 100 * f1, 100 * prec, 100 * rec, k, med))


def main():
    for a, b in (("J03WPY", "J03WOY"), ("J03WOY", "J03WPY")):
        Xa, ya, oka = build(a)
        Xb, yb, okb = build(b)
        tr = np.flatnonzero(oka)[::10]
        mu, sd = Xa[tr].mean(axis=0), Xa[tr].std(axis=0) + 1e-9
        w = fit_logistic((Xa[tr] - mu) / sd, ya[tr])
        te = np.flatnonzero(okb)
        p = predict_probability(w, (Xb[te] - mu) / sd)
        ps = pd.Series(p).rolling(125, center=True, min_periods=1).mean().to_numpy()   # lissage 5 s
        j = NAMES.index("v_med_5s")                      # regle a un seul seuil, seuil choisi sur le match d'entrainement
        grid = np.quantile(Xa[tr, j], np.linspace(0.05, 0.95, 46))
        best = max(grid, key=lambda t: balanced_accuracy(ya[tr], (Xa[tr, j] < t).astype(int)))
        print("\n===== entrainement {} -> test {} ({:.0f} % de ballon mort au test) =====".format(a, b, 100 * yb[te].mean()))
        print("vrais arrets : {} segments, mediane {:.1f} s".format(*seg_stats(yb[te])))
        line("regle : vitesse mediane 5 s < {:.2f}".format(best), yb[te], (Xb[te, j] < best).astype(int))
        line("logistique, seuil 0.5", yb[te], (p >= 0.5).astype(int))
        line("logistique lissee 5 s, seuil 0.5", yb[te], (ps >= 0.5).astype(int))
        print("poids : " + ", ".join("{} {:+.2f}".format(n, c) for n, c in zip(NAMES, w[:-1])))


if __name__ == "__main__":
    main()