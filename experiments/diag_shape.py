"""La forme collective des equipes (sans ballon) predit-elle le possesseur ?
Pseudo-etiquettes automatiques (ballon tres proche d'un joueur). Sans verite terrain."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import GAME, half_video_blocks  # noqa: E402
from starter.load_smart_tagging import load as load_tags  # noqa: E402
from diagnostics import load_tracking  # noqa: E402
from diag_margin import tag_possessor  # noqa: E402

FPS = 25
NAMES = ["dcx", "mid", "dsx", "dsy", "dcy", "vdx", "vmid"]


def team_stats(players, team, t):
    P = players[:, team == t, :]
    cnt = (~np.isnan(P[:, :, 0])).sum(axis=1)
    with np.errstate(all="ignore"):
        cx, cy = np.nanmean(P[:, :, 0], axis=1), np.nanmean(P[:, :, 1], axis=1)
        sx, sy = np.nanstd(P[:, :, 0], axis=1), np.nanstd(P[:, :, 1], axis=1)
    bad = cnt < 7
    for a in (cx, cy, sx, sy):
        a[bad] = np.nan
    return cx, cy, sx, sy


def velocity(c, win=25):
    v = np.full_like(c, np.nan)
    v[win:-win] = (c[2 * win:] - c[:-2 * win]) / (2 * win / FPS)
    return v


def shape_features(players, team):
    cxH, cyH, sxH, syH = team_stats(players, team, 0)
    cxA, cyA, sxA, syA = team_stats(players, team, 1)
    vH, vA = velocity(cxH), velocity(cxA)
    return np.column_stack([cxH - cxA, (cxH + cxA) / 2 - 52.65, sxH - sxA, syH - syA,
                            cyH - cyA, vH - vA, (vH + vA) / 2])


def fit_logreg(X, y, l2=1.0, iters=25):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) + l2 * w
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w


def predict(w, X):
    return 1 / (1 + np.exp(-np.clip(np.hstack([X, np.ones((len(X), 1))]) @ w, -30, 30)))


def main():
    ball, players, team = load_tracking()
    n = len(ball)
    F = shape_features(players, team)
    feat_ok = ~np.isnan(F).any(axis=1)

    d = np.linalg.norm(players - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    dH, dA = d[:, team == 0].min(axis=1), d[:, team == 1].min(axis=1)
    both = np.isfinite(dH) & np.isfinite(dA)
    sure = both & (np.minimum(dH, dA) < 1.0) & (np.abs(dA - dH) > 2.0)
    y_all = (dH < dA).astype(float)                        # 1 : Home possede
    tag_p = tag_possessor(load_tags(), n)
    has_ball = ~np.isnan(ball[:, 0])

    for half, a, b in half_video_blocks(GAME):
        idx = np.arange(a, b + 1)
        train_pool = idx[sure[idx] & feat_ok[idx]][::5]     # 1 image sur 5 (images voisines tres liees)
        fold = (train_pool // 7500) % 5                     # blocs de 5 min
        accs, majs = [], []
        for f in range(5):
            tr, te = train_pool[fold != f], train_pool[fold == f]
            if len(te) < 50:
                continue
            mu, sd = F[tr].mean(axis=0), F[tr].std(axis=0) + 1e-9
            w = fit_logreg((F[tr] - mu) / sd, y_all[tr])
            pred = predict(w, (F[te] - mu) / sd) > 0.5
            accs.append((pred == (y_all[te] > 0.5)).mean())
            majs.append(max(y_all[te].mean(), 1 - y_all[te].mean()))
        mu, sd = F[train_pool].mean(axis=0), F[train_pool].std(axis=0) + 1e-9
        w = fit_logreg((F[train_pool] - mu) / sd, y_all[train_pool])
        gap = idx[(~has_ball[idx]) & feat_ok[idx] & (tag_p[idx] >= 0)]
        pg = predict(w, (F[gap] - mu) / sd) > 0.5
        agree_gap = (pg == (tag_p[gap] == 0)).mean()
        print("\n== Mi-temps {} ==".format(half))
        print("pseudo-etiquettes sures : {} images ({:.0f} s), Home {:.0f} %".format(
            len(train_pool) * 5, len(train_pool) * 5 / FPS, 100 * y_all[train_pool].mean()))
        print("precision sur blocs de 5 min non vus : {:.1f} % (classe majoritaire : {:.1f} %)".format(
            100 * np.mean(accs), 100 * np.mean(majs)))
        print("accord avec les tags sur images SANS ballon : {:.1f} % ({} s)".format(
            100 * agree_gap, len(gap) // FPS))
        print("poids (standardises) :", ", ".join("{} {:+.2f}".format(k, v) for k, v in zip(NAMES, w[:-1])))


if __name__ == "__main__":
    main()