"""Pourquoi le rappel de D est-il bas sur la fenetre publique ? Diagnostic en lecture seule des arrets reels.
Compare la vitesse des joueurs pendant les arrets, chez nous et dans IDSSE. Ne regle aucun parametre."""
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from games import GAME, half_video_blocks  # noqa: E402
from make_submission import load_tracking, dead_probability, dead_mask, HW, FPS  # noqa: E402


def motion(players):
    n = len(players)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disp = np.full(players.shape[:2], np.nan, np.float32)
        disp[HW:n - HW] = np.linalg.norm(players[2 * HW:] - players[:n - 2 * HW], axis=2)
        v = disp / (2 * HW / FPS)
        v[v > 12] = np.nan
        med = np.nanmedian(v, axis=1)
        p75 = np.nanpercentile(v, 75, axis=1)
    sm = lambda a: pd.Series(a).rolling(125, center=True, min_periods=25).mean().to_numpy()
    return sm(med), sm(p75)


def clock(frame):
    s = frame / FPS
    return "{:d}:{:02d}".format(int(s // 60), int(s % 60))


def pcts(a):
    a = a[~np.isnan(a)]
    return "  ".join("{:4.2f}".format(x) for x in np.percentile(a, [10, 25, 50, 75, 90]))


def runs(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def main():
    ball, players, team = load_tracking()
    n = len(ball)
    blocks = half_video_blocks(GAME)
    dead = dead_mask(dead_probability(ball, players), blocks, n)
    med, p75 = motion(players)
    cnt = (~np.isnan(players[:, :, 0])).sum(axis=1)
    vis = ~np.isnan(ball[:, 0])
    gt = pd.read_csv(os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv"))["label"].astype(str).values
    isD, isAlive = gt == "D", (gt == "H") | (gt == "A")

    print("== Les arrets reels de la fenetre publique ==")
    print("{:>10} {:>7} {:>10} {:>8} {:>10} {:>8} {:>9}".format(
        "debut", "duree", "detecte", "ballon", "v med.", "v p75", "joueurs"))
    for a, b in runs(isD):
        print("{:>10} {:>6.1f}s {:>9.0f}% {:>7.0f}% {:>9.2f} {:>8.2f} {:>9.1f}".format(
            clock(a), (b - a) / FPS, 100 * dead[a:b].mean(), 100 * vis[a:b].mean(),
            np.nanmean(med[a:b]), np.nanmean(p75[a:b]), cnt[a:b].mean()))
    print("rappel global dans la fenetre : {:.1f} %".format(100 * dead[isD].mean()))

    z = np.load(os.path.join(ROOT, "data", "external", "cache", "J03WPY.npz"))
    zmed, zp75 = motion(z["players"])
    print("\n== Vitesse mediane des joueurs (fenetre 5 s), percentiles 10 / 25 / 50 / 75 / 90 (m/s) ==")
    print("{:<42} {}".format("notre fenetre publique, jeu vivant", pcts(med[isAlive])))
    print("{:<42} {}".format("notre fenetre publique, ballon mort", pcts(med[isD])))
    print("{:<42} {}".format("IDSSE J03WPY, jeu vivant", pcts(zmed[z["alive"]])))
    print("{:<42} {}".format("IDSSE J03WPY, ballon mort", pcts(zmed[~z["alive"]])))
    print("\n== Joueurs suivis par image ==")
    print("{:<42} moyenne {:.1f}".format("notre fenetre publique, jeu vivant", cnt[isAlive].mean()))
    print("{:<42} moyenne {:.1f}".format("notre fenetre publique, ballon mort", cnt[isD].mean()))


if __name__ == "__main__":
    main()