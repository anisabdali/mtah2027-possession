"""Intervalles anormaux a l'interieur des mi-temps notees : trous du ballon, immobilite collective.
Sans verite terrain. L'heure affichee est le temps video (mm:ss) pour que tu puisses controler."""
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks  # noqa: E402
from diagnostics import load_tracking  # noqa: E402

FPS = 25
H = 12
LOW_SPEED = 0.8     # m/s, descriptif


def runs(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def clock(frame):
    s = frame / FPS
    return "{:d}:{:02d}".format(int(s // 60), int(s % 60))


def main():
    warnings.simplefilter("ignore")
    ball, players, team = load_tracking()
    n = len(ball)
    width = GAMES[GAME]["pitch"][1]
    disp = np.full(players.shape[:2], np.nan, np.float32)
    disp[H:n - H] = np.linalg.norm(players[2 * H:] - players[:n - 2 * H], axis=2)
    pspeed = np.nanmedian(disp, axis=1) / (2 * H / FPS)
    smooth = pd.Series(pspeed).rolling(125, center=True, min_periods=25).median().to_numpy()
    valid = ~np.isnan(players[:, :, 0])
    npl = valid.sum(axis=1)
    y = players[:, :, 1]
    side = np.where(npl > 0, (((y < 3) | (y > width - 3)) & valid).sum(axis=1) / np.maximum(npl, 1), np.nan)
    visible = ~np.isnan(ball[:, 0])

    def describe(a, b):
        return "vitesse med. {:.2f} m/s | joueurs suivis {:.1f} | pres des touches {:.0f} % | ballon visible {:.0f} %".format(
            np.nanmedian(pspeed[a:b]), npl[a:b].mean(), 100 * np.nanmean(side[a:b]), 100 * visible[a:b].mean())

    for half, a0, b0 in half_video_blocks(GAME):
        print("\n===== mi-temps {} : images {}..{}, video {} -> {} ({:.1f} min) =====".format(
            half, a0, b0, clock(a0), clock(b0), (b0 - a0) / FPS / 60))

        s, e = runs(~visible[a0:b0 + 1])
        order = np.argsort(e - s)[::-1][:5]
        print("-- plus longs trous du ballon --")
        for i in order:
            a, b = a0 + s[i], a0 + e[i]
            print("  {} -> {}  duree {:5.1f} s | {}".format(clock(a), clock(b), (b - a) / FPS, describe(a, b)))

        s, e = runs(smooth[a0:b0 + 1] < LOW_SPEED)
        keep = [(x, y_) for x, y_ in zip(s, e) if (y_ - x) / FPS >= 20]
        keep.sort(key=lambda t: t[1] - t[0], reverse=True)
        print("-- plus longues periodes de vitesse mediane < {} m/s (au moins 20 s) --".format(LOW_SPEED))
        for x, y_ in keep[:5]:
            a, b = a0 + x, a0 + y_
            print("  {} -> {}  duree {:5.1f} s | {}".format(clock(a), clock(b), (b - a) / FPS, describe(a, b)))
        if not keep:
            print("  aucune")

        low = npl[a0:b0 + 1] <= 5
        s, e = runs(low)
        print("-- images avec 5 joueurs suivis ou moins : {} ({:.2f} %) --".format(
            int(low.sum()), 100 * low.mean()))
        for i in np.argsort(e - s)[::-1][:3]:
            if e[i] - s[i] >= FPS:
                print("  {} -> {}  duree {:.1f} s".format(clock(a0 + s[i]), clock(a0 + e[i]), (e[i] - s[i]) / FPS))


if __name__ == "__main__":
    main()