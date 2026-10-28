"""Statistiques de bruit sans verite terrain : notre tracking contre IDSSE (propre et degrade).
  python experiments/noise_stats.py"""
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from games import GAME, half_video_blocks  # noqa: E402
from diagnostics import load_tracking  # noqa: E402
from inspect_open import EXT  # noqa: E402
from bench_dfl import markov_mask, drop_players  # noqa: E402

FPS = 25


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def stats(players, ball, use):
    n = len(players)
    ok3 = np.zeros(n, bool)
    ok3[1:-1] = use[:-2] & use[1:-1] & use[2:]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cnt = (~np.isnan(players[use][:, :, 0])).sum(axis=1)
        v = np.linalg.norm(players[1:] - players[:-1], axis=2) * FPS            # m/s entre deux images
        v = v[use[1:] & use[:-1]]
        v = v[~np.isnan(v)]
        acc = np.linalg.norm(players[2:] - 2 * players[1:-1] + players[:-2], axis=2) * FPS * FPS
        acc = acc[ok3[1:-1]]
        acc = acc[~np.isnan(acc)]
        bv = np.linalg.norm(ball[1:] - ball[:-1], axis=1) * FPS
        bv = bv[use[1:] & use[:-1]]
        bv = bv[~np.isnan(bv)]
        vis = (~np.isnan(ball[use, 0])).mean()
    return {
        "joueurs suivis par image, moyenne": float(cnt.mean()),
        "joueurs suivis par image, 5e percentile": pct(cnt, 5),
        "images avec moins de 18 joueurs (%)": 100 * float((cnt < 18).mean()),
        "vitesse des joueurs, mediane (m/s)": pct(v, 50),
        "vitesse des joueurs, 99e percentile (m/s)": pct(v, 99),
        "deplacements > 12 m/s en une image (%)": 100 * float((v > 12).mean()),
        "deplacements > 25 m/s en une image (%)": 100 * float((v > 25).mean()),
        "acceleration, mediane (m/s2)": pct(acc, 50),
        "acceleration, 90e percentile (m/s2)": pct(acc, 90),
        "acceleration, 99e percentile (m/s2)": pct(acc, 99),
        "ballon visible (%)": 100 * float(vis),
        "vitesse du ballon, mediane (m/s)": pct(bv, 50),
        "vitesse du ballon, 99e percentile (m/s)": pct(bv, 99),
        "ballon > 40 m/s en une image (%)": 100 * float((bv > 40).mean()),
    }


def main():
    cols = {}
    ball, players, team = load_tracking()
    use = np.zeros(len(ball), bool)
    for _, a, b in half_video_blocks(GAME):
        use[a:b + 1] = True
    cols["notre match"] = stats(players, ball, use)
    for m in ("J03WPY", "J03WOY"):
        z = np.load(os.path.join(EXT, "cache", m + ".npz"))
        use = np.ones(len(z["alive"]), bool)
        cols[m + " propre"] = stats(z["players"], z["ball"], use)
        rng = np.random.default_rng(0)                      # meme degradation que les bancs d'essai
        b = z["ball"].copy()
        b[markov_mask(z["alive"], rng)] = np.nan
        cols[m + " degrade"] = stats(drop_players(z["players"], rng), b, use)
    names = list(cols)
    print("{:<44}".format("") + "".join("{:>17}".format(k) for k in names))
    for key in cols[names[0]]:
        print("{:<44}".format(key) + "".join("{:>17.2f}".format(cols[k][key]) for k in names))


if __name__ == "__main__":
    main()