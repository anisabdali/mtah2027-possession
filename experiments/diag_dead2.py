"""Pour chaque coup de pied arrete : la plus longue periode de vitesse basse autour du debut du tag.
Sans verite terrain."""
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import GAME, GAMES  # noqa: E402
from starter.load_smart_tagging import load  # noqa: E402
from diagnostics import load_tracking  # noqa: E402

FPS = 25
H = 12
V_LOW = 1.33        # m/s : mediane globale des images notees (reference, pas optimisee)
SEARCH_S = 12       # on cherche dans +-12 s autour du debut du tag


def longest_run(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    if len(s) == 0:
        return None
    i = np.argmax(e - s)
    return s[i], e[i]


def q(v):
    return "{:5.1f} / {:5.1f} / {:5.1f}".format(*np.percentile(v, [25, 50, 75]))


def main():
    warnings.simplefilter("ignore")
    ball, players, team = load_tracking()
    n = len(ball)
    length, width = GAMES[GAME]["pitch"]
    disp = np.full(players.shape[:2], np.nan, np.float32)
    disp[H:n - H] = np.linalg.norm(players[2 * H:] - players[:n - 2 * H], axis=2)
    pspeed = np.nanmedian(disp, axis=1) / (2 * H / FPS)
    smooth = pd.Series(pspeed).rolling(FPS, center=True, min_periods=5).median().to_numpy()
    visible = ~np.isnan(ball[:, 0])
    outside = visible & ((ball[:, 0] < 0) | (ball[:, 0] > length) | (ball[:, 1] < 0) | (ball[:, 1] > width))

    by_type = {}
    for e in load():
        if (e["code"] or "").strip().lower() != "set pieces":
            continue
        typ = (e["tags"].get("Type") or "?").strip()
        if typ.lower().startswith("kick off"):
            continue
        by_type.setdefault(typ, []).append(e["start_frame"])

    print("seuil de vitesse basse : {} m/s ; recherche dans +-{} s autour du debut du tag".format(V_LOW, SEARCH_S))
    print("(les trois nombres sont les percentiles 25 / 50 / 75)\n")
    for typ, starts in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        runs = []
        for s0 in starts:
            a, b = s0 - SEARCH_S * FPS, s0 + SEARCH_S * FPS
            if a < 0 or b > n:
                continue
            r = longest_run(smooth[a:b] < V_LOW)
            if r is None:
                continue
            ra, rb = a + r[0], a + r[1]
            runs.append(((ra - s0) / FPS, (rb - s0) / FPS, (rb - ra) / FPS,
                         100 * visible[ra:rb].mean(), 100 * outside[ra:rb].mean()))
        print("== {} : {} instances, periode basse trouvee dans {} ==".format(typ, len(starts), len(runs)))
        if not runs:
            continue
        r = np.array(runs)
        print("  debut de la periode (s / debut du tag) : " + q(r[:, 0]))
        print("  fin de la periode   (s / debut du tag) : " + q(r[:, 1]))
        print("  duree de la periode (s)                : " + q(r[:, 2]))
        print("  periodes d'au moins 5 s : {:.0f} %  | ballon visible {:.0f} %  hors du terrain {:.0f} %".format(
            100 * (r[:, 2] >= 5).mean(), r[:, 3].mean(), r[:, 4].mean()))


if __name__ == "__main__":
    main()