"""A quoi ressemble un arret de jeu dans le tracking ? Profil autour des coups de pied arretes.
Sans verite terrain."""
import os
import sys
import warnings

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks  # noqa: E402
from starter.load_smart_tagging import load  # noqa: E402
from diagnostics import load_tracking  # noqa: E402

FPS = 25
H = 12                      # vitesse sur +-12 images (~1 s)
OFFSETS = [-20, -10, -5, -2, 0, 2, 5, 10, 20]   # secondes


def main():
    warnings.simplefilter("ignore")
    ball, players, team = load_tracking()
    n = len(ball)
    length, width = GAMES[GAME]["pitch"]

    disp = np.full(players.shape[:2], np.nan, np.float32)
    disp[H:n - H] = np.linalg.norm(players[2 * H:] - players[:n - 2 * H], axis=2)
    pspeed = np.nanmedian(disp, axis=1) / (2 * H / FPS)        # vitesse mediane des joueurs (m/s)

    visible = ~np.isnan(ball[:, 0])
    outside = visible & ((ball[:, 0] < 0) | (ball[:, 0] > length) | (ball[:, 1] < 0) | (ball[:, 1] > width))

    graded = np.zeros(n, bool)
    for _, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True
    print("vitesse mediane des joueurs, toutes images notees : {:.2f} m/s (mediane), {:.2f} (moyenne)".format(
        np.nanmedian(pspeed[graded]), np.nanmean(pspeed[graded])))
    print("ballon hors du terrain, images notees : {:.1f} % des images avec ballon".format(
        100 * outside[graded].sum() / visible[graded].sum()))

    events = [e for e in load() if (e["code"] or "").strip().lower() == "set pieces"]
    by_type = {}
    for e in events:
        by_type.setdefault((e["tags"].get("Type") or "?").strip(), []).append(e)

    def profile(frames):
        out = []
        for off in OFFSETS:
            idx = np.clip(frames + off * FPS, 0, n - 1)
            out.append((np.nanmean(pspeed[idx]), 100 * visible[idx].mean(), 100 * outside[idx].mean()))
        return out

    for typ, evs in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
        if len(evs) < 5 or typ.lower().startswith("kick off"):
            continue
        starts = np.array([e["start_frame"] for e in evs])
        ends = np.array([e["end_frame"] for e in evs])
        dur = np.mean(ends - starts) / FPS
        ps, pe = profile(starts), profile(ends)
        print("\n== {} : {} instances, fenetre moyenne {:.1f} s ==".format(typ, len(evs), dur))
        print("{:>6} | {:>20} | {:>20}".format("s", "alignee sur le DEBUT", "alignee sur la FIN"))
        print("{:>6} | {:>6} {:>6} {:>6} | {:>6} {:>6} {:>6}".format(
            "", "v m/s", "vis %", "hors %", "v m/s", "vis %", "hors %"))
        for off, s, e in zip(OFFSETS, ps, pe):
            print("{:>6} | {:>6.2f} {:>6.0f} {:>6.0f} | {:>6.2f} {:>6.0f} {:>6.0f}".format(off, *s, *e))


if __name__ == "__main__":
    main()