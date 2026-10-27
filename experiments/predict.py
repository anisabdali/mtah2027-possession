"""v1 : possession par modele a deux etats (H, A), chemin de Viterbi.
Preuves : joueur le plus proche (pondere par la distance) + a priori Smart Tagging.
Aucune verite terrain n'est utilisee. Pas de D dans cette version (X hors mi-temps notees)."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import GAME, half_video_blocks  # noqa: E402
from starter.baseline import write_submission  # noqa: E402
from starter.load_smart_tagging import load as load_tags  # noqa: E402
from diagnostics import load_tracking  # noqa: E402  (meme dossier src/)

FPS = 25
MEAN_POSSESSION_S = 14.0      # hypothese : duree moyenne d'une possession
EVIDENCE_PER_SECOND = 1.0     # hypothese : ~1 observation independante par seconde
SWITCH_COST = float(np.log(MEAN_POSSESSION_S * FPS))   # cout d'un changement de possession
NEAR_SCALE_M = 2.0            # hypothese : decroissance de la fiabilite avec la distance

# code Smart Tagging -> (+1 : le tag Team est le possesseur ; -1 : c'est l'adversaire, fiabilite)
# Fiabilites tirees de l'accord tag / joueur le plus proche (diagnostics_tagging.py), pas de GT.
TAG_TABLE = {
    "build up": (+1, 0.76), "progression": (+1, 0.70), "attacking transition": (+1, 0.62),
    "ball in final third": (+1, 0.62), "long ball": (+1, 0.62),
    "defending in attacking third": (-1, 0.76), "defending in middle third": (-1, 0.70),
    "defensive transition": (-1, 0.62),
}
TEAM_CODE = {"barcelona": 0, "atletic club": 1, "athletic club": 1}   # 0 = H, 1 = A


def emissions(ball, players, team, events):
    n = len(ball)
    d = np.linalg.norm(players - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    dmin = d.min(axis=1)
    ok = np.isfinite(dmin)
    idx = np.where(ok)[0]
    near = team[d[idx].argmin(axis=1)].astype(int)
    r = 0.5 + 0.45 * np.exp(-dmin[idx] / NEAR_SCALE_M)     # fiabilite du plus proche
    e = np.zeros((n, 2))
    e[idx, near] += np.log(r)
    e[idx, 1 - near] += np.log(1 - r)

    best_q = np.zeros(n)
    best_p = np.full(n, -1, np.int8)
    for ev in events:                                      # une seule phase par image : la plus fiable
        rule = TAG_TABLE.get((ev["code"] or "").strip().lower())
        tag_team = TEAM_CODE.get((ev["tags"].get("Team") or "").strip().lower())
        if rule is None or tag_team is None:
            continue
        sign, q = rule
        poss = tag_team if sign > 0 else 1 - tag_team
        a, b = max(0, ev["start_frame"]), min(n, ev["end_frame"])
        better = best_q[a:b] < q
        best_q[a:b][better] = q
        best_p[a:b][better] = poss
    idx = np.where(best_p >= 0)[0]
    p = best_p[idx].astype(int)
    e[idx, p] += np.log(best_q[idx])
    e[idx, 1 - p] += np.log(1 - best_q[idx])
    return e * (EVIDENCE_PER_SECOND / FPS)


def viterbi(e, cost):
    """Chemin le plus probable a 2 etats, cout fixe par changement."""
    n = len(e)
    e0, e1 = e[:, 0].tolist(), e[:, 1].tolist()
    back0, back1 = bytearray(n), bytearray(n)
    s0, s1 = e0[0], e1[0]
    for t in range(1, n):
        a, b = s0, s1 - cost
        if b > a:
            back0[t] = 1
            a = b
        c, d = s1, s0 - cost
        if d > c:
            back1[t] = 1
            c = d
        s0, s1 = a + e0[t], c + e1[t]
    state = 0 if s0 >= s1 else 1
    path = np.empty(n, np.int8)
    for t in range(n - 1, -1, -1):
        path[t] = state
        if (back0 if state == 0 else back1)[t]:
            state = 1 - state
    return path


def main():
    ball, players, team = load_tracking()
    e = emissions(ball, players, team, load_tags())
    labels = np.full(len(ball), "X", dtype="<U1")
    for half, a, b in half_video_blocks(GAME):
        path = viterbi(e[a:b + 1], SWITCH_COST)
        labels[a:b + 1] = np.where(path == 0, "H", "A")
        k = int(np.count_nonzero(np.diff(path))) + 1
        print("mi-temps {} : {} possessions, duree moyenne {:.1f} s".format(
            half, k, (b - a + 1) / FPS / k))
    out = os.path.join(ROOT, "evaluation", "results", "v1_hmm.csv")
    write_submission(out, labels)
    print("ecrit :", out)


if __name__ == "__main__":
    main()