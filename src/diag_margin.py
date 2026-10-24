"""Accord avec le possesseur indique par les tags, selon la distance au ballon et la marge m.
Sans verite terrain."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import GAME, half_video_blocks  # noqa: E402
from starter.load_smart_tagging import load as load_tags  # noqa: E402
from diagnostics import load_tracking  # noqa: E402
from predict import TAG_TABLE, TEAM_CODE  # noqa: E402


def tag_possessor(events, n):
    best_q = np.zeros(n)
    best_p = np.full(n, -1, np.int8)
    for ev in events:
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
    return best_p


def main():
    ball, players, team = load_tracking()
    n = len(ball)
    graded = np.zeros(n, bool)
    for _, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True

    d = np.linalg.norm(players - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    dH = d[:, team == 0].min(axis=1)
    dA = d[:, team == 1].min(axis=1)
    p = tag_possessor(load_tags(), n)

    ok = graded & np.isfinite(dH) & np.isfinite(dA) & (p >= 0)
    dmin = np.minimum(dH, dA)[ok]
    m = (dA - dH)[ok]                       # > 0 : un joueur de Home est plus proche
    p = p[ok]
    agree = ((m > 0) & (p == 0)) | ((m < 0) & (p == 1))
    keep = m != 0

    dbins = [0, 1, 2, 4, 8, np.inf]
    mbins = [0, 0.5, 1, 2, 4, np.inf]
    drow = np.digitize(dmin, dbins[1:-1])
    mcol = np.digitize(np.abs(m), mbins[1:-1])

    print("Frames avec ballon, 2 equipes suivies et un tag fiable : {} ({:.0f} s)".format(
        keep.sum(), keep.sum() / 25))
    print("\nAccord avec le possesseur des tags : % (secondes d'observation)")
    print("lignes : distance du plus proche (m) ; colonnes : |marge| entre les deux equipes (m)\n")
    header = "{:>10} {:>10}".format("dist", "toutes")
    for lo, hi in zip(mbins[:-1], mbins[1:]):
        header += " {:>11}".format("{}-{}".format(lo, "inf" if hi == np.inf else hi))
    print(header)
    for r, (lo, hi) in enumerate(zip(dbins[:-1], dbins[1:])):
        row = keep & (drow == r)
        label = "{}-{}".format(lo, "inf" if hi == np.inf else hi)
        line = "{:>10} {:>10}".format(
            label, "{:.0f}% ({}s)".format(100 * agree[row].mean(), row.sum() // 25) if row.sum() > 100 else "-")
        for c in range(len(mbins) - 1):
            cell = row & (mcol == c)
            line += " {:>11}".format(
                "{:.0f}% ({}s)".format(100 * agree[cell].mean(), cell.sum() // 25) if cell.sum() > 100 else "-")
        print(line)


if __name__ == "__main__":
    main()