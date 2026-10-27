"""v2 : preuves ponderees par la distance ET la marge entre les deux equipes.
Poids lus dans le tableau d'accord avec les tags (diag_margin.py). Sans verite terrain."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import GAME, half_video_blocks  # noqa: E402
from starter.baseline import write_submission  # noqa: E402
from starter.load_smart_tagging import load as load_tags  # noqa: E402
from diagnostics import load_tracking  # noqa: E402
from predict import TAG_TABLE, TEAM_CODE, viterbi, SWITCH_COST, FPS  # noqa: E402

TAG_PER_SECOND = 1.0          # poids fixe des tags : ~1 observation par seconde
DIST_EDGES = [1, 2, 4, 8]     # lignes : 0-1, 1-2, 2-4, 4-8, 8+ (m)
MARGIN_EDGES = [0.5, 1, 2, 4] # colonnes : 0-0.5, 0.5-1, 1-2, 2-4, 4+ (m)
AGREEMENT = np.array([
    [0.64, 0.70, 0.72, 0.80, 0.83],
    [0.52, 0.55, 0.62, 0.69, 0.80],
    [0.49, 0.51, 0.55, 0.61, 0.68],
    [0.48, 0.49, 0.49, 0.57, 0.60],
    [0.50, 0.50, 0.50, 0.50, 0.50],   # >8 m : echantillon trop petit, aucune preuve
])
WEIGHT = np.where(AGREEMENT > 0.5, np.log(AGREEMENT / (1 - AGREEMENT)), 0.0)


def track_evidence(ball, players, team):
    n = len(ball)
    d = np.linalg.norm(players - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    dH = d[:, team == 0].min(axis=1)
    dA = d[:, team == 1].min(axis=1)
    idx = np.where(np.isfinite(dH) & np.isfinite(dA))[0]
    dmin = np.minimum(dH[idx], dA[idx])
    margin = np.abs(dA[idx] - dH[idx])
    w = WEIGHT[np.digitize(dmin, DIST_EDGES), np.digitize(margin, MARGIN_EDGES)]
    near = (dA[idx] < dH[idx]).astype(int)       # 0 : Home plus proche, 1 : Away
    e = np.zeros((n, 2))
    e[idx, near] += w / 2
    e[idx, 1 - near] -= w / 2
    return e


def tag_prior(events, n):
    best_q = np.zeros(n)
    best_p = np.full(n, -1, np.int8)
    for ev in events:                             # une seule phase par image : la plus fiable
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
    e = np.zeros((n, 2))
    e[idx, p] += np.log(best_q[idx])
    e[idx, 1 - p] += np.log(1 - best_q[idx])
    return e


def decode(e_track, e_tag, k, blocks):
    e = (e_track * k + e_tag * TAG_PER_SECOND) / FPS
    return [(half, a, b, viterbi(e[a:b + 1], SWITCH_COST)) for half, a, b in blocks]


def stats(paths):
    durs = []
    for _, a, b, path in paths:
        cuts = np.concatenate(([0], np.flatnonzero(np.diff(path)) + 1, [len(path)]))
        durs.extend(np.diff(cuts) / FPS)
    return np.array(durs)


def main():
    ball, players, team = load_tracking()
    n = len(ball)
    blocks = half_video_blocks(GAME)
    e_track = track_evidence(ball, players, team)
    e_tag = tag_prior(load_tags(), n)
    graded = np.zeros(n, bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    print("images notees portant une preuve de tracking : {:.1f} %".format(
        100 * (np.abs(e_track[graded]).sum(axis=1) > 0).mean()))

    arg = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if arg == "sweep":
        print("\n{:>5} {:>7} {:>9} {:>9} {:>8} {:>7}".format(
            "k", "poss.", "moy (s)", "mediane", "p10 (s)", "% H"))
        for k in (2, 4, 8, 12, 16, 25, 40):
            paths = decode(e_track, e_tag, k, blocks)
            d = stats(paths)
            home = sum(int((p == 0).sum()) for *_, p in paths)
            tot = sum(len(p) for *_, p in paths)
            print("{:>5} {:>7} {:>9.1f} {:>9.1f} {:>8.1f} {:>6.1f}%".format(
                k, len(d), d.mean(), np.median(d), np.percentile(d, 10), 100 * home / tot))
    else:
        k = float(arg)
        labels = np.full(n, "X", dtype="<U1")
        for _, a, b, path in decode(e_track, e_tag, k, blocks):
            labels[a:b + 1] = np.where(path == 0, "H", "A")
        out = os.path.join(ROOT, "evaluation", "results", "v2_k{:g}.csv".format(k))
        write_submission(out, labels)
        print("ecrit :", out)


if __name__ == "__main__":
    main()