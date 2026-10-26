"""v4 : trois etats (H, A, D). Ballon mort detecte par le mouvement collectif (appris sur les matchs DFL,
probabilite sauvegardee par dead_transfer.py), possession par tracking + forme + tags.
Aucune verite terrain du hackathon.
  python src/predict_v4.py sweep
  python src/predict_v4.py write <k> <ks> <kd>"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import GAME, half_video_blocks  # noqa: E402
from starter.baseline import write_submission  # noqa: E402
from starter.load_smart_tagging import load as load_tags  # noqa: E402
from diagnostics import load_tracking  # noqa: E402
from predict import SWITCH_COST, FPS  # noqa: E402
from predict_v2 import track_evidence, tag_prior, TAG_PER_SECOND  # noqa: E402
from predict_v3 import shape_evidence  # noqa: E402

DEAD_COST = float(np.log(25.0 * FPS))     # segments vivant / mort d'environ 25 s (matchs DFL)
PROB_CLIP = 0.02


def emissions(e_track, e_shape, e_tag, ps, k, ks, kd):
    team = (e_track * k + e_shape * ks + e_tag * TAG_PER_SECOND) / FPS
    p = np.clip(np.nan_to_num(ps, nan=0.5), PROB_CLIP, 1 - PROB_CLIP)
    z = np.log(p / (1 - p)) / 2 * kd / FPS          # > 0 : arret probable
    return (team[:, 0] - z).tolist(), (team[:, 1] - z).tolist(), z.tolist()


def viterbi3(eH, eA, eD, c_ha, c_d):
    """Etats 0 = H, 1 = A, 2 = D. Cout c_ha entre H et A, c_d entre vivant et mort."""
    n = len(eH)
    back = [bytearray(n), bytearray(n), bytearray(n)]
    sH, sA, sD = eH[0], eA[0], eD[0]
    for t in range(1, n):
        best, arg = sH, 0
        if sA - c_ha > best:
            best, arg = sA - c_ha, 1
        if sD - c_d > best:
            best, arg = sD - c_d, 2
        nH = best + eH[t]
        back[0][t] = arg
        best, arg = sA, 1
        if sH - c_ha > best:
            best, arg = sH - c_ha, 0
        if sD - c_d > best:
            best, arg = sD - c_d, 2
        nA = best + eA[t]
        back[1][t] = arg
        best, arg = sD, 2
        if sH - c_d > best:
            best, arg = sH - c_d, 0
        if sA - c_d > best:
            best, arg = sA - c_d, 1
        nD = best + eD[t]
        back[2][t] = arg
        sH, sA, sD = nH, nA, nD
    state = max(range(3), key=lambda s: (sH, sA, sD)[s])
    path = np.empty(n, np.int8)
    for t in range(n - 1, -1, -1):
        path[t] = state
        state = back[state][t]
    return path


def decode(args, blocks):
    eH, eA, eD = emissions(*args)
    return [(h, a, b, viterbi3(eH[a:b + 1], eA[a:b + 1], eD[a:b + 1], SWITCH_COST, DEAD_COST))
            for h, a, b in blocks]


def summarize(paths):
    alive, dead = [], []
    nd = tot = 0
    for _, a, b, path in paths:
        cuts = np.concatenate(([0], np.flatnonzero(np.diff(path)) + 1, [len(path)]))
        for x, y in zip(cuts[:-1], cuts[1:]):
            (dead if path[x] == 2 else alive).append((y - x) / FPS)
        nd += int((path == 2).sum())
        tot += len(path)
    return np.array(alive), np.array(dead), nd / tot


def main():
    ball, players, team = load_tracking()
    n = len(ball)
    blocks = half_video_blocks(GAME)
    ps = np.load(os.path.join(ROOT, "data", "cache", "dead_prob.npy"))
    e_track = track_evidence(ball, players, team)
    e_shape = shape_evidence(ball, players, team, blocks)
    e_tag = tag_prior(load_tags(), n)

    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if mode == "sweep":
        print("{:>4} {:>4} {:>4} | {:>6} {:>8} | {:>9} {:>8} {:>8} {:>8}".format(
            "k", "ks", "kd", "% D", "arrets", "possess.", "moy (s)", "med (s)", "p10 (s)"))
        for k, ks, kd in [(12, 12, 0.5), (12, 12, 1), (12, 12, 2), (12, 12, 4),
                          (25, 12, 1), (25, 24, 1), (40, 24, 1)]:
            alive, dead, share = summarize(decode((e_track, e_shape, e_tag, ps, k, ks, kd), blocks))
            print("{:>4} {:>4} {:>4} | {:>5.1f}% {:>8} | {:>9} {:>8.1f} {:>8.1f} {:>8.1f}".format(
                k, ks, kd, 100 * share, len(dead), len(alive), alive.mean(),
                np.median(alive), np.percentile(alive, 10)))
    else:
        k, ks, kd = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
        labels = np.full(n, "X", dtype="<U1")
        for _, a, b, path in decode((e_track, e_shape, e_tag, ps, k, ks, kd), blocks):
            labels[a:b + 1] = np.array(["H", "A", "D"])[path]
        out = os.path.join(ROOT, "evaluation", "results", "v4_k{:g}_ks{:g}_kd{:g}.csv".format(k, ks, kd))
        write_submission(out, labels)
        print("ecrit :", out)


if __name__ == "__main__":
    main()