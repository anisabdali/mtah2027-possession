"""v4b : deux etages. 1) ballon mort (detecteur de mouvement collectif, probabilite de dead_transfer.py);
2) possession H/A decodee separement dans chaque phase vivante. Aucune verite terrain du hackathon.
  python src/predict_v4b.py sweep
  python src/predict_v4b.py write <k> <ks>"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import GAME, half_video_blocks  # noqa: E402
from starter.baseline import write_submission  # noqa: E402
from starter.load_smart_tagging import load as load_tags  # noqa: E402
from diagnostics import load_tracking  # noqa: E402
from predict import viterbi, SWITCH_COST, FPS  # noqa: E402
from predict_v2 import track_evidence, tag_prior, TAG_PER_SECOND  # noqa: E402
from predict_v3 import shape_evidence  # noqa: E402
from predict_v4 import summarize  # noqa: E402

DEAD_THRESHOLD = 0.5     # probabilite de ballon mort (detecteur valide sur la DFL)
MIN_STOP_S = 3.0         # hypothese : un arret de moins de 3 s est ignore
MIN_PLAY_S = 3.0         # hypothese : une phase vivante de moins de 3 s entre deux arrets est absorbee


def remove_short(mask, min_len, value):
    m = mask.copy()
    d = np.diff(np.concatenate(([0], (m == value).astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if b - a < min_len and a > 0 and b < len(m):      # on ne touche pas aux bords du bloc
            m[a:b] = not value
    return m


def dead_mask(ps, blocks, n):
    raw = np.nan_to_num(ps, nan=0.0) >= DEAD_THRESHOLD
    out = np.zeros(n, bool)
    for _, a, b in blocks:
        m = remove_short(raw[a:b + 1], int(MIN_STOP_S * FPS), True)
        m = remove_short(m, int(MIN_PLAY_S * FPS), False)
        out[a:b + 1] = m
    return out


def decode(e_track, e_shape, e_tag, dead, k, ks, blocks):
    e = (e_track * k + e_shape * ks + e_tag * TAG_PER_SECOND) / FPS
    out = []
    for half, a, b in blocks:
        d = dead[a:b + 1]
        labels = np.full(b - a + 1, 2, np.int8)
        edges = np.diff(np.concatenate(([0], (~d).astype(np.int8), [0])))
        for x, y in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            labels[x:y] = viterbi(e[a + x:a + y], SWITCH_COST)
        out.append((half, a, b, labels))
    return out


def main():
    ball, players, team = load_tracking()
    n = len(ball)
    blocks = half_video_blocks(GAME)
    ps = np.load(os.path.join(ROOT, "data", "cache", "dead_prob.npy"))
    dead = dead_mask(ps, blocks, n)
    e_track = track_evidence(ball, players, team)
    e_shape = shape_evidence(ball, players, team, blocks)
    e_tag = tag_prior(load_tags(), n)

    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if mode == "sweep":
        print("{:>4} {:>4} | {:>6} {:>7} | {:>9} {:>8} {:>8} {:>8}".format(
            "k", "ks", "% D", "arrets", "possess.", "moy (s)", "med (s)", "p10 (s)"))
        for k, ks in [(12, 12), (12, 0), (8, 12), (25, 12), (25, 24)]:
            alive, stops, share = summarize(decode(e_track, e_shape, e_tag, dead, k, ks, blocks))
            print("{:>4} {:>4} | {:>5.1f}% {:>7} | {:>9} {:>8.1f} {:>8.1f} {:>8.1f}".format(
                k, ks, 100 * share, len(stops), len(alive), alive.mean(),
                np.median(alive), np.percentile(alive, 10)))
    else:
        k, ks = float(sys.argv[2]), float(sys.argv[3])
        labels = np.full(n, "X", dtype="<U1")
        for _, a, b, path in decode(e_track, e_shape, e_tag, dead, k, ks, blocks):
            labels[a:b + 1] = np.array(["H", "A", "D"])[path]
        out = os.path.join(ROOT, "evaluation", "results", "v4b_k{:g}_ks{:g}.csv".format(k, ks))
        write_submission(out, labels)
        print("ecrit :", out)


if __name__ == "__main__":
    main()