"""Balayage du poids des preuves : durees de possession predites, sans verite terrain."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import GAME, half_video_blocks  # noqa: E402
from starter.load_smart_tagging import load as load_tags  # noqa: E402
from diagnostics import load_tracking  # noqa: E402
from predict import emissions, viterbi, SWITCH_COST, FPS  # noqa: E402


def main():
    ball, players, team = load_tracking()
    e1 = emissions(ball, players, team, load_tags())   # poids de base : 1 observation par seconde
    blocks = half_video_blocks(GAME)
    print("{:>6} {:>8} {:>10} {:>10} {:>10} {:>8}".format(
        "k/s", "poss.", "moy (s)", "mediane", "p10 (s)", "% H"))
    for k in (1, 2, 3, 4, 6, 8, 12, 16, 25):
        durs, home, total = [], 0, 0
        for _, a, b in blocks:
            path = viterbi(e1[a:b + 1] * k, SWITCH_COST)
            cuts = np.concatenate(([0], np.flatnonzero(np.diff(path)) + 1, [len(path)]))
            durs.extend(np.diff(cuts) / FPS)
            home += int((path == 0).sum())
            total += len(path)
        d = np.array(durs)
        print("{:>6} {:>8} {:>10.1f} {:>10.1f} {:>10.1f} {:>7.1f}%".format(
            k, len(d), d.mean(), np.median(d), np.percentile(d, 10), 100 * home / total))


if __name__ == "__main__":
    main()