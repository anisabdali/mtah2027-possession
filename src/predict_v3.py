"""v3 : tracking (joueur le plus proche + marge) + forme collective + tags.
Aucune verite terrain. Modele de forme entraine sur pseudo-etiquettes automatiques, par mi-temps.
  python src/predict_v3.py sweep
  python src/predict_v3.py write <k> <ks>"""
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
from predict_v2 import track_evidence, tag_prior, stats, TAG_PER_SECOND  # noqa: E402
from diag_shape import shape_features, fit_logreg  # noqa: E402

LOGIT_CLIP = 3.0


def shape_evidence(ball, players, team, blocks):
    n = len(ball)
    F = shape_features(players, team)
    feat_ok = ~np.isnan(F).any(axis=1)
    with np.errstate(invalid="ignore"):
        d = np.linalg.norm(players - ball[:, None, :], axis=2)
        d = np.where(np.isnan(d), np.inf, d)
        dH, dA = d[:, team == 0].min(axis=1), d[:, team == 1].min(axis=1)
        sure = np.isfinite(dH) & np.isfinite(dA) & (np.minimum(dH, dA) < 1.0) & (np.abs(dA - dH) > 2.0)
    y = (dH < dA).astype(float)
    e = np.zeros((n, 2))
    for _, a, b in blocks:
        idx = np.arange(a, b + 1)
        pool = idx[sure[idx] & feat_ok[idx]][::5]
        mu, sd = F[pool].mean(axis=0), F[pool].std(axis=0) + 1e-9
        w = fit_logreg((F[pool] - mu) / sd, y[pool])
        ok = idx[feat_ok[idx]]
        z = np.hstack([(F[ok] - mu) / sd, np.ones((len(ok), 1))]) @ w
        z = np.clip(z, -LOGIT_CLIP, LOGIT_CLIP)
        e[ok, 0] += z / 2
        e[ok, 1] -= z / 2
    return e


def decode(e_track, e_shape, e_tag, k, ks, blocks):
    e = (e_track * k + e_shape * ks + e_tag * TAG_PER_SECOND) / FPS
    return [(h, a, b, viterbi(e[a:b + 1], SWITCH_COST)) for h, a, b in blocks]


def main():
    ball, players, team = load_tracking()
    n = len(ball)
    blocks = half_video_blocks(GAME)
    e_track = track_evidence(ball, players, team)
    e_shape = shape_evidence(ball, players, team, blocks)
    e_tag = tag_prior(load_tags(), n)

    mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
    if mode == "sweep":
        print("{:>4} {:>5} {:>7} {:>9} {:>9} {:>8} {:>7}".format(
            "k", "ks", "poss.", "moy (s)", "mediane", "p10 (s)", "% H"))
        for k, ks in [(12, 12), (25, 12), (25, 16), (25, 24), (40, 24), (40, 32)]:
            paths = decode(e_track, e_shape, e_tag, k, ks, blocks)
            d = stats(paths)
            home = sum(int((p == 0).sum()) for *_, p in paths)
            tot = sum(len(p) for *_, p in paths)
            print("{:>4} {:>5} {:>7} {:>9.1f} {:>9.1f} {:>8.1f} {:>6.1f}%".format(
                k, ks, len(d), d.mean(), np.median(d), np.percentile(d, 10), 100 * home / tot))
    else:
        k, ks = float(sys.argv[2]), float(sys.argv[3])
        labels = np.full(n, "X", dtype="<U1")
        for _, a, b, path in decode(e_track, e_shape, e_tag, k, ks, blocks):
            labels[a:b + 1] = np.where(path == 0, "H", "A")
        out = os.path.join(ROOT, "evaluation", "results", "v3_k{:g}_ks{:g}.csv".format(k, ks))
        write_submission(out, labels)
        print("ecrit :", out)


if __name__ == "__main__":
    main()