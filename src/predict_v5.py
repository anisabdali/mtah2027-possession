"""v5 : v4b avec le detecteur de ballon mort v2 (fenetres gauche/droite + signaux du ballon).
Detecteur entraine sur les DEUX matchs DFL degrades, applique au match du hackathon. Sans verite terrain.
  python src/predict_v5.py diag
  python src/predict_v5.py write <k> <ks>"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from games import GAME, half_video_blocks  # noqa: E402
from starter.baseline import write_submission  # noqa: E402
from starter.load_smart_tagging import load as load_tags  # noqa: E402
from diagnostics import load_tracking  # noqa: E402
from inspect_open import EXT  # noqa: E402
from eval_learned import fit_logistic, predict_probability  # noqa: E402
from predict_v2 import track_evidence, tag_prior  # noqa: E402
from predict_v3 import shape_evidence  # noqa: E402
from predict_v4 import summarize  # noqa: E402
import predict_v4b as v4b  # noqa: E402
from bench_dfl import markov_mask, drop_players  # noqa: E402
from bench_dead2 import features2  # noqa: E402

FPS = 25
KICKOFF = {1: 11185, 2: 113742}      # coups d'envoi reels (controle de load_smart_tagging.py)
CACHE = os.path.join(ROOT, "data", "cache", "dead_prob2.npy")


def train_detector():
    Xs, ys = [], []
    for i, m in enumerate(("J03WPY", "J03WOY")):
        z = np.load(os.path.join(EXT, "cache", m + ".npz"))
        rng = np.random.default_rng(100 + i)
        ball = z["ball"].copy()
        ball[markov_mask(z["alive"], rng)] = np.nan
        players = drop_players(z["players"], rng)
        X = features2(ball, players)
        ok = np.flatnonzero(~np.isnan(X).any(axis=1))[::10]
        Xs.append(X[ok])
        ys.append((~z["alive"][ok]).astype(int))
    X, y = np.vstack(Xs), np.concatenate(ys)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    print("detecteur v2 : {} images d'entrainement, {:.0f} % de ballon mort".format(len(y), 100 * y.mean()))
    return fit_logistic((X - mu) / sd, y), mu, sd


def dead_probability(ball, players):
    if os.path.exists(CACHE):
        return np.load(CACHE)
    w, mu, sd = train_detector()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = features2(ball, players)
    ok = ~np.isnan(X).any(axis=1)
    p = np.full(len(ball), np.nan)
    p[ok] = predict_probability(w, (X[ok] - mu) / sd)
    ps = pd.Series(p).rolling(25, center=True, min_periods=5).mean().to_numpy()
    np.save(CACHE, ps)
    return ps


def clock(frame):
    s = frame / FPS
    return "{:d}:{:02d}".format(int(s // 60), int(s % 60))


def stops(mask, a0, b0):
    d = np.diff(np.concatenate(([0], mask[a0:b0 + 1].astype(np.int8), [0])))
    return [(a0 + x, a0 + y) for x, y in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def main():
    ball, players, team = load_tracking()
    n = len(ball)
    blocks = half_video_blocks(GAME)
    ps = dead_probability(ball, players)
    dead2 = v4b.dead_mask(ps, blocks, n)
    mode = sys.argv[1] if len(sys.argv) > 1 else "diag"

    if mode == "diag":
        dead1 = v4b.dead_mask(np.load(os.path.join(ROOT, "data", "cache", "dead_prob.npy")), blocks, n)
        g = np.zeros(n, bool)
        for _, a, b in blocks:
            g[a:b + 1] = True
        a1, a2 = dead1[g], dead2[g]
        po = (a1 == a2).mean()
        pe = a1.mean() * a2.mean() + (1 - a1.mean()) * (1 - a2.mean())
        vis = ~np.isnan(ball[:, 0])
        print("\nballon mort predit sur les images notees : v1 {:.1f} %   v2 {:.1f} %".format(100 * a1.mean(), 100 * a2.mean()))
        print("accord v1 / v2 : {:.1f} %  (kappa {:.2f})".format(100 * po, (po - pe) / (1 - pe)))
        print("ballon visible : {:.0f} % si v2 predit mort, {:.0f} % si v2 predit vivant".format(
            100 * vis[g & dead2].mean(), 100 * vis[g & ~dead2].mean()))
        allstops = []
        for h, a, b in blocks:
            allstops += [(h, x, y) for x, y in stops(dead2, a, b)]
        L = np.array([(y - x) / FPS for _, x, y in allstops])
        print("arrets v2 : {}  duree (s) mediane {:.1f}  p90 {:.1f}  max {:.0f}".format(
            len(L), np.median(L), np.percentile(L, 90), L.max()))
        print("\narrets v2 les plus longs, par ordre chronologique (temps ecoule depuis le coup d'envoi, mi-temps) :")
        print("{:>8} {:>13} {:>16} {:>6} {:>10} {:>14}".format(
            "mi-t.", "video", "ecoule (min:s)", "duree", "ballon vis.", "v1 couvre"))
        top = sorted(sorted(allstops, key=lambda t: t[1] - t[2])[:14], key=lambda t: t[1])
        for h, x, y in top:
            e0, e1 = (x - KICKOFF[h]) / FPS, (y - KICKOFF[h]) / FPS
            cover = dead1[x:y].mean()
            print("{:>8} {:>6}-{:>6} {:>7}-{:>7} {:>5.0f}s {:>9.0f}% {:>13.0f}%".format(
                h, clock(x), clock(y), "{:d}:{:02d}".format(int(e0 // 60), int(e0 % 60)),
                "{:d}:{:02d}".format(int(e1 // 60), int(e1 % 60)), (y - x) / FPS, 100 * vis[x:y].mean(), 100 * cover))
    else:
        k, ks = float(sys.argv[2]), float(sys.argv[3])
        e_track = track_evidence(ball, players, team)
        e_shape = shape_evidence(ball, players, team, blocks)
        e_tag = tag_prior(load_tags(), n)
        alive, st, share = summarize(v4b.decode(e_track, e_shape, e_tag, dead2, k, ks, blocks))
        print("possessions : {}  moyenne {:.1f} s  mediane {:.1f}  p10 {:.1f} | D {:.1f} % en {} arrets".format(
            len(alive), alive.mean(), np.median(alive), np.percentile(alive, 10), 100 * share, len(st)))
        labels = np.full(n, "X", dtype="<U1")
        for _, a, b, path in v4b.decode(e_track, e_shape, e_tag, dead2, k, ks, blocks):
            labels[a:b + 1] = np.array(["H", "A", "D"])[path]
        out = os.path.join(ROOT, "evaluation", "results", "v5_k{:g}_ks{:g}.csv".format(k, ks))
        write_submission(out, labels)
        print("ecrit :", out)


if __name__ == "__main__":
    main()