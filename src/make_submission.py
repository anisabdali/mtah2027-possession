"""Pipeline final : donnees tactiques -> CSV de possession (H, A, D, X).
Une commande :  python src/make_submission.py [--out chemin.csv]
Aucune verite terrain n'est utilisee. Entierement deterministe (pas d'aleatoire)."""
import argparse
import hashlib
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import GAME, GAMES, half_video_blocks, require, source_frames  # noqa: E402
from starter.baseline import team_of_each_track, write_submission  # noqa: E402
from starter.load_smart_tagging import load as load_tags  # noqa: E402

FPS = 25
K_TRACK, K_SHAPE, TAG_PER_SECOND = 12.0, 12.0, 1.0      # poids des trois sources de preuves
SWITCH_COST = float(np.log(14.0 * FPS))                  # changement de possession : ~14 s de duree moyenne
LOGIT_CLIP = 3.0
DEAD_THRESHOLD, MIN_STOP_S, MIN_PLAY_S = 0.5, 3.0, 3.0
MODEL = os.path.join(ROOT, "models", "dead_detector.json")
CACHE = os.path.join(ROOT, "data", "cache", "atd.npz")   # derive du tracking : reste dans data/, ignore par Git
HW, VMAX, PITCH_L, PITCH_W = 12, 12.0, 105.0, 68.0

# Tags Smart Tagging : (+1 : le tag Team est le possesseur ; -1 : c'est l'adversaire), fiabilite
TAG_TABLE = {
    "build up": (+1, 0.76), "progression": (+1, 0.70), "attacking transition": (+1, 0.62),
    "ball in final third": (+1, 0.62), "long ball": (+1, 0.62),
    "defending in attacking third": (-1, 0.76), "defending in middle third": (-1, 0.70),
    "defensive transition": (-1, 0.62),
}
TEAM_CODE = {"barcelona": 0, "atletic club": 1, "athletic club": 1}   # 0 = H, 1 = A

# Accord "joueur le plus proche" / possesseur selon la distance et la marge (tableau de diag_margin.py)
DIST_EDGES = [1, 2, 4, 8]
MARGIN_EDGES = [0.5, 1, 2, 4]
AGREEMENT = np.array([
    [0.64, 0.70, 0.72, 0.80, 0.83],
    [0.52, 0.55, 0.62, 0.69, 0.80],
    [0.49, 0.51, 0.55, 0.61, 0.68],
    [0.48, 0.49, 0.49, 0.57, 0.60],
    [0.50, 0.50, 0.50, 0.50, 0.50],
])
WEIGHT = np.where(AGREEMENT > 0.5, np.log(AGREEMENT / (1 - AGREEMENT)), 0.0)


# ---------------------------------------------------------------- tracking
def load_tracking():
    """ball (n,2), players (n,slots,2) en metres (NaN si absent), team (slots,)."""
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return z["ball"], z["players"], z["team"]
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, None)
    team = team_of_each_track(meta_path)
    length, width = GAMES[GAME]["pitch"]
    n = source_frames("tactical", GAME)
    ball = np.full((n, 2), np.nan, np.float32)
    players = np.full((n, len(team), 2), np.nan, np.float32)
    print("lecture du tracking (1 a 3 minutes, une seule fois)...")
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            i = int(parts[0]) - 1
            b = parts[2]
            if b[0] != "N":
                a = b.split(",")
                ball[i] = (float(a[0]) * length, float(a[1]) * width)
            for slot, g in enumerate(parts[1].split(";")):
                if slot >= len(team) or g[0] == "N":
                    continue
                a = g.split(",")
                players[i, slot] = (float(a[0]) * length, float(a[1]) * width)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, ball=ball, players=players, team=team)
    return ball, players, team


# ---------------------------------------------------------------- source 1 : joueur le plus proche
def track_evidence(ball, players, team):
    n = len(ball)
    d = np.linalg.norm(players - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    dH, dA = d[:, team == 0].min(axis=1), d[:, team == 1].min(axis=1)
    idx = np.where(np.isfinite(dH) & np.isfinite(dA))[0]
    dmin = np.minimum(dH[idx], dA[idx])
    margin = np.abs(dA[idx] - dH[idx])
    w = WEIGHT[np.digitize(dmin, DIST_EDGES), np.digitize(margin, MARGIN_EDGES)]
    near = (dA[idx] < dH[idx]).astype(int)
    e = np.zeros((n, 2))
    e[idx, near] += w / 2
    e[idx, 1 - near] -= w / 2
    return e


# ---------------------------------------------------------------- source 2 : forme collective
def team_stats(players, team, t):
    P = players[:, team == t, :]
    cnt = (~np.isnan(P[:, :, 0])).sum(axis=1)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cx, cy = np.nanmean(P[:, :, 0], axis=1), np.nanmean(P[:, :, 1], axis=1)
        sx, sy = np.nanstd(P[:, :, 0], axis=1), np.nanstd(P[:, :, 1], axis=1)
    bad = cnt < 7
    for a in (cx, cy, sx, sy):
        a[bad] = np.nan
    return cx, cy, sx, sy


def velocity(c, win=25):
    v = np.full_like(c, np.nan)
    v[win:-win] = (c[2 * win:] - c[:-2 * win]) / (2 * win / FPS)
    return v


def shape_features(players, team):
    cxH, cyH, sxH, syH = team_stats(players, team, 0)
    cxA, cyA, sxA, syA = team_stats(players, team, 1)
    vH, vA = velocity(cxH), velocity(cxA)
    return np.column_stack([cxH - cxA, (cxH + cxA) / 2 - 52.65, sxH - sxA, syH - syA,
                            cyH - cyA, vH - vA, (vH + vA) / 2])


def fit_newton(X, y, l2=1.0, iters=25):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) + l2 * w
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w


def shape_evidence(ball, players, team, blocks):
    """Modele de forme entraine par mi-temps sur des pseudo-etiquettes automatiques (ballon tres proche d'un joueur)."""
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
        w = fit_newton((F[pool] - mu) / sd, y[pool])
        ok = idx[feat_ok[idx]]
        z = np.hstack([(F[ok] - mu) / sd, np.ones((len(ok), 1))]) @ w
        z = np.clip(z, -LOGIT_CLIP, LOGIT_CLIP)
        e[ok, 0] += z / 2
        e[ok, 1] -= z / 2
    return e


# ---------------------------------------------------------------- source 3 : Smart Tagging
def tag_prior(events, n):
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
    idx = np.where(best_p >= 0)[0]
    p = best_p[idx].astype(int)
    e = np.zeros((n, 2))
    e[idx, p] += np.log(best_q[idx])
    e[idx, 1 - p] += np.log(1 - best_q[idx])
    return e


# ---------------------------------------------------------------- ballon mort
def roll(a, w, kind):
    s = pd.Series(a)
    mp = max(1, w // 5)
    if kind == "c":
        return s.rolling(w, center=True, min_periods=mp).mean().to_numpy()
    if kind == "l":
        return s.rolling(w, min_periods=mp).mean().to_numpy()
    return s[::-1].rolling(w, min_periods=mp).mean()[::-1].to_numpy()


def dead_features(ball, players):
    n = len(ball)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        disp = np.full(players.shape[:2], np.nan, np.float32)
        disp[HW:n - HW] = np.linalg.norm(players[2 * HW:] - players[:n - 2 * HW], axis=2)
        v = disp / (2 * HW / FPS)
        v[v > VMAX] = np.nan
        valid = ~np.isnan(v)
        med = np.nanmedian(v, axis=1)
        p75 = np.nanpercentile(v, 75, axis=1)
        slow = ((v < 1.0) & valid).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
        slow[~valid.any(axis=1)] = np.nan
        bd = np.full(n, np.nan, np.float32)
        bd[HW:n - HW] = np.linalg.norm(ball[2 * HW:] - ball[:n - 2 * HW], axis=1)
        bspeed = bd / (2 * HW / FPS)
        bspeed[bspeed > 40] = np.nan
        cols = []
        for b in (med, slow, p75):
            cols.append(roll(b, 375, "c"))
            for w in (50, 150):
                cols.append(roll(b, w, "l"))
                cols.append(roll(b, w, "r"))
        vis = (~np.isnan(ball[:, 0])).astype(float)
        out = ((ball[:, 0] < 0) | (ball[:, 0] > PITCH_L) | (ball[:, 1] < 0) | (ball[:, 1] > PITCH_W)).astype(float)
        sp, m = np.nan_to_num(bspeed), (~np.isnan(bspeed)).astype(float)
        num, den = roll(sp, 50, "c"), roll(m, 50, "c")
        cols.append(roll(vis, 125, "c"))
        cols.append(np.where(den > 0.02, num / np.maximum(den, 1e-9), 0.0))
        cols.append(roll(out, 125, "c"))
    return np.column_stack(cols)


def dead_probability(ball, players):
    with open(MODEL) as handle:
        d = json.load(handle)
    w, mu, sd = np.array(d["w"]), np.array(d["mu"]), np.array(d["sd"])
    X = dead_features(ball, players)
    ok = ~np.isnan(X).any(axis=1)
    p = np.full(len(ball), np.nan)
    Xn = (X[ok] - mu) / sd
    p[ok] = 1.0 / (1.0 + np.exp(-np.clip(np.column_stack([Xn, np.ones(len(Xn))]) @ w, -30, 30)))
    return pd.Series(p).rolling(25, center=True, min_periods=5).mean().to_numpy()


def remove_short(mask, min_len, value):
    m = mask.copy()
    d = np.diff(np.concatenate(([0], (m == value).astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if b - a < min_len and a > 0 and b < len(m):
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


# ---------------------------------------------------------------- decodage de la possession
def viterbi(e, cost):
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


def decode(e_track, e_shape, e_tag, dead, blocks):
    e = (e_track * K_TRACK + e_shape * K_SHAPE + e_tag * TAG_PER_SECOND) / FPS
    out = []
    for half, a, b in blocks:
        d = dead[a:b + 1]
        labels = np.full(b - a + 1, 2, np.int8)               # 2 = D
        edges = np.diff(np.concatenate(([0], (~d).astype(np.int8), [0])))
        for x, y in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            labels[x:y] = viterbi(e[a + x:a + y], SWITCH_COST)   # H/A a l'interieur de chaque phase vivante
        out.append((half, a, b, labels))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "submission.csv"))
    args = parser.parse_args()
    ball, players, team = load_tracking()
    n = len(ball)
    blocks = half_video_blocks(GAME)
    dead = dead_mask(dead_probability(ball, players), blocks, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        e_track = track_evidence(ball, players, team)
        e_shape = shape_evidence(ball, players, team, blocks)
    e_tag = tag_prior(load_tags(), n)
    labels = np.full(n, "X", dtype="<U1")                    # X en dehors des mi-temps (jamais notees)
    for _, a, b, path in decode(e_track, e_shape, e_tag, dead, blocks):
        labels[a:b + 1] = np.array(["H", "A", "D"])[path]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_submission(args.out, labels)
    with open(args.out, "rb") as handle:
        digest = hashlib.sha256(handle.read()).hexdigest()
    print("ecrit : {}\nsha256 : {}".format(args.out, digest))


if __name__ == "__main__":
    main()