"""Convertit un match DFL (kloppy) en tableaux numpy en metres, mis en cache. Sans verite terrain du hackathon.
  python src/dfl_cache.py J03WPY J03WOY"""
import os
import sys
import time

import numpy as np

from inspect_open import load, EXT, PITCH_L, PITCH_W

FPS = 25


def is_home(g):
    return str(g).lower().endswith("home")


def build(match_id):
    out = os.path.join(EXT, "cache", "{}.npz".format(match_id))
    if os.path.exists(out):
        return out
    t0 = time.time()
    print("{} : chargement kloppy (plusieurs minutes, plusieurs Go de memoire)...".format(match_id))
    ds = load(None, match_id)
    slot, team = {}, []
    for t in ds.metadata.teams:
        g = 0 if is_home(getattr(t, "ground", "")) else 1
        for p in t.players:
            slot[p.player_id] = len(team)
            team.append(g)
    n, K = len(ds.frames), len(team)
    ball = np.full((n, 2), np.nan, np.float32)
    players = np.full((n, K, 2), np.nan, np.float32)
    alive = np.zeros(n, bool)
    owner = np.full(n, -1, np.int8)
    period = np.zeros(n, np.int8)
    for i, f in enumerate(ds.frames):
        alive[i] = "alive" in str(f.ball_state).lower()
        g = str(getattr(f.ball_owning_team, "ground", "")).lower()
        owner[i] = 0 if g.endswith("home") else (1 if g.endswith("away") else -1)
        period[i] = f.period.id
        b = f.ball_coordinates
        if b is not None:
            ball[i] = (b.x * PITCH_L, b.y * PITCH_W)
        for p, pos in f.players_coordinates.items():
            j = slot.get(p.player_id)
            if j is not None and pos is not None:
                players[i, j] = (pos.x * PITCH_L, pos.y * PITCH_W)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, ball=ball, players=players, alive=alive, owner=owner,
                        period=period, team=np.array(team, np.int8))
    print("  ecrit {} en {:.0f} s".format(out, time.time() - t0))
    return out


def run_lengths(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def stats(match_id):
    z = np.load(build(match_id))
    alive, owner, period, ball = z["alive"], z["owner"], z["period"], z["ball"]
    n = len(alive)
    print("\n===== {} : {} images ({:.1f} min) =====".format(match_id, n, n / FPS / 60))
    for p in sorted(set(period.tolist())):
        idx = np.flatnonzero(period == p)
        a = idx[0] + int(np.argmax(alive[idx])) if alive[idx].any() else idx[0]
        print("periode {} : images {}..{}, premiere image vivante {} (+{:.0f} s apres le debut)".format(
            p, idx[0], idx[-1], a, (a - idx[0]) / FPS))
    started = np.zeros(n, bool)
    for p in sorted(set(period.tolist())):
        idx = np.flatnonzero(period == p)
        if alive[idx].any():
            started[idx[0] + int(np.argmax(alive[idx])):idx[-1] + 1] = True
    print("ballon mort : {:.1f} % de toutes les images, {:.1f} % apres le premier coup d'envoi".format(
        100 * (~alive).mean(), 100 * (~alive[started]).mean()))
    print("ballon sans coordonnees (images vivantes) : {:.2f} %".format(100 * np.isnan(ball[alive, 0]).mean()))
    tracked = (~np.isnan(z["players"][alive][:, :, 0])).sum(axis=1)
    print("joueurs localises par image vivante : moyenne {:.1f}, min {}, max {}".format(
        tracked.mean(), tracked.min(), tracked.max()))
    print("possession (images vivantes) : Home {:.1f} %".format(100 * (owner[alive] == 0).mean()))
    for name, mask in (("home", alive & (owner == 0)), ("away", alive & (owner == 1))):
        pass
    lab = np.where(~alive, 2, owner)           # 0 H, 1 A, 2 D
    durs, dead = [], []
    for p in sorted(set(period.tolist())):
        seg = lab[period == p]
        cuts = np.concatenate(([0], np.flatnonzero(np.diff(seg)) + 1, [len(seg)]))
        for a, b in zip(cuts[:-1], cuts[1:]):
            (dead if seg[a] == 2 else durs).append((b - a) / FPS)
    d, dd = np.array(durs), np.array(dead)
    print("possessions (suites H ou A, D separe) : {}  duree (s) : moyenne {:.1f} mediane {:.1f} p10 {:.1f} p90 {:.1f}".format(
        len(d), d.mean(), np.median(d), np.percentile(d, 10), np.percentile(d, 90)))
    print("arrets (suites D) : {}  duree (s) : mediane {:.1f} p90 {:.1f}".format(
        len(dd), np.median(dd), np.percentile(dd, 90)))
    sp = np.diff(np.flatnonzero(np.concatenate(([True], np.diff(lab) != 0, [True]))))
    print("segments H/A/D de moins de 1 s : {:.0f} %".format(100 * (sp < FPS).mean()))


if __name__ == "__main__":
    for m in (sys.argv[1:] or ["J03WPY"]):
        stats(m)