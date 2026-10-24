"""Diagnostic du tracking : ballon absent, joueurs suivis, distances.
N'utilise aucune vérité terrain.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import GAME, GAMES, require, source_frames, half_video_blocks
from starter.baseline import team_of_each_track

FPS = 25
CACHE = os.path.join(ROOT, "data", "cache", "atd.npz")


def load_tracking():
    """Retourne les positions du ballon, des joueurs et leurs équipes."""
    if os.path.exists(CACHE):
        with np.load(CACHE) as z:
            return z["ball"], z["players"], z["team"]

    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, None)
    team = team_of_each_track(meta_path)
    length, width = GAMES[GAME]["pitch"]
    n = source_frames("tactical", GAME)

    ball = np.full((n, 2), np.nan, dtype=np.float32)
    players = np.full((n, len(team), 2), np.nan, dtype=np.float32)

    print("Lecture du tracking (une seule fois)...")

    with open(raw_path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue

            try:
                i = int(parts[0]) - 1  # Les identifiants commencent à 1.
            except ValueError:
                continue

            if not 0 <= i < n:
                continue

            b = parts[2]
            if b and b[0] != "N":
                xy = b.split(",")
                if len(xy) >= 2:
                    ball[i] = float(xy[0]) * length, float(xy[1]) * width

            for slot, p in enumerate(parts[1].split(";")):
                if slot >= len(team):
                    break
                if not p or p[0] == "N":
                    continue

                xy = p.split(",")
                if len(xy) >= 2:
                    players[i, slot] = (
                        float(xy[0]) * length,
                        float(xy[1]) * width,
                    )

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, ball=ball, players=players, team=team)

    return ball, players, team


def run_lengths(mask):
    """Longueurs des séquences consécutives de True."""
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    diff = np.diff(padded)
    return np.where(diff == -1)[0] - np.where(diff == 1)[0]


def main():
    ball, players, team = load_tracking()
    blocks = half_video_blocks(GAME)

    graded = np.zeros(len(ball), dtype=bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True

    if not graded.any():
        raise ValueError("Aucune image à analyser.")

    has_ball = np.isfinite(ball).all(axis=1)
    has_player = np.isfinite(players).all(axis=2)
    n_players = has_player.sum(axis=1)

    print(f"\n== Images analysées : {graded.sum()} ==")
    print(f"Ballon visible : {100 * has_ball[graded].mean():.1f} %")

    print("\n== Joueurs suivis par image ==")
    counts = n_players[graded]
    print(f"Moyenne {counts.mean():.1f}  min {counts.min()}  max {counts.max()}")
    print(f"Plus de 22 joueurs : {100 * (counts > 22).mean():.1f} %")
    print(f"Moins de 18 joueurs : {100 * (counts < 18).mean():.1f} %")

    print("\n== Trous du ballon ==")
    gaps = np.concatenate([
        run_lengths(~has_ball[a:b + 1])
        for _, a, b in blocks if b >= a
    ])
    missing_total = gaps.sum()

    if len(gaps) == 0:
        print("Aucun trou détecté.")
    else:
        sec = gaps / FPS
        print(f"Nombre de trous : {len(gaps)}")
        print(
            f"Durée (s) : médiane {np.median(sec):.2f}  "
            f"p90 {np.percentile(sec, 90):.2f}  max {sec.max():.1f}"
        )

        for s in (0.5, 1, 3, 5, 10):
            share = gaps[sec > s].sum() / missing_total if missing_total else 0
            print(f"  Part des images manquantes dans les trous > {s:g} s : "
                  f"{100 * share:.1f} %")

    print("\n== Joueur le plus proche du ballon ==")
    idx = np.flatnonzero(graded & has_ball)

    if len(idx) == 0:
        print("Aucune image avec ballon visible.")
        return

    d = np.linalg.norm(players[idx] - ball[idx, None, :], axis=2)
    d[~has_player[idx]] = np.inf

    nearest_slot = d.argmin(axis=1)
    nearest = d[np.arange(len(idx)), nearest_slot]
    ok = np.isfinite(nearest)

    if not ok.any():
        print("Aucune distance exploitable.")
        return

    q = np.percentile(nearest[ok], [25, 50, 75, 90, 99])
    print(
        f"Distance (m) : p25 {q[0]:.1f}  médiane {q[1]:.1f}  "
        f"p75 {q[2]:.1f}  p90 {q[3]:.1f}  p99 {q[4]:.1f}"
    )
    print(f"Plus proche à plus de 3 m : {100 * (nearest[ok] > 3).mean():.1f} %")

    # Convertit les équipes en codes numériques, quel que soit leur type initial.
    _, team_code = np.unique(team, return_inverse=True)

    side = np.full(len(ball), -1, dtype=np.int32)
    side[idx[ok]] = team_code[nearest_slot[ok]]

    both = (
        graded[:-1] & graded[1:]
        & (side[:-1] >= 0) & (side[1:] >= 0)
    )
    minutes = both.sum() / FPS / 60
    flips = int((side[:-1][both] != side[1:][both]).sum())

    rate = flips / minutes if minutes else 0
    print(f"Changements d'équipe du plus proche : {flips} ({rate:.1f} par minute)")


if __name__ == "__main__":
    main()