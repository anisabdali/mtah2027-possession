"""Smart Tagging vs tracking : couverture, sens du tag Team, lien avec les trous du ballon.
N'utilise aucune verite terrain."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from games import GAME, half_video_blocks  # noqa: E402
from starter.load_smart_tagging import load  # noqa: E402
from diagnostics import load_tracking, run_lengths  # noqa: E402  (meme dossier src/)

TEAM_CODE = {"barcelona": 0, "atletic club": 1, "athletic club": 1}  # 0 = H, 1 = A


def main():
    ball, players, team = load_tracking()
    n = len(ball)
    graded = np.zeros(n, bool)
    for _, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True
    has_ball = ~np.isnan(ball[:, 0])

    # equipe du joueur le plus proche, image par image (-1 si pas de ballon)
    d = np.linalg.norm(players - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    nearest = np.full(n, -1, np.int8)
    ok = has_ball & np.isfinite(d.min(axis=1))
    nearest[ok] = team[d[ok].argmin(axis=1)]

    events = load()
    covered_any = np.zeros(n, bool)
    covered_team = np.zeros(n, bool)
    set_piece = np.zeros(n, bool)
    per_code = {}

    for e in events:
        a, b = max(0, e["start_frame"]), min(n, e["end_frame"])
        if b <= a:
            continue
        covered_any[a:b] = True
        code = (e["code"] or "").strip()
        if code.lower() == "set pieces":
            set_piece[a:b] = True
        t = TEAM_CODE.get((e["tags"].get("Team") or "").strip().lower())
        stat = per_code.setdefault(code, {"n": 0, "dur": 0.0, "agree": 0, "total": 0, "na": 0})
        stat["n"] += 1
        stat["dur"] += (b - a) / 25
        if t is None:
            stat["na"] += 1
            continue
        covered_team[a:b] = True
        seg = nearest[a:b]
        seg = seg[seg >= 0]
        stat["total"] += len(seg)
        stat["agree"] += int((seg == t).sum())

    g = graded
    missing = g & ~has_ball
    print("== Couverture par le Smart Tagging ==")
    print("images notees couvertes par une phase : {:.1f} %".format(100 * covered_any[g].mean()))
    print("images notees couvertes par une phase avec equipe : {:.1f} %".format(100 * covered_team[g].mean()))
    print("images SANS ballon couvertes par une phase : {:.1f} %".format(100 * covered_any[missing].mean()))
    print("images SANS ballon couvertes par une phase avec equipe : {:.1f} %".format(
        100 * covered_team[missing].mean()))

    print("\n== Coups de pied arretes (SET PIECES) et ballon absent ==")
    print("part des images notees dans un coup de pied arrete : {:.1f} %".format(100 * set_piece[g].mean()))
    print("part des images SANS ballon dans un coup de pied arrete : {:.1f} %".format(
        100 * set_piece[missing].mean()))
    print("part des images AVEC ballon dans un coup de pied arrete : {:.1f} %".format(
        100 * set_piece[g & has_ball].mean()))

    print("\n== Accord tag Team / equipe du joueur le plus proche (ballon visible) ==")
    print("{:<30} {:>4} {:>8} {:>5} {:>8}".format("code", "n", "duree s", "N/A", "accord"))
    for code, s in sorted(per_code.items(), key=lambda kv: -kv[1]["n"]):
        acc = "{:6.1f}%".format(100 * s["agree"] / s["total"]) if s["total"] else "   -"
        print("{:<30} {:>4} {:>8.1f} {:>5} {:>8}".format(code, s["n"], s["dur"] / s["n"], s["na"], acc))

    seg = np.concatenate([run_lengths(has_ball[a:b + 1]) for _, a, b in half_video_blocks(GAME)]) / 25
    print("\n== Segments ou le ballon est visible ==")
    print("nombre {}  duree (s) : mediane {:.1f}  p90 {:.1f}  max {:.0f}".format(
        len(seg), np.median(seg), np.percentile(seg, 90), seg.max()))


if __name__ == "__main__":
    main()