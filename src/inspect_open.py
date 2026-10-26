"""Inspecter un match ouvert DFL (IDSSE) charge depuis data/external : etat du ballon, possession, durees.
  python src/inspect_open.py 30000     # 30000 premieres images (essai)
  python src/inspect_open.py           # match entier"""
import glob
import os
import sys
from collections import Counter

import numpy as np

try:
    from kloppy import sportec
except ModuleNotFoundError:
    raise SystemExit("kloppy absent : python -m pip install -r requirements-starter.txt")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(ROOT, "data", "external")
PITCH_L, PITCH_W = 105.0, 68.0   # metres, cf. ds.metadata.pitch_dimensions


def dist_m(a, b):
    """Distance en metres entre deux points aux coordonnees normalisees (0-1)."""
    return float(np.hypot((a.x - b.x) * PITCH_L, (a.y - b.y) * PITCH_W))



def load(limit=None, match_id="J03WPY"):
    raw_matches = glob.glob(
        os.path.join(EXT, f"*positions_raw*{match_id}.xml")
    )
    meta_matches = glob.glob(
        os.path.join(EXT, f"*matchinformation*{match_id}.xml")
    )

    if len(raw_matches) != 1 or len(meta_matches) != 1:
        raise FileNotFoundError(
            f"Match {match_id}: trouvé {len(raw_matches)} fichier(s) "
            f"de positions et {len(meta_matches)} fichier(s) de métadonnées."
        )

    kw = {
        "raw_data": raw_matches[0],
        "meta_data": meta_matches[0],
        "only_alive": False,
    }
    if limit is not None:
        kw["limit"] = limit

    return sportec.load_tracking(**kw)



def runs(labels, targets):
    out, cur, start = [], None, 0
    for i, lab in enumerate(labels):
        t = lab if lab in targets else None
        if t != cur:
            if cur is not None:
                out.append(i - start)
            cur, start = t, i
    if cur is not None:
        out.append(len(labels) - start)
    return out


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    ds = load(limit)
    fps = float(ds.metadata.frame_rate)
    print("images : {}  fps : {}".format(len(ds.frames), fps))
    states, owners, labels, no_ball = Counter(), Counter(), [], 0
    for f in ds.frames:
        st = getattr(f, "ball_state", None)
        ow = getattr(f, "ball_owning_team", None)
        g = str(getattr(ow, "ground", ow)).lower()
        states[str(st)] += 1
        owners[g] += 1
        if f.ball_coordinates is None:
            no_ball += 1
        if not str(st).lower().endswith("alive"):
            labels.append("D")
        elif g.endswith("home"):
            labels.append("H")
        elif g.endswith("away"):
            labels.append("A")
        else:
            labels.append("?")
    print("\netat du ballon :", dict(states))
    print("equipe en possession :", dict(owners))
    print("ballon sans coordonnees : {:.1f} %".format(100 * no_ball / len(ds.frames)))
    c = Counter(labels)
    print("etiquettes : " + ", ".join("{} {:.1f} %".format(k, 100 * v / len(labels)) for k, v in sorted(c.items())))
    r = np.array(runs(labels, {"H", "A"})) / fps
    if len(r):
        print("\npossessions (suites de H ou de A, D separe) : {}".format(len(r)))
        print("duree (s) : moyenne {:.1f}  mediane {:.1f}  p10 {:.1f}  p90 {:.1f}".format(
            r.mean(), np.median(r), np.percentile(r, 10), np.percentile(r, 90)))


if __name__ == "__main__":
    main()