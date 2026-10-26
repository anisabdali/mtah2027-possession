
"""Evaluer la possession par le joueur le plus proche du ballon."""
from collections import defaultdict

import numpy as np

from inspect_open import load


def ground_name(team):
    if team is None:
        return None
    return str(getattr(team, "ground", "")).lower()


def main():
    ds = load(30000)

    # Associer chaque joueur à son équipe : home ou away.
    player_ground = {}
    for team in ds.metadata.teams:
        ground = ground_name(team)
        for player in team.players:
            player_ground[player.player_id] = ground

    # Statistiques globales et résultats par image.
    rows = []
    skipped = defaultdict(int)

    for frame in ds.frames:
        if "alive" not in str(frame.ball_state).lower():
            skipped["ballon mort"] += 1
            continue

        owner = ground_name(frame.ball_owning_team)
        if owner not in ("home", "away"):
            skipped["possession inconnue"] += 1
            continue

        ball = frame.ball_coordinates
        if ball is None:
            skipped["coordonnees ballon absentes"] += 1
            continue

        distances = {"home": [], "away": []}

        for player, pos in frame.players_coordinates.items():
            team = player_ground.get(player.player_id)
            if team not in distances or pos is None:
                continue

            d = np.hypot(pos.x - ball.x, pos.y - ball.y)
            if np.isfinite(d):
                distances[team].append(d)

        if not distances["home"] or not distances["away"]:
            skipped["equipe sans joueur localise"] += 1
            continue

        dh = min(distances["home"])
        da = min(distances["away"])

        prediction = "home" if dh < da else "away"
        nearest_distance = min(dh, da)
        margin = abs(dh - da)

        rows.append((
            owner,
            prediction,
            nearest_distance,
            margin,
            dh,
            da,
        ))

    if not rows:
        print("Aucune image exploitable.")
        print("Images ignorees :", dict(skipped))
        return

    data = np.asarray(rows, dtype=object)
    owners = data[:, 0]
    preds = data[:, 1]
    nearest_d = data[:, 2].astype(float)
    margins = data[:, 3].astype(float)

    correct = owners == preds
    n = len(rows)

    print("Images exploitables :", n)
    print("Images ignorees :", dict(skipped))
    print("Accord joueur le plus proche / DFL : {:.1f} %".format(
        100 * correct.mean()))

    majority = max(
        np.mean(owners == "home"),
        np.mean(owners == "away"),
    )
    print("Reference majoritaire (baseline) : {:.1f} %".format(
        100 * majority))

    for label, values in (
        ("distance du joueur le plus proche", nearest_d),
        ("ecart entre les distances home et away", margins),
    ):
        print("\n== Accord par quartile : {} ==".format(label))
        edges = np.quantile(values, [0, .25, .5, .75, 1])

        for i in range(4):
            if i < 3:
                mask = (values >= edges[i]) & (values < edges[i + 1])
            else:
                mask = (values >= edges[i]) & (values <= edges[i + 1])

            if not mask.any():
                continue

            print(
                "Q{} [{:.5f}, {:.5f}] : {:5d} images, accord {:.1f} %".format(
                    i + 1,
                    edges[i],
                    edges[i + 1],
                    int(mask.sum()),
                    100 * correct[mask].mean(),
                )
            )

    print("\n== Accord selon l'equipe possedante ==")
    for team in ("home", "away"):
        mask = owners == team
        print("{} : {:.1f} % ({} images)".format(
            team, 100 * correct[mask].mean(), int(mask.sum())))


if __name__ == "__main__":
    main()
