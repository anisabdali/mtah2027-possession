"""Mesure precision/couverture de la prediction geometrique sur DFL."""

import numpy as np
from inspect_open import load


def ground_name(team):
    if team is None:
        return None
    return str(getattr(team, "ground", "")).lower()


def main():
    ds = load(30000)

    player_ground = {}
    for team in ds.metadata.teams:
        ground = ground_name(team)
        for player in team.players:
            player_ground[player.player_id] = ground

    rows = []

    for frame_idx, frame in enumerate(ds.frames):
        if "alive" not in str(frame.ball_state).lower():
            continue

        owner = ground_name(frame.ball_owning_team)
        ball = frame.ball_coordinates

        if owner not in ("home", "away") or ball is None:
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
            continue

        dh = min(distances["home"])
        da = min(distances["away"])

        pred = "home" if dh < da else "away"
        nearest = min(dh, da)
        margin = abs(dh - da)

        rows.append((frame_idx, owner, pred, nearest, margin))

    if not rows:
        print("Aucune frame exploitable.")
        return

    data = np.asarray(rows, dtype=object)
    frame_idx = data[:, 0].astype(int)
    owners = data[:, 1]
    preds = data[:, 2]
    nearest = data[:, 3].astype(float)
    margins = data[:, 4].astype(float)
    print("Nombre de marges :", len(margins))
    print("Marges distinctes :", len(np.unique(margins)))
    print("Min :", np.min(margins))
    print("Quantiles :", np.quantile(margins, [0, 0.25, 0.5, 0.75, 1]))
    correct = owners == preds
    n = len(rows)

    print(f"Frames exploitables : {n}")
    print(f"Accuracy sans seuil : {100 * correct.mean():.2f} %")
    print()
    print("Seuil distance = distance maximale au joueur le plus proche.")
    print("Seuil marge = ecart minimal entre les deux equipes.")
    print()

    # Seuils descriptifs : ils servent a explorer le compromis,
    # pas a prouver que ces seuils generaliseront a un autre match.
    distance_thresholds = np.unique(
        np.quantile(nearest, [0.25, 0.50, 0.75, 1.00])
    )

    margin_thresholds = np.unique(
        np.quantile(margins, [0.00, 0.25, 0.50, 0.75])
    )

    print("Seuils de distance :", distance_thresholds)
    print("Seuils de marge :", margin_thresholds)

    print(
        f"{'Dist. max':>10} {'Marge min':>10} "
        f"{'Couverture':>12} {'Accuracy':>10} {'Frames':>8}"
    )
    print("-" * 56)

    results = []

    for dmax in distance_thresholds:
        for mmin in margin_thresholds:
            selected = (nearest <= dmax) & (margins >= mmin)
            count = int(selected.sum())

            print(
                f"dmax={dmax:.5f}, mmin={mmin:.5f}, "
                f"count={count}"
            )

            if count == 0:
                continue

            coverage = count / n
            accuracy = correct[selected].mean()

            results.append((accuracy, coverage, dmax, mmin, count))

            print(
                f"{100 * coverage:6.1f}% de couverture, "
                f"{100 * accuracy:6.2f}% de précision"
            )

    # Robustesse temporelle : seuils fixes, blocs de 5 minutes.
    rules = [
        ("A - confiance moyenne", nearest <= 0.01780),
        (
            "B - confiance elevee",
            (nearest <= 0.01780) & (margins >= 0.04116),
        ),
    ]

    block_size = 5 * 60 * 25  # 5 minutes a 25 images/s

    print("\n=== Robustesse temporelle (blocs de 5 minutes) ===")
    print("Couverture calculee parmi les frames exploitables.")

    for start in range(0, len(ds.frames), block_size):
        end = min(start + block_size, len(ds.frames))
        in_block = (frame_idx >= start) & (frame_idx < end)

        total = int(in_block.sum())
        if total == 0:
            continue

        print(f"\nBloc {start / 25 / 60:.0f}–{end / 25 / 60:.0f} min")

        for name, accepted in rules:
            selected = in_block & accepted
            count = int(selected.sum())

            if count == 0:
                print(f"{name}: aucune prediction ({total} frames exploitables)")
                continue

            coverage = count / total
            accuracy = correct[selected].mean()

            print(
                f"{name}: precision {100 * accuracy:.1f} %, "
                f"couverture {100 * coverage:.1f} %, "
                f"{count}/{total} frames exploitables"
            )

    print("\nMeilleurs compromis exploratoires :")
    # Affiche plusieurs points de couverture plutôt que de retenir
    # uniquement la precision la plus elevee.
    for target in (0.25, 0.50, 0.75, 0.90):
        eligible = [r for r in results if r[1] >= target]
        if not eligible:
            continue

        best = max(eligible, key=lambda r: r[0])
        accuracy, coverage, dmax, mmin, count = best

        print(
            f"Couverture >= {target:.0%} : "
            f"accuracy {100 * accuracy:.2f} %, "
            f"couverture {100 * coverage:.1f} %, "
            f"distance <= {dmax:.5f}, marge >= {mmin:.5f}, "
            f"{count} frames"
        )

    

if __name__ == "__main__":
    main()