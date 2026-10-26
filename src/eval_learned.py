
"""Compare la geometrie seule a une regression logistique.
Entrainement : 70 % du match au debut.
Evaluation : 30 % a la fin, jamais vus pendant l'entrainement.
Usage : python src/eval_learned.py
"""
import numpy as np
from inspect_open import load


def ground_name(team):
    """Renvoie home/away depuis les metadonnees DFL."""
    if team is None:
        return None
    return str(getattr(team, "ground", "")).lower()


def extract_data(ds):
    """Construit les caracteristiques geometriques et les labels de possession."""
    player_ground = {}
    for team in ds.metadata.teams:
        ground = ground_name(team)
        for player in team.players:
            player_ground[player.player_id] = ground

    rows, labels, frame_indices = [], [], []

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
                distances[team].append(float(d))

        if not distances["home"] or not distances["away"]:
            continue

        dh = np.sort(distances["home"])
        da = np.sort(distances["away"])

        # Les premieres caracteristiques decrivent les distances au ballon.
        features = [
            dh[0], da[0], dh[0] - da[0], abs(dh[0] - da[0]),
            min(dh[0], da[0]),
            dh[1] if len(dh) > 1 else 1.0,
            da[1] if len(da) > 1 else 1.0,
        ]

        # Combien de joueurs de chaque equipe sont proches du ballon ?
        for threshold in (0.01, 0.02, 0.04, 0.08):
            features.append(sum(d <= threshold for d in distances["home"]))
            features.append(sum(d <= threshold for d in distances["away"]))

        # Position du ballon : permet au modele d'apprendre des effets spatiaux.
        features.extend([ball.x, ball.y])

        rows.append(features)
        labels.append(1 if owner == "home" else 0)
        frame_indices.append(frame_idx)

    return (
        np.asarray(rows, dtype=float),
        np.asarray(labels, dtype=int),
        np.asarray(frame_indices, dtype=int),
    )


def fit_logistic(X, y, l2=1.0, iterations=30):
    """Ajuste une regression logistique binaire par Newton."""
    Xb = np.column_stack([X, np.ones(len(X))])
    w = np.zeros(Xb.shape[1])
    penalty = np.eye(len(w)) * l2
    penalty[-1, -1] = 1e-8  # penalise tres peu le biais

    for _ in range(iterations):
        z = np.clip(Xb @ w, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        gradient = Xb.T @ (p - y) + penalty @ w
        weights = p * (1 - p)
        hessian = (Xb * weights[:, None]).T @ Xb + penalty
        w -= np.linalg.solve(hessian, gradient)

    return w


def predict_probability(w, X):
    """Calcule la probabilite estimee que home possede le ballon."""
    Xb = np.column_stack([X, np.ones(len(X))])
    return 1.0 / (1.0 + np.exp(-np.clip(Xb @ w, -30, 30)))


def report(name, y_true, y_pred):
    """Affiche accuracy, baseline majoritaire et matrice de confusion."""
    accuracy = np.mean(y_true == y_pred)
    majority = max(np.mean(y_true == 0), np.mean(y_true == 1))

    # Lignes = vraie equipe ; colonnes = equipe predite.
    cm = np.zeros((2, 2), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1

    print(f"\n=== {name} ===")
    print(f"Frames evaluees : {len(y_true)}")
    print(f"Accuracy : {100 * accuracy:.2f} %")
    print(f"Baseline majoritaire : {100 * majority:.2f} %")
    print("Matrice de confusion (lignes=reel, colonnes=pred):")
    print("             predit Away  predit Home")
    print(f"Reel Away     {cm[0, 0]:8d}   {cm[0, 1]:8d}")
    print(f"Reel Home     {cm[1, 0]:8d}   {cm[1, 1]:8d}")

    for label, idx in (("Away", 0), ("Home", 1)):
        mask = y_true == idx
        if mask.any():
            recall = np.mean(y_pred[mask] == idx)
            print(f"Rappel {label} : {100 * recall:.2f} % ({mask.sum()} frames)")



def balanced_accuracy(y_true, y_pred):
    """Moyenne des rappels Away et Home."""
    recalls = []
    for label in (0, 1):
        mask = y_true == label
        if not mask.any():
            return 0.0
        recalls.append(np.mean(y_pred[mask] == label))
    return float(np.mean(recalls))


def main():
    print("Chargement des 30 000 premieres frames DFL...")
    ds = load(30000)
    X, y, frame_indices = extract_data(ds)

    if len(y) < 100:
        raise SystemExit("Pas assez de frames exploitables.")

    # Découpage chronologique sur les indices des frames du match.
    n_frames = len(ds.frames)
    train_end = int(0.60 * n_frames)
    val_end = int(0.70 * n_frames)

    train = frame_indices < train_end
    val = (frame_indices >= train_end) & (frame_indices < val_end)
    test = frame_indices >= val_end

    if min(train.sum(), val.sum(), test.sum()) < 50:
        raise SystemExit("Pas assez de donnees dans un des trois blocs.")

    X_train_raw, X_val_raw, X_test_raw = X[train], X[val], X[test]
    y_train, y_val, y_test = y[train], y[val], y[test]

    # Baseline géométrique : distances brutes, sans normalisation.
    nearest_test = (
        X_test_raw[:, 0] < X_test_raw[:, 1]
    ).astype(int)

    # La normalisation est apprise uniquement sur l'entraînement.
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std < 1e-10] = 1.0

    X_train = (X_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std
    X_test = (X_test_raw - mean) / std

    # Entraînement du modèle.
    w = fit_logistic(X_train, y_train)

    # Validation : choix du seuil uniquement sur ces données.
    p_val = predict_probability(w, X_val)
    thresholds = np.linspace(0.10, 0.90, 161)

    scores = []
    for threshold in thresholds:
        pred = (p_val >= threshold).astype(int)
        scores.append(balanced_accuracy(y_val, pred))

    best_score = max(scores)

    # En cas d'égalité, privilégier le seuil le plus proche de 0.5.
    best_threshold = min(
        (t for t, score in zip(thresholds, scores)
         if np.isclose(score, best_score)),
        key=lambda t: abs(t - 0.5),
    )

    # Évaluation finale : aucun réglage sur le bloc de test.
    p_test = predict_probability(w, X_test)
    pred_default = (p_test >= 0.5).astype(int)
    pred_calibrated = (p_test >= best_threshold).astype(int)

    print(f"\nFrames valides : {len(y)}")
    print(f"Entrainement : {len(y_train)}")
    print(f"Validation : {len(y_val)}")
    print(f"Test final : {len(y_test)}")
    print(f"Seuil choisi sur validation : {best_threshold:.3f}")
    print(f"Balanced accuracy validation : {best_score * 100:.2f} %")

    report("Joueur le plus proche (test)", y_test, nearest_test)
    report("Regression logistique seuil 0.5 (test)",
           y_test, pred_default)
    report("Regression logistique seuil optimise (test)",
           y_test, pred_calibrated)

    print(
        "\nBalanced accuracy sur le test : "
        f"nearest={balanced_accuracy(y_test, nearest_test) * 100:.2f} %, "
        f"logistique 0.5={balanced_accuracy(y_test, pred_default) * 100:.2f} %, "
        f"logistique optimisee="
        f"{balanced_accuracy(y_test, pred_calibrated) * 100:.2f} %"
    )


if __name__ == "__main__":
    main()
