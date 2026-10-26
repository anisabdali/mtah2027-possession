
"""Evaluation inter-match : entrainement sur J03WPY, test sur J03WOY.
Usage : python src/eval_cross_match.py
"""
import numpy as np
from inspect_open import load, dist_m
from eval_learned import (
    extract_data,
    fit_logistic,
    predict_probability,
    balanced_accuracy,
    report,
)


def main():
    limit = 30000

    # 1. Chargement independant des deux matchs.
    print(f"Chargement de J03WPY ({limit} premieres frames)...")
    source = load(limit, match_id="J03WPY")
    X_src, y_src, idx_src = extract_data(source)

    print(f"Chargement de J03WOY ({limit} premieres frames)...")
    target = load(limit, match_id="J03WOY")
    X_tgt, y_tgt, idx_tgt = extract_data(target)

    if len(y_src) < 100 or len(y_tgt) < 100:
        raise SystemExit("Pas assez de frames valides dans un des matchs.")

    # 2. Decoupage chronologique du match source uniquement.
    n_frames = len(source.frames)
    train_end = int(0.60 * n_frames)
    val_end = int(0.70 * n_frames)

    train = idx_src < train_end
    val = (idx_src >= train_end) & (idx_src < val_end)

    if min(train.sum(), val.sum()) < 50:
        raise SystemExit("Pas assez de donnees pour entrainement/validation.")

    X_train_raw, y_train = X_src[train], y_src[train]
    X_val_raw, y_val = X_src[val], y_src[val]

    # 3. Normalisation apprise exclusivement sur l'entrainement source.
    mean = X_train_raw.mean(axis=0)
    std = X_train_raw.std(axis=0)
    std[std < 1e-10] = 1.0

    X_train = (X_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std
    X_target = (X_tgt - mean) / std

    # 4. Entrainement sur J03WPY.
    w = fit_logistic(X_train, y_train)

    # 5. Choix du seuil sur la validation de J03WPY seulement.
    p_val = predict_probability(w, X_val)
    thresholds = np.linspace(0.10, 0.90, 161)

    scores = [
        balanced_accuracy(y_val, (p_val >= t).astype(int))
        for t in thresholds
    ]
    best_score = max(scores)
    best_threshold = min(
        (t for t, score in zip(thresholds, scores)
         if np.isclose(score, best_score)),
        key=lambda t: abs(t - 0.5),
    )

    # 6. Evaluation finale sur J03WOY, sans aucun ajustement.
    nearest = (X_tgt[:, 0] < X_tgt[:, 1]).astype(int)
    p_target = predict_probability(w, X_target)
    pred_default = (p_target >= 0.5).astype(int)
    pred_calibrated = (p_target >= best_threshold).astype(int)

    print("\n========== PROTOCOLE ==========")
    print(f"Match source : J03WPY ({len(y_src)} frames valides)")
    print(f"  Entrainement : {len(y_train)}")
    print(f"  Validation   : {len(y_val)}")
    print(f"Match cible : J03WOY ({len(y_tgt)} frames valides)")
    print(f"Seuil choisi sur J03WPY : {best_threshold:.3f}")
    print(
        "Balanced accuracy de validation : "
        f"{100 * best_score:.2f} %"
    )

    print("\n========== RESULTATS SUR J03WOY ==========")
    report("Joueur le plus proche", y_tgt, nearest)
    report("Regression logistique, seuil 0.5",
           y_tgt, pred_default)
    report("Regression logistique, seuil source optimise",
           y_tgt, pred_calibrated)

    print("\n========== COMPARAISON FINALE ==========")
    print(
        f"Baseline nearest : "
        f"{100 * balanced_accuracy(y_tgt, nearest):.2f} %"
    )
    print(
        f"Logistique seuil 0.5 : "
        f"{100 * balanced_accuracy(y_tgt, pred_default):.2f} %"
    )
    print(
        f"Logistique seuil {best_threshold:.3f} : "
        f"{100 * balanced_accuracy(y_tgt, pred_calibrated):.2f} %"
    )


if __name__ == "__main__":
    main()
