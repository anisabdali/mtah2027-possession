"""Entraine le detecteur de ballon mort v2 sur les deux matchs DFL degrades et sauvegarde ses coefficients."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

from predict_v5 import train_detector  # noqa: E402


def main():
    w, mu, sd = train_detector()
    out = os.path.join(ROOT, "models", "dead_detector.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = {
        "description": "Regression logistique de ballon mort (18 caracteristiques de mouvement et de ballon).",
        "training": "matchs ouverts DFL J03WPY et J03WOY, tracking degrade (graines 100 et 101)",
        "w": [float(x) for x in w], "mu": [float(x) for x in mu], "sd": [float(x) for x in sd],
    }
    with open(out, "w") as handle:
        json.dump(payload, handle, indent=1)
    print("ecrit :", out)


if __name__ == "__main__":
    main()