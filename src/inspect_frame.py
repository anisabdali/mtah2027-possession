
"""Inspecter les coordonnées et les identifiants des joueurs."""
from inspect_open import load

ds = load(30000)
f = ds.frames[0]

print("Équipes du match :")
for team in ds.metadata.teams:
    print(repr(team))

print("\nCoordonnées des joueurs :")
print("Type :", type(f.players_coordinates))
print("Contenu :", repr(f.players_coordinates)[:5000])

print("\nDonnées des joueurs :")
print("Type :", type(f.players_data))
print("Contenu :", repr(f.players_data)[:5000])

print("\nÉquipe en possession :", repr(f.ball_owning_team))
print("État du ballon :", repr(f.ball_state))
