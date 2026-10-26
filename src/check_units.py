from inspect_open import load, dist_m

ds = load(200)
print("systeme de coordonnees :", ds.metadata.coordinate_system)
print("dimensions du terrain  :", ds.metadata.pitch_dimensions)
f = ds.frames[0]
print("ballon, image 0        :", f.ball_coordinates)
print("une position joueur    :", next(iter(f.players_coordinates.values())))