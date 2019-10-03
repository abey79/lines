# Cube geometries

CUBE_VERTICES = (
    (-0.5, -0.5, -0.5),
    (-0.5, 0.5, -0.5),
    (0.5, 0.5, -0.5),
    (0.5, -0.5, -0.5),
    (-0.5, -0.5, 0.5),
    (-0.5, 0.5, 0.5),
    (0.5, 0.5, 0.5),
    (0.5, -0.5, 0.5),
)

CUBE_SEGMENTS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
)

CUBE_FACES = (
    (0, 1, 2),
    (0, 2, 3),
    (0, 1, 5),
    (0, 5, 4),
    (0, 3, 7),
    (0, 7, 4),
    (3, 2, 6),
    (3, 6, 7),
    (1, 2, 6),
    (1, 6, 5),
    (4, 5, 6),
    (4, 6, 7),
)
