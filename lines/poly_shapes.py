import math
from typing import Sequence, Tuple

import numpy as np

from .math import vertices_matmul
from .shapes import Shape
from .skins import SilhouetteSkin


class PolyShape(Shape):
    """
    PolyShape is a base class for any Shape made of segments and polygonal masks, such as
    cubes, OBJ-based models, etc. Here, no abstract shape is used before compilation.
    """

    def __init__(
        self,
        vertices: Sequence[Tuple[float, float, float]],
        segments: Sequence[Tuple[int, int]],
        faces: Sequence[Tuple[int, int, int]],
        **kwargs,
    ):
        """
        :param vertices: [Nx3] floats
        :param segments: [Mx2] uint indices
        :param faces: [Px3] uint indices
        """

        super().__init__(**kwargs)

        # Store vertices in homogeneous coordinate
        self._vertices = np.hstack(
            (np.array(vertices, dtype=np.double), np.ones((len(vertices), 1)))
        )
        self._segments = np.reshape(np.array(segments, dtype=np.uint32), (len(segments), 2))
        self._faces = np.reshape(np.array(faces, dtype=np.uint32), (len(faces), 3))

    def _compile_impl(self, camera_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Project vertices to camera space and normalise to 3D
        vertices = vertices_matmul(self._vertices, camera_matrix @ self.transform)
        vertices = np.divide(vertices[:, 0:3], np.tile(vertices[:, -1:], (1, 3)))

        # Return segments and faces
        return vertices[self._segments], vertices[self._faces]


class Cube(PolyShape):
    """
    This shape represent a cube centered on (0, 0, 0) with unit side length
    """

    def __init__(self, **kwargs):
        from .tables import CUBE_VERTICES, CUBE_SEGMENTS, CUBE_FACES

        super().__init__(CUBE_VERTICES, CUBE_SEGMENTS, CUBE_FACES, **kwargs)


class Cylinder(PolyShape):
    """
    This shape represent a vertical cylinder, unit height and radius, centered on (0, 0, 0)
    """

    SEGMENT_COUNT = 36

    def __init__(self, vertical_segs: bool = False, silhouette: bool = True, **kwargs):
        n = self.SEGMENT_COUNT

        # create vertices
        t = 2 * math.pi * np.arange(0, n) / n
        circle = np.array([np.cos(t), np.sin(t), np.ones_like(t) * -0.5]).transpose()
        vertices = np.vstack((circle, circle, np.array([(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)])))
        vertices[n:-2, 2] = 0.5

        # create segments
        s = [(i, (i + 1) % n) for i in range(n)]
        s.extend([(i + n, (i + 1) % n + n) for i in range(n)])
        if vertical_segs:
            s.extend([(i, i + n) for i in range(n)])
        segments = np.array(s)

        # create faces
        f = []
        for i in range(n):
            f.append([i, (i + 1) % n, i + n])  # vertical face 1
            f.append([i + n, (i + 1) % n + n, (i + 1) % n])  # vertical face 2
            f.append([i, (i + 1) % n, 2 * n])  # bottom plate
            f.append([i + n, (i + 1) % n + n, 2 * n + 1])  # top plate
        faces = np.array(f)

        super().__init__(vertices, segments, faces, **kwargs)

        if silhouette:
            self.add(SilhouetteSkin())


class OBJShape(PolyShape):
    """
    PolyShape whose content is loaded from an .OBJ file.
    """

    # TODO
