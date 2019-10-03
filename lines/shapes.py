from typing import Sequence, Tuple, Union

import numpy as np

from .transform import Transform


class Shape:
    """
    Base class for a 3D object that can be "compiled" (projected) to 2D vector representation.
    A Shape instance models a single 3D object and allows the user to act upon it in the
    following ways:
    - Apply transforms on the object (scaling, rotation, translation).
    - Compile the object into 3D segments and faces that can subsequently be used for occlusion
    computation and final projection to 2D.

    Instances of the Shape class are valid, but compile into empty segment/face sets. The Shape
    class is thus mostly useful as a base class.
    """

    def transform(self, t: Union[np.ndarray, Transform, Sequence[Sequence[float]]]) -> None:
        """
        :param t: Transform object or 4x4 array of float
        """
        if type(t) == Transform:
            m = t.get()
        else:
            m = np.array(t)
        self._apply_transform(m)

    def _apply_transform(self, m: np.ndarray) -> None:
        """
        Apply the homogeneous transformation matrix m to the the shape.
        :param m: 4x4 homogeneous transformation matrix
        """

    # noinspection PyMethodMayBeStatic
    def compile(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: ([Nx2x3] ndarray of segments, [Mx3x3] ndarray of triangles
        """
        return np.zeros(shape=(0, 2, 3)), np.zeros(shape=(0, 3, 3))


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
    ):
        """
        :param vertices: [Nx3] floats
        :param segments: [Mx2] uint indices
        :param faces: [Px3] uint indices
        """
        # Store vertices in homogeneous coordinate
        self._vertices = np.hstack(
            (np.array(vertices, dtype=np.double), np.ones((len(vertices), 1)))
        )
        self._segments = np.array(segments, dtype=np.uint32)
        self._faces = np.array(faces, dtype=np.uint32)

    def _apply_transform(self, m: np.ndarray) -> None:
        new_vertices = np.empty_like(self._vertices)

        # TODO: this loop can most def be optimized with numpy
        for i in range(len(self._vertices)):
            new_vertices[i] = m @ self._vertices[i]
        self._vertices = new_vertices

    def compile(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._vertices[self._segments], self._vertices[self._faces]


class Cube(PolyShape):
    """
    This shape represent a cube centered on (0, 0, 0) with unit side length
    """

    def __init__(self):
        from .tables import CUBE_VERTICES, CUBE_SEGMENTS, CUBE_FACES

        super().__init__(CUBE_VERTICES, CUBE_SEGMENTS, CUBE_FACES)


class OBJShape(PolyShape):
    pass


class Sphere(Shape):
    """
    Spheres are projected first before lines and masking polygons can be created, in order to
    properly render the silhouette.
    """
