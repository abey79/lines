import math
from typing import Sequence, Tuple, Union

import numpy as np

from .skins import Skin


class Shape:
    """
    Base class for a 3D object that can be "compiled" (projected) to 2D vector representation.
    A Shape instance models a single 3D object and allows the user to act upon it in the
    following ways:
    - Apply transforms on the object (scaling, rotation, translation).
    - Apply one or more skins, which may affect the compilation process
    - Compile the object into 3D segments and faces according to a camera projection matrix.

    Instances of the Shape class are valid, but compile into empty segment/face sets.
    """

    def __init__(
        self,
        scale: Union[float, Sequence[float]] = None,
        rotate_x: float = None,
        rotate_y: float = None,
        rotate_z: float = None,
        translate: Sequence[float] = None,
    ):
        """
        Initialize a Shape with a default transform matrix. If parameters are passed, the
        corresponding transform a applied. If multiple parameters are passed, the transforms
        are applied in the order listed here:
        :param scale: scale of the the shape (provide a 3-tuple for per axis scaling)
        :param rotate_x: rotation around x axis (rad)
        :param rotate_y: rotation around y axis (rad)
        :param rotate_z: rotation around z axis (rad)
        :param translate: translation
        """
        self._skins = []

        self.transform = np.identity(4)
        if scale is not None:
            self.scale(scale)
        if rotate_x is not None:
            self.rotate_x(rotate_x)
        if rotate_y is not None:
            self.rotate_y(rotate_y)
        if rotate_z is not None:
            self.rotate_z(rotate_z)
        if translate is not None:
            self.translate(translate)

    def add(self, item: Skin) -> None:
        """
        Add an item to the shape. Shape handles only skins. Node also handles sub-shapes
        :param item: skin to add
        :return:
        """
        if isinstance(item, Skin):
            self._skins.append(item)
        else:
            raise ValueError("only Skin instances may be added to a Shape")

    @property
    def transform(self) -> np.ndarray:
        return self.__transform

    @transform.setter
    def transform(self, transform: np.ndarray) -> None:
        self.__transform = transform

    def scale(
        self,
        x: Union[float, Sequence[float]],
        y: Union[float, None] = None,
        z: Union[float, None] = None,
    ) -> None:
        try:
            if y is None or z is None:
                # noinspection PyBroadException
                try:
                    scale_vec = [x[0], x[1], x[2]]
                except:
                    scale_vec = [float(x)] * 3
            else:
                scale_vec = [float(x), float(y), float(z)]
        except Exception as exc:
            raise ValueError(
                "Argument may be one float, one size-3 sequence or 3 floats"
            ) from exc

        for i in range(3):
            self.__transform[i][i] *= scale_vec[i]

    def rotate_x(self, angle: float) -> None:
        s, c = math.sin(angle), math.cos(angle)
        r = np.array(([1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]))
        self.__transform = r @ self.__transform

    def rotate_y(self, angle: float) -> None:
        s, c = math.sin(angle), math.cos(angle)
        r = np.array(([c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]))
        self.__transform = r @ self.__transform

    def rotate_z(self, angle: float) -> None:
        s, c = math.sin(angle), math.cos(angle)
        r = np.array(([c, s, 0, 0], [-s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]))
        self.__transform = r @ self.__transform

    def translate(
        self,
        x: Union[Sequence[float], float],
        y: Union[float, None] = None,
        z: Union[float, None] = None,
    ) -> None:
        """
        Apply a translation to the current transform. Either one 3-tuple or 3 float can be
        passed as arguments.
        :param x: either a 3-tuple of coordinate or the x coordinate
        :param y: y coordinate
        :param z: z coordinate
        """
        try:
            if y is None or z is None:
                v_x, v_y, v_z = float(x[0]), float(x[1]), float(x[2])
            else:
                v_x, v_y, v_z = float(x), float(y), float(z)
        except Exception as exc:
            raise ValueError(
                "Argument must be either one vector or three coordinates"
            ) from exc

        self.__transform[0][3] += v_x
        self.__transform[1][3] += v_y
        self.__transform[2][3] += v_z

    def compile(self, camera_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform the shape into segments and (opaque) faces in camera space, possibly applying
        skins in the process. The actual compilation is delegated to _compile_impl(), which
        subclasses should override.
        :param camera_matrix: (4x4) camera view and projection matrix
        :return: ([Nx2x3] ndarray of segments, [Mx3x3] ndarray of triangles
        """

        segs, faces = self._compile_impl(camera_matrix)

        for skin in self._skins:
            segs, faces = skin.apply(segs, faces, camera_matrix)

        return segs, faces

    # noinspection PyMethodMayBeStatic
    def _compile_impl(self, camera_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Subclass must implement this method to compile it into a set of 3D segments and faces
        in camera space. The following steps are typically applied:
        - the shape's transform matrix is applied
        - the geometry is projected to camera space with camera_matrix
        - a list of segment and face is generated
        :param camera_matrix: (4x4) camera view and projection matrix
        :return: ([Nx2x3] ndarray of segments, [Mx3x3] ndarray of triangles
        """
        # return emptiness
        return np.zeros(shape=(0, 2, 3)), np.zeros(shape=(0, 3, 3))


class Sphere(Shape):
    """
    Spheres are projected first before lines and masking polygons can be created, in order to
    properly render the silhouette.
    """

    # TODO


class Node(Shape):
    """
    Empty shape that does not generate any geometry but contains other shapes, permitting the
    construction of a scene graph. The node transform matrix are passed on their children
    shapes.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._shapes = []

    def add(self, item: Union[Shape, Skin]) -> None:
        """
        Add a sub-shape or a skin to the node.
        :param item: the shape to add
        """
        if isinstance(item, Skin):
            super().add(item)
        elif isinstance(item, Shape):
            self._shapes.append(item)
        else:
            raise ValueError("only Skin or Shape instances may be added to a Node")

    def _compile_impl(self, camera_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Delegate compilation to sub-shapes. We apply the Node's transform to the camera matrix
        to apply it globally to sub-shapes.
        """
        if not self._shapes:
            return np.empty(shape=(0, 2, 3)), np.empty(shape=(0, 3, 3))

        segment_set = []
        face_set = []
        for shape in self._shapes:
            segments, faces = shape.compile(camera_matrix @ self.transform)
            segment_set.append(segments)
            face_set.append(faces)

        return np.vstack(segment_set), np.vstack(face_set)
