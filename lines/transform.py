import math
from typing import Union, Sequence

import numpy as np


class Transform:
    """
    Wrapper on a homogeneous 4x4 transformation matrix.
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
        Initialize a transform defaulting to identity. If parameters are passed, the
        corresponding transform a applied. If multiple parameters are passed, the transforms
        are applied in the order listed here:
        :param rotate_x: rotation around x axis (rad)
        :param rotate_y: rotation around y axis (rad)
        :param rotate_z:
        :param translate:
        """
        self._matrix = np.identity(4)
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
            self._matrix[i][i] *= scale_vec[i]

    def rotate_x(self, angle: float) -> None:
        s, c = math.sin(angle), math.cos(angle)
        r = np.array(([1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]))
        self._matrix = r @ self._matrix

    def rotate_y(self, angle: float) -> None:
        s, c = math.sin(angle), math.cos(angle)
        r = np.array(([c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]))
        self._matrix = r @ self._matrix

    def rotate_z(self, angle: float) -> None:
        s, c = math.sin(angle), math.cos(angle)
        r = np.array(([c, s, 0, 0], [-s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]))
        self._matrix = r @ self._matrix

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

        self._matrix[0][3] += v_x
        self._matrix[1][3] += v_y
        self._matrix[2][3] += v_z

    def get(self) -> np.ndarray:
        return self._matrix

    def __call__(self) -> np.ndarray:
        return self.get()
