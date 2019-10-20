import logging
from typing import Union

import numpy as np
import matplotlib.pyplot as plt
import shapely.ops
from shapely.geometry import MultiLineString, asMultiLineString

import lines


class RenderedScene:
    """
    Instances of lines.RenderedScene are returned by the lines.Scene.render() function. They
    contain the rendered 2D vector data as well as intermediate 3D segments and faces. This
    class implements various display and export methods.
    """

    def __init__(
        self,
        scene: "lines.Scene",
        vectors: np.ndarray,
        merge_lines: bool = True,
        projected_segments: Union[np.ndarray, None] = None,
        projected_faces: Union[np.ndarray, None] = None,
    ):
        """
        Initialized a lines.RenderedScene object. It should not be called by client code as
        lines.Scene is exclusively responsible to create instances.
        """
        self._scene = scene
        self._vectors = vectors
        self._projected_segments = projected_segments
        self._projected_faces = projected_faces
        self._mls = None

        if merge_lines:
            self.merge_lines()

    @property
    def vectors(self) -> np.ndarray:
        """
        Return the vector data as a Nx2x2 NumPy array.
        :return: Nx2x2 vector data
        """
        return self._vectors

    @property
    def mls(self) -> MultiLineString:
        """
        Return vector data as a shapely.geometry.MultiLineString object
        :return: the MultiLineString object
        """
        if self._mls is None:
            self._mls = asMultiLineString(self._vectors)
        return self._mls

    def merge_lines(self) -> None:
        """
        Optimise the vector data using shapely.ops.linemerge(). This may be useful of the
        rendering is to be drawn by a plotter.
        """
        tot_seg_count = len(self.mls)
        merged = shapely.ops.linemerge(self.mls)
        if merged.geom_type == "LineString":
            self._mls = MultiLineString([merged])
        elif merged.geom_type == "MultiLineString":
            self._mls = merged
        else:
            logging.warning(
                f"inconsistent geom_type {merged.geom_type} returned by linemerge()"
            )
        logging.info(f"optimized {tot_seg_count} to {len(self._mls)} line strings")

    def show(
        self, black: bool = True, show_axes: bool = False, show_grid: bool = False
    ) -> None:
        """
        Display the rendered scene with matplotlib.
        :param black: if True (default), the scene is rendered in black, otherwise let
        :param show_axes: if True, axes are displayed
        :param show_grid: if True, grid are displayed
        matplotlib assign a per-segment colour
        """
        for ls in self.mls:
            plt.plot(*ls.xy, "k-" if black else "-", solid_capstyle="round")
        plt.axis("equal")
        if not show_axes:
            plt.axis("off")
        if show_grid:
            plt.grid("on")
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.show()

    def save(self, file_name: str) -> None:
        """
        Export the scene to a file. The provided file name's extension is used to determine
        the file format.
        Supported format: SVG, PNG, JPG
        :param file_name: file name
        """
        # TODO: implement this
        pass
