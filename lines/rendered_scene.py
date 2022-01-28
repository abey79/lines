import logging
from typing import Sequence, Set, Union

import matplotlib.pyplot as plt
import numpy as np
import shapely.ops
import svgwrite
from shapely.geometry import MultiLineString

import lines


class RenderedScene:
    """Instances of lines.RenderedScene are returned by the lines.Scene.render() function. They
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
            self._mls = MultiLineString(self._vectors.tolist())
        return self._mls

    def merge_lines(self) -> None:
        """
        Optimise the vector data using shapely.ops.linemerge(). This may be useful of the
        rendering is to be drawn by a plotter.
        """
        tot_seg_count = len(self.mls.geoms)
        merged = shapely.ops.linemerge(self.mls)
        if merged.geom_type == "LineString":
            self._mls = MultiLineString([merged])
        elif merged.geom_type == "MultiLineString":
            self._mls = merged
        else:
            logging.warning(
                f"inconsistent geom_type {merged.geom_type} returned by linemerge()"
            )
        logging.info(f"optimized {tot_seg_count} to {len(self._mls.geoms)} line strings")

    def show(
        self,
        black: bool = True,
        show_axes: bool = False,
        show_grid: bool = False,
        show_hidden: Union[int, Sequence[int], bool] = False,
        show_faces: Union[int, Sequence[int], bool] = False,
        highlighted_segments: Union[int, Sequence[int], None] = None,
    ) -> None:
        """
        Display the rendered scene with matplotlib.
        :param black: if True (default), the scene is rendered in black, otherwise let
            matplotlib assign a per-segment colour
        :param show_axes: if True, axes are displayed
        :param show_grid: if True, grid are displayed
        :param show_hidden: if True, hidden segment are displayed with dotted lines
        :param show_faces: if True, faces are displayed in transparent green, alternatively you
            may pass one or more int to specify which face to plot
        :param highlighted_segments: one or more index of segment(s) to highlight (note that
            ``merge_lines()`` may drastically affect the length and order of segment list as
            those are merged into line strings).
        """

        if show_faces is not False:
            if type(show_faces) == int:
                indices = [show_faces]
            elif type(show_faces) == bool:
                indices = list(range(len(self._projected_faces)))
            else:
                indices = show_faces

            for face in self._projected_faces[indices, :, :]:
                plt.plot(face[[0, 1, 2, 0], 0], face[[0, 1, 2, 0], 1], "g-", lw=0.5, alpha=0.5)
                plt.fill(face[[0, 1, 2, 0], 0], face[[0, 1, 2, 0], 1], "g", alpha=0.2)
        if show_hidden is not False:
            if type(show_hidden) == int:
                indices = [show_hidden]
            elif type(show_hidden) == bool:
                indices = list(range(len(self._projected_segments)))
            else:
                indices = show_hidden
            for seg in self._projected_segments[indices, :, :]:
                plt.plot(seg[:, 0], seg[:, 1], "k:", lw=0.5)

        if highlighted_segments is None:
            highlighted_segments = []
        elif type(highlighted_segments) == int:
            highlighted_segments = [highlighted_segments]
        for i, ls in enumerate(self.mls.geoms):
            kwargs = {}
            if i in highlighted_segments:
                kwargs = {"lw": 1.5}
                style = "r.-"
            elif black:
                style = "k-"
            else:
                style = "-"
            plt.plot(*ls.xy, style, solid_capstyle="round", **kwargs)
        plt.axis("equal")
        if not show_axes:
            plt.axis("off")
        if show_grid:
            plt.grid("on")
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.show()

    def save(self, file_name: str, **kwargs) -> None:
        """
        Export the scene to a file. The provided file name's extension is used to determine
        the file format.
        Supported format: SVG, PNG, JPG
        :param file_name: file name
        """

        if file_name.endswith(".svg"):
            self.save_svg(file_name, **kwargs)
        else:
            # TODO: support more format
            logging.warning(f"Unsupported file format for '{file_name}'")

    def save_svg(self, file_name: str, **kwargs) -> None:
        """
        TODO: hacked dimensions
        """
        dwg = svgwrite.Drawing(file_name, size=(1000, 1000), profile="tiny", debug=False)

        for line in self.mls.geoms:
            dwg.add(
                dwg.path(
                    "M"
                    + " L".join(f"{500 * (x + 1)},{500 * (1 - y)}" for x, y in line.coords),
                    fill="none",
                    stroke="black",
                    **kwargs,
                )
            )

        dwg.save()

    def find_indices(self) -> Set[int]:
        """
        **(Debug/test function)** Find the indices in the list of projected segment
        corresponding to the final list of displayed segment. This function is mainly used
        for debugging purposes and tests.
        :return: set of indices
        """
        idx = set()
        for ls in self.mls:
            segment = np.array(ls)
            segments = np.tile(
                segment.reshape((1, 2, 2)), (len(self._projected_segments), 1, 1)
            )

            res = np.logical_or(
                np.all(
                    np.all(np.isclose(self._projected_segments[:, :, 0:2], segments), axis=-1),
                    axis=-1,
                ),
                np.all(
                    np.all(
                        np.isclose(self._projected_segments[:, [1, 0], 0:2], segments), axis=-1
                    ),
                    axis=-1,
                ),
            )

            if np.sum(res) == 1:
                idx.add(int(np.argmax(res)))

        return idx
