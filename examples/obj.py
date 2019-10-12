import math
import timeit
from typing import Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import MultiLineString
import pywavefront

from lines import Scene, PolyShape


def plot_mls(mls: MultiLineString) -> None:
    for ls in mls:
        plt.plot(*ls.xy, "k-", solid_capstyle="round", lw=1)
    plt.axis("equal")
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()


class OBJShape(PolyShape):
    def __init__(self, file_name, **kwargs):
        model = pywavefront.Wavefront(file_name, create_materials=True, collect_faces=True)

        # normalize everything
        vertices = np.array(model.vertices)
        vertices = vertices / np.max(np.max(vertices))

        # collect all segment
        faces = model.mesh_list[0].faces
        segs = set()

        def sort_int(a: int, b: int) -> Tuple[int, int]:
            return (b, a) if a < b else (a, b)

        for face in faces:
            segs.add(sort_int(face[0], face[1]))
            segs.add(sort_int(face[1], face[2]))
            segs.add(sort_int(face[2], face[0]))

        super().__init__(vertices, list(segs), faces, **kwargs)


def main():
    # TODO: empty scene crashes
    # TODO: cow.obj crashes with
    #       scene.look_at((25, 22.5, 15), (0, 0, 0))
    #       scene.perspective(50 / 180 * math.pi, 0.1, 10)

    scene = Scene()

    obj = OBJShape("deer.obj")
    obj.rotate_x(-math.pi / 2)
    obj.rotate_z(-math.pi / 2)
    scene.add(obj)
    scene.look_at((2, 2, 2), (0, 0, 0.5))
    scene.perspective(25 / 180 * math.pi, 0.1, 10)

    mls = scene.render()
    plot_mls(mls)


if __name__ == "__main__":
    print(f"Execution time: {timeit.timeit(main, number=1)}")
