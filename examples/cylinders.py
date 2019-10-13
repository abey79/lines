import math
import timeit

import matplotlib.pyplot as plt
from shapely.geometry import MultiLineString

from lines import Scene
from lines.poly_shapes import Cylinder


def plot_mls(mls: MultiLineString) -> None:
    for ls in mls:
        plt.plot(*ls.xy, "k-", solid_capstyle="round")
    plt.axis("equal")
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()


def main():
    scene = Scene()
    scene.add(Cylinder(vertical_segs=True, silhouette=False, translate=(0, -3, 0)))
    scene.add(Cylinder(vertical_segs=False, silhouette=False))
    scene.add(Cylinder(vertical_segs=False, silhouette=True, translate=(0, 3, 0)))

    scene.look_at((12, 0, 4), (0, 0, 0))
    scene.perspective(50, 0.1, 20)

    mls = scene.render()
    plot_mls(mls)


if __name__ == "__main__":
    print(f"Execution time: {timeit.timeit(main, number=1)}")
