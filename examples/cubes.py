import math
import random
import timeit

import matplotlib.pyplot as plt
from shapely.geometry import MultiLineString

from lines import Cube, Scene, Node, SilhouetteSkin


def plot_mls(mls: MultiLineString) -> None:
    for ls in mls:
        plt.plot(*ls.xy, "k-", solid_capstyle="round", lw=1)
    plt.axis("equal")
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()


def main(silhouette: bool = False):
    scene = Scene()
    n = Node()

    for i in range(-10, 11, 2):
        for j in range(-10, 11, 2):
            h = 1 + random.random() * 2
            c = Cube(scale=(1, 1, h), translate=(i, j, h / 2))
            n.add(c)

    if silhouette:
        n.add(SilhouetteSkin(keep_segments=False))
    scene.add(n)
    scene.look_at((25, 22.5, 15), (0, 0, 0))
    scene.perspective(50, 0.1, 10)

    mls = scene.render()
    plot_mls(mls)


if __name__ == "__main__":
    print(f"Execution time: {timeit.timeit(lambda : main(True), number=1)}")
