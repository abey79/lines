"""
This example replicates Michael Fogleman's city drawing made with his ln library.
"""
import random

import matplotlib.pyplot as plt
from shapely.geometry import MultiLineString

from lines import Scene, StrippedCube


def plot_mls(mls: MultiLineString) -> None:
    for ls in mls:
        plt.plot(*ls.xy, "k-", solid_capstyle="round")
    plt.axis("off")
    plt.axis("tight")
    plt.axis("equal")
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()


def main():
    scene = Scene()

    for i in range(-12, 13, 2):
        for j in range(-12, 13, 2):
            h = 1 + random.random() * 5
            c = StrippedCube(scale=(1, 1, h), translate=(i, j, h / 2))
            scene.add(c)

    scene.look_at((1.1, 0.8, 8.2), (0, 0.2, 0))
    scene.perspective(90, 0.1, 10)

    mls = scene.render()
    plot_mls(mls)


if __name__ == "__main__":
    main()
