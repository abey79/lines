import math
import timeit

import matplotlib.pyplot as plt
from shapely.geometry import MultiLineString

from lines import PolyShape, Cube, Transform, Scene


def plot_polyshape(ps: PolyShape) -> None:
    segments, _ = ps.compile()
    for seg in segments:
        plt.plot(*((seg[0][i], seg[1][i]) for i in range(2)), "k-")
    plt.axis("equal")
    plt.show()


def plot_mls(mls: MultiLineString) -> None:
    for ls in mls:
        plt.plot(*ls.xy, "k-", solid_capstyle="round")
    plt.axis("equal")
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()


def main():
    scene = Scene()

    for i in range(-4, 5, 2):
        for j in range(-4, 5, 2):
            c = Cube()
            h = 1  # + random.random() * 2
            c.transform(Transform(scale=(1, 1, h), translate=(i, j, h / 2)))
            scene.add(c)

    scene.look_at((15, 12.5, 5), (0, 0, 0))
    scene.perspective(50 / 180 * math.pi, 0.1, 10)

    mls = scene.render()
    plot_mls(mls)


if __name__ == "__main__":
    # timeit.timeit(main, number=1)
    main()
