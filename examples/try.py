import math
import matplotlib.pyplot as plt
from shapely.geometry import MultiLineString

from lines import PolyShape, Cube, Transform, Scene


class SegmentShape(PolyShape):
    def __init__(self, p0, p1):
        super().__init__([p0, p1], [(0, 1)], [])


class TriangleShape(PolyShape):
    def __init__(self, p0, p1, p2):
        super().__init__([p0, p1, p2], [(0, 1), (1, 2), (2, 0)], [(0, 1, 2)])


def plot_mls(mls: MultiLineString) -> None:
    for ls in mls:
        plt.plot(*ls.xy, "-")
    plt.axis("equal")
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    plt.show()


def main():
    scene = Scene()
    scene.add(SegmentShape((0, -2, 1), (0, 1, 1)))
    scene.add(SegmentShape((1, -1, 0.5), (-3, -1, 0.5)))  # behind

    scene.add(TriangleShape((1, 0, 0), (-1, 0, 0), (0, 0, 2)))

    scene.look_at((2, 2, 1), (0, 0, 1))
    scene.perspective(90, 0.1, 10)
    mls = scene.render()

    plot_mls(mls)


if __name__ == "__main__":
    main()
