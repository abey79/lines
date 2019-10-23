import timeit

from lines import Scene
from lines.poly_shapes import Cylinder


def main():
    scene = Scene()
    scene.add(Cylinder(vertical_segs=True, silhouette=False, translate=(0, -3, 0)))
    scene.add(Cylinder(vertical_segs=False, silhouette=False))
    scene.add(Cylinder(vertical_segs=False, silhouette=True, translate=(0, 3, 0)))

    scene.look_at((12, 0, 4), (0, 0, 0))
    scene.perspective(50, 0.1, 20)
    scene.render().show(show_hidden=True)


if __name__ == "__main__":
    print(f"Execution time: {timeit.timeit(main, number=1)}")
