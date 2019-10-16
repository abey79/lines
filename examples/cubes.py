import random
import timeit

from lines import Cube, Scene, Node, SilhouetteSkin


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
    scene.render().show()


if __name__ == "__main__":
    print(f"Execution time: {timeit.timeit(lambda : main(), number=1)}")
