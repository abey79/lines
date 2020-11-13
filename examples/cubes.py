import timeit

from lines import Cube, Node, Scene, SilhouetteSkin

# FIXME: renderer v2 has a hole in the silhouette


def main(silhouette: bool = True):
    scene = Scene()
    n = Node()

    for i in range(0, 3, 2):
        for j in range(0, 1, 2):
            h = 1
            c = Cube(scale=(1, 1, h), translate=(i, j, h / 2))
            n.add(c)

    if silhouette:
        n.add(SilhouetteSkin(keep_segments=False))
    scene.add(n)
    scene.look_at((25, 22.5, 15), (0, 0, 0))
    scene.perspective(50, 0.1, 10)
    rs = scene.render("v2", merge_lines=False)
    print(rs.find_indices())
    rs.show(show_faces=23, show_grid=True, show_axes=True)


if __name__ == "__main__":
    print(f"Execution time: {timeit.timeit(lambda : main(), number=1)}")
