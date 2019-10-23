from lines import Scene
from lines import SegmentShape, TriangleShape


def main():
    scene = Scene()
    scene.add(SegmentShape((0, -2, 1), (0, 1, 1)))
    scene.add(SegmentShape((1, -.1, 0.5), (-3, -.1, 0.5)))  # behind
    scene.add(SegmentShape((-3, .1, 0.8), (1, .1, 0.8)))

    scene.add(TriangleShape((1, 0, 0), (-1, 0, 0), (0, 0, 2)))

    scene.look_at((2, 2, 1), (0, 0, 1))
    scene.perspective(90, 0.1, 10)
    scene.render().show(show_axes=True, show_grid=True)


if __name__ == "__main__":
    main()
