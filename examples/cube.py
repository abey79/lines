import math

import matplotlib.pyplot as plt

from lines import Scene, Cube


def main():
    # Setup the scene
    scene = Scene()
    scene.add(Cube())
    scene.look_at((2, 1, 1.5), (0, 0, 0))
    scene.perspective(50, 0.1, 10)

    # Render the scene
    mls = scene.render()

    # Plot the scene
    for ls in mls:
        plt.plot(*ls.xy, "k-", solid_capstyle="round")
    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
