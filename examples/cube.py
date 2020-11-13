from lines import Cube, Scene


def main():
    # Setup the scene
    scene = Scene()
    scene.add(Cube())
    scene.look_at((2, 1, 1.5), (0, 0, 0))
    scene.perspective(50, 0.1, 10)

    # Render and display the scene
    scene.render().show(show_hidden=True)


if __name__ == "__main__":
    main()
