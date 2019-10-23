from lines import Scene, Cube, Pyramid, Cylinder


def main():
    # Setup the scene
    scene = Scene()
    scene.add(Cube(translate=(2, 0, 0)))
    scene.add(Pyramid())
    scene.add(Cylinder(scale=(0.5, 0.5, 1), translate=(-2, 0, 0)))
    scene.look_at((2, 6, 1.5), (0, 0, 0))
    scene.perspective(70, 0.1, 10)

    # Render and display the scene
    scene.render().show(show_hidden=True)


if __name__ == "__main__":
    main()
