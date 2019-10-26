from lines import Scene, Cube, Node, Cylinder


def main():
    # Setup the scene
    scene = Scene()
    n = Node()
    n.add(Cube(scale=(10, 20, 6)))
    for i in [2.5, -2.5]:
        for j in [-7.5, -2.5, 2.5, 7.5]:
            n.add(Cylinder(scale=(1.5, 1.5, 1), translate=(i, j, 3.5)))
    scene.add(n)
    scene.look_at((30, 30, 20), (0, 0, 0))
    scene.perspective(50, 0.1, 10)

    # Render and display the scene
    scene.render().show()


if __name__ == "__main__":
    main()
