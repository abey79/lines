# _lines_ – the plotter-friendly 3D engine

_lines_ is a vector 3D engine writen in Python that outputs vector data compatible with plotter such as the
[AxiDraw V3](https://axidraw.com). Renders are constructed with shapes made of 3D segments (e.g. the edges of a cube) and
faces that make the shape opaque to objects behind. Segments and faces are initially projected in camera space using
linear algebra, much in the same way as traditional raster-based 3D engines (OpenGL, etc.). Then, instead of generating
a bitmap image, 2.5D geometrical computation is applied to "hide" (or, rather, "cut off") the segments that should be
hidden behind faces. The result is a lean set of vector data well suited for 2D plotters.

This tool has been inspired by the excellent [ln](https://github.com/fogleman/ln) project from Michael Fogleman.  


## Getting Started

### Installation

To play with _lines_, you need to checkout the source and install the dependencies in a virtual environment, for
example with the following steps:

```bash
$ git clone https://github.com/abey79/lines.git
$ cd lines
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

### Running examples

Example can then be run by executing the corresponding file:

```bash
$ cd examples
$ python cube.py
```

You should see this image rendered in a matplotlib window:

<img src="https://i.imgur.com/z0jEq33.png" alt="cube" width=200>


### Running tests

This project uses [pytest](https://docs.pytest.org), so running tests is a matter of executing this command in the
project root directory:

```bash
$ pytest
```


## Documentation

### Scene

Creating a `Scene` object is the starting point to using _lines_. Scenes are used to collect shapes to render, and 
configure the camera.

```python
from lines import Scene

scene = Scene()
scene.look_at(
    (2, 2, 2),  # this is where the camera eye is
    (0, 0, 0),  # this is where the camera is looking
)
scene.perspective(50, 0.1, 20)  # setup a perspective projection with 50° FOV
rendered_scene = scene.render()  # don't expect much of this until you add some shapes to the scene
```

The `render()` member function returns an instance of the`RenderedScene` class. This object contains among other things
the result of the rendering process (a collection of vector data). Its primary purpose is to display and export the 
rendered data.

```python
rendered_scene.show()  # use matplotlib to display the rendered scene
rendered_scene.save('my_render.svg')  # export the rendered scene to a svg file
```

### Shapes

Shapes are 3D objects that you can add to the scene. Here is how to add a cube to the scene for example:

```python
from lines import Cube

cube = Cube()
scene.add(cube)
```

A `Cube` is, well, a simple cube with unit side length and centered on coordinates (0, 0, 0). You probably want cubes
of various sizes, orientation and locations though. You can achieve this by acting on the shape's transform matrix, 
which can be done easily with the shape API:

```python
cube.scale(2)  # the cube is now twice bigger
cube.rotate_z(30)  # the cube is now rotated of 30° around the vertical axis
cube.translate(10, 2, 1)  # the cube is now elsewhere
```

You can scale on a per-axis basis as well:

```python
cube.scale(1, 1, 10)  # this will now look like a skyscraper
```

**Important:** scaling and rotations are always operate around coordinate (0, 0, 0) and may lead to unexpected results
if applied after a translation. The easiest is generally to first scale the object to its final size, then rotate it
and, finally, translate it to its intended location in space.

For convenience, shapes' constructors accept optional transform keyword parameters:

```python
cube = Cube(scale=2, rotate_z=30, translate=(10, 2, 1))
```

In this case, scaling is always applied before rotation, which is applied before translation, regardless of the order
of the parameters.

Various shapes are readily available and it is easy to create new ones:

<img src="https://i.imgur.com/ZggktLI.png" alt="cube, pyramid, cylinder" width="400px">

_TODO: list available shapes_


### Nodes

_to be completed..._


### Skins

_to be completed..._


### Rendered scenes

_to be completed..._


## Built With

* [NumPy](https://numpy.org) - Most of the data is stored and processed as NumPy arrays
* [Shapely](https://github.com/Toblerity/Shapely) - Used for most of the geometry computation
* [svgwrite](https://github.com/mozman/svgwrite) - SVG output


## Contributing

Pull requests are welcome.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
