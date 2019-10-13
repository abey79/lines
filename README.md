# _lines_ – the plotter-friendly 3D engine

_lines_ is a vector 3D engine that output vector data compatible with plotter such as the
[AxiDraw V3](https://axidraw.com). Scenes are constructed with shapes made of 3D segments (e.g. the edges of a cube) and
faces that make the shape opaque to objects behind. Segments and faces are initially projected in camera space using
linear algebra, much in the same way as traditional raster-based 3D engines (OpenGL, etc.). Then, instead of generating
a bitmap image, 2.5D geometrical computation is applied to "hide" (or, rather, "cut off") the segments that should be
hidden behind faces. The result is a lean set of vector data well suited to 2D plotters.

This tool has been inspired by the excellent [ln](https://github.com/fogleman/ln) project by Michael Fogleman.


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

s = Scene()
s.look_at(
    (2, 2, 2),  # this is where the camera eye is
    (0, 0, 0),  # this is where the camera is looking
)
s.perspective(50, 0.1, 20)  # setup a perspective projection with 50° FOV
mls = s.render()  # don't expect much of this until you add some shapes to the scene
```

_TODO: describe what render() returns_

### Shapes

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
