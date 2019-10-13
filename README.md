# _lines_ – the plotter-friendly 3D engine

_lines_ is a vector 3D engine that output vector data that is compatible with plotter such as the [AxiDraw V3](https://axidraw.com). This project is similar to – and much inspired from – the [ln](https://github.com/fogleman/ln) project of Michael Fogleman.


## Getting Started

To play with _lines_, you will need to checkout the source and install the dependencies in a virtual environment with the following steps:

```bash
$ git clone https://github.com/abey79/lines.git
$ cd lines
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

Example can then be run by executing the corresponding file:

```bash
$ cd examples
$ python cube.py
```

You should see this image rendered in a matplotlib window:

<img src="https://i.imgur.com/z0jEq33.png" width=200>


## Running the tests

This project uses [pytest](https://docs.pytest.org), so running tests is a matter of execuing this command in the project root directory:

```bash
$ pytest
```


## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Merge requests are welcome.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
