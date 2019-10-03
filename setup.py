from setuptools import setup, find_packages


with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="lines",
    version="0.1.0",
    description="3D to plotter line engine",
    long_description=readme,
    author="Antoine Beyeler",
    url="https://github.com/abey79/lines",
    license=license,
    packages=find_packages(exclude=("examples", "tests")),
)
