import random

import pytest

from lines import Shape


@pytest.fixture(scope="session")
def root_directory(request):
    return str(request.config.rootdir)


@pytest.fixture
def random_transform():
    u = random.uniform
    shape = Shape(
        scale=(u(0.1, 10), u(0.1, 10), u(0.1, 10)),
        rotate_x=u(-180, 180),
        rotate_y=u(-180, 180),
        rotate_z=u(-180, 180),
        translate=(u(-10, 10), u(-10, 10), u(-10, 10)),
    )
    return shape.transform
