import math

import numpy as np
import pytest

from lines import Shape


@pytest.fixture(params=[Shape.rotate_x, Shape.rotate_y, Shape.rotate_z])
def rotate_xyz(request):
    return request.param


def test_init():
    # Bare transform should be identity
    shape = Shape()
    v = np.array([4, 3, 2, 1])
    assert np.array_equal(shape.transform, np.identity(4))
    assert np.array_equal(v, shape.transform @ v)


def test_scale_args():
    shape1 = Shape()
    shape1.scale(5)
    shape2 = Shape()
    shape2.scale(5, 5, 5)
    shape3 = Shape()
    shape3.scale([5, 5, 5])
    shape4 = Shape()
    shape4.scale(np.array([5, 5, 5]))
    assert np.array_equal(shape1.transform, shape2.transform)
    assert np.array_equal(shape1.transform, shape3.transform)
    assert np.array_equal(shape1.transform, shape4.transform)


def test_scale():
    shape = Shape(scale=[3, 2, 4])
    v = np.array([2, 3, 1, 1])

    assert np.array_equal(shape.transform @ v, np.array([6, 6, 4, 1]))


def test_translate_args():
    shape1 = Shape()
    shape1.translate([1, 2, 3])
    shape2 = Shape()
    shape2.translate(1, 2, 3)
    shape3 = Shape(translate=(1, 2, 3))
    shape4 = Shape(translate=np.array([1, 2, 3]))
    assert np.array_equal(shape1.transform, shape2.transform)
    assert np.array_equal(shape1.transform, shape3.transform)
    assert np.array_equal(shape1.transform, shape4.transform)


def test_translate_bad_args():
    shape = Shape()
    with pytest.raises(ValueError):
        shape.translate(1, 2)
    with pytest.raises(ValueError):
        shape.translate([1, 2])
    with pytest.raises(ValueError):
        shape.translate("a", 2, 3)
    with pytest.raises(ValueError):
        shape.translate([1, 2], 3)


def test_translate():
    v = np.array([1, 2, 3, 1])
    shape = Shape()
    shape.translate(3, 2, 5)
    assert np.array_equal(shape.transform @ v, np.array([4, 4, 8, 1]))


def test_rotate_args(rotate_xyz):
    shape1 = Shape(**{rotate_xyz.__name__: 2.5})
    shape2 = Shape()
    rotate_xyz(shape2, 2.5)
    assert np.array_equal(shape1.transform, shape2.transform)


def test_rotate_identity(rotate_xyz):
    shape1 = Shape()
    rotate_xyz(shape1, 10)
    rotate_xyz(shape1, -10)
    assert np.allclose(shape1.transform, Shape().transform)

    shape2 = Shape()
    rotate_xyz(shape2, 2 * math.pi)
    assert np.allclose(shape2.transform, Shape().transform)


def test_rotate_norm(rotate_xyz):
    shape = Shape()

    rotate_xyz(shape, 3)
    v = np.array((3, 5, 2, 1))
    v_r = shape.transform @ v

    assert pytest.approx(np.linalg.norm(v)) == pytest.approx(np.linalg.norm(v_r))
