import math

import numpy as np
import pytest

from lines import Transform


@pytest.fixture(params=[Transform.rotate_x, Transform.rotate_y, Transform.rotate_z])
def rotate_xyz(request):
    return request.param


def test_init():
    # Bare transform should be identity
    t = Transform()
    v = np.array([4, 3, 2, 1])
    assert np.array_equal(t.get(), np.identity(4))
    assert np.array_equal(v, t.get() @ v)


def test_scale_args():
    t1 = Transform()
    t1.scale(5)
    t2 = Transform()
    t2.scale(5, 5, 5)
    t3 = Transform()
    t3.scale([5, 5, 5])
    t4 = Transform()
    t4.scale(np.array([5, 5, 5]))
    assert np.array_equal(t1.get(), t2.get())
    assert np.array_equal(t1.get(), t3.get())
    assert np.array_equal(t1.get(), t4.get())


def test_scale():
    t = Transform(scale=[3, 2, 4])
    v = np.array([2, 3, 1, 1])

    assert np.array_equal(t.get() @ v, np.array([6, 6, 4, 1]))


def test_translate_args():
    t1 = Transform()
    t1.translate([1, 2, 3])
    t2 = Transform()
    t2.translate(1, 2, 3)
    t3 = Transform(translate=(1, 2, 3))
    t4 = Transform(translate=np.array([1, 2, 3]))
    assert np.array_equal(t1.get(), t2.get())
    assert np.array_equal(t1.get(), t3.get())
    assert np.array_equal(t1.get(), t4.get())


def test_translate_bad_args():
    t = Transform()
    with pytest.raises(ValueError):
        t.translate(1, 2)
    with pytest.raises(ValueError):
        t.translate([1, 2])
    with pytest.raises(ValueError):
        t.translate("a", 2, 3)
    with pytest.raises(ValueError):
        t.translate([1, 2], 3)


def test_translate():
    v = np.array([1, 2, 3, 1])
    t = Transform()
    t.translate(3, 2, 5)
    assert np.array_equal(t.get() @ v, np.array([4, 4, 8, 1]))


def test_rotate_args(rotate_xyz):
    t1 = Transform(**{rotate_xyz.__name__: 2.5})
    t2 = Transform()
    rotate_xyz(t2, 2.5)
    assert np.array_equal(t1.get(), t2.get())


def test_rotate_identity(rotate_xyz):
    t1 = Transform()
    rotate_xyz(t1, 10)
    rotate_xyz(t1, -10)
    assert np.allclose(t1.get(), Transform().get())

    t2 = Transform()
    rotate_xyz(t2, 2 * math.pi)
    assert np.allclose(t2.get(), Transform().get())


def test_rotate_norm(rotate_xyz):
    t = Transform()

    rotate_xyz(t, 3)
    v = np.array((3, 5, 2, 1))
    v_r = t.get() @ v

    assert pytest.approx(np.linalg.norm(v)) == pytest.approx(np.linalg.norm(v_r))
