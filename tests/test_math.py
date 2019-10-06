import numpy as np
import pytest

from lines.math import vertices_matmul
from lines.tables import CUBE_VERTICES, CUBE_SEGMENTS, CUBE_FACES


def test_vertices_matmul_empty():
    # Empty input is accepted, and returns empty output
    m = np.random.rand(3, 3)
    i = np.reshape(np.array([]), (0, 3))
    o = vertices_matmul(i, m)

    assert len(o) == 0
    assert np.all(o == i)


@pytest.mark.parametrize("n", (range(1, 5)))
def test_vertices_matmul_single_vertex(n):
    # a single vertex is accepted and is multiplied by a matrix
    m = np.random.rand(n, n)
    i = np.random.rand(n)
    o = vertices_matmul(i, m)
    assert np.all(m @ i == o)


@pytest.mark.parametrize("n", (range(1, 5)))
def test_vertices_matmul_single_vertex_array(n):
    # an array of a  single vertex is accepted and is multiplied by a matrix
    m = np.random.rand(n, n)
    i = np.random.rand(1, n)
    o = vertices_matmul(i, m)
    expected = m @ i.reshape((n,))
    assert np.all(expected == o)


def test_vertices_matmul_non_square_matrix():
    # non square matrices are not accepted and throw ValueError
    m = np.random.rand(3, 4)
    i = np.random.rand(1, 3)

    with pytest.raises(ValueError):
        vertices_matmul(i, m)


@pytest.mark.parametrize(("nm", "ni"), ((3, 4), (4, 3), (1, 5), (5, 1)))
def test_vertices_matmul_single_vertex_dim_mismatch(nm, ni):
    # when a single vertex is passed, mismatching dimension throw ValueError
    m = np.random.rand(nm, nm)
    i = np.random.rand(1, ni)
    with pytest.raises(ValueError):
        vertices_matmul(i, m)


def test_vertices_matmul_two_dimensions():
    segs = np.array(CUBE_VERTICES)
    m = np.random.rand(3, 3)
    o = vertices_matmul(segs, m)

    for i in range(len(segs)):
        assert np.all(o[i] == m @ segs[i])


def test_vertices_matmul_three_dimensions():
    vert = np.array(CUBE_VERTICES)
    s_idx = np.reshape(np.array(CUBE_SEGMENTS, dtype=np.uint32), (len(CUBE_SEGMENTS), 2))
    f_idx = np.reshape(np.array(CUBE_FACES, dtype=np.uint32), (len(CUBE_FACES), 3))
    segs = vert[s_idx]
    faces = vert[f_idx]

    m = np.random.rand(3, 3)
    o_segs = vertices_matmul(segs, m)
    o_faces = vertices_matmul(faces, m)

    for i in range(len(segs)):
        for j in range(2):
            assert np.all(o_segs[i][j] == m @ segs[i][j])

    for i in range(len(faces)):
        for j in range(3):
            assert np.all(o_faces[i][j] == m @ faces[i][j])
