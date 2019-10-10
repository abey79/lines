import numpy as np
import pytest

from lines.math import vertices_matmul, segments_parallel_to_face, ParallelType, mask_segments
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


def test_segments_parallel_to_face_single_segment():
    p0 = np.array([1, 2, 4])
    n = np.array([0, 0, 5])
    segments = np.array([[(100, 100, 100), (200, 300, 100)]])
    assert np.all(
        segments_parallel_to_face(segments, p0, n)
        == np.array([ParallelType.PARALLEL_FRONT.value])
    )


def test_segments_parallel_to_face_ground_plane():
    # ground plane
    p0 = np.array([0, 0, 0])
    n = np.array([0, 0, 3])

    segments = np.array(
        [
            [(0, 0, 0), (1, 0, 0)],  # coincident
            [(0, 0, 1), (1, 0, 1)],  # front
            [(0, 0, -1), (1, 0, -1)],  # back
            [(0, 0, -1), (1, 0, 1)],  # not parallel
        ]
    )

    assert np.all(
        segments_parallel_to_face(segments, p0, n)
        == np.array(
            [
                ParallelType.PARALLEL_COINCIDENT.value,
                ParallelType.PARALLEL_FRONT.value,
                ParallelType.PARALLEL_BACK.value,
                ParallelType.NOT_PARALLEL.value,
            ]
        )
    )


def test_segments_parallel_to_face_z_parallel_plane():
    with pytest.raises(ValueError):
        segments_parallel_to_face(
            np.random.rand(10, 2, 3), np.random.rand(3), np.array([3, 4, 0])
        )


@pytest.mark.parametrize(
    ("seg", "p0", "n"),
    [
        (np.random.rand(10, 2, 4), np.random.rand(3), np.array([1, 2, 3])),
        (np.random.rand(10, 2, 3), np.random.rand(4), np.array([1, 2, 3])),
        (np.random.rand(10, 2, 4), np.random.rand(3), np.array([1, 2, 3, 4])),
    ],
)
def test_segments_parallel_to_face_wrong_dimensions(seg, p0, n):
    with pytest.raises(ValueError):
        segments_parallel_to_face(seg, p0, n)


@pytest.mark.parametrize(
    ("seg", "mask"),
    [
        (np.random.rand(10, 2, 4), np.random.rand(5, 2)),
        (np.random.rand(10, 2, 3), np.random.rand(5, 3)),
        (np.random.rand(10, 3), np.random.rand(5, 2)),
        (np.random.rand(10, 2, 3), np.random.rand(5, 2, 2)),
    ],
)
def test_mask_segments_wrong_dimensions(seg, mask):
    with pytest.raises(ValueError):
        mask_segments(seg, mask)


def test_mask_segments_diff_then_intersect():
    segments = np.random.rand(500, 2, 3)
    mask = np.array([(0.1, 0.1), (0.2, 0.8), (0.8, 0.5)])

    seg_diff = mask_segments(segments, mask, True)
    assert len(seg_diff) > 0
    seg_diff_excl = mask_segments(seg_diff, mask, False)
    assert len(seg_diff_excl) == 0


def test_mask_segments_empty_results():
    pass


def test_mask_segments_empty_input():
    segments = np.array([]).reshape((0, 2, 3))
    mask = np.array([(0.5, -1), (0.5, 1), (10, 0)])

    assert np.all(mask_segments(segments, mask) == segments)


def test_mask_segment_single_segment():
    segments = np.array([[(0, 0, 0), (1, 0, 0)]])
    mask = np.array([(0.5, -1), (0.5, 1), (10, 0)])

    seg_diff = mask_segments(segments, mask, True)
    seg_excl = mask_segments(segments, mask, False)

    assert np.all(seg_diff == np.array([[(0, 0, 0), (0.5, 0, 0)]]))
    assert np.all(seg_excl == np.array([[(0.5, 0, 0), (1, 0, 0)]]))


def test_mask_segment_bug_disappearing_segment(root_directory):
    # Because of https://github.com/Toblerity/Shapely/issues/780, some z values are
    # randomly affected. This is test data extracted from a actual failing situation where
    # segments ended up missing because of the affected Z value.

    path = root_directory + "/tests/fixtures/"
    segments = np.load(path + "bug_segments.pickle", allow_pickle=True)
    face = np.load(path + "bug_face.pickle", allow_pickle=True)
    seg = np.load(path + "bug_seg_of_interest.pickle", allow_pickle=True)

    # the segment of interest "seg" should be present in "segments"
    assert np.any(np.all(np.all(segments == seg, axis=2), axis=1))

    results = mask_segments(segments, face, True)

    # the segment of interest should NOT be masked away with this particular mask
    assert np.any(np.all(np.all(results == seg, axis=2), axis=1))


# TODO: add tests for split_segments()