import collections
import enum

import numpy as np
from shapely.geometry import Polygon, asLineString


def _validate_segments(segments: np.ndarray) -> None:
    if len(segments.shape) != 3 or segments.shape[1:] != (2, 3):
        raise ValueError(f"segments array has shape {segments.shape} instead of (N, 2, 3)")


def _validate_shape(a: np.ndarray, *args):
    if len(a.shape) != len(args):
        return False

    for s, v in zip(a.shape, args):
        if isinstance(v, int):
            if s != v:
                return False
        elif isinstance(v, collections.abc.Iterable):
            if not s in v:
                return False
        else:
            if v is not None:
                return False

    return True


def vertices_matmul(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """
    Apply a matrix multiplication to all vertices in the input array. The vertices coordinates
    are assumed to be stored in the input's last dimension.
    :param vertices: [d0 x ... x dN x M] N-dimensional array of M-sized vertices
    :param matrix: [M x M] matrix
    :return: transformed vertices (identical shape as input)
    """

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")

    if vertices.shape[-1] != matrix.shape[0]:
        raise ValueError(
            f"matrix dimension ({matrix.shape[0]}x{matrix.shape[1]}) does not match vertex "
            f"dimension ({vertices.shape[-1]})"
        )

    if len(vertices) == 0:
        return np.empty_like(vertices)

    if len(vertices.shape) == 1:
        return matrix @ vertices

    # vertices needs to be reshaped such that the last two dimensions are (..., N, 1)
    # then matmul can be applied as it broadcast the matrix on the last two dimension of the
    # other operand
    shape = vertices.shape
    output = matrix @ vertices.reshape((*shape, 1))
    output.shape = shape
    return output


# noinspection DuplicatedCode
def segments_outside_triangle_2d(segments: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """
    Compute which segments are outside the face, considering only 2D projection along Z axis.
    As a results, the input's Z data is disregarded and optional.
    :param segments: (M x 2 x 2-3) segments
    :param triangle: (3 x 2-3) face
    :return: Mx1 array of boolean, true for segments outside of the triangle
    """

    if len(segments.shape) != 3 or segments.shape[1] != 2 or not segments.shape[2] in (2, 3):
        raise ValueError(
            f"segments array has shape {segments.shape} instead of (N, 2, 2 or 3)"
        )

    if len(triangle.shape) != 2 or not triangle.shape[1] in (2, 3):
        raise ValueError(f"triangle array has shape {triangle.shape} instead of (3, 2 or 3)")

    """
    https://gamedev.stackexchange.com/a/21110
    if t0, t1 and t2 are all on the same side of line P0P1, return NOT INTERSECTING
    if P0 AND P1 are on the other side of line t0t1 as t2, return NOT INTERSECTING
    if P0 AND P1 are on the other side of line t1t2 as t0, return NOT INTERSECTING
    if P0 AND P1 are on the other side of line t2t0 as t1, return NOT INTERSECTING
    """

    p0 = segments[:, 0, 0:2]
    p1 = segments[:, 1, 0:2]
    t0 = triangle[0, 0:2]
    t1 = triangle[1, 0:2]
    t2 = triangle[2, 0:2]

    p0p1 = p1 - p0

    p0t0 = t0 - p0
    p0t1 = t1 - p0
    p0t2 = t2 - p0

    t0p0 = -p0t0
    t1p0 = -p0t1
    t2p0 = -p0t2

    t0p1 = p1 - t0
    t1p1 = p1 - t1
    t2p1 = p1 - t2

    t0t1 = t1 - t0
    t1t2 = t2 - t1
    t2t0 = t0 - t2

    t1t0 = -t0t1
    t2t1 = -t1t2
    t0t2 = -t2t0

    f1 = np.cross(t0t1, t0p0) * np.cross(t0t1, t0t2)
    f2 = np.cross(t0t1, t0t2) * np.cross(t0t1, t0p1)

    f3 = np.cross(t1t2, t1p0) * np.cross(t1t2, t1t0)
    f4 = np.cross(t1t2, t1t0) * np.cross(t1t2, t1p1)

    f5 = np.cross(t2t0, t2p0) * np.cross(t2t0, t2t1)
    f6 = np.cross(t2t0, t2t1) * np.cross(t2t0, t2p1)

    p0p1_cross_p0t1 = np.cross(p0p1, p0t1)
    f7 = np.cross(p0p1, p0t0) * p0p1_cross_p0t1
    f8 = p0p1_cross_p0t1 * np.cross(p0p1, p0t2)

    # some tolerance is accepted to ensure that face do not hide segment running behind
    # along the edge, as it regularly happens with SilhouetteSkin
    return (
        np.logical_and(f1 <= 1e-12, f2 <= 1e-12)
        | np.logical_and(f3 <= 1e-12, f4 <= 1e-12)
        | np.logical_and(f5 <= 1e-12, f6 <= 1e-12)
        | np.logical_and(f7 >= 1e-12, f8 >= 1e-12)
    )


# noinspection DuplicatedCode
def triangles_overlap_segment_2d(triangles: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """
    Compute which triangles overlap a segment, considering only 2D projection along Z axis.
    The input's Z data is disregarded and optional.
    :param triangles: (M x 3 x 2-3) triangles
    :param segment: (2 x 2-3) segment
    :return: Mx1 array of boolean, true faces overlapping the segment
    """

    if (
        len(triangles.shape) != 3
        or triangles.shape[1] != 3
        or not triangles.shape[2] in (2, 3)
    ):
        raise ValueError(
            f"triangles array has shape {triangles.shape} instead of (N, 3, 2 or 3)"
        )

    if len(segment.shape) != 2 or segment.shape[0] != 2 or not segment.shape[1] in (2, 3):
        raise ValueError(f"segment array has shape {segment.shape} instead of (2, 2 or 3)")

    """
    https://gamedev.stackexchange.com/a/21110
    if t0, t1 and t2 are all on the same side of line P0P1, return NOT INTERSECTING
    if P0 AND P1 are on the other side of line t0t1 as t2, return NOT INTERSECTING
    if P0 AND P1 are on the other side of line t1t2 as t0, return NOT INTERSECTING
    if P0 AND P1 are on the other side of line t2t0 as t1, return NOT INTERSECTING
    """

    p0 = segment[0, 0:2]
    p1 = segment[1, 0:2]
    t0 = triangles[:, 0, 0:2]
    t1 = triangles[:, 1, 0:2]
    t2 = triangles[:, 2, 0:2]

    p0p1 = p1 - p0

    p0t0 = t0 - p0
    p0t1 = t1 - p0
    p0t2 = t2 - p0

    t0p0 = -p0t0
    t1p0 = -p0t1
    t2p0 = -p0t2

    t0p1 = p1 - t0
    t1p1 = p1 - t1
    t2p1 = p1 - t2

    t0t1 = t1 - t0
    t1t2 = t2 - t1
    t2t0 = t0 - t2

    t1t0 = -t0t1
    t2t1 = -t1t2
    t0t2 = -t2t0

    f1 = np.cross(t0t1, t0p0) * np.cross(t0t1, t0t2)
    f2 = np.cross(t0t1, t0t2) * np.cross(t0t1, t0p1)

    f3 = np.cross(t1t2, t1p0) * np.cross(t1t2, t1t0)
    f4 = np.cross(t1t2, t1t0) * np.cross(t1t2, t1p1)

    f5 = np.cross(t2t0, t2p0) * np.cross(t2t0, t2t1)
    f6 = np.cross(t2t0, t2t1) * np.cross(t2t0, t2p1)

    p0p1_cross_p0t1 = np.cross(p0p1, p0t1)
    f7 = np.cross(p0p1, p0t0) * p0p1_cross_p0t1
    f8 = p0p1_cross_p0t1 * np.cross(p0p1, p0t2)

    return (
        np.logical_and(f1 >= 0, f2 >= 0)
        | np.logical_and(f3 >= 0, f4 >= 0)
        | np.logical_and(f5 >= 0, f6 >= 0)
        | np.logical_and(f7 <= 0, f8 <= 0)
    )


class ParallelType(enum.Enum):
    """
    Enum to describe parallelism information.
    Front/back axis is z, and front means higher z value
    """

    NOT_PARALLEL = 0
    PARALLEL_COINCIDENT = 1
    PARALLEL_BACK = 2
    PARALLEL_FRONT = 3


def segments_parallel_to_face(
    segments: np.ndarray, p0: np.ndarray, n: np.ndarray
) -> np.ndarray:
    """
    Compare an array of segments to a plane defined by a point and a normal, and return an
    array containing ParallelType information for each segment. The plan must not be parallel
    to the z axis.
    :param segments: Mx2x3 array of segment
    :param p0: plane reference point (length 3)
    :param n: plane normal (length 3, does not need to be normalized)
    :return: length M array of ParallelType
    """

    _validate_segments(segments)

    if p0.shape != (3,) or n.shape != (3,):
        raise ValueError(f"p0 and n must be of length 3")

    if np.isclose(n[2], 0):
        raise ValueError(f"plane should not be parallel to z axis")

    # make sure n points up
    if n[2] < 0:
        n = -n

    cnt = len(segments)
    s0 = segments[:, 0]
    sv = np.diff(segments, axis=1).reshape(cnt, 3)

    output = np.empty(cnt)
    output.fill(ParallelType.NOT_PARALLEL.value)

    # Test for parallelism: dot(sv, n) == 0
    para_idx, = np.where(np.isclose(np.dot(sv, n), 0))

    # dot(s0 - p0, n) is:
    # 0 for coincidence
    # >0 if s0 is same side as n
    # <0 if s0 is other side as n

    prod = np.dot(s0[para_idx] - p0, n)
    idx1 = np.isclose(prod, 0)
    idx2 = np.logical_and(~idx1, prod > 0)
    idx3 = np.logical_and(~idx1, prod < 0)

    output[para_idx[idx1]] = ParallelType.PARALLEL_COINCIDENT.value
    output[para_idx[idx2]] = ParallelType.PARALLEL_FRONT.value
    output[para_idx[idx3]] = ParallelType.PARALLEL_BACK.value

    return output


def segment_parallel_to_planes(
    segment: np.ndarray, p0: np.ndarray, n: np.ndarray
) -> np.ndarray:
    """
    Compare a segments to an array of  plane defined by a point and a normal, and return an
    array containing ParallelType information for each segment. PARALLEL_FRONT is returned when
    the segment is on the positive side w.r.t. the normal.
    :param segment: 2x3 segment
    :param p0: Mx3 plane reference points
    :param n: Mx3 plane normals (not necessarily normalized
    :return: length M array of ParallelType
    """

    if not _validate_shape(segment, 2, 3):
        raise ValueError(f"segment has shape {segment.shape} instead of (2, 3)")

    if not _validate_shape(p0, None, 3):
        raise ValueError(f"p0 has shape {p0.shape} instead of (M, 3)")

    if not _validate_shape(n, None, 3):
        raise ValueError(f"n has shape {n.shape} instead of (M, 3)")

    if p0.shape[0] != n.shape[0]:
        raise ValueError(f"p0 and n must have the same first dimension")

    s0 = segment[0]
    s1 = segment[1]
    sv = s1 - s0

    output = np.empty(len(p0), dtype=int)
    output.fill(ParallelType.NOT_PARALLEL.value)

    # Test for parallelism: dot(sv, n) == 0
    para_idx, = np.where(np.isclose(np.dot(n, sv), 0, atol=1e-16))

    # dot(s0 - p0, n) is:
    # 0 for coincidence
    # >0 if s0 is same side as n
    # <0 if s0 is other side as n

    if len(para_idx) > 0:
        # prod = np.tensordot(s0 - p0[para_idx], n[para_idx], axes=(1, 1))
        prod = np.sum((s0 - p0[para_idx]) * n[para_idx], axis=1)
        idx = np.isclose(prod, 0, atol=1e-16)
        output[para_idx[idx]] = ParallelType.PARALLEL_COINCIDENT.value
        output[para_idx[np.logical_and(~idx, prod > 0)]] = ParallelType.PARALLEL_FRONT.value
        output[para_idx[np.logical_and(~idx, prod < 0)]] = ParallelType.PARALLEL_BACK.value

    return output


def mask_segments(segments: np.array, mask: np.array, diff: bool = True) -> np.array:
    """
    Take a array of 3D segments and masks them with a 2D polygon. The polygon is assumed to be
    perpendicular to the Z axis and the masking is done along the Z axis
    :param segments: Nx2x3 array of segments
    :param mask: Mx2 polygon to be used for masking
    :param diff: if True, the mask area is removed from segment, otherwise the intersection is
    kept
    :return: Lx2x3 array of segments whose length might differ from input
    """

    _validate_segments(segments)

    if len(mask.shape) != 2 or mask.shape[1] != 2:
        raise ValueError(f"mask array must be of dimension (Nx2) instead of {mask.shape}")

    if len(segments) == 0:
        return np.array(segments)

    poly = Polygon(mask)

    # The following is the parallel implementation, which passes the tests but generates
    # a lot of lines (https://github.com/Toblerity/Shapely/issues/779)

    # mls = asMultiLineString(segments)
    # if diff:
    #     mls2 = mls.difference(poly)
    # else:
    #     mls2 = mls.intersection(poly)
    #
    # if mls2.geom_type == "LineString":
    #     return np.array([mls2.coords])
    # elif mls2.geom_type == "MultiLineString":
    #     return np.array([np.array(ls.coords) for ls in mls2])
    # elif mls2.geom_type == "MultiPoint":
    #     return np.empty((0, 2, 3))
    # else:
    #     raise ValueError(f"Unexpected geometry: {mls2.geom_type}")

    # The following is a slower (?), segment-by-segment implementation that does not exhibit
    # the line multiplication issue

    op = "difference" if diff else "intersection"

    output = []
    for i in range(len(segments)):
        ls = asLineString(segments[i, :, :])
        res = getattr(ls, op)(poly)

        # segments with identical start/stop location are sometime returned
        if np.isclose(res.length, 0):
            continue

        if res.geom_type == "LineString":
            output.append(np.array([res.coords]))
        elif res.geom_type == "MultiLineString":
            output.append(np.array([np.array(l.coords) for l in res]))

    if output:
        res = np.vstack(output)

        # https://github.com/Toblerity/Shapely/issues/780
        # Some z-coordinate are being erroneously affected by the difference. This ugly
        # hack attempts to restore them.
        for s in segments:
            idx, = np.where(np.all(np.all(s[:, 0:2] == res[:, :, 0:2], axis=2), axis=1))
            res[idx, :, 2] = s[:, 2]

        return res
    else:
        return np.empty((0, 2, 3))


def split_segments(
    segments: np.ndarray, p0: np.ndarray, n: np.ndarray
) -> (np.ndarray, np.ndarray):
    """
    Compute segment to plan intersection and gather them (or their split parts) in two
    categories: behind and in front of the plane along z axis (front: higher z value).
    Segments must not be parallel to the plane and plane must not be parallel to z axis
    Ref: http://geomalgorithms.com/a05-_intersect-1.html
    :param segments: Mx2x3 array of segments
    :param p0: plane reference point (length 3)
    :param n: plane normal (length 3, does not need to be normalized)
    :return: tuple of ?x2x3 array containing segments in front, resp. behind of the plane
    """

    _validate_segments(segments)

    # make sure n points "up"
    if n[2] < 0:
        n = -n

    # if both ends of segments are behind or in front, they can readily be classified as such
    d0 = np.dot(segments[:, 0] - p0, n)
    d1 = np.dot(segments[:, 1] - p0, n)

    behind = np.logical_and(d0 < 0, d1 < 0)
    front = np.logical_and(d0 >= 0, d1 >= 0)
    split_idx, = np.where(np.logical_and(~behind, ~front))

    # Segment parametrized as: s0 + s * sv, where sv = s1 - s0, and s in [0, 1]
    # Plane parametrized as: p0 and n
    # Intersection with plane: s_i = -dot(n, s0-p0) / dot(n, sv)

    split_segs = segments[split_idx, :, :]

    sv = np.diff(split_segs, axis=1).reshape(len(split_segs), 3)
    s0 = split_segs[:, 0]
    s1 = split_segs[:, 1]

    si = -np.dot(s0 - p0, n) / np.dot(sv, n)
    s_mid = s0 + sv * si.reshape((len(s0), 1))
    front_condition = (np.dot(s0 - p0, n) >= 0).reshape((len(s0), 1))

    split_segs_front = np.empty_like(split_segs)
    split_segs_front[:, 0, :] = np.where(front_condition, s0, s_mid)
    split_segs_front[:, 1, :] = np.where(front_condition, s_mid, s1)

    split_segs_back = np.empty_like(split_segs)
    split_segs_back[:, 0, :] = np.where(front_condition, s_mid, s0)
    split_segs_back[:, 1, :] = np.where(front_condition, s1, s_mid)

    # gather all segments together
    front_segments = np.vstack([segments[front], split_segs_front])
    back_segments = np.vstack([segments[behind], split_segs_back])
    return front_segments, back_segments


def mask_segment(segment: np.ndarray, faces: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """
    Compute the visibility of the segment given the provided faces and pre-computed normals.
    Segments are hidden when the are on the opposite side of the normal
    :param segment: (2x3) the segment to process
    :param faces: (Nx3x3) the faces to consider
    :param normals: (Nx3) the pre-computed normals
    :return: (Mx2x3) list of visible sub-segments
    """

    if not _validate_shape(segment, 2, 3):
        raise ValueError(f"segment has shape {segment.shape} instead of (2, 3)")

    if not _validate_shape(faces, None, 3, 3):
        raise ValueError(f"faces has shape {faces.shape} instead of (M, 3, 3)")

    if not _validate_shape(normals, None, 3):
        raise ValueError(f"normals has shape {normals.shape} instead of (M, 3)")

    if faces.shape[0] != normals.shape[0]:
        raise ValueError(f"faces and normals must have the same first dimension")

    s0 = segment[0]
    s1 = segment[1]
    sv = s1 - s0

    # Check 2D overlap, all non-overlapping faces can be ignored
    active = triangles_overlap_segment_2d(faces, segment)
    if not np.any(active):
        return segment.reshape((1, 2, 3))

    # Check if faces are on a plane behind both ends of the segment
    p0 = faces[active, 0]
    n = normals[active]
    d0 = np.sum((s0 - p0) * n, axis=1)
    d1 = np.sum((s1 - p0) * n, axis=1)
    face_behind = np.logical_and(d0 >= 0, d1 >= 0)
    active[face_behind] = False
    if not np.any(active):
        return segment.reshape((1, 2, 3))

    face_in_front = np.logical_and(d0 < 0, d1 < 0)
    split_idx, = np.where(np.logical_and(~behind, ~front))

    # Check parallelism
    para = segment_parallel_to_planes(segment, p0, n)

    print(overlap, para)

    return np.empty(shape=(0, 2, 3))
