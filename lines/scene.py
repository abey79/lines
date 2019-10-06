import math
from typing import Sequence

import numpy as np
import shapely.ops
from shapely.geometry import MultiLineString, Polygon

from lines.math import vertices_matmul
from .shapes import Shape


class Scene:
    def __init__(self):
        self._shapes = []
        self._camera_matrix = np.identity(4)

    def add(self, shape: Shape) -> None:
        """
        Add a shape to the scene.
        :param shape: the shape to add
        """
        self._shapes.append(shape)

    def look_at(self, eye: Sequence[float], to: Sequence[float]) -> None:
        eye = np.array(eye, dtype="float64")
        to = np.array(to, dtype="float64")
        up = np.array([0, 0, 1], dtype="float64")

        f = to - eye
        f = f / np.linalg.norm(f)
        s = np.cross(f, up)
        s = s / np.linalg.norm(s)
        u = np.cross(s, f)
        u = u / np.linalg.norm(u)

        m = np.zeros((4, 4))
        m[0, :-1] = s
        m[1, :-1] = u
        m[2, :-1] = -f
        m[3, :-1] = eye
        m[-1, -1] = 1.0

        m = np.transpose(np.linalg.inv(m))

        self._camera_matrix = m

    def frustum(self, l: float, r: float, b: float, t: float, n: float, f: float) -> None:
        t1 = 2 * n
        t2 = r - l
        t3 = t - b
        t4 = f - n
        frustum_matrix = np.array(
            [
                [t1 / t2, 0, (r + l) / t2, 0],
                [0, t1 / t3, (t + b) / t3, 0],
                [0, 0, (-f - n) / t4, (-t1 * f) / t4],
                [0, 0, -1, 0],
            ]
        )
        self._camera_matrix = frustum_matrix @ self._camera_matrix

    def perspective(self, fov: float, near: float, far: float) -> None:
        y_max = near * math.tan(fov / 2)
        x_max = y_max * 1  # y_max * aspect
        self.frustum(-x_max, x_max, -y_max, y_max, near, far)

    def render(self) -> MultiLineString:
        """
        Renders the scene with the current camera projection. Returns a
        shapely.geometry.MultiLineString object containing all the visible 2D segment
        :return: the rendered 2D lines
        """
        # (A) Gather all segments and all faces

        segment_set = []
        face_set = []
        for shape in self._shapes:
            segments, faces = shape.compile()
            segment_set.append(segments)
            face_set.append(faces)

        all_segments = np.vstack(segment_set)
        all_faces = np.vstack(face_set)

        #####TODO!!!!! Camera matrix must be passed to compile directely
        #####otherwise non-PolyShape cannot have a chance to adjust their model before
        #####producing the segments/faces

        # (B) Project everything with the camera matrix and convert to regular coordinates
        proj_segments = vertices_matmul(all_segments, self._camera_matrix)

        all_segments = np.divide(
            proj_segments[:, :, 0:3], np.tile(proj_segments[:, :, -1:], (1, 1, 3))
        )

        # (C) Process all face/segment occlusion

        # For each face/segment combination, the following steps are done:
        # (1) Check if segment and face are parallel
        #     -> Segment contained in face plan -> unmasked
        #     -> Segment in front of face plan -> unmasked
        #     -> Segment behind the face plan -> to be masked
        #
        # (2) Segment is not parallel, split it at plan intersection
        #     -> half segment in front -> unmasked
        #     -> half segment behind -> to be masked
        #
        # (3) All (sub-)segment flagged as to be masked are... masked with face.

        for p0, p1, p2 in all_faces:
            print(p1, p2, p0)

        # (D) Crop to camera view
        segments_2d = all_segments[:, :, 0:2]
        mls = MultiLineString(list(segments_2d))
        cam_view = Polygon(((-1, -1), (-1, 1), (1, 1), (1, -1)))
        # mls = mls.intersection(cam_view)

        # (E) Convert to 2D data and  merge line strings
        tot_seg_count = len(mls)
        segments_optimized = shapely.ops.linemerge(mls)
        #segments_optimized = mls
        print(f"Seg count: {tot_seg_count}, optimized seg count: {len(segments_optimized)}")
        return segments_optimized
