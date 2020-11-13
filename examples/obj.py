import math
import timeit
from typing import Tuple

import numpy as np
import pywavefront

from lines import PolyShape, Scene, SilhouetteSkin


class OBJShape(PolyShape):
    def __init__(self, file_name, **kwargs):
        model = pywavefront.Wavefront(file_name, create_materials=True, collect_faces=True)

        # normalize everything
        vertices = np.array(model.vertices)
        vertices = vertices / np.max(np.max(vertices))

        # collect all segment
        faces = model.mesh_list[0].faces
        segs = set()

        def sort_int(a: int, b: int) -> Tuple[int, int]:
            return (b, a) if a < b else (a, b)

        for face in faces:
            segs.add(sort_int(face[0], face[1]))
            segs.add(sort_int(face[1], face[2]))
            segs.add(sort_int(face[2], face[0]))

        super().__init__(vertices, list(segs), faces, **kwargs)


def main(silhouette: bool = False):
    scene = Scene()

    obj = OBJShape("cow.obj")
    obj.rotate_x(-math.pi / 2)
    obj.rotate_z(-math.pi / 2)
    if silhouette:
        obj.add(SilhouetteSkin(keep_segments=False))

    scene.add(obj)
    scene.look_at((0.5, 2, 0.2), (0, 0, 0))
    scene.perspective(40, 0.1, 10)
    rs = scene.render("v2")
    rs.show()
    rs.save("cow_new.svg")


if __name__ == "__main__":
    print(f"Execution time: {timeit.timeit(lambda : main(False), number=1)}")
