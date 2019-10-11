# https://github.com/Toblerity/Shapely/issues/780

import numpy as np
import matplotlib.pyplot as plt
from shapely.wkt import loads, dumps
import shapely.geos


def main():
    ls = loads(
        "LINESTRING Z (-0.08365735370739519 0.08221315533414053 -1.010732607765682, -0.1533588930799144 0.0653358494958651 -1.010390339846296)"
    )
    p = loads(
        "POLYGON ((-0.2239294216623028 -0.01825216019427908, -0.1515331919718202 -0.03513918728914136, -0.1533588930799144 0.0653358494958651, -0.2239294216623028 -0.01825216019427908))"
    )
    result = ls.difference(p)

    plt.plot(*ls.coords.xy, 'r-')
    plt.plot(*p.boundary.xy, 'g-')
    plt.axis("equal")
    plt.show()

    print(dumps(result))
    print(np.array(result) == np.array(ls))
    print(shapely.geos.geos_version_string)
    print(shapely.__version__)



if __name__ == "__main__":
    main()
