from shapely.geometry import MultiLineString, Polygon
import matplotlib.pyplot as plt


mls = MultiLineString([[(0, 1), (5, 1)], [(1, 2), (1, 0)]])
p = Polygon([(0.5, 0.5), (0.5, 1.5), (2, 1.5), (2, 0.5)])
results = mls.intersection(p)

plt.subplot(1, 2, 1)
for ls in mls:
    plt.plot(*ls.xy)
plt.plot(*p.boundary.xy, "-.k")
plt.xlim([0, 5])
plt.ylim([0, 2])

plt.subplot(1, 2, 2)
for ls in results:
    plt.plot(*ls.xy)
plt.xlim([0, 5])
plt.ylim([0, 2])

plt.show()
