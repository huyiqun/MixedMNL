import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x1 = [1, -1]
x2 = [-1, 1]
p = 0
z = np.arange(100)
yy = []
for z in np.arange(-50, 50, 1):
    for w in np.arange(-50, 50, 1):
        t = [np.exp(x1[0] * z + x1[1] * w - p), np.exp(x2[0] * z + x2[1] * w - p), 1]
        y = [a / np.sum(t) for a in t]
        yy.append(y)

yy = np.array(yy)
print(yy)

fig = plt.figure(figsize=(10, 8))
for dd in range(4):
    ax = fig.add_subplot(2, 2, dd + 1, projection="3d")
    ax.view_init(elev=45.0, azim=dd * 45)
    ax.scatter(yy[:, 0], yy[:, 1], yy[:, 2])
    # ax.plot([0,0,1,0, []0,1,0,0], [1,0,0,1], c="r")
    ax.plot_surface(
        np.array([[0, 1], [0, 0]]),
        np.array([[0, 0], [1, 1]]),
        np.array([[1, 0], [0, 0]]),
        alpha=0.2,
    )


x = np.arange(1, 1e6)

y1 = (x ** 2) * np.log(x)
y2 = (np.log(x)) ** 2

plt.plot(x, y1, label="y1")
plt.plot(x, y2, label="y2")
plt.legend()

