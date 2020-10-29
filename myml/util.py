import numpy as np
import matplotlib.pyplot as plot
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
            np.arange(x1_min, x1_max, resolution),
            np.arange(x2_min, x2_max, resolution)
            )
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plot.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plot.xlim(xx1.min(), xx1.max())
    plot.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plot.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                alpha=0.8,
                color=cmap(idx),
                marker=markers[idx],
                label=cl)
    plot.show()

def plot_decision_regions_3d(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the separator plane
    fig = plot.figure()
    axis = fig.add_subplot(111, projection='3d')
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
            np.arange(x1_min, x1_max, resolution),
            np.arange(x2_min, x2_max, resolution)
            )
    w = classifier.weights[classifier.weights.shape[0]-1]
    xx3 = (-w[0] - w[1] * xx1 - w[2] * xx2) / w[3]
    # calculate the points on the plane
    axis.plot_surface(xx1, xx2, xx3)

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        axis.scatter(xs=X[y == cl, 0], ys=X[y == cl, 1], zs=X[y == cl, 2],
                alpha=0.8,
                color=cmap(idx),
                marker=markers[idx],
                label=cl
                )

    plot.show()

