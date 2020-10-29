import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

from myml import Perceptron
import myml.util as util

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# these values are really close together, so it takes a while to learn
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-versicolor', -1, 1)

X = df.iloc[0:100, [0, 1]].values

pn = Perceptron(0.1, 1000)
pn.fit(X, y)

if (X.shape[1] == 2):
    util.plot_decision_regions(X, y, pn)

# here I take 3 features, though it's not hard to learn
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-versicolor', -1, 1)

X = df.iloc[0:100, [0, 1, 2]].values

pn = Perceptron(0.1, 1000)
pn.fit(X, y)

if (X.shape[1] == 3):
    util.plot_decision_regions_3d(X, y, pn)

