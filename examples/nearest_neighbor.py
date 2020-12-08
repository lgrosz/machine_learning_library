import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

from myml import NearestNeighbor
import myml.util as util

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

y = df.iloc[0:150, 4].values
X = np.array([x for x in df.iloc[0:150, 2:3].values])

learner = NearestNeighbor(10)
learner.fit(X, y)

print("Classification: ",learner.predict(np.array([4, 1.5])))

y = df.iloc[0:150, 1].values
X = np.array([x for x in df.iloc[0:150, 2:3].values])

learner.fit(X, y)

print("Regression: ",learner.predict(np.array([4, 1.5])))

