import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

from myml import LinearRegressor
from myml import DecisionStump
import myml.util as util

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

y = df.iloc[50:150, 4].values # labels, versicolor-virginica
X = np.array([x for x in df.iloc[50:150, 0:3].values]) # all features, decision stump will choose one of them

ds = DecisionStump()
ds.fit(X, y)

print(ds.predict(np.array([[7, 3.2, 4.7]])))

