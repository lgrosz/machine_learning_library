import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

from myml import LinearRegressor
from myml import DecisionStump
import myml.util as util

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

#y = df.iloc[50:100, 2].values # versicolor, petal length
## needs to be a matrix where there are n (number of data points) rows are independant variables
#X = np.array([[x] for x in df.iloc[50:100, 3].values]) # versicolor, petal length
#
#lr = LinearRegressor(0.01, 1000)
#lr.fit(X, y)
#
#util.plot_linear_regression(X, y, lr)

y = df.iloc[50:150, 4].values # labels, versicolor-virginica
X = np.array([[x] for x in df.iloc[50:150, 2].values]) # petal length only

ds = DecisionStump()
ds.fit(X, y)

