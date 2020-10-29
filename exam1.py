import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

from myml import LinearRegressor
import myml.util as util

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

y = df.iloc[51:100, 2].values # versicolor, petal length
x = df.iloc[51:100, 3].values # versicolor, petal width

# learning rate + max epochs?
lr = LinearRegressor(0.01, 1000)
lr.fit(x, y)

util.plot_linear_regression(x, y, lr)

