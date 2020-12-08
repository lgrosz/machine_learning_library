import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

from myml import LogisticRegressor
from myml import LinearRegressor
import myml.util as util

yList = [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1]
XList = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]
y = np.array([y for y in yList])
X = np.array([[x] for x in XList])

print(y)
print(X)

lr = LogisticRegressor(0.01, 1000)
lr.fit(X, y)

util.plot_linear_regression(X, y, lr)

