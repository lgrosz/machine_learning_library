import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

from myml import Threshold
import myml.util as util

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

y = df.iloc[50:150, 4].values # labels, versicolor-virginica
X = np.array([x for x in df.iloc[50:150, 0:3].values])

threshold = Threshold()
threshold.fit(X, y)

print(threshold.predict([7, 3.2, 4.7]))

