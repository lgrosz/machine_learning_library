import numpy as np
from math import exp
import matplotlib.pyplot as plot

from .linear_regressor import LinearRegressor


class LogisticRegressor:
    """
    Takes real value inputs and makes a prediction as to the probability of the
    input belonging to some class
    """
    def __init__(self, rate, epochs):
        self.base_regressor = LinearRegressor(rate, epochs)

    def fit(self, X, d):
        """
        Creates the hypothesis beta from X and d. n
        is this number of samples and p is the number
        of independant variables

        Keyword arguments:
        ;param numpy.ndarray X: the independant variables (n, p)
        ;param numpy.ndarray y: the dependant variables (n)
        """
        self.base_regressor.fit(X, d)

    @staticmethod
    def _logistic_function(a):
        return (1 / (1 + exp(-1 * a)))

    def predict(self, X):
        """
        Returns numpy array of predicted values

        todo: it'd be cool if I can pass in a
        number and it'd predict as if it was
        that epoch

        ;param numpy.ndarray X: an array of inputs
        """
        linear = self.base_regressor.predict(X)
        return np.array([self._logistic_function(a) for a in linear])

def plot_linear_regression(x, y, lr):
    # 2d plot
    if (x.shape[1] == 1):
        plot.scatter(x, y)
        plotX = np.array([[t] for t in np.linspace(np.min(x), np.max(x), 1000)])
        ploty = lr.predict(plotX)
        plot.plot(plotX, ploty, '-r')
        plot.show()

