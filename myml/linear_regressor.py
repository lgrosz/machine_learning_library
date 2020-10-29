import numpy as np

class LinearRegressor:
    def __init__(self, rate, epochs):
        self.rate = rate
        self.epochs = epochs

    def fit(self, X, d):
        # make x[j, 0] = 1, so we can use w[0] as a bias
        X = np.array([np.insert(x, 0, 1) for x in X])

        # number of regressands/dependant variables
        n = d.size

        # number of regressors/independant variables
        p = X[0].size

        # need beta, the weights, for each epoch
        beta = np.zeros((self.epochs, p))
        # need epsilon, the errors, for each epoch
        epsilon = np.zeros((self.epochs, n))

        # we will use a gradient descent algorithm to
        # minimize the cost
        for i in range(self.epochs):
            # start with the previous weight
            if (i > 0):
                beta[i, :] = beta[i-1, :]

            y = np.dot(beta[i], X[i])
            print(y)

            # calculate partial derivatives for each independant variable
            derivatives = np.array([(-2 / p * sum(x_i * (d - y))) for x_i in X])

            # calculate new weights
            beta[i, :] = beta[i, :] - np.array([self.rate * d for d in derivatives])

        print(beta)

