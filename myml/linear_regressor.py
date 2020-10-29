import numpy as np

class LinearRegressor:
    def __init__(self, rate, epochs):
        self.rate = rate
        self.epochs = epochs

    def fit(self, X, d):
        """
        Creates the hypothesis beta from X and d. n
        is this number of samples and p is the number
        of independant variables

        Keyword arguments:
        ;param numpy.ndarray X: the independant variables (n, p)
        ;param numpy.ndarray y: the dependant variables (n)
        """
        # make x[j, 0] = 1, so we can use w[0] as a bias
        X = np.array([np.insert(x, 0, 1) for x in X])

        # number of regressands/dependant variables
        n = d.size

        # number of regressors/independant variables
        p = X.shape[1]

        # need beta, the weights, for each epoch
        beta = np.zeros((self.epochs, p))
        # need epsilon, the errors, for each epoch
        epsilon = np.zeros((self.epochs, n))

        # gradient descent minimization algorithms
        for i in range(self.epochs):
            # start with the previous weight
            if (i > 0):
                beta[i, :] = beta[i-1, :]

            y = np.array([np.dot(beta[i], x) for x in X])

            # calculate partial derivatives for each independant variable
            derivatives = np.array([(-2 / p * sum(X[:, j] * (d - y))) for j in range(X.shape[1])])

            # calculate new weights
            beta[i, :] = beta[i, :] - np.array([self.rate * d for d in derivatives])

            # calculate error for the new weights
            newy = np.array([np.dot(beta[i], x) for x in X])
            epsilon[i, :] = d - newy
            print(epsilon[i])

