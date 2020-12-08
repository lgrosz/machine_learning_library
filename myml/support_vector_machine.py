import numpy as np


class SupportVectorMachine:
    def __init__(self, rate, epochs):
        self.epochs = epochs
        self.rate = rate

    def fit(self, X, d):
        self.weights = stochastic_gradient_descent(X, d, self.rate, self.epochs)

    def predict(self, x):
        return np.sign(np.dot(self.weights, x))

def cost_gradient(weights, features, label):
    distance = 1 - (label * np.dot(features, weights))

    if (max(0, distance) == 0):
        return weights
    else:
        # reg strength is 10000 todo make this variable
        return weights - (10000 * label * features)

def stochastic_gradient_descent(features, labels, rate, epochs):
    weights = np.zeros(features.shape[1])

    for _ in range(1, epochs):
        for i, x in enumerate(features):
            ascent = cost_gradient(weights, x, labels[i])
            weights = weights - (rate * ascent)

            # can put convergence threshold here or something to speed it up for
            # large number of epochs

    return weights

