import numpy as np

class Perceptron:
    def __init__(self, rate, niter):
        self.rate = rate
        self.niter = niter

    def fit(self, X, d):
        """
        Creates the hypothesis w from X and d.

        Keyword arguments:
        ;param numpy.ndarray X: the training input domain
        ;param numpy.ndarray d: the desired output for the training input domain
        """
        # todo check types, this should be done for all public facing functions

        # make x[j, 0] = 1, so we can use w[0] as the bias
        X = np.array([np.insert(x, 0, 1) for x in X])

        self.errors = np.array([])
        nfeatures = X[0].size
        self.weights = np.empty((0, nfeatures))


        for i in range(self.niter):
            # initialize weight array
            w = None
            if (self.weights.size < 1):
                w = np.array([0 for _ in range(nfeatures)])
            else:
                w = np.copy(self.weights[i-1])

            # calculate y's for each sample and set the weight
            itererrors = np.array([])
            for j, x_j in enumerate(X):
                d_j = d[j]
                y_j = Perceptron.f(w, x_j)
                w = np.array([w[i] + self.rate * (d_j - y_j) * x_j[i] for i in range(nfeatures)])
                itererrors = np.append(itererrors, abs(d_j-y_j))

            # record the weight for this iteration
            self.weights = np.append(self.weights, [w], axis=0)

            # record the number of errors
            itererror = np.sum(itererrors)# / X.shape[0]
            self.errors = np.append(self.errors, itererror)
            if (itererror == 0):
                break

    def net_input(self, X):
        """
        Returns numpy array weighted samples

        ;param numpy.ndarray X: sample array
        """
        w = self.weights[self.weights.shape[0]-1]
        X = np.array([np.insert(x, 0, 1) for x in X])
        return np.array([np.dot(w, x_j) for x_j in X])


    def predict(self, X):
        """
        Returns numpy array of labels

        ;param numpy.ndarray X: sample array
        """
        w = self.weights[self.weights.shape[0]-1]
        X = np.array([np.insert(x, 0, 1) for x in X])
        return np.array([Perceptron.f(w, x_j) for x_j in X])


    @staticmethod
    def f(w, x):
        """
        Returns 1 if w dot x > 0, -1 otherwise

        Since x[0] is 1, w[0] acts as a bias

        ;param numpy.ndarray w: weight vector
        ;param numpy.ndarray x: input features
        """
        if (np.dot(w, x) > 0):
            return 1
        else:
            return -1


