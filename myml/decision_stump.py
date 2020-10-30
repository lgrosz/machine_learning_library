import numpy as np

class DecisionStump:
    """
    Finds a single feature and a threshhold which
    provides the best information gain on a single
    decision.
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        best_gain = 0 # best information gain
        best_question = None # best question
        uncertainty = gini(X, y) # uncertainty of the labels right now
        p = X.shape[1] # number of features

        for feature in range(p):
            # we only need to generate questions for the unique feature values
            values = unique_vals(X, feature) 

            for val in values:
                question = Question(feature, val)

                leftX, lefty, rightX, righty = partition(X, y, question)

                # we don't need a question that does nothing
                if (lefty.size == 0 or righty.size == 0):
                    continue

                gain = info_gain(leftX, lefty, rightX, righty, uncertainty)

                if (gain >= best_gain):
                    best_gain, best_question = gain, question
                    # self.left = probabilities
                    # self.right = probabilities

        self.q = best_question
        self.left = {}
        self.right = {}

    def predict(self, X):
        return np.array([self.left if self.q.match(x) else self.right for x in X])

class Question:
    def __init__(self, feature, threshold):
        self.feature = feature
        self.threshold = threshold

    def match(self, row):
        val = row[self.feature]
        if (is_numeric(val)):
            return val < self.threshold
        else:
            return val == self.threshold


def is_numeric(val):
    return isinstance(val, int) or isinstance(val, float)

def unique_vals(rows, feature):
    return np.array(list(set([row[feature] for row in rows])))

def gini(X, y):
    counts = class_counts(X, y)
    impurity = 1
    for label in counts:
        probability = counts[label] / float(X.size)
        impurity -= probability**2
    return impurity

def class_counts(X, y):
    counts = {}
    for i in range(X.size):
        label = y[i]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

def partition(X, y, q):
    print(X)
    print(y)
    leftX = np.array((0, X.shape[1]))
    lefty = np.array([])
    rightX = np.array(0, X.shape[1])
    righty = np.array([])

    for i in range(X.size):
        if q.match(X):
            leftX = np.append(leftX, X[i], axis=0)
            lefty = np.append(lefty, y[i])
        else:
            rightX = np.append(rightX, X[i], axis=0)
            righty = np.append(righty, y[i])

    return leftX, lefty, rightX, righty

def info_gain(leftX, lefty, rightX, righty, current_uncertainty):
    p = float(leftX.size) / (rightX.size + leftX.size)
    return current_uncertainty - p * gini(leftX, lefty) - (1-p) * gini(rightX, righty)

