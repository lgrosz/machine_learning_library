import numpy as np
import operator

from .decision_stump import DecisionStump

class Threshold:
    """
    Uses the decision stump to make a conclusive deduction about the
    classification of a given set of features
    """
    def __init__(self):
        self._stump = DecisionStump()

    def fit(self, X, y):
        self._stump.fit(X, y)

    def predict(self, x):
        # return the key of the largest value
        return max(self._stump.predict(x).items(), key=operator.itemgetter(1))[0]

