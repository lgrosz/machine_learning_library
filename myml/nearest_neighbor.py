import numpy as np

class NearestNeighbor:
    def __init__(self, k):
        self.k = k

    def fit(self, X, d):
        self.X = X
        self.d = d
        if (is_numeric(d[0])):
            self.regression = True
        else:
            self.regression = False
        pass

    def predict(self, q):
        ordered_collection = []
        for i, x in enumerate(self.X):
            dist = self._dist(x, q)
            index = i
            ordered_collection.append(OrderedItem(index, dist))

        ordered_collection.sort()

        # use k nearest neighbors
        k_labels = [self.d[x.index] for x in ordered_collection[0:self.k]]

        if (self.regression):
            # mean
            return sum(k_labels) / len(k_labels)
        else:
            # mode
            return max(k_labels, key=k_labels.count)

    @staticmethod
    def _dist(x, q):
        """
        returns absolute euclidean distance between x and q
        """
        return np.linalg.norm(q - x)

def is_numeric(val):
    return isinstance(val, int) or isinstance(val, float)

class OrderedItem:
    def __init__(self, index, value):
        self.index = index
        self.value = value

    def __lt__(self, other):
        if not isinstance(other, OrderedItem):
            return NotImplemented

        return self.value < other.value

    def __eq__(self, other):
        if not isinstance(other, OrderedItem):
            return NotImplemented

        return self.value == other.value

