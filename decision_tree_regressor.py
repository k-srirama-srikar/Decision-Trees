import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.max_depth=max_depth
        self.tree=None

    def mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def split(self, X, y, feature_index, threshold):
        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold
        return X[left_indices], X[right_indices], y[left_indices], y[right_indices]

    def best_split(self, X, y):
        best_feature, best_threshold = None, None
        best_mse = float("inf")
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self.split(X, y, feature_index, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                mse_left = self.mse(y_left)
                mse_right = self.mse(y_right)
                mse_split = (len(y_left) * mse_left + len(y_right) * mse_right) / len(y)
                if mse_split < best_mse:
                    best_mse = mse_split
                    best_feature = feature_index
                    best_threshold = threshold
        return best_feature, best_threshold

    def build_tree(self, X, y, depth=0):
        if len(y) <= 1 or depth == self.max_depth:
            return np.mean(y)
        feature_index, threshold = self.best_split(X, y)
        if feature_index is None:
            return np.mean(y)
        X_left, X_right, y_left, y_right = self.split(X, y, feature_index, threshold)
        left_subtree = self.build_tree(X_left, y_left, depth + 1)
        right_subtree = self.build_tree(X_right, y_right, depth + 1)
        return (feature_index, threshold, left_subtree, right_subtree)

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict_sample(self, x, node):
        if isinstance(node, (float, np.float64)):  # Leaf node
            return node
        feature_index, threshold, left_subtree, right_subtree = node
        if x[feature_index] <= threshold:
            return self.predict_sample(x, left_subtree)
        else:
            return self.predict_sample(x, right_subtree)

    def predict(self, X):
        return np.array([self.predict_sample(x, self.tree) for x in X])


# Example Usage
X = np.array([[4.9], [1.6], [2.1], [0.7], [4.1], [2.3]])
y = np.array([11, 3, 5, 1, 9, 6])

tree = DecisionTreeRegressor(max_depth=2)
tree.fit(X, y)
predictions = tree.predict(X)
print("Predictions:", predictions)
X2 = np.array([[4.2], [1.4], [3.1], [1.7], [2.1]])
print("Predictions 2:", tree.predict(X2))
