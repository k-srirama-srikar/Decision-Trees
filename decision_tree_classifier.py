import numpy as np

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree=None

    def gini(self,y):
        classes = np.unique(y)
        # this gives an array of unique elements
        gini = 1.0
        for cls in classes:
            p_cls = np.sum(y == cls) / len(y)
            # np.sum(y == cls) returns the number of cls' are present in y
            gini -= p_cls**2
        return gini

    def split(self, X, y, feature_index, threshold):
        left_indices = X[:, feature_index] <= threshold
        right_indices = X[:, feature_index] > threshold
        # Note that X is a numpy array
        # So, X[:, feature_index] is different from the general list slicing
        # Consider a 2-D numpy array,
        # say, X = array([[1, 2],
        #                [2, 3],
        #                [4, 5],
        #                [6, 7]])
        # if we perform X[:,0] this results in array([1,2,4,6])
        # and X[:,0]<5(say the threshold is 5) results in array([True, True, True, False])
        # and X[array([True, True, True, False])] results in array([[1, 2],
        #                                                           [2, 3],
        #                                                           [4, 5]])
        return X[left_indices], X[right_indices], y[left_indices], y[right_indices]

    def best_split(self, X,y):
        best_feature, best_threshold = None, None
        best_gini = 1.0 # keeping it maximum initially and then changing to the the least possible value iteratively
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self.split(X, y, feature_index, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    # i.e., the split resulted in the same array or the converse...
                    continue
                gini_left = self.gini(y_left)
                gini_right = self.gini(y_right)
                gini_split = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)
                if gini_split < best_gini:
                    best_gini = gini_split
                    best_feature = feature_index
                    best_threshold = threshold
        return best_feature, best_threshold


    def build_tree(self, X, y, depth = 0):
        if len(np.unique(y))==1 or depth==self.max_depth:
            return np.unique(y)[0]

        feature_index, threshold = self.best_split(X, y)
        if feature_index == None:
            return np.unique(y)[0]
        X_left, X_right, y_left, y_right = self.split(X,y,feature_index,threshold)
        left_subtree = self.build_tree(X_left, y_left, depth + 1)
        right_subtree = self.build_tree(X_right, y_right, depth + 1)
        return (feature_index, threshold, left_subtree, right_subtree)

    def fit(self, X,y):
        self.tree = self.build_tree(X,y)

    def predict_sample(self, x, node):
        if isinstance(node, (int, np.int32)):
            return node
        feature_index, threshold, left_subtree, right_subtree = node
        if x[feature_index] <= threshold:
            return self.predict_sample(x, left_subtree)
        else:
            return self.predict_sample(x, right_subtree)

    def predict(self, X):
        return np.array([self.predict_sample(x, self.tree) for x in X])


# Example
X = np.array([[2.7], [1.5], [3.0], [0.8], [4.1], [2.3]])
y = np.array([0, 1, 0, 1, 0, 1])

tree = DecisionTreeClassifier(max_depth=2)
tree.fit(X, y)
pred_X = np.array([[2.4], [1.9], [3.9], [1.8], [2.1], [1.3]])
predictions = tree.predict(X)
print("Predictions:", predictions)

