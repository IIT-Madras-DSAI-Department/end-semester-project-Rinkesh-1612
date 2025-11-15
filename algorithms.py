import numpy as np
from collections import Counter
from scipy.signal import convolve2d

def f1_score(y_true, y_pred):
    f1s = []
    for label in np.unique(y_true):
        tp = np.sum((y_pred == label) & (y_true == label))
        fp = np.sum((y_pred == label) & (y_true != label))
        fn = np.sum((y_pred != label) & (y_true == label))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1s.append(f1)
    return np.mean(f1s)

def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

class KNearestNeighbors:
    def __init__(self, k=5, weights='uniform', metric='euclidean'):
        self.k = k
        self.weights = weights
        self.metric = metric
        if self.weights not in ['uniform', 'distance']:
            raise ValueError("weights not recognized: should be 'uniform' or 'distance'")
        if self.metric not in ['euclidean', 'manhattan']:
            raise ValueError("metric not recognized: should be 'euclidean' or 'manhattan'")

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        if self.metric == 'euclidean':
            distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        elif self.metric == 'manhattan':
            distances = [np.sum(np.abs(x - x_train)) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        k_nearest_distances = [distances[i] for i in k_indices]
        if self.weights == 'uniform':
            most_common = Counter(k_nearest_labels).most_common(1)
            return most_common[0][0]
        elif self.weights == 'distance':
            votes = {}
            epsilon = 1e-6
            for i in range(self.k):
                label = k_nearest_labels[i]
                dist = k_nearest_distances[i]
                weight = 1 / (dist + epsilon)
                votes[label] = votes.get(label, 0) + weight
            return max(votes, key=votes.get)

class LogisticRegressionOvR:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.models = {}

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            y_binary = np.where(y == cls, 1, 0)
            w = np.zeros(n_features)
            b = 0
            for _ in range(self.n_iterations):
                linear_model = np.dot(X, w) + b
                y_predicted = self._sigmoid(linear_model)
                dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y_binary))
                db = (1 / n_samples) * np.sum(y_predicted - y_binary)
                w -= self.learning_rate * dw
                b -= self.learning_rate * db
            self.models[cls] = {'w': w, 'b': b}

    def predict_proba(self, X):
        probas = np.zeros((X.shape[0], len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            model = self.models[cls]
            linear_model = np.dot(X, model['w']) + model['b']
            probas[:, i] = self._sigmoid(linear_model)
        return probas

    def predict(self, X):
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=10, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def get_params(self, deep=True):
        return {
            'min_samples_split': self.min_samples_split,
            'max_depth': self.max_depth,
            'n_features': self.n_features
        }

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            return Node(value=self._most_common_label(y))
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        if best_feat is None:
            return Node(value=self._most_common_label(y))
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return Node(value=self._most_common_label(y))
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            if len(thresholds) > 15:
                thresholds = np.random.choice(thresholds, size=15, replace=False)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        parent_gini = self._gini(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        g_l, g_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
        child_gini = (n_l / n) * g_l + (n_r / n) * g_r
        return parent_gini - child_gini

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _gini(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return 1 - np.sum([p ** 2 for p in ps])

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

class BaggingClassifier:
    def __init__(self, base_estimator, n_estimators=100):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators = []

    def fit(self, X, y):
        self.estimators = []
        n_samples = X.shape[0]
        for _ in range(self.n_estimators):
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[idxs], y[idxs]
            estimator = type(self.base_estimator)(**self.base_estimator.get_params())
            estimator.fit(X_sample, y_sample)
            self.estimators.append(estimator)

    def predict(self, X):
        predictions = np.array([estimator.predict(X) for estimator in self.estimators])
        predictions = np.swapaxes(predictions, 0, 1)
        final_predictions = [Counter(pred).most_common(1)[0][0] for pred in predictions]
        return np.array(final_predictions)

    def get_params(self, deep=True):
        return {"base_estimator": self.base_estimator, "n_estimators": self.n_estimators}

class StackingClassifier:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, X, y):
        for model in self.base_models:
            model.fit(X, y)
        meta_features = np.column_stack([
            model.predict(X) for model in self.base_models
        ])
        self.meta_model.fit(meta_features, y)

    def predict(self, X):
        base_predictions = np.column_stack([
            model.predict(X) for model in self.base_models
        ])
        return self.meta_model.predict(base_predictions)

class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column > self.threshold] = -1
        return predictions

class AdaBoost:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.models = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_search_features = 60
        for cls in self.classes_:
            y_binary = np.where(y == cls, 1, -1)
            weights = np.full(n_samples, (1 / n_samples))
            clfs = []
            for _ in range(self.n_estimators):
                clf = DecisionStump()
                min_error = float('inf')
                feature_indices = np.random.choice(n_features, size=n_search_features, replace=False)
                for feature_i in feature_indices:
                    X_column = X[:, feature_i]
                    thresholds = np.unique(X_column)
                    if len(thresholds) > 15:
                        thresholds = np.random.choice(thresholds, size=15, replace=False)
                    for threshold in thresholds:
                        p = 1
                        predictions = np.ones(n_samples)
                        predictions[X_column < threshold] = -1
                        misclassified = weights[y_binary != predictions]
                        error = sum(misclassified)
                        if error > 0.5:
                            error = 1 - error
                            p = -1
                        if error < min_error:
                            clf.polarity = p
                            clf.threshold = threshold
                            clf.feature_idx = feature_i
                            min_error = error
                EPS = 1e-10
                clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))
                predictions = clf.predict(X)
                weights *= np.exp(-clf.alpha * y_binary * predictions)
                weights /= np.sum(weights)
                clfs.append(clf)
            self.models[cls] = clfs

    def predict(self, X):
        all_class_preds = []
        for cls in self.classes_:
            clfs = self.models[cls]
            y_pred = np.zeros(X.shape[0])
            for clf in clfs:
                y_pred += clf.alpha * clf.predict(X)
            all_class_preds.append(y_pred)
        all_class_preds = np.array(all_class_preds).T
        return self.classes_[np.argmax(all_class_preds, axis=1)]

class DecisionTreeRegressor:
    def __init__(self, max_depth=3, min_samples_split=2, max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.tree_ = None
        self.feature_importances_ = None

    def fit(self, X, y):
        n_features = X.shape[1]
        self.feature_importances_ = np.zeros(n_features)
        if self.max_features is None:
            self.n_features_to_consider_ = n_features
        elif self.max_features == 'sqrt':
            self.n_features_to_consider_ = int(np.sqrt(n_features))
        else:
            self.n_features_to_consider_ = self.max_features
        self.tree_ = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            len(np.unique(y)) == 1):
            return np.mean(y)
        best_split = self._best_split(X, y, n_samples, n_features)
        if best_split is None:
            return np.mean(y)
        feature_idx, threshold = best_split
        left_idxs = X[:, feature_idx] <= threshold
        right_idxs = ~left_idxs
        left_subtree = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_subtree = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return {'feature_idx': feature_idx, 'threshold': threshold, 'left': left_subtree, 'right': right_subtree}

    def _best_split(self, X, y, n_samples, n_features):
        best_gain = -1
        best_feature_idx, best_threshold = None, None
        parent_variance = np.var(y) * n_samples
        feature_indices = np.random.choice(n_features, self.n_features_to_consider_, replace=False)
        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])
            if len(thresholds) > 15:
                thresholds = np.random.choice(thresholds, size=15, replace=False)
            for threshold in thresholds:
                left_idxs = X[:, feature_idx] <= threshold
                right_idxs = ~left_idxs
                y_left, y_right = y[left_idxs], y[right_idxs]
                if len(y_left) < self.min_samples_split or len(y_right) < self.min_samples_split:
                    continue
                child_variance = (len(y_left) * np.var(y_left) + 
                                  len(y_right) * np.var(y_right))
                gain = parent_variance - child_variance
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        if best_feature_idx is not None:
            self.feature_importances_[best_feature_idx] += best_gain
        if best_gain > 0:
            return best_feature_idx, best_threshold
        return None

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree_) for x in X])

    def _traverse_tree(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node['feature_idx']] <= node['threshold']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])

class GradientBoostingBinaryClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_split=2, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees_ = []
        self.initial_prediction_ = None

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        n_samples, _ = X.shape
        p_initial = np.mean(y)
        self.initial_prediction_ = np.log(p_initial / (1 - p_initial))
        F = np.full(n_samples, self.initial_prediction_)
        for i in range(self.n_estimators):
            probabilities = self._sigmoid(F)
            residuals = y - probabilities
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            )
            tree.fit(X, residuals)
            update = tree.predict(X)
            F += self.learning_rate * update
            self.trees_.append(tree)

    def predict_proba(self, X):
        F = np.full(X.shape[0], self.initial_prediction_)
        for tree in self.trees_:
            F += self.learning_rate * tree.predict(X)
        return self._sigmoid(F)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

class GradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_features = max_features
        self.models = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            y_binary = np.where(y == cls, 1, 0)
            binary_gbc = GradientBoostingBinaryClassifier(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                max_features=self.max_features
            )
            binary_gbc.fit(X, y_binary)
            self.models[cls] = binary_gbc

    def predict(self, X):
        probas = np.zeros((X.shape[0], len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            model = self.models[cls]
            probas[:, i] = model.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov = np.cov(X_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        self.components = eigenvectors[idxs[:self.n_components]]

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)

class SVM_Binary:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.models = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for cls in self.classes_:
            y_binary = np.where(y == cls, 1, -1)
            model = SVM_Binary(
                learning_rate=self.learning_rate, 
                lambda_param=self.lambda_param, 
                n_iters=self.n_iters
            )
            model.fit(X, y_binary)
            self.models[cls] = model

    def predict(self, X):
        scores = np.zeros((X.shape[0], len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            model = self.models[cls]
            scores[:, i] = np.dot(X, model.w) - model.b
        return self.classes_[np.argmax(scores, axis=1)]

class HOG:
    def __init__(self, cell_size=(8, 8), n_bins=9):
        self.cell_size = cell_size
        self.n_bins = n_bins
        self.bin_centers = np.linspace(0, 180, n_bins + 1, endpoint=False) + (180 / (2 * n_bins))

    def _get_gradients(self, image):
        dx_kernel = np.array([[-1, 0, 1]])
        dy_kernel = np.array([[-1], [0], [1]])
        dx = convolve2d(image, dx_kernel, mode='same', boundary='symm')
        dy = convolve2d(image, dy_kernel, mode='same', boundary='symm')
        magnitude = np.sqrt(dx**2 + dy**2)
        orientation = np.rad2deg(np.arctan2(dy, dx)) % 180
        return magnitude, orientation

    def _create_cell_histograms(self, magnitude, orientation):
        n_rows, n_cols = magnitude.shape
        cell_rows, cell_cols = self.cell_size
        n_cells_row = n_rows // cell_rows
        n_cells_col = n_cols // cell_cols
        histograms = np.zeros((n_cells_row, n_cells_col, self.n_bins))
        for i in range(n_cells_row):
            for j in range(n_cells_col):
                row_start, row_end = i * cell_rows, (i + 1) * cell_rows
                col_start, col_end = j * cell_cols, (j + 1) * cell_cols
                cell_mag = magnitude[row_start:row_end, col_start:col_end]
                cell_ori = orientation[row_start:row_end, col_start:col_end]
                hist, _ = np.histogram(cell_ori, bins=self.n_bins, range=(0, 180), weights=cell_mag)
                histograms[i, j, :] = hist
        return histograms

    def transform(self, X):
        n_samples, n_features = X.shape
        image_height, image_width = 28, 28
        hog_features_list = []
        for i in range(n_samples):
            image = X[i].reshape(image_height, image_width)
            magnitude, orientation = self._get_gradients(image)
            histograms = self._create_cell_histograms(magnitude, orientation)
            histograms = (histograms - np.mean(histograms)) / (np.std(histograms) + 1e-5)
            hog_features_list.append(histograms.flatten())
        return np.array(hog_features_list)