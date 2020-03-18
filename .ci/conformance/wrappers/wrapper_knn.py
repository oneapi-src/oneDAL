from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifier_original
from sklearn.neighbors.base import _check_weights
from sklearn import neighbors

from scipy import sparse


daal4pyCallsCounter = 0
sklCallsCounter = 0

csr_data = 0
weights_uniform_calls = 0
weights_distance_calls = 0
algorithm_auto_calls = 0
algorithm_ball_tree_calls = 0
algorithm_kdtree_calls = 0
algorithm_brute_calls = 0
metric_manhattan_calls = 0
metric_euclidean_calls = 0
metric_minkowski_calls = 0

class KNeighborsClassifier(KNeighborsClassifier_original):

    def __init__(self, n_neighbors=5,
                 weights='uniform', algorithm='auto', leaf_size=30,
                 p=2, metric='minkowski', metric_params=None, n_jobs=None,
                 **kwargs):

        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size, metric=metric, p=p,
            metric_params=metric_params,
            n_jobs=n_jobs, **kwargs)
        self.weights = _check_weights(weights)

    def fit(self, X, y):
        global daal4pyCallsCounter
        global sklCallsCounter

        global csr_data
        global weights_uniform_calls
        global weights_distance_calls
        global algorithm_auto_calls
        global algorithm_ball_tree_calls
        global algorithm_kdtree_calls
        global algorithm_brute_calls
        global metric_manhattan_calls
        global metric_euclidean_calls
        global metric_minkowski_calls

        if sparse.issparse(X):
            csr_data += 1
        if self.weights is 'uniform':
            weights_uniform_calls += 1
        if self.weights is 'distance':
            weights_distance_calls += 1

        if self.algorithm is 'auto':
            algorithm_auto_calls += 1
        if self.algorithm is 'ball_tree':
            algorithm_ball_tree_calls += 1
        if self.algorithm is 'kd_tree':
            algorithm_kdtree_calls += 1
        if self.algorithm is 'brute':
            algorithm_brute_calls += 1

        if self.metric is 'minkowski' and self.p > 2:
            metric_minkowski_calls += 1
        if self.metric is 'minkowski' and self.p == 2:
            metric_euclidean_calls += 1
        if self.metric is 'minkowski' and self.p < 2:
            metric_manhattan_calls += 1

        sklCallsCounter += 1
        print('skl_calls=', sklCallsCounter)
        print('daal_calls=', daal4pyCallsCounter)
        print('csr_calls=', csr_data)
        print('weights_uniform_calls=', weights_uniform_calls)
        print('weights_distance_calls=', weights_distance_calls)
        print('algorithm_auto_calls=', algorithm_auto_calls)
        print('algorithm_ball_calls=', algorithm_ball_tree_calls)
        print('algorithm_kdtree_calls=', algorithm_kdtree_calls)
        print('algorithm_brute_calls=', algorithm_brute_calls)
        print('metric_manhattan_calls=', metric_manhattan_calls)
        print('metric_euclidean_calls=', metric_euclidean_calls)
        print('metric_minkowski_calls=', metric_minkowski_calls)
        return super().fit(X, y)

    def predict(self, X):
        global daal4pyCallsCounter
        global sklCallsCounter

        global csr_data
        global weights_uniform_calls
        global weights_distance_calls
        global algorithm_auto_calls
        global algorithm_ball_tree_calls
        global algorithm_kdtree_calls
        global algorithm_brute_calls
        global metric_manhattan_calls
        global metric_euclidean_calls
        global metric_minkowski_calls

        if sparse.issparse(X):
            csr_data += 1
        if self.weights is 'uniform':
            weights_uniform_calls += 1
        if self.weights is 'distance':
            weights_distance_calls += 1

        if self.algorithm is 'auto':
            algorithm_auto_calls += 1
        if self.algorithm is 'ball_tree':
            algorithm_ball_tree_calls += 1
        if self.algorithm is 'kd_tree':
            algorithm_kdtree_calls += 1
        if self.algorithm is 'brute':
            algorithm_brute_calls += 1

        if self.metric is 'minkowski' and self.p > 2:
            metric_minkowski_calls += 1
        if self.metric is 'minkowski' and self.p == 2:
            metric_euclidean_calls += 1
        if self.metric is 'minkowski' and self.p < 2:
            metric_manhattan_calls += 1

        sklCallsCounter += 1
        print('skl_calls=', sklCallsCounter)
        print('daal_calls=', daal4pyCallsCounter)
        print('csr_calls=', csr_data)
        print('weights_uniform_calls=', weights_uniform_calls)
        print('weights_distance_calls=', weights_distance_calls)
        print('algorithm_auto_calls=', algorithm_auto_calls)
        print('algorithm_ball_calls=', algorithm_ball_tree_calls)
        print('algorithm_kdtree_calls=', algorithm_kdtree_calls)
        print('algorithm_brute_calls=', algorithm_brute_calls)
        print('metric_manhattan_calls=', metric_manhattan_calls)
        print('metric_euclidean_calls=', metric_euclidean_calls)
        print('metric_minkowski_calls=', metric_minkowski_calls)
        return super().predict(X)

    def predict_proba(self, X):
        global daal4pyCallsCounter
        global sklCallsCounter
        global csr_data
        global weights_uniform_calls
        global weights_distance_calls
        global algorithm_auto_calls
        global algorithm_ball_tree_calls
        global algorithm_kdtree_calls
        global algorithm_brute_calls
        global metric_manhattan_calls
        global metric_euclidean_calls
        global metric_minkowski_calls

        if sparse.issparse(X):
            csr_data += 1
        if self.weights is 'uniform':
            weights_uniform_calls += 1
        if self.weights is 'distance':
            weights_distance_calls += 1

        if self.algorithm is 'auto':
            algorithm_auto_calls += 1
        if self.algorithm is 'ball_tree':
            algorithm_ball_tree_calls += 1
        if self.algorithm is 'kd_tree':
            algorithm_kdtree_calls += 1
        if self.algorithm is 'brute':
            algorithm_brute_calls += 1

        if self.metric is 'minkowski' and self.p > 2:
            metric_minkowski_calls += 1
        if self.metric is 'minkowski' and self.p == 2:
            metric_euclidean_calls += 1
        if self.metric is 'minkowski' and self.p < 2:
            metric_manhattan_calls += 1

        sklCallsCounter += 1
        print('skl_calls=', sklCallsCounter)
        print('daal_calls=', daal4pyCallsCounter)
        print('csr_calls=', csr_data)
        print('weights_uniform_calls=', weights_uniform_calls)
        print('weights_distance_calls=', weights_distance_calls)
        print('algorithm_auto_calls=', algorithm_auto_calls)
        print('algorithm_ball_calls=', algorithm_ball_tree_calls)
        print('algorithm_kdtree_calls=', algorithm_kdtree_calls)
        print('algorithm_brute_calls=', algorithm_brute_calls)
        print('metric_manhattan_calls=', metric_manhattan_calls)
        print('metric_euclidean_calls=', metric_euclidean_calls)
        print('metric_minkowski_calls=', metric_minkowski_calls)
        return super().predict_proba(X)

neighbors.KNeighborsClassifier = KNeighborsClassifier