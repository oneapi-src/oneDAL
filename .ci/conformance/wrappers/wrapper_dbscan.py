from daal4py import dbscan as d4p_dbscan
from sklearn.cluster.dbscan_ import dbscan as skl_dbscan
from sklearn.utils.validation import check_array
from scipy.spatial import distance as distance_scipy
from scipy import sparse
import numpy as np

#Global counters for counting calls from d4py and Skl
daal4pyCallsCounter = 0
sklCallsCounter = 0

'''
Global counters for counting percent usage of incompatible
paramenters by daal4py
'''
csr_counter = 0
minkowski_counter = 0
precomputed_counter = 0
manhattan_counter = 0


def dbscan(X, eps=0.5, min_samples=5, metric='minkowski', metric_params=None,
           algorithm='auto', leaf_size=30, p=2, sample_weight=None, n_jobs=1):
    global daal4pyCallsCounter
    global sklCallsCounter

    global csr_counter
    global minkowski_counter
    global precomputed_counter
    global manhattan_counter

    if sparse.issparse(X) == True:
        csr_counter+=1
    if metric == 'precomputed':
        precomputed_counter+=1
    if metric == 'minkowski' and p != 2:
        minkowski_counter+=1
    if metric == 'manhattan':
        manhattan_counter+=1

    core_sample_indices = 0
    labels = 0

    #Daal4py implementation
    if  (sparse.issparse(X) == False and
         metric != 'precomputed'
        # (
        # (metric == 'euclidean' or metric == distance_scipy.euclidean) or 
        # (metric == 'minkowski' and p==2)
        # )
        ):

        daal4pyCallsCounter+=1

        # Check the algorithm parameters
        if eps <= 0:
            raise ValueError('Parameter "eps" must be '
                             'non-zero positive value.')
        if leaf_size <= 0:
            raise ValueError('Parameter "leaf_size" must be '
                             'non-zero positive value.')

        if algorithm != 'ball_tree' and algorithm != 'kd_tree' and algorithm != 'brute' and algorithm != 'auto':
            raise ValueError('Algorithm is incorrect.')
        
        if metric == 'blah':
            raise ValueError('Metric is incorrect.')
        
        if type(p) == int and p < 0:
            raise ValueError('Parameter "p" must be '
                             'non-zero positive value.')

        # Converting X to np.array if X is a list or tuple
        X = check_array(X, accept_sparse=False)
        elements_number = X.shape[0]

        if type(sample_weight) == list:
            sample_weight = np.array(sample_weight)
        if sample_weight is None:
            sample_weight = np.ones([elements_number, 1])
        if len(sample_weight.shape) == 1:
            sample_weight = np.reshape(sample_weight, (sample_weight.shape[0], 1))

        if X.shape[0] != sample_weight.shape[0]:
                raise ValueError('Incorrect sizes.')

        dbscan_result = d4p_dbscan(epsilon=eps, minObservations=min_samples, resultsToCompute="computeCoreIndices|computeCoreObservations"
        ).compute(data=X, weights=sample_weight)
        # labels = np.reshape(dbscan_result.assignments, elements_number)
        labels = dbscan_result.assignments.ravel()
        #core_sample_indices = 0
        if dbscan_result.coreIndices is not None:
            number_core_samples = dbscan_result.coreIndices.shape[0]
            core_sample_indices = np.reshape(dbscan_result.coreIndices, number_core_samples)
            components = dbscan_result.coreObservations
        else:
            number_core_samples = 0
            core_sample_indices = np.empty(shape=(0,))
            components = np.empty(shape=(0, X.shape[1]))

    #Scikit-learn implementation
    else:
        sklCallsCounter+=1
        core_sample_indices, labels = skl_dbscan(X=X, eps=eps, min_samples=min_samples, metric=metric, metric_params=metric_params,
        algorithm=algorithm, leaf_size=leaf_size, p=p, sample_weight=sample_weight, n_jobs=n_jobs)

    print('skl_calls=', sklCallsCounter)
    print('daal_calls=', daal4pyCallsCounter)
    print('data_sparse_using=', csr_counter)
    print('param_metric_precomputed=', precomputed_counter)
    print('param_metric_minkowski=', minkowski_counter)
    print('param_metric_manhattan=', manhattan_counter, '\n')
    return core_sample_indices, labels

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean',
                 metric_params=None, algorithm='auto', leaf_size=30, p=None,
                 n_jobs=1):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs

    def fit(self, X, y=None, sample_weight=None):
        X = check_array(X, accept_sparse='csr')
        clust = dbscan(X=X, eps=self.eps, min_samples=self.min_samples, metric=self.metric, metric_params=self.metric_params,
        algorithm=self.algorithm, leaf_size=self.leaf_size, p=self.p, sample_weight=sample_weight, n_jobs=self.n_jobs)
        self.core_sample_indices_, self.labels_ = clust
        if len(self.core_sample_indices_):
            # fix for scipy sparse indexing issue
            self.components_ = X[self.core_sample_indices_].copy()
        else:
            # no core samples
            self.components_ = np.empty((0, X.shape[1]))
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        self.fit(X, sample_weight=sample_weight)
        return self.labels_
