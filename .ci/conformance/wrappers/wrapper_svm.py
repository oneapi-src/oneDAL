from sklearn import svm, linear_model, datasets, metrics, base

from scipy import sparse

daal4pyCallsCounter = 0
sklCallsCounter = 0

csr_data_calls = 0
penalty_calls = 0
kernel_rbf_calls = 0
kernel_linear_calls = 0
kernel_poly_calls = 0
kernel_sigmoid_calls = 0
kernel_precomputed_calls = 0
shrinking_calls = 0
probability_calls = 0
class_weight_calls = 0
verbose_calls = 0
df_shape_ovo_calls = 0
df_shape_ovr_calls = 0
random_state_calls = 0

def print_counters():
    global daal4pyCallsCounter
    global sklCallsCounter

    global csr_data_calls
    global penalty_calls
    global kernel_rbf_calls
    global kernel_linear_calls
    global kernel_poly_calls
    global kernel_sigmoid_calls
    global kernel_precomputed_calls
    global shrinking_calls
    global probability_calls
    global class_weight_calls
    global verbose_calls
    global df_shape_ovo_calls
    global df_shape_ovr_calls
    global random_state_calls

    print('skl_calls=', sklCallsCounter)
    print('daal_calls=', daal4pyCallsCounter)

    print('csr_data_calls=', csr_data_calls)
    print('penalty_calls=', penalty_calls)
    print('kernel_rbf_calls=', kernel_rbf_calls)
    print('kernel_linear_calls=', kernel_linear_calls)
    print('kernel_poly_calls=', kernel_poly_calls)
    print('kernel_sigmoid_calls=', kernel_sigmoid_calls)
    print('kernel_precomputed_calls=', kernel_precomputed_calls)
    print('shrinking_calls=', shrinking_calls)
    print('probability_calls=', probability_calls)
    print('class_weight_calls=', class_weight_calls)
    print('verbose_calls=', verbose_calls)
    print('df_shape_ovo_calls=', df_shape_ovo_calls)
    print('df_shape_ovr_calls=', df_shape_ovr_calls)
    print('random_state_calls=', random_state_calls)

class SVC(svm.SVC):
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto_deprecated',
                 coef0=0.0, shrinking=True, probability=False,
                 tol=1e-3, cache_size=200, class_weight=None,
                 verbose=False, max_iter=-1, decision_function_shape='ovr',
                 random_state=None):

        super().__init__(
            kernel=kernel, degree=degree, gamma=gamma,
            coef0=coef0, tol=tol, C=C, shrinking=shrinking,
            probability=probability, cache_size=cache_size,
            class_weight=class_weight, verbose=verbose, max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            random_state=random_state)

    def decision_function(self, X):
        global daal4pyCallsCounter
        global sklCallsCounter

        global csr_data_calls
        global penalty_calls
        global kernel_rbf_calls
        global kernel_linear_calls
        global kernel_poly_calls
        global kernel_sigmoid_calls
        global kernel_precomputed_calls
        global shrinking_calls
        global probability_calls
        global class_weight_calls
        global verbose_calls
        global df_shape_ovo_calls
        global df_shape_ovr_calls
        global random_state_calls

        sklCallsCounter += 1

        if sparse.issparse(X):
            csr_data_calls += 1

        if self.kernel is 'rbf':
            kernel_rbf_calls += 1

        if self.kernel is 'linear':
            kernel_linear_calls  += 1

        if self.kernel is 'poly':
            kernel_poly_calls += 1

        if self.kernel is 'sigmoid':
            kernel_sigmoid_calls += 1

        if self.kernel is 'precomputed':
            kernel_precomputed_calls += 1

        if self.shrinking is True:
            shrinking_calls += 1

        if self.probability is True:
            probability_calls += 1

        if self.class_weight is not None:
            class_weight_calls += 1

        if self.verbose is True:
            verbose_calls += 1

        if self.decision_function_shape is 'ovo':
            df_shape_ovo_calls += 1

        if self.decision_function_shape is 'ovr':
            df_shape_ovr_calls += 1

        if self.random_state is not None:
            random_state_calls += 1

        print_counters()
        return super().decision_function(X)

    def predict(self, X):
        global daal4pyCallsCounter
        global sklCallsCounter

        global csr_data_calls
        global penalty_calls
        global kernel_rbf_calls
        global kernel_linear_calls
        global kernel_poly_calls
        global kernel_sigmoid_calls
        global kernel_precomputed_calls
        global shrinking_calls
        global probability_calls
        global class_weight_calls
        global verbose_calls
        global df_shape_ovo_calls
        global df_shape_ovr_calls
        global random_state_calls

        sklCallsCounter += 1

        if sparse.issparse(X):
            csr_data_calls += 1

        if self.kernel is 'rbf':
            kernel_rbf_calls += 1

        if self.kernel is 'linear':
            kernel_linear_calls  += 1

        if self.kernel is 'poly':
            kernel_poly_calls += 1

        if self.kernel is 'sigmoid':
            kernel_sigmoid_calls += 1

        if self.kernel is 'precomputed':
            kernel_precomputed_calls += 1

        if self.shrinking is True:
            shrinking_calls += 1

        if self.probability is True:
            probability_calls += 1

        if self.class_weight is not None:
            class_weight_calls += 1

        if self.verbose is True:
            verbose_calls += 1

        if self.decision_function_shape is 'ovo':
            df_shape_ovo_calls += 1

        if self.decision_function_shape is 'ovr':
            df_shape_ovr_calls += 1

        if self.random_state is not None:
            random_state_calls += 1

        print_counters()
        return super().predict(X)

    def _predict_proba(self, X):
        global daal4pyCallsCounter
        global sklCallsCounter

        global csr_data_calls
        global penalty_calls
        global kernel_rbf_calls
        global kernel_linear_calls
        global kernel_poly_calls
        global kernel_sigmoid_calls
        global kernel_precomputed_calls
        global shrinking_calls
        global probability_calls
        global class_weight_calls
        global verbose_calls
        global df_shape_ovo_calls
        global df_shape_ovr_calls
        global random_state_calls

        sklCallsCounter += 1

        if sparse.issparse(X):
            csr_data_calls += 1

        if self.kernel is 'rbf':
            kernel_rbf_calls += 1

        if self.kernel is 'linear':
            kernel_linear_calls  += 1

        if self.kernel is 'poly':
            kernel_poly_calls += 1

        if self.kernel is 'sigmoid':
            kernel_sigmoid_calls += 1

        if self.kernel is 'precomputed':
            kernel_precomputed_calls += 1

        if self.shrinking is True:
            shrinking_calls += 1

        if self.probability is True:
            probability_calls += 1

        if self.class_weight is not None:
            class_weight_calls += 1

        if self.verbose is True:
            verbose_calls += 1

        if self.decision_function_shape is 'ovo':
            df_shape_ovo_calls += 1

        if self.decision_function_shape is 'ovr':
            df_shape_ovr_calls += 1

        if self.random_state is not None:
            random_state_calls += 1

        print_counters()
        return super()._predict_proba(X)

    def _predict_log_proba(self, X):
        global daal4pyCallsCounter
        global sklCallsCounter

        global csr_data_calls
        global penalty_calls
        global kernel_rbf_calls
        global kernel_linear_calls
        global kernel_poly_calls
        global kernel_sigmoid_calls
        global kernel_precomputed_calls
        global shrinking_calls
        global probability_calls
        global class_weight_calls
        global verbose_calls
        global df_shape_ovo_calls
        global df_shape_ovr_calls
        global random_state_calls

        sklCallsCounter += 1

        if sparse.issparse(X):
            csr_data_calls += 1

        if self.kernel is 'rbf':
            kernel_rbf_calls += 1

        if self.kernel is 'linear':
            kernel_linear_calls  += 1

        if self.kernel is 'poly':
            kernel_poly_calls += 1

        if self.kernel is 'sigmoid':
            kernel_sigmoid_calls += 1

        if self.kernel is 'precomputed':
            kernel_precomputed_calls += 1

        if self.shrinking is True:
            shrinking_calls += 1

        if self.probability is True:
            probability_calls += 1

        if self.class_weight is not None:
            class_weight_calls += 1

        if self.verbose is True:
            verbose_calls += 1

        if self.decision_function_shape is 'ovo':
            df_shape_ovo_calls += 1

        if self.decision_function_shape is 'ovr':
            df_shape_ovr_calls += 1

        if self.random_state is not None:
            random_state_calls += 1

        print_counters()
        return super()._predict_log_proba(X)

svm.SVC = SVC
