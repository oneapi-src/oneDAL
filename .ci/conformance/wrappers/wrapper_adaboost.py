from sklearn.ensemble import AdaBoostClassifier as AdaBoostClassifier_skl
from sklearn.ensemble import AdaBoostRegressor
from scipy import sparse

#Global counters for counting calls from d4py and Skl
daal4pyCallsCounter = 0
sklCallsCounter = 0

'''
Global counters for counting percent usage of incompatible
paramenters by daal4py
'''
csr_counter = 0
samme_counter = 0
sammer_counter = 0
weight_counter = 0


class AdaBoostClassifier(AdaBoostClassifier_skl):
    def __init__(self,
                 base_estimator=None,
                 n_estimators=50,
                 learning_rate=1.,
                 algorithm='SAMME.R',
                 random_state=None):

        super().__init__(
            base_estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
            algorithm=algorithm)

        # self.algorithm = algorithm

    def fit(self, X, y, sample_weight=None):
        global sklCallsCounter
        global csr_counter
        global sammer_counter
        global samme_counter
        global weight_counter
        sklCallsCounter += 1

        if sparse.issparse(X):
            csr_counter += 1
        if sample_weight != None:
            weight_counter += 1

        if self.algorithm == 'SAMME.R':
            sammer_counter += 1
        else:
            samme_counter += 1
        print('sammer_calls=', sammer_counter)
        print('samme_calls=', samme_counter)
        print('skl_calls=', sklCallsCounter)
        print('daal_calls=', daal4pyCallsCounter)
        print('sparse_calls=', csr_counter)
        print('weight_calls=', weight_counter)
        return super().fit(X, y, sample_weight)

    def _validate_estimator(self):
        super()._validate_estimator()

    def _boost(self, iboost, X, y, sample_weight, random_state):
        return super()._boost(iboost, X, y, sample_weight, random_state)

    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        return super()._boost_real(iboost, X, y, sample_weight, random_state)

    def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
        return super()._boost_discrete(iboost, X, y, sample_weight, random_state)

    def predict(self, X):
        global sklCallsCounter
        global sammer_counter
        global samme_counter
        global csr_counter
        sklCallsCounter += 1

        if self.algorithm == 'SAMME.R':
            sammer_counter += 1
        else:
            samme_counter += 1
        if sparse.issparse(X):
            csr_counter += 1

        print('skl_calls=', sklCallsCounter)
        print('daal_calls=', daal4pyCallsCounter)
        print('sammer_calls=', sammer_counter)
        print('samme_calls=', samme_counter)
        print('sparse_calls=', csr_counter)
        return super().predict(X)

    def decision_function(self, X):
        return super().decision_function(X)

    @staticmethod
    def _compute_proba_from_decision(decision, n_classes):
        return super()._compute_proba_from_decision(decision, n_classes)

    def predict_proba(self, X):
        global sklCallsCounter
        global sammer_counter
        global samme_counter
        global csr_counter
        sklCallsCounter += 1

        if self.algorithm == 'SAMME.R':
            sammer_counter += 1
        else:
            samme_counter += 1
        if sparse.issparse(X):
            csr_counter += 1

        print('skl_calls=', sklCallsCounter)
        print('daal_calls=', daal4pyCallsCounter)
        print('sammer_calls=', sammer_counter)
        print('samme_calls=', samme_counter)
        print('sparse_calls=', csr_counter)
        return super().predict_proba(X)

    def predict_log_proba(self, X):
        global sklCallsCounter
        global sammer_counter
        global samme_counter
        global csr_counter
        sklCallsCounter += 1

        if self.algorithm == 'SAMME.R':
            sammer_counter += 1
        else:
            samme_counter += 1

        if sparse.issparse(X):
            csr_counter += 1

        print('skl_calls=', sklCallsCounter)
        print('daal_calls=', daal4pyCallsCounter)
        print('sammer_calls=', sammer_counter)
        print('samme_calls=', samme_counter)
        print('sparse_calls=', csr_counter)
        return super().predict_log_proba(X)