#
#*******************************************************************************
# Copyright 2014-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#******************************************************************************/

import numpy as np
from scipy import sparse
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.base import RegressorMixin, MultiOutputMixin
from sklearn.linear_model.base import LinearModel, _pre_fit
from sklearn.linear_model.coordinate_descent import enet_path
from sklearn.linear_model.coordinate_descent import ElasticNet as ElasticNet_original

import daal4py

csc_counter = 0
fit_intercept_counter = 0
normalize_counter = 0
precompute_counter = 0
precompute_array_counter = 0
copy_X_counter = 0
warm_start_counter = 0
positive_counter = 0
random_state_counter = 0
selection_cyclic_counter = 0
selection_random_counter = 0


#Global counters for counting calls from d4py and Skl
daal4pyCallsCounter = 0
sklCallsCounter = 0

def print_counters():
    global daal4pyCallsCounter
    global sklCallsCounter

    global csc_counter
    global fit_intercept_counter
    global normalize_counter
    global precompute_counter
    global precompute_array_counter
    global copy_X_counter
    global warm_start_counter
    global positive_counter
    global random_state_counter
    global selection_cyclic_counter
    global selection_random_counter

    print('skl_calls=', sklCallsCounter)
    print('daal_calls=', daal4pyCallsCounter)

    print('data_sparse_using=', csc_counter)
    print('intercept_estimation=', fit_intercept_counter)
    print('normalize_data=', normalize_counter)
    print('precomputed_Gram_matrix=', precompute_counter)
    print('pass_precomputed_matrix=', precompute_array_counter)
    print('copy_X_data=', copy_X_counter)
    print('reuse_previous_solution=', warm_start_counter)
    print('positive_coefficients=', positive_counter)
    print('random_state_instance=', random_state_counter)
    print('cyclic_selection=', selection_cyclic_counter)
    print('random_selection=', selection_random_counter, '\n')


class ElasticNet(ElasticNet_original):
    __doc__ = ElasticNet_original.__doc__

    path = staticmethod(enet_path)

    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                 normalize=False, precompute=False, max_iter=1000,
                 copy_X=True, tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic'):
        super(ElasticNet, self).__init__(
            alpha = alpha, l1_ratio = l1_ratio, fit_intercept = fit_intercept,
            normalize = normalize, precompute = precompute, max_iter = max_iter,
            copy_X = copy_X, tol = tol, warm_start = warm_start,
            positive = positive, random_state = random_state, selection = selection)

    def fit(self, X, y, check_input=True):
        global daal4pyCallsCounter
        global sklCallsCounter

        global csc_counter
        global fit_intercept_counter
        global normalize_counter
        global precompute_counter
        global precompute_array_counter
        global copy_X_counter
        global warm_start_counter
        global positive_counter
        global random_state_counter
        global selection_cyclic_counter
        global selection_random_counter

        if sparse.issparse(X) == True:
            csc_counter+=1
        if self.fit_intercept:
            fit_intercept_counter += 1
        if self.normalize:
            normalize_counter += 1
        if ((type(self.precompute) == np.ndarray) or (type(self.precompute) == bool and self.precompute == True)):
            precompute_counter += 1
        if (type(self.precompute) == np.ndarray):
            precompute_array_counter += 1
        if self.copy_X:
            copy_X_counter += 1
        if self.warm_start:
            warm_start_counter += 1
        if self.positive:
            positive_counter += 1
        if not (self.random_state == None):
            random_state_counter += 1
        if self.selection == 'cyclic':
            selection_cyclic_counter += 1
        if self.selection == 'random':
            selection_random_counter += 1

        sklCallsCounter+=1

        print_counters()

        return super().fit(X, y, check_input)

    def sparse_coef_(self):
        super().sparse_coef_()

    def s_decision_function(self, X):
        super().s_decision_function(X)

    def predict(self, X):
        global daal4pyCallsCounter
        global sklCallsCounter

        global csc_counter
        global fit_intercept_counter
        global normalize_counter
        global precompute_counter
        global precompute_array_counter
        global copy_X_counter
        global warm_start_counter
        global positive_counter
        global random_state_counter
        global selection_cyclic_counter
        global selection_random_counter

        if sparse.issparse(X) == True:
            csc_counter+=1
        if self.fit_intercept:
            fit_intercept_counter += 1
        if self.normalize:
            normalize_counter += 1
        if ((type(self.precompute) == np.ndarray) or (type(self.precompute) == bool and self.precompute == True)):
            precompute_counter += 1
        if (type(self.precompute) == np.ndarray):
            precompute_array_counter += 1
        if self.copy_X:
            copy_X_counter += 1
        if self.warm_start:
            warm_start_counter += 1
        if self.positive:
            positive_counter += 1
        if not (self.random_state == None):
            random_state_counter += 1
        if self.selection == 'cyclic':
            selection_cyclic_counter += 1
        if self.selection == 'random':
            selection_random_counter += 1

        sklCallsCounter+=1

        print_counters()

        return super().predict(X)
