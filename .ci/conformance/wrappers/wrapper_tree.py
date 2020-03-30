from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifierSkl
from sklearn.tree import DecisionTreeRegressor as DecisionTreeRegressorSkl
from scipy import sparse
from sklearn import __version__ as sklearn_version
from distutils.version import LooseVersion

#Global counters for counting calls from d4py and Skl
daal4pyCallsCounter = 0
sklCallsCounter = 0

'''
Global counters for counting percent usage of incompatible
paramenters by daal4py
'''
csr_counter = 0
weight_counter = 0
check_input_counter = 0
X_idx_sorted_counter = 0

# decision tree classification criterion
gini_criterion_counter = 0
entropy_criterion_counter = 0

# decision tree regression criterion
mse_criterion_counter = 0
friedman_mse_criterion_counter = 0
mae_criterion_counter = 0

best_splitter_counter = 0
random_splitter_counter = 0
max_depth_counter = 0
min_samples_split_counter = 0
min_samples_leaf_counter = 0
min_weight_fraction_leaf_counter = 0
max_features_counter = 0
auto_max_features_counter = 0
sqrt_max_features_counter = 0
log2_max_features_counter = 0
max_leaf_node_counter = 0
min_impurity_decrease_counter = 0
min_impurity_split_counter = 0
class_weight_counter = 0
ccp_alpha_counter = 0
presort_counter = 0

def print_counters():
    global sklCallsCounter
    global daal4pyCallsCounter

    global csr_counter
    global weight_counter
    global check_input_counter
    global X_idx_sorted_counter

    global mse_criterion_counter
    global friedman_mse_criterion_counter
    global mae_criterion_counter

    global gini_criterion_counter
    global entropy_criterion_counter
    global best_splitter_counter
    global random_splitter_counter
    global max_depth_counter
    global min_samples_split_counter
    global min_samples_leaf_counter
    global min_weight_fraction_leaf_counter
    global max_features_counter
    global auto_max_features_counter
    global sqrt_max_features_counter
    global log2_max_features_counter
    global max_leaf_node_counter
    global min_impurity_decrease_counter
    global min_impurity_split_counter
    global class_weight_counter
    global ccp_alpha_counter
    global presort_counter

    print('skl_calls=', sklCallsCounter)
    print('daal_calls=', daal4pyCallsCounter)

    print("sparse_calls=", csr_counter)
    print("weight_calls=", weight_counter)
    print("check_input_calls=", check_input_counter)
    print("X_idx_sorted_calls=", X_idx_sorted_counter)
    print("mse_criterion_calls=", mse_criterion_counter)
    print("friedman_mse_criterion_calls=", friedman_mse_criterion_counter)
    print("mae_criterion_calls=", mae_criterion_counter)
    print("gini_criterion_counter=", gini_criterion_counter)
    print("entropy_criterion_counter=", entropy_criterion_counter)
    print("best_splitter_calls=", best_splitter_counter)
    print("random_splitter_calls=", random_splitter_counter)
    print("max_depth_calls=", max_depth_counter)
    print("min_samples_split_calls=", min_samples_split_counter)
    print("min_samples_leaf_calls=", min_samples_leaf_counter)
    print("min_weight_fraction_leaf_calls=", min_weight_fraction_leaf_counter)
    print("max_features_calls=", max_features_counter)
    print("auto_max_features_calls=", auto_max_features_counter)
    print("sqrt_max_features_calls=", sqrt_max_features_counter)
    print("log2_max_features_calls=", log2_max_features_counter)
    print("max_leaf_node_calls=", max_leaf_node_counter)
    print("min_impurity_decrease_calls=", min_impurity_decrease_counter)
    print("min_impurity_split_calls=", min_impurity_split_counter)
    print("class_weight_calls=", class_weight_counter)
    print("ccp_alpha_counter=", ccp_alpha_counter)
    print("presort_calls=", presort_counter, "\n")



if (LooseVersion(sklearn_version) >= LooseVersion("0.22")):
    class DecisionTreeClassifier(DecisionTreeClassifierSkl):
        def __init__(self,
                     criterion="gini",
                     splitter="best",
                     max_depth=None,
                     min_samples_split=2,
                     min_samples_leaf=1,
                     min_weight_fraction_leaf=0.,
                     max_features=None,
                     random_state=None,
                     max_leaf_nodes=None,
                     min_impurity_decrease=0.,
                     min_impurity_split=None,
                     class_weight=None,
                     presort='deprecated',
                     ccp_alpha=0.0):
            super().__init__(
                criterion=criterion,
                splitter=splitter,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                class_weight=class_weight,
                random_state=random_state,
                min_impurity_decrease=min_impurity_decrease,
                min_impurity_split=min_impurity_split,
                presort=presort,
                ccp_alpha=ccp_alpha)

        def fit(self, X, y, sample_weight=None, check_input=True,
                X_idx_sorted=None):

            global sklCallsCounter
            global daal4pyCallsCounter

            global csr_counter
            global weight_counter
            global check_input_counter
            global X_idx_sorted_counter

            global mse_criterion_counter
            global friedman_mse_criterion_counter
            global mae_criterion_counter

            global gini_criterion_counter
            global entropy_criterion_counter
            global best_splitter_counter
            global random_splitter_counter
            global max_depth_counter
            global min_samples_split_counter
            global min_samples_leaf_counter
            global min_weight_fraction_leaf_counter
            global max_features_counter
            global auto_max_features_counter
            global sqrt_max_features_counter
            global log2_max_features_counter
            global max_leaf_node_counter
            global min_impurity_decrease_counter
            global min_impurity_split_counter
            global class_weight_counter
            global ccp_alpha_counter
            global presort_counter

            if self.criterion == "gini":
                gini_criterion_counter += 1
            if self.criterion == "entropy":
                entropy_criterion_counter += 1
            if self.splitter == "best":
                best_splitter_counter += 1
            if self.splitter == "random":
                random_splitter_counter += 1
            if self.max_depth is not None:
                max_depth_counter += 1

            min_samples_leaf_counter += 1

            min_weight_fraction_leaf_counter += 1
            if self.max_features is not None:
                if self.max_features == "auto":
                    auto_max_features_counter += 1
                if self.max_features == "sqrt":
                    sqrt_max_features_counter += 1
                if self.max_features == "log2":
                    log2_max_features_counter += 1
            else:
                max_features_counter += 1

            if self.max_leaf_nodes is not None:
                max_leaf_node_counter += 1
            min_impurity_decrease_counter += 1
            min_samples_split_counter += 1
            if self.class_weight is not None:
                class_weight_counter += 1

            ccp_alpha_counter += 1

            sklCallsCounter += 1

            if sparse.issparse(X):
                csr_counter += 1
            if sample_weight is not None:
                weight_counter += 1

            if check_input:
                check_input_counter += 1
            if X_idx_sorted is not None:
                X_idx_sorted_counter += 1

            print_counters()

            super().fit(
                X, y,
                sample_weight=sample_weight,
                check_input=check_input,
                X_idx_sorted=X_idx_sorted)
            return self

        def predict(self, X, check_input=True):
            global sklCallsCounter
            global daal4pyCallsCounter

            global csr_counter
            global weight_counter
            global check_input_counter
            global X_idx_sorted_counter

            global mse_criterion_counter
            global friedman_mse_criterion_counter
            global mae_criterion_counter

            global gini_criterion_counter
            global entropy_criterion_counter
            global best_splitter_counter
            global random_splitter_counter
            global max_depth_counter
            global min_samples_split_counter
            global min_samples_leaf_counter
            global min_weight_fraction_leaf_counter
            global max_features_counter
            global auto_max_features_counter
            global sqrt_max_features_counter
            global log2_max_features_counter
            global max_leaf_node_counter
            global min_impurity_decrease_counter
            global min_impurity_split_counter
            global class_weight_counter
            global ccp_alpha_counter
            global presort_counter

            sklCallsCounter += 1
            if check_input:
                check_input_counter += 1
            if sparse.issparse(X):
                csr_counter += 1

            if self.criterion == "gini":
                gini_criterion_counter += 1
            if self.criterion == "entropy":
                entropy_criterion_counter += 1
            if self.splitter == "best":
                best_splitter_counter += 1
            if self.splitter == "random":
                random_splitter_counter += 1
            if self.max_depth is not None:
                max_depth_counter += 1

            min_samples_leaf_counter += 1

            min_weight_fraction_leaf_counter += 1
            if self.max_features is not None:
                if self.max_features == "auto":
                    auto_max_features_counter += 1
                if self.max_features == "sqrt":
                    sqrt_max_features_counter += 1
                if self.max_features == "log2":
                    log2_max_features_counter += 1
            else:
                max_features_counter += 1

            if self.max_leaf_nodes is not None:
                max_leaf_node_counter += 1
            min_impurity_decrease_counter += 1
            min_samples_split_counter += 1
            if self.class_weight is not None:
                class_weight_counter += 1

            ccp_alpha_counter += 1

            print_counters()

            return super().predict(X, check_input=check_input)

        def predict_proba(self, X, check_input=True):
            global sklCallsCounter
            global daal4pyCallsCounter

            global csr_counter
            global weight_counter
            global check_input_counter
            global X_idx_sorted_counter

            global mse_criterion_counter
            global friedman_mse_criterion_counter
            global mae_criterion_counter

            global gini_criterion_counter
            global entropy_criterion_counter
            global best_splitter_counter
            global random_splitter_counter
            global max_depth_counter
            global min_samples_split_counter
            global min_samples_leaf_counter
            global min_weight_fraction_leaf_counter
            global max_features_counter
            global auto_max_features_counter
            global sqrt_max_features_counter
            global log2_max_features_counter
            global max_leaf_node_counter
            global min_impurity_decrease_counter
            global min_impurity_split_counter
            global class_weight_counter
            global ccp_alpha_counter
            global presort_counter

            sklCallsCounter += 1
            if sparse.issparse(X):
                csr_counter += 1
            if check_input:
                check_input_counter += 1

            if self.criterion == "gini":
                gini_criterion_counter += 1
            if self.criterion == "entropy":
                entropy_criterion_counter += 1
            if self.splitter == "best":
                best_splitter_counter += 1
            if self.splitter == "random":
                random_splitter_counter += 1
            if self.max_depth is not None:
                max_depth_counter += 1

            min_samples_leaf_counter += 1

            min_weight_fraction_leaf_counter += 1
            if self.max_features is not None:
                if self.max_features == "auto":
                    auto_max_features_counter += 1
                if self.max_features == "sqrt":
                    sqrt_max_features_counter += 1
                if self.max_features == "log2":
                    log2_max_features_counter += 1
            else:
                max_features_counter += 1

            if self.max_leaf_nodes is not None:
                max_leaf_node_counter += 1
            min_impurity_decrease_counter += 1
            min_samples_split_counter += 1
            if self.class_weight is not None:
                class_weight_counter += 1

            ccp_alpha_counter += 1

            print_counters()

            return super().predict_proba(X, check_input=check_input)

        def predict_log_proba(self, X):

            global sklCallsCounter
            global daal4pyCallsCounter

            global csr_counter
            global weight_counter
            global check_input_counter
            global X_idx_sorted_counter

            global mse_criterion_counter
            global friedman_mse_criterion_counter
            global mae_criterion_counter

            global gini_criterion_counter
            global entropy_criterion_counter
            global best_splitter_counter
            global random_splitter_counter
            global max_depth_counter
            global min_samples_split_counter
            global min_samples_leaf_counter
            global min_weight_fraction_leaf_counter
            global max_features_counter
            global auto_max_features_counter
            global sqrt_max_features_counter
            global log2_max_features_counter
            global max_leaf_node_counter
            global min_impurity_decrease_counter
            global min_impurity_split_counter
            global class_weight_counter
            global ccp_alpha_counter
            global presort_counter

            sklCallsCounter += 1

            if sparse.issparse(X):
                csr_counter += 1
            if self.criterion == "gini":
                gini_criterion_counter += 1
            if self.criterion == "entropy":
                entropy_criterion_counter += 1
            if self.splitter == "best":
                best_splitter_counter += 1
            if self.splitter == "random":
                random_splitter_counter += 1
            if self.max_depth is not None:
                max_depth_counter += 1

            min_samples_leaf_counter += 1

            min_weight_fraction_leaf_counter += 1
            if self.max_features is not None:
                if self.max_features == "auto":
                    auto_max_features_counter += 1
                if self.max_features == "sqrt":
                    sqrt_max_features_counter += 1
                if self.max_features == "log2":
                    log2_max_features_counter += 1
            else:
                max_features_counter += 1

            if self.max_leaf_nodes is not None:
                max_leaf_node_counter += 1
            min_impurity_decrease_counter += 1
            min_samples_split_counter += 1
            if self.class_weight is not None:
                class_weight_counter += 1

            ccp_alpha_counter += 1

            print_counters()

            return super().predict_log_proba(X)


    class DecisionTreeRegressor(DecisionTreeRegressorSkl):
        def __init__(self,
                     criterion="mse",
                     splitter="best",
                     max_depth=None,
                     min_samples_split=2,
                     min_samples_leaf=1,
                     min_weight_fraction_leaf=0.,
                     max_features=None,
                     random_state=None,
                     max_leaf_nodes=None,
                     min_impurity_decrease=0.,
                     min_impurity_split=None,
                     presort='deprecated',
                     ccp_alpha=0.0):
            super().__init__(
                criterion=criterion,
                splitter=splitter,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                random_state=random_state,
                min_impurity_decrease=min_impurity_decrease,
                min_impurity_split=min_impurity_split,
                presort=presort,
                ccp_alpha=ccp_alpha)


        def fit(self, X, y, sample_weight=None, check_input=True,
                X_idx_sorted=None):

            global sklCallsCounter
            global daal4pyCallsCounter

            global csr_counter
            global weight_counter
            global check_input_counter
            global X_idx_sorted_counter

            global mse_criterion_counter
            global friedman_mse_criterion_counter
            global mae_criterion_counter

            global gini_criterion_counter
            global entropy_criterion_counter
            global best_splitter_counter
            global random_splitter_counter
            global max_depth_counter
            global min_samples_split_counter
            global min_samples_leaf_counter
            global min_weight_fraction_leaf_counter
            global max_features_counter
            global auto_max_features_counter
            global sqrt_max_features_counter
            global log2_max_features_counter
            global max_leaf_node_counter
            global min_impurity_decrease_counter
            global min_impurity_split_counter
            global class_weight_counter
            global ccp_alpha_counter
            global presort_counter

            if self.criterion == "mse":
                mse_criterion_counter += 1
            if self.criterion == "friedman_mse":
                friedman_mse_criterion_counter += 1
            if self.criterion == "mae":
                mae_criterion_counter += 1
            if self.splitter == "best":
                best_splitter_counter += 1
            if self.splitter == "random":
                random_splitter_counter += 1
            if self.max_depth is not None:
                max_depth_counter += 1

            min_samples_leaf_counter += 1

            min_weight_fraction_leaf_counter += 1
            if self.max_features is not None:
                if self.max_features == "auto":
                    auto_max_features_counter += 1
                if self.max_features == "sqrt":
                    sqrt_max_features_counter += 1
                if self.max_features == "log2":
                    log2_max_features_counter += 1
            else:
                max_features_counter += 1

            if self.max_leaf_nodes is not None:
                max_leaf_node_counter += 1
            min_impurity_decrease_counter += 1
            min_samples_split_counter += 1

            ccp_alpha_counter += 1

            sklCallsCounter += 1

            if sparse.issparse(X):
                csr_counter += 1
            if sample_weight is not None:
                weight_counter += 1

            if check_input:
                check_input_counter += 1
            if X_idx_sorted is not None:
                X_idx_sorted_counter += 1

            print_counters()

            super().fit(
                X, y,
                sample_weight=sample_weight,
                check_input=check_input,
                X_idx_sorted=X_idx_sorted)
            return self

        def predict(self, X, check_input=True):

            global sklCallsCounter
            global daal4pyCallsCounter

            global csr_counter
            global weight_counter
            global check_input_counter
            global X_idx_sorted_counter

            global mse_criterion_counter
            global friedman_mse_criterion_counter
            global mae_criterion_counter

            global gini_criterion_counter
            global entropy_criterion_counter
            global best_splitter_counter
            global random_splitter_counter
            global max_depth_counter
            global min_samples_split_counter
            global min_samples_leaf_counter
            global min_weight_fraction_leaf_counter
            global max_features_counter
            global auto_max_features_counter
            global sqrt_max_features_counter
            global log2_max_features_counter
            global max_leaf_node_counter
            global min_impurity_decrease_counter
            global min_impurity_split_counter
            global class_weight_counter
            global ccp_alpha_counter
            global presort_counter

            if self.criterion == "mse":
                mse_criterion_counter += 1
            if self.criterion == "friedman_mse":
                friedman_mse_criterion_counter += 1
            if self.criterion == "mae":
                mae_criterion_counter += 1
            if self.splitter == "best":
                best_splitter_counter += 1
            if self.splitter == "random":
                random_splitter_counter += 1
            if self.max_depth is not None:
                max_depth_counter += 1

            min_samples_leaf_counter += 1

            min_weight_fraction_leaf_counter += 1
            if self.max_features is not None:
                if self.max_features == "auto":
                    auto_max_features_counter += 1
                if self.max_features == "sqrt":
                    sqrt_max_features_counter += 1
                if self.max_features == "log2":
                    log2_max_features_counter += 1
            else:
                max_features_counter += 1

            if self.max_leaf_nodes is not None:
                max_leaf_node_counter += 1
            min_impurity_decrease_counter += 1
            min_samples_split_counter += 1

            ccp_alpha_counter += 1

            sklCallsCounter += 1

            if sparse.issparse(X):
                csr_counter += 1

            if check_input:
                check_input_counter += 1

            print_counters()
            return super().predict(X, check_input=check_input)
else:
    class DecisionTreeClassifier(DecisionTreeClassifierSkl):
        def __init__(self,
                     criterion="gini",
                     splitter="best",
                     max_depth=None,
                     min_samples_split=2,
                     min_samples_leaf=1,
                     min_weight_fraction_leaf=0.,
                     max_features=None,
                     random_state=None,
                     max_leaf_nodes=None,
                     min_impurity_decrease=0.,
                     min_impurity_split=None,
                     class_weight=None,
                     presort=False):
            super().__init__(
                criterion=criterion,
                splitter=splitter,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                class_weight=class_weight,
                random_state=random_state,
                min_impurity_decrease=min_impurity_decrease,
                min_impurity_split=min_impurity_split,
                presort=presort)

        def fit(self, X, y, sample_weight=None, check_input=True,
                X_idx_sorted=None):

            global sklCallsCounter
            global daal4pyCallsCounter

            global csr_counter
            global weight_counter
            global check_input_counter
            global X_idx_sorted_counter

            global mse_criterion_counter
            global friedman_mse_criterion_counter
            global mae_criterion_counter

            global gini_criterion_counter
            global entropy_criterion_counter
            global best_splitter_counter
            global random_splitter_counter
            global max_depth_counter
            global min_samples_split_counter
            global min_samples_leaf_counter
            global min_weight_fraction_leaf_counter
            global max_features_counter
            global auto_max_features_counter
            global sqrt_max_features_counter
            global log2_max_features_counter
            global max_leaf_node_counter
            global min_impurity_decrease_counter
            global min_impurity_split_counter
            global class_weight_counter
            global ccp_alpha_counter
            global presort_counter

            if self.criterion == "gini":
                gini_criterion_counter += 1
            if self.criterion == "entropy":
                entropy_criterion_counter += 1
            if self.splitter == "best":
                best_splitter_counter += 1
            if self.splitter == "random":
                random_splitter_counter += 1
            if self.max_depth is not None:
                max_depth_counter += 1

            min_samples_leaf_counter += 1

            min_weight_fraction_leaf_counter += 1
            if self.max_features is not None:
                if self.max_features == "auto":
                    auto_max_features_counter += 1
                if self.max_features == "sqrt":
                    sqrt_max_features_counter += 1
                if self.max_features == "log2":
                    log2_max_features_counter += 1
            else:
                max_features_counter += 1

            if self.max_leaf_nodes is not None:
                max_leaf_node_counter += 1
            min_impurity_decrease_counter += 1
            min_samples_split_counter += 1
            if self.class_weight is not None:
                class_weight_counter += 1

            if self.presort:
                presort_counter += 1

            sklCallsCounter += 1

            if sparse.issparse(X):
                csr_counter += 1
            if sample_weight is not None:
                weight_counter += 1

            if check_input:
                check_input_counter += 1
            if X_idx_sorted is not None:
                X_idx_sorted_counter += 1

            print_counters()

            super().fit(
                X, y,
                sample_weight=sample_weight,
                check_input=check_input,
                X_idx_sorted=X_idx_sorted)
            return self

        def predict(self, X, check_input=True):
            global sklCallsCounter
            global daal4pyCallsCounter

            global csr_counter
            global weight_counter
            global check_input_counter
            global X_idx_sorted_counter

            global mse_criterion_counter
            global friedman_mse_criterion_counter
            global mae_criterion_counter

            global gini_criterion_counter
            global entropy_criterion_counter
            global best_splitter_counter
            global random_splitter_counter
            global max_depth_counter
            global min_samples_split_counter
            global min_samples_leaf_counter
            global min_weight_fraction_leaf_counter
            global max_features_counter
            global auto_max_features_counter
            global sqrt_max_features_counter
            global log2_max_features_counter
            global max_leaf_node_counter
            global min_impurity_decrease_counter
            global min_impurity_split_counter
            global class_weight_counter
            global ccp_alpha_counter
            global presort_counter

            sklCallsCounter += 1
            if check_input:
                check_input_counter += 1
            if sparse.issparse(X):
                csr_counter += 1

            if self.criterion == "gini":
                gini_criterion_counter += 1
            if self.criterion == "entropy":
                entropy_criterion_counter += 1
            if self.splitter == "best":
                best_splitter_counter += 1
            if self.splitter == "random":
                random_splitter_counter += 1
            if self.max_depth is not None:
                max_depth_counter += 1

            min_samples_leaf_counter += 1

            min_weight_fraction_leaf_counter += 1
            if self.max_features is not None:
                if self.max_features == "auto":
                    auto_max_features_counter += 1
                if self.max_features == "sqrt":
                    sqrt_max_features_counter += 1
                if self.max_features == "log2":
                    log2_max_features_counter += 1
            else:
                max_features_counter += 1

            if self.max_leaf_nodes is not None:
                max_leaf_node_counter += 1
            min_impurity_decrease_counter += 1
            min_samples_split_counter += 1
            if self.class_weight is not None:
                class_weight_counter += 1

            if self.presort:
                presort_counter += 1

            print_counters()

            return super().predict(X, check_input=check_input)

        def predict_proba(self, X, check_input=True):
            global sklCallsCounter
            global daal4pyCallsCounter

            global csr_counter
            global weight_counter
            global check_input_counter
            global X_idx_sorted_counter

            global mse_criterion_counter
            global friedman_mse_criterion_counter
            global mae_criterion_counter

            global gini_criterion_counter
            global entropy_criterion_counter
            global best_splitter_counter
            global random_splitter_counter
            global max_depth_counter
            global min_samples_split_counter
            global min_samples_leaf_counter
            global min_weight_fraction_leaf_counter
            global max_features_counter
            global auto_max_features_counter
            global sqrt_max_features_counter
            global log2_max_features_counter
            global max_leaf_node_counter
            global min_impurity_decrease_counter
            global min_impurity_split_counter
            global class_weight_counter
            global ccp_alpha_counter
            global presort_counter

            sklCallsCounter += 1
            if sparse.issparse(X):
                csr_counter += 1
            if check_input:
                check_input_counter += 1

            if self.criterion == "gini":
                gini_criterion_counter += 1
            if self.criterion == "entropy":
                entropy_criterion_counter += 1
            if self.splitter == "best":
                best_splitter_counter += 1
            if self.splitter == "random":
                random_splitter_counter += 1
            if self.max_depth is not None:
                max_depth_counter += 1

            min_samples_leaf_counter += 1

            min_weight_fraction_leaf_counter += 1
            if self.max_features is not None:
                if self.max_features == "auto":
                    auto_max_features_counter += 1
                if self.max_features == "sqrt":
                    sqrt_max_features_counter += 1
                if self.max_features == "log2":
                    log2_max_features_counter += 1
            else:
                max_features_counter += 1

            if self.max_leaf_nodes is not None:
                max_leaf_node_counter += 1
            min_impurity_decrease_counter += 1
            min_samples_split_counter += 1
            if self.class_weight is not None:
                class_weight_counter += 1

            if self.presort:
                presort_counter += 1

            print_counters()

            return super().predict_proba(X, check_input=check_input)

        def predict_log_proba(self, X):
            global sklCallsCounter
            global daal4pyCallsCounter

            global csr_counter
            global weight_counter
            global check_input_counter
            global X_idx_sorted_counter

            global mse_criterion_counter
            global friedman_mse_criterion_counter
            global mae_criterion_counter

            global gini_criterion_counter
            global entropy_criterion_counter
            global best_splitter_counter
            global random_splitter_counter
            global max_depth_counter
            global min_samples_split_counter
            global min_samples_leaf_counter
            global min_weight_fraction_leaf_counter
            global max_features_counter
            global auto_max_features_counter
            global sqrt_max_features_counter
            global log2_max_features_counter
            global max_leaf_node_counter
            global min_impurity_decrease_counter
            global min_impurity_split_counter
            global class_weight_counter
            global ccp_alpha_counter
            global presort_counter

            sklCallsCounter += 1

            if sparse.issparse(X):
                csr_counter += 1
            if self.criterion == "gini":
                gini_criterion_counter += 1
            if self.criterion == "entropy":
                entropy_criterion_counter += 1
            if self.splitter == "best":
                best_splitter_counter += 1
            if self.splitter == "random":
                random_splitter_counter += 1
            if self.max_depth is not None:
                max_depth_counter += 1

            min_samples_leaf_counter += 1

            min_weight_fraction_leaf_counter += 1
            if self.max_features is not None:
                if self.max_features == "auto":
                    auto_max_features_counter += 1
                if self.max_features == "sqrt":
                    sqrt_max_features_counter += 1
                if self.max_features == "log2":
                    log2_max_features_counter += 1
            else:
                max_features_counter += 1

            if self.max_leaf_nodes is not None:
                max_leaf_node_counter += 1
            min_impurity_decrease_counter += 1
            min_samples_split_counter += 1
            if self.class_weight is not None:
                class_weight_counter += 1

            if self.presort:
                presort_counter += 1

            print_counters()

            return super().predict_log_proba(X)


    class DecisionTreeRegressor(DecisionTreeRegressorSkl):
        def __init__(self,
                     criterion="mse",
                     splitter="best",
                     max_depth=None,
                     min_samples_split=2,
                     min_samples_leaf=1,
                     min_weight_fraction_leaf=0.,
                     max_features=None,
                     random_state=None,
                     max_leaf_nodes=None,
                     min_impurity_decrease=0.,
                     min_impurity_split=None,
                     presort=False):
            super().__init__(
                criterion=criterion,
                splitter=splitter,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                random_state=random_state,
                min_impurity_decrease=min_impurity_decrease,
                min_impurity_split=min_impurity_split,
                presort=presort)


        def fit(self, X, y, sample_weight=None, check_input=True,
                X_idx_sorted=None):

            global sklCallsCounter
            global daal4pyCallsCounter

            global csr_counter
            global weight_counter
            global check_input_counter
            global X_idx_sorted_counter

            global mse_criterion_counter
            global friedman_mse_criterion_counter
            global mae_criterion_counter

            global gini_criterion_counter
            global entropy_criterion_counter
            global best_splitter_counter
            global random_splitter_counter
            global max_depth_counter
            global min_samples_split_counter
            global min_samples_leaf_counter
            global min_weight_fraction_leaf_counter
            global max_features_counter
            global auto_max_features_counter
            global sqrt_max_features_counter
            global log2_max_features_counter
            global max_leaf_node_counter
            global min_impurity_decrease_counter
            global min_impurity_split_counter
            global class_weight_counter
            global ccp_alpha_counter
            global presort_counter

            if self.criterion == "mse":
                mse_criterion_counter += 1
            if self.criterion == "friedman_mse":
                friedman_mse_criterion_counter += 1
            if self.criterion == "mae":
                mae_criterion_counter += 1
            if self.splitter == "best":
                best_splitter_counter += 1
            if self.splitter == "random":
                random_splitter_counter += 1
            if self.max_depth is not None:
                max_depth_counter += 1

            min_samples_leaf_counter += 1

            min_weight_fraction_leaf_counter += 1
            if self.max_features is not None:
                if self.max_features == "auto":
                    auto_max_features_counter += 1
                if self.max_features == "sqrt":
                    sqrt_max_features_counter += 1
                if self.max_features == "log2":
                    log2_max_features_counter += 1
            else:
                max_features_counter += 1

            if self.max_leaf_nodes is not None:
                max_leaf_node_counter += 1
            min_impurity_decrease_counter += 1
            min_samples_split_counter += 1

            if self.presort:
                presort_counter += 1

            sklCallsCounter += 1

            if sparse.issparse(X):
                csr_counter += 1
            if sample_weight is not None:
                weight_counter += 1

            if check_input:
                check_input_counter += 1
            if X_idx_sorted is not None:
                X_idx_sorted_counter += 1

            print_counters()

            super().fit(
                X, y,
                sample_weight=sample_weight,
                check_input=check_input,
                X_idx_sorted=X_idx_sorted)
            return self

        def predict(self, X, check_input=True):

            global sklCallsCounter
            global daal4pyCallsCounter

            global csr_counter
            global weight_counter
            global check_input_counter
            global X_idx_sorted_counter

            global mse_criterion_counter
            global friedman_mse_criterion_counter
            global mae_criterion_counter

            global gini_criterion_counter
            global entropy_criterion_counter
            global best_splitter_counter
            global random_splitter_counter
            global max_depth_counter
            global min_samples_split_counter
            global min_samples_leaf_counter
            global min_weight_fraction_leaf_counter
            global max_features_counter
            global auto_max_features_counter
            global sqrt_max_features_counter
            global log2_max_features_counter
            global max_leaf_node_counter
            global min_impurity_decrease_counter
            global min_impurity_split_counter
            global class_weight_counter
            global ccp_alpha_counter
            global presort_counter

            if self.criterion == "mse":
                mse_criterion_counter += 1
            if self.criterion == "friedman_mse":
                friedman_mse_criterion_counter += 1
            if self.criterion == "mae":
                mae_criterion_counter += 1
            if self.splitter == "best":
                best_splitter_counter += 1
            if self.splitter == "random":
                random_splitter_counter += 1
            if self.max_depth is not None:
                max_depth_counter += 1

            min_samples_leaf_counter += 1

            min_weight_fraction_leaf_counter += 1
            if self.max_features is not None:
                if self.max_features == "auto":
                    auto_max_features_counter += 1
                if self.max_features == "sqrt":
                    sqrt_max_features_counter += 1
                if self.max_features == "log2":
                    log2_max_features_counter += 1
            else:
                max_features_counter += 1

            if self.max_leaf_nodes is not None:
                max_leaf_node_counter += 1
            min_impurity_decrease_counter += 1
            min_samples_split_counter += 1

            if self.presort:
                presort_counter += 1

            sklCallsCounter += 1

            if sparse.issparse(X):
                csr_counter += 1

            if check_input:
                check_input_counter += 1

            print_counters()
            return super().predict(X, check_input=check_input)
            