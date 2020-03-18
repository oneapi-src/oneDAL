from sklearn.ensemble import RandomForestClassifier as RandomForestClassifierSkl
from sklearn.ensemble import RandomForestRegressor as RandomForestRegressorSkl
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

# decision tree classification criterion
gini_criterion_counter = 0
entropy_criterion_counter = 0

# decision tree regression criterion
mse_criterion_counter = 0
mae_criterion_counter = 0

n_estimators_counter=0
max_depth_counter = 0
min_samples_split_counter = 0
min_samples_leaf_counter = 0
min_weight_fraction_leaf_counter = 0
auto_max_features_counter = 0
sqrt_max_features_counter = 0
log2_max_features_counter = 0
max_features_counter = 0
max_leaf_nodes_counter = 0
min_impurity_decrease_counter = 0
min_impurity_split_counter = 0
bootstrap_counter = 0
oob_score_counter = 0
warm_start_counter = 0
class_weight_counter = 0
ccp_alpha_counter = 0
max_samples_counter = 0


if (LooseVersion(sklearn_version) >= LooseVersion("0.22")):
    class RandomForestClassifier(RandomForestClassifierSkl):
        def __init__(self,
                     n_estimators=100,
                     criterion="gini",
                     max_depth=None,
                     min_samples_split=2,
                     min_samples_leaf=1,
                     min_weight_fraction_leaf=0.,
                     max_features="auto",
                     max_leaf_nodes=None,
                     min_impurity_decrease=0.,
                     min_impurity_split=None,
                     bootstrap=True,
                     oob_score=False,
                     n_jobs=None,
                     random_state=None,
                     verbose=0,
                     warm_start=False,
                     class_weight=None,
                     ccp_alpha=0.0,
                     max_samples=None):
             super().__init__(
                     n_estimators=n_estimators,
                     criterion=criterion,
                     max_depth=max_depth,
                     min_samples_split=min_samples_split,
                     min_samples_leaf=min_samples_leaf,
                     min_weight_fraction_leaf=min_weight_fraction_leaf,
                     max_features=max_features,
                     max_leaf_nodes=max_leaf_nodes,
                     min_impurity_decrease=min_impurity_decrease,
                     min_impurity_split=min_impurity_split,
                     bootstrap=bootstrap,
                     oob_score=oob_score,
                     n_jobs=n_jobs,
                     random_state=random_state,
                     verbose=verbose,
                     warm_start=warm_start,
                     class_weight=class_weight,
                     ccp_alpha=ccp_alpha,
                     max_samples=max_samples)

        def fit(self, X, y, sample_weight=None):

            global sklCallsCounter

            global csr_counter
            global weight_counter
            global gini_criterion_counter
            global entropy_criterion_counter
            global n_estimators_counter
            global max_depth_counter
            global min_samples_split_counter
            global min_samples_leaf_counter
            global min_weight_fraction_leaf_counter
            global auto_max_features_counter
            global sqrt_max_features_counter
            global log2_max_features_counter
            global max_features_counter
            global min_impurity_decrease_counter
            global min_impurity_split_counter
            global max_leaf_nodes_counter
            global bootstrap_counter
            global oob_score_counter
            global warm_start_counter
            global class_weight_counter
            global ccp_alpha_counter
            global max_samples_counter

            if sparse.issparse(X):
                csr_counter += 1
            if sample_weight is not None:
                weight_counter += 1

            n_estimators_counter += 1
            if self.criterion == "gini":
                gini_criterion_counter += 1
            if self.criterion == "entropy":
                entropy_criterion_counter += 1

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
                max_leaf_nodes_counter += 1
            min_impurity_decrease_counter += 1
            min_impurity_split_counter += 1
            min_samples_split_counter += 1
            if self.bootstrap:
                bootstrap_counter += 1
            if self.oob_score:
                oob_score_counter += 1
            if self.warm_start:
                warm_start_counter += 1
            if self.class_weight is not None:
                class_weight_counter += 1

            ccp_alpha_counter += 1

            if self.max_samples is not None:
                max_samples_counter += 1

            sklCallsCounter += 1

            print('skl_calls=', sklCallsCounter)
            print('daal_calls=', daal4pyCallsCounter)

            print("sparse_calls=", csr_counter)
            print("weight_calls=", weight_counter)
            print("gini_criterion_calls=", gini_criterion_counter)
            print("entropy_criterion_calls=", entropy_criterion_counter)
            print("n_estimators_calls=", n_estimators_counter)
            print("max_depth_calls=", max_depth_counter)
            print("min_samples_split_calls=", min_samples_split_counter)
            print("min_samples_leaf_calls=", min_samples_leaf_counter)
            print("min_weight_fraction_leaf_calls=", min_weight_fraction_leaf_counter)
            print("auto_max_features_calls=", auto_max_features_counter)
            print("sqrt_max_features_calls=", sqrt_max_features_counter)
            print("log2_max_features_calls=", log2_max_features_counter)
            print("max_features_calls=", max_features_counter)
            print("max_leaf_nodes_calls=", max_leaf_nodes_counter)
            print("min_impurity_decrease_calls=", min_impurity_decrease_counter)
            print("min_impurity_split_calls=", min_impurity_split_counter)
            print("bootstrap_calls=", bootstrap_counter)
            print("oob_score_calls=", oob_score_counter)
            print("warm_start_calls=", warm_start_counter)
            print("class_weight_calls=", class_weight_counter)
            print("ccp_alpha_calls=", ccp_alpha_counter)
            print("max_samples_calls=", max_samples_counter, "\n")

            super().fit(
                X, y,
                sample_weight=sample_weight)
            return self

        def predict(self, X):
            global sklCallsCounter

            global csr_counter
            global gini_criterion_counter
            global entropy_criterion_counter
            global n_estimators_counter
            global max_depth_counter
            global min_samples_split_counter
            global min_samples_leaf_counter
            global min_weight_fraction_leaf_counter
            global auto_max_features_counter
            global sqrt_max_features_counter
            global log2_max_features_counter
            global max_features_counter
            global max_leaf_nodes_counter
            global min_impurity_decrease_counter
            global min_impurity_split_counter
            global bootstrap_counter
            global oob_score_counter
            global warm_start_counter
            global class_weight_counter
            global ccp_alpha_counter
            global max_samples_counter

            if sparse.issparse(X):
                csr_counter += 1

            n_estimators_counter += 1
            if self.criterion == "gini":
                gini_criterion_counter += 1
            if self.criterion == "entropy":
                entropy_criterion_counter += 1

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
                max_leaf_nodes_counter += 1
            min_impurity_decrease_counter += 1
            min_impurity_split_counter += 1
            min_samples_split_counter += 1
            if self.bootstrap:
                bootstrap_counter += 1
            if self.oob_score:
                oob_score_counter += 1
            if self.warm_start:
                warm_start_counter += 1
            if self.class_weight is not None:
                class_weight_counter += 1

            ccp_alpha_counter += 1

            if self.max_samples is not None:
                max_samples_counter += 1

            sklCallsCounter += 1

            print('skl_calls=', sklCallsCounter)
            print('daal_calls=', daal4pyCallsCounter)

            print("sparse_calls=", csr_counter)
            print("gini_criterion_calls=", gini_criterion_counter)
            print("entropy_criterion_calls=", entropy_criterion_counter)
            print("n_estimators_calls=", n_estimators_counter)
            print("max_depth_calls=", max_depth_counter)
            print("min_samples_split_calls=", min_samples_split_counter)
            print("min_samples_leaf_calls=", min_samples_leaf_counter)
            print("min_weight_fraction_leaf_calls=", min_weight_fraction_leaf_counter)
            print("auto_max_features_calls=", auto_max_features_counter)
            print("sqrt_max_features_calls=", sqrt_max_features_counter)
            print("log2_max_features_calls=", log2_max_features_counter)
            print("max_features_calls=", max_features_counter)
            print("max_leaf_nodes_calls=", max_leaf_nodes_counter)
            print("min_impurity_decrease_calls=", min_impurity_decrease_counter)
            print("min_impurity_split_calls=", min_impurity_split_counter)
            print("bootstrap_calls=", bootstrap_counter)
            print("oob_score_calls=", oob_score_counter)
            print("warm_start_calls=", warm_start_counter)
            print("class_weight_calls=", class_weight_counter)
            print("ccp_alpha_calls=", ccp_alpha_counter)
            print("max_samples_calls=", max_samples_counter, "\n")
            return super().predict(X)

        def predict_log_proba(self, X):

            global sklCallsCounter

            global csr_counter
            global gini_criterion_counter
            global entropy_criterion_counter
            global n_estimators_counter
            global max_depth_counter
            global min_samples_split_counter
            global min_samples_leaf_counter
            global min_weight_fraction_leaf_counter
            global auto_max_features_counter
            global sqrt_max_features_counter
            global log2_max_features_counter
            global max_features_counter
            global max_leaf_nodes_counter
            global min_impurity_decrease_counter
            global min_impurity_split_counter
            global bootstrap_counter
            global oob_score_counter
            global warm_start_counter
            global class_weight_counter
            global ccp_alpha_counter
            global max_samples_counter

            if sparse.issparse(X):
                csr_counter += 1

            n_estimators_counter += 1
            if self.criterion == "gini":
                gini_criterion_counter += 1
            if self.criterion == "entropy":
                entropy_criterion_counter += 1

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
                max_leaf_nodes_counter += 1
            min_impurity_decrease_counter += 1
            min_impurity_split_counter += 1
            min_samples_split_counter += 1
            if self.bootstrap:
                bootstrap_counter += 1
            if self.oob_score:
                oob_score_counter += 1
            if self.warm_start:
                warm_start_counter += 1
            if self.class_weight is not None:
                class_weight_counter += 1

            ccp_alpha_counter += 1

            if self.max_samples is not None:
                max_samples_counter += 1

            sklCallsCounter += 1

            print('skl_calls=', sklCallsCounter)
            print('daal_calls=', daal4pyCallsCounter)

            print("sparse_calls=", csr_counter)
            print("gini_criterion_calls=", gini_criterion_counter)
            print("entropy_criterion_calls=", entropy_criterion_counter)
            print("n_estimators_calls=", n_estimators_counter)
            print("max_depth_calls=", max_depth_counter)
            print("min_samples_split_calls=", min_samples_split_counter)
            print("min_samples_leaf_calls=", min_samples_leaf_counter)
            print("min_weight_fraction_leaf_calls=", min_weight_fraction_leaf_counter)
            print("auto_max_features_calls=", auto_max_features_counter)
            print("sqrt_max_features_calls=", sqrt_max_features_counter)
            print("log2_max_features_calls=", log2_max_features_counter)
            print("max_features_calls=", max_features_counter)
            print("max_leaf_nodes_calls=", max_leaf_nodes_counter)
            print("min_impurity_decrease_calls=", min_impurity_decrease_counter)
            print("min_impurity_split_calls=", min_impurity_split_counter)
            print("bootstrap_calls=", bootstrap_counter)
            print("oob_score_calls=", oob_score_counter)
            print("warm_start_calls=", warm_start_counter)
            print("class_weight_calls=", class_weight_counter)
            print("ccp_alpha_calls=", ccp_alpha_counter)
            print("max_samples_calls=", max_samples_counter, "\n")

            return super().predict_log_proba(X)

        def predict_proba(self, X):

            global sklCallsCounter

            global csr_counter
            global gini_criterion_counter
            global entropy_criterion_counter
            global n_estimators_counter
            global max_depth_counter
            global min_samples_split_counter
            global min_samples_leaf_counter
            global min_weight_fraction_leaf_counter
            global auto_max_features_counter
            global sqrt_max_features_counter
            global log2_max_features_counter
            global max_features_counter
            global max_leaf_nodes_counter
            global min_impurity_decrease_counter
            global min_impurity_split_counter
            global bootstrap_counter
            global oob_score_counter
            global warm_start_counter
            global class_weight_counter
            global ccp_alpha_counter
            global max_samples_counter

            if sparse.issparse(X):
                csr_counter += 1

            n_estimators_counter += 1
            if self.criterion == "gini":
                gini_criterion_counter += 1
            if self.criterion == "entropy":
                entropy_criterion_counter += 1

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
                max_leaf_nodes_counter += 1
            min_impurity_decrease_counter += 1
            min_impurity_split_counter += 1
            min_samples_split_counter += 1
            if self.bootstrap:
                bootstrap_counter += 1
            if self.oob_score:
                oob_score_counter += 1
            if self.warm_start:
                warm_start_counter += 1
            if self.class_weight is not None:
                class_weight_counter += 1

            ccp_alpha_counter += 1

            if self.max_samples is not None:
                max_samples_counter += 1

            sklCallsCounter += 1

            print('skl_calls=', sklCallsCounter)
            print('daal_calls=', daal4pyCallsCounter)

            print("sparse_calls=", csr_counter)
            print("gini_criterion_calls=", gini_criterion_counter)
            print("entropy_criterion_calls=", entropy_criterion_counter)
            print("n_estimators_calls=", n_estimators_counter)
            print("max_depth_calls=", max_depth_counter)
            print("min_samples_split_calls=", min_samples_split_counter)
            print("min_samples_leaf_calls=", min_samples_leaf_counter)
            print("min_weight_fraction_leaf_calls=", min_weight_fraction_leaf_counter)
            print("auto_max_features_calls=", auto_max_features_counter)
            print("sqrt_max_features_calls=", sqrt_max_features_counter)
            print("log2_max_features_calls=", log2_max_features_counter)
            print("max_features_calls=", max_features_counter)
            print("max_leaf_nodes_calls=", max_leaf_nodes_counter)
            print("min_impurity_decrease_calls=", min_impurity_decrease_counter)
            print("min_impurity_split_calls=", min_impurity_split_counter)
            print("bootstrap_calls=", bootstrap_counter)
            print("oob_score_calls=", oob_score_counter)
            print("warm_start_calls=", warm_start_counter)
            print("class_weight_calls=", class_weight_counter)
            print("ccp_alpha_calls=", ccp_alpha_counter)
            print("max_samples_calls=", max_samples_counter, "\n")
            return super().predict_proba(X)

    class RandomForestRegressor(RandomForestRegressorSkl):
        def __init__(self,
                     n_estimators=100,
                     criterion="mse",
                     max_depth=None,
                     min_samples_split=2,
                     min_samples_leaf=1,
                     min_weight_fraction_leaf=0.,
                     max_features="auto",
                     max_leaf_nodes=None,
                     min_impurity_decrease=0.,
                     min_impurity_split=None,
                     bootstrap=True,
                     oob_score=False,
                     n_jobs=None,
                     random_state=None,
                     verbose=0,
                     warm_start=False,
                     ccp_alpha=0.0,
                     max_samples=None):
                super().__init__(
                     n_estimators=n_estimators,
                     criterion=criterion,
                     max_depth=max_depth,
                     min_samples_split=min_samples_split,
                     min_samples_leaf=min_samples_leaf,
                     min_weight_fraction_leaf=min_weight_fraction_leaf,
                     max_features=max_features,
                     max_leaf_nodes=max_leaf_nodes,
                     min_impurity_decrease=min_impurity_decrease,
                     min_impurity_split=min_impurity_split,
                     bootstrap=bootstrap,
                     oob_score=oob_score,
                     n_jobs=n_jobs,
                     random_state=random_state,
                     verbose=verbose,
                     warm_start=warm_start,
                     ccp_alpha=ccp_alpha,
                     max_samples=max_samples)


        def fit(self, X, y, sample_weight=None):

            global sklCallsCounter

            global csr_counter
            global weight_counter
            global mse_criterion_counter
            global mae_criterion_counter
            global n_estimators_counter
            global max_depth_counter
            global min_samples_split_counter
            global min_samples_leaf_counter
            global min_weight_fraction_leaf_counter
            global auto_max_features_counter
            global sqrt_max_features_counter
            global log2_max_features_counter
            global max_features_counter
            global max_leaf_nodes_counter
            global min_impurity_decrease_counter
            global min_impurity_split_counter
            global bootstrap_counter
            global oob_score_counter
            global warm_start_counter
            global ccp_alpha_counter
            global max_samples_counter

            if sparse.issparse(X):
                csr_counter += 1
            if sample_weight is not None:
                weight_counter += 1

            n_estimators_counter += 1
            if self.criterion == "mse":
                mse_criterion_counter += 1
            if self.criterion == "mae":
                mae_criterion_counter += 1

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
                max_leaf_nodes_counter += 1
            min_impurity_decrease_counter += 1
            min_impurity_split_counter += 1
            min_samples_split_counter += 1
            if self.bootstrap:
                bootstrap_counter += 1
            if self.oob_score:
                oob_score_counter += 1
            if self.warm_start:
                warm_start_counter += 1

            ccp_alpha_counter += 1

            if self.max_samples is not None:
                max_samples_counter += 1

            sklCallsCounter += 1

            print('skl_calls=', sklCallsCounter)
            print('daal_calls=', daal4pyCallsCounter)

            print("sparse_calls=", csr_counter)
            print("weight_calls=", weight_counter)
            print("mse_criterion_calls=", mse_criterion_counter)
            print("mae_criterion_calls=", mae_criterion_counter)
            print("n_estimators_calls=", n_estimators_counter)
            print("max_depth_calls=", max_depth_counter)
            print("min_samples_split_calls=", min_samples_split_counter)
            print("min_samples_leaf_calls=", min_samples_leaf_counter)
            print("min_weight_fraction_leaf_calls=", min_weight_fraction_leaf_counter)
            print("auto_max_features_calls=", auto_max_features_counter)
            print("sqrt_max_features_calls=", sqrt_max_features_counter)
            print("log2_max_features_calls=", log2_max_features_counter)
            print("max_features_calls=", max_features_counter)
            print("max_leaf_nodes_calls=", max_leaf_nodes_counter)
            print("min_impurity_decrease_calls=", min_impurity_decrease_counter)
            print("min_impurity_split_calls=", min_impurity_split_counter)
            print("bootstrap_calls=", bootstrap_counter)
            print("oob_score_calls=", oob_score_counter)
            print("warm_start_calls=", warm_start_counter)
            print("ccp_alpha_calls=", ccp_alpha_counter)
            print("max_samples_calls=", max_samples_counter, "\n")

            return super().fit(
                X, y,
                sample_weight=sample_weight)

        def predict(self, X):

            global sklCallsCounter

            global csr_counter
            global mse_criterion_counter
            global mae_criterion_counter
            global n_estimators_counter
            global max_depth_counter
            global min_samples_split_counter
            global min_samples_leaf_counter
            global min_weight_fraction_leaf_counter
            global auto_max_features_counter
            global sqrt_max_features_counter
            global log2_max_features_counter
            global max_features_counter
            global max_leaf_nodes_counter
            global min_impurity_decrease_counter
            global min_impurity_split_counter
            global bootstrap_counter
            global oob_score_counter
            global warm_start_counter
            global ccp_alpha_counter
            global max_samples_counter

            if sparse.issparse(X):
                csr_counter += 1

            n_estimators_counter += 1
            if self.criterion == "mse":
                mse_criterion_counter += 1
            if self.criterion == "mae":
                mae_criterion_counter += 1

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
                max_leaf_nodes_counter += 1
            min_impurity_decrease_counter += 1
            min_impurity_split_counter += 1
            min_samples_split_counter += 1
            if self.bootstrap:
                bootstrap_counter += 1
            if self.oob_score:
                oob_score_counter += 1
            if self.warm_start:
                warm_start_counter += 1

            ccp_alpha_counter += 1

            if self.max_samples is not None:
                max_samples_counter += 1

            sklCallsCounter += 1
            print('skl_calls=', sklCallsCounter)
            print('daal_calls=', daal4pyCallsCounter)

            print("sparse_calls=", csr_counter)
            print("mse_criterion_calls=", mse_criterion_counter)
            print("mae_criterion_calls=", mae_criterion_counter)
            print("n_estimators_calls=", n_estimators_counter)
            print("max_depth_calls=", max_depth_counter)
            print("min_samples_split_calls=", min_samples_split_counter)
            print("min_samples_leaf_calls=", min_samples_leaf_counter)
            print("min_weight_fraction_leaf_calls=", min_weight_fraction_leaf_counter)
            print("auto_max_features_calls=", auto_max_features_counter)
            print("sqrt_max_features_calls=", sqrt_max_features_counter)
            print("log2_max_features_calls=", log2_max_features_counter)
            print("max_features_calls=", max_features_counter)
            print("max_leaf_nodes_calls=", max_leaf_nodes_counter)
            print("min_impurity_decrease_calls=", min_impurity_decrease_counter)
            print("min_impurity_split_calls=", min_impurity_split_counter)
            print("bootstrap_calls=", bootstrap_counter)
            print("oob_score_calls=", oob_score_counter)
            print("warm_start_calls=", warm_start_counter)
            print("ccp_alpha_calls=", ccp_alpha_counter)
            print("max_samples_calls=", max_samples_counter, "\n")
            return super().predict(X)

else:
    class RandomForestClassifier(RandomForestClassifierSkl):
        def __init__(self,
                     n_estimators='warn',
                     criterion="gini",
                     max_depth=None,
                     min_samples_split=2,
                     min_samples_leaf=1,
                     min_weight_fraction_leaf=0.,
                     max_features="auto",
                     max_leaf_nodes=None,
                     min_impurity_decrease=0.,
                     min_impurity_split=None,
                     bootstrap=True,
                     oob_score=False,
                     n_jobs=None,
                     random_state=None,
                     verbose=0,
                     warm_start=False,
                     class_weight=None):
            super().__init__(
                     n_estimators=n_estimators,
                     criterion=criterion,
                     max_depth=max_depth,
                     min_samples_split=min_samples_split,
                     min_samples_leaf=min_samples_leaf,
                     min_weight_fraction_leaf=min_weight_fraction_leaf,
                     max_features=max_features,
                     max_leaf_nodes=max_leaf_nodes,
                     min_impurity_decrease=min_impurity_decrease,
                     min_impurity_split=min_impurity_split,
                     bootstrap=bootstrap,
                     oob_score=oob_score,
                     n_jobs=n_jobs,
                     random_state=random_state,
                     verbose=verbose,
                     warm_start=warm_start,
                     class_weight=class_weight)


        def fit(self, X, y, sample_weight=None):

            global sklCallsCounter

            global csr_counter
            global weight_counter
            global gini_criterion_counter
            global entropy_criterion_counter
            global n_estimators_counter
            global max_depth_counter
            global min_samples_split_counter
            global min_samples_leaf_counter
            global min_weight_fraction_leaf_counter
            global auto_max_features_counter
            global sqrt_max_features_counter
            global log2_max_features_counter
            global max_features_counter
            global max_leaf_nodes_counter
            global min_impurity_decrease_counter
            global min_impurity_split_counter
            global bootstrap_counter
            global oob_score_counter
            global warm_start_counter
            global class_weight_counter

            if sparse.issparse(X):
                csr_counter += 1
            if sample_weight is not None:
                weight_counter += 1

            n_estimators_counter += 1
            if self.criterion == "gini":
                gini_criterion_counter += 1
            if self.criterion == "entropy":
                entropy_criterion_counter += 1

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
                max_leaf_nodes_counter += 1
            min_impurity_decrease_counter += 1
            min_impurity_split_counter += 1
            min_samples_split_counter += 1
            if self.bootstrap:
                bootstrap_counter += 1
            if self.oob_score:
                oob_score_counter += 1
            if self.warm_start:
                warm_start_counter += 1
            if self.class_weight is not None:
                class_weight_counter += 1

            sklCallsCounter += 1

            print('skl_calls=', sklCallsCounter)
            print('daal_calls=', daal4pyCallsCounter)

            print("sparse_calls=", csr_counter)
            print("weight_calls=", weight_counter)
            print("gini_criterion_calls=", gini_criterion_counter)
            print("entropy_criterion_calls=", entropy_criterion_counter)
            print("n_estimators_calls=", n_estimators_counter)
            print("max_depth_calls=", max_depth_counter)
            print("min_samples_split_calls=", min_samples_split_counter)
            print("min_samples_leaf_calls=", min_samples_leaf_counter)
            print("min_weight_fraction_leaf_calls=", min_weight_fraction_leaf_counter)
            print("auto_max_features_calls=", auto_max_features_counter)
            print("sqrt_max_features_calls=", sqrt_max_features_counter)
            print("log2_max_features_calls=", log2_max_features_counter)
            print("max_features_calls=", max_features_counter)
            print("max_leaf_nodes_calls=", max_leaf_nodes_counter)
            print("min_impurity_decrease_calls=", min_impurity_decrease_counter)
            print("min_impurity_split_calls=", min_impurity_split_counter)
            print("bootstrap_calls=", bootstrap_counter)
            print("oob_score_calls=", oob_score_counter)
            print("warm_start_calls=", warm_start_counter)
            print("class_weight_calls=", class_weight_counter, "\n")

            super().fit(
                X, y,
                sample_weight=sample_weight)
            return self

        def predict(self, X):
            global sklCallsCounter

            global csr_counter
            global gini_criterion_counter
            global entropy_criterion_counter
            global n_estimators_counter
            global max_depth_counter
            global min_samples_split_counter
            global min_samples_leaf_counter
            global min_weight_fraction_leaf_counter
            global auto_max_features_counter
            global sqrt_max_features_counter
            global log2_max_features_counter
            global max_features_counter
            global max_leaf_nodes_counter
            global min_impurity_decrease_counter
            global min_impurity_split_counter
            global bootstrap_counter
            global oob_score_counter
            global warm_start_counter
            global class_weight_counter

            if sparse.issparse(X):
                csr_counter += 1

            n_estimators_counter += 1
            if self.criterion == "gini":
                gini_criterion_counter += 1
            if self.criterion == "entropy":
                entropy_criterion_counter += 1

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
                max_leaf_nodes_counter += 1
            min_impurity_decrease_counter += 1
            min_impurity_split_counter += 1
            min_samples_split_counter += 1
            if self.bootstrap:
                bootstrap_counter += 1
            if self.oob_score:
                oob_score_counter += 1
            if self.warm_start:
                warm_start_counter += 1
            if self.class_weight is not None:
                class_weight_counter += 1

            sklCallsCounter += 1

            print('skl_calls=', sklCallsCounter)
            print('daal_calls=', daal4pyCallsCounter)

            print("sparse_calls=", csr_counter)
            print("gini_criterion_calls=", gini_criterion_counter)
            print("entropy_criterion_calls=", entropy_criterion_counter)
            print("n_estimators_calls=", n_estimators_counter)
            print("max_depth_calls=", max_depth_counter)
            print("min_samples_split_calls=", min_samples_split_counter)
            print("min_samples_leaf_calls=", min_samples_leaf_counter)
            print("min_weight_fraction_leaf_calls=", min_weight_fraction_leaf_counter)
            print("auto_max_features_calls=", auto_max_features_counter)
            print("sqrt_max_features_calls=", sqrt_max_features_counter)
            print("log2_max_features_calls=", log2_max_features_counter)
            print("max_features_calls=", max_features_counter)
            print("max_leaf_nodes_calls=", max_leaf_nodes_counter)
            print("min_impurity_decrease_calls=", min_impurity_decrease_counter)
            print("min_impurity_split_calls=", min_impurity_split_counter)
            print("bootstrap_calls=", bootstrap_counter)
            print("oob_score_calls=", oob_score_counter)
            print("warm_start_calls=", warm_start_counter)
            print("class_weight_calls=", class_weight_counter, "\n")

            return super().predict(X)

        def predict_log_proba(self, X):

            global sklCallsCounter

            global csr_counter
            global gini_criterion_counter
            global entropy_criterion_counter
            global n_estimators_counter
            global max_depth_counter
            global min_samples_split_counter
            global min_samples_leaf_counter
            global min_weight_fraction_leaf_counter
            global auto_max_features_counter
            global sqrt_max_features_counter
            global log2_max_features_counter
            global max_features_counter
            global max_leaf_nodes_counter
            global min_impurity_decrease_counter
            global min_impurity_split_counter
            global bootstrap_counter
            global oob_score_counter
            global warm_start_counter
            global class_weight_counter

            if sparse.issparse(X):
                csr_counter += 1

            n_estimators_counter += 1
            if self.criterion == "gini":
                gini_criterion_counter += 1
            if self.criterion == "entropy":
                entropy_criterion_counter += 1

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
                max_leaf_nodes_counter += 1
            min_impurity_decrease_counter += 1
            min_impurity_split_counter += 1
            min_samples_split_counter += 1
            if self.bootstrap:
                bootstrap_counter += 1
            if self.oob_score:
                oob_score_counter += 1
            if self.warm_start:
                warm_start_counter += 1
            if self.class_weight is not None:
                class_weight_counter += 1

            sklCallsCounter += 1

            print('skl_calls=', sklCallsCounter)
            print('daal_calls=', daal4pyCallsCounter)

            print("sparse_calls=", csr_counter)
            print("gini_criterion_calls=", gini_criterion_counter)
            print("entropy_criterion_calls=", entropy_criterion_counter)
            print("n_estimators_calls=", n_estimators_counter)
            print("max_depth_calls=", max_depth_counter)
            print("min_samples_split_calls=", min_samples_split_counter)
            print("min_samples_leaf_calls=", min_samples_leaf_counter)
            print("min_weight_fraction_leaf_calls=", min_weight_fraction_leaf_counter)
            print("auto_max_features_calls=", auto_max_features_counter)
            print("sqrt_max_features_calls=", sqrt_max_features_counter)
            print("log2_max_features_calls=", log2_max_features_counter)
            print("max_features_calls=", max_features_counter)
            print("max_leaf_nodes_calls=", max_leaf_nodes_counter)
            print("min_impurity_decrease_calls=", min_impurity_decrease_counter)
            print("min_impurity_split_calls=", min_impurity_split_counter)
            print("bootstrap_calls=", bootstrap_counter)
            print("oob_score_calls=", oob_score_counter)
            print("warm_start_calls=", warm_start_counter)
            print("class_weight_calls=", class_weight_counter, "\n")
            return super().predict_log_proba(X)

        def predict_proba(self, X):

            global sklCallsCounter

            global csr_counter
            global gini_criterion_counter
            global entropy_criterion_counter
            global n_estimators_counter
            global max_depth_counter
            global min_samples_split_counter
            global min_samples_leaf_counter
            global min_weight_fraction_leaf_counter
            global auto_max_features_counter
            global sqrt_max_features_counter
            global log2_max_features_counter
            global max_features_counter
            global max_leaf_nodes_counter
            global min_impurity_decrease_counter
            global min_impurity_split_counter
            global bootstrap_counter
            global oob_score_counter
            global warm_start_counter
            global class_weight_counter
            global ccp_alpha_counter
            global max_samples_counter

            if sparse.issparse(X):
                csr_counter += 1

            n_estimators_counter += 1
            if self.criterion == "gini":
                gini_criterion_counter += 1
            if self.criterion == "entropy":
                entropy_criterion_counter += 1

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
                max_leaf_nodes_counter += 1
            min_impurity_decrease_counter += 1
            min_impurity_split_counter += 1
            min_samples_split_counter += 1
            if self.bootstrap:
                bootstrap_counter += 1
            if self.oob_score:
                oob_score_counter += 1
            if self.warm_start:
                warm_start_counter += 1
            if self.class_weight is not None:
                class_weight_counter += 1

            sklCallsCounter += 1

            print('skl_calls=', sklCallsCounter)
            print('daal_calls=', daal4pyCallsCounter)

            print("sparse_calls=", csr_counter)
            print("gini_criterion_calls=", gini_criterion_counter)
            print("entropy_criterion_calls=", entropy_criterion_counter)
            print("n_estimators_calls=", n_estimators_counter)
            print("max_depth_calls=", max_depth_counter)
            print("min_samples_split_calls=", min_samples_split_counter)
            print("min_samples_leaf_calls=", min_samples_leaf_counter)
            print("min_weight_fraction_leaf_calls=", min_weight_fraction_leaf_counter)
            print("auto_max_features_calls=", auto_max_features_counter)
            print("sqrt_max_features_calls=", sqrt_max_features_counter)
            print("log2_max_features_calls=", log2_max_features_counter)
            print("max_features_calls=", max_features_counter)
            print("max_leaf_nodes_calls=", max_leaf_nodes_counter)
            print("min_impurity_decrease_calls=", min_impurity_decrease_counter)
            print("min_impurity_split_calls=", min_impurity_split_counter)
            print("bootstrap_calls=", bootstrap_counter)
            print("oob_score_calls=", oob_score_counter)
            print("warm_start_calls=", warm_start_counter)
            print("class_weight_calls=", class_weight_counter, "\n")
            return super().predict_proba(X)

    class RandomForestRegressor(RandomForestRegressorSkl):
        def __init__(self,
                     n_estimators='warn',
                     criterion="mse",
                     max_depth=None,
                     min_samples_split=2,
                     min_samples_leaf=1,
                     min_weight_fraction_leaf=0.,
                     max_features="auto",
                     max_leaf_nodes=None,
                     min_impurity_decrease=0.,
                     min_impurity_split=None,
                     bootstrap=True,
                     oob_score=False,
                     n_jobs=None,
                     random_state=None,
                     verbose=0,
                     warm_start=False):

            super().__init__(
                     n_estimators=n_estimators,
                     criterion=criterion,
                     max_depth=max_depth,
                     min_samples_split=min_samples_split,
                     min_samples_leaf=min_samples_leaf,
                     min_weight_fraction_leaf=min_weight_fraction_leaf,
                     max_features=max_features,
                     max_leaf_nodes=max_leaf_nodes,
                     min_impurity_decrease=min_impurity_decrease,
                     min_impurity_split=min_impurity_split,
                     bootstrap=bootstrap,
                     oob_score=oob_score,
                     n_jobs=n_jobs,
                     random_state=random_state,
                     verbose=verbose,
                     warm_start=warm_start)

        def fit(self, X, y, sample_weight=None):

            global sklCallsCounter

            global csr_counter
            global weight_counter
            global mse_criterion_counter
            global mae_criterion_counter
            global n_estimators_counter
            global max_depth_counter
            global min_samples_split_counter
            global min_samples_leaf_counter
            global min_weight_fraction_leaf_counter
            global auto_max_features_counter
            global sqrt_max_features_counter
            global log2_max_features_counter
            global max_features_counter
            global max_leaf_nodes_counter
            global min_impurity_decrease_counter
            global min_impurity_split_counter
            global bootstrap_counter
            global oob_score_counter
            global warm_start_counter

            if sparse.issparse(X):
                csr_counter += 1
            if sample_weight is not None:
                weight_counter += 1

            n_estimators_counter += 1
            if self.criterion == "mse":
                mse_criterion_counter += 1
            if self.criterion == "mae":
                mae_criterion_counter += 1

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
                max_leaf_nodes_counter += 1
            min_impurity_decrease_counter += 1
            min_impurity_split_counter += 1
            min_samples_split_counter += 1
            if self.bootstrap:
                bootstrap_counter += 1
            if self.oob_score:
                oob_score_counter += 1
            if self.warm_start:
                warm_start_counter += 1

            sklCallsCounter += 1

            print('skl_calls=', sklCallsCounter)
            print('daal_calls=', daal4pyCallsCounter)

            print("sparse_calls=", csr_counter)
            print("weight_calls=", weight_counter)
            print("mse_criterion_calls=", mse_criterion_counter)
            print("mae_criterion_calls=", mae_criterion_counter)
            print("n_estimators_calls=", n_estimators_counter)
            print("max_depth_calls=", max_depth_counter)
            print("min_samples_split_calls=", min_samples_split_counter)
            print("min_samples_leaf_calls=", min_samples_leaf_counter)
            print("min_weight_fraction_leaf_calls=", min_weight_fraction_leaf_counter)
            print("auto_max_features_calls=", auto_max_features_counter)
            print("sqrt_max_features_calls=", sqrt_max_features_counter)
            print("log2_max_features_calls=", log2_max_features_counter)
            print("max_features_calls=", max_features_counter)
            print("max_leaf_nodes_calls=", max_leaf_nodes_counter)
            print("min_impurity_decrease_calls=", min_impurity_decrease_counter)
            print("min_impurity_split_calls=", min_impurity_split_counter)
            print("bootstrap_calls=", bootstrap_counter)
            print("oob_score_calls=", oob_score_counter)
            print("warm_start_calls=", warm_start_counter, "\n")

            return super().fit(
                X, y,
                sample_weight=sample_weight)

        def predict(self, X):

            global sklCallsCounter

            global csr_counter
            global mse_criterion_counter
            global mae_criterion_counter
            global n_estimators_counter
            global max_depth_counter
            global min_samples_split_counter
            global min_samples_leaf_counter
            global min_weight_fraction_leaf_counter
            global auto_max_features_counter
            global sqrt_max_features_counter
            global log2_max_features_counter
            global max_features_counter
            global max_leaf_nodes_counter
            global min_impurity_decrease_counter
            global min_impurity_split_counter
            global bootstrap_counter
            global oob_score_counter
            global warm_start_counter

            if sparse.issparse(X):
                csr_counter += 1

            n_estimators_counter += 1
            if self.criterion == "mse":
                mse_criterion_counter += 1
            if self.criterion == "mae":
                mae_criterion_counter += 1

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
                max_leaf_nodes_counter += 1
            min_impurity_decrease_counter += 1
            min_impurity_split_counter += 1
            min_samples_split_counter += 1
            if self.bootstrap:
                bootstrap_counter += 1
            if self.oob_score:
                oob_score_counter += 1
            if self.warm_start:
                warm_start_counter += 1

            sklCallsCounter += 1
            print('skl_calls=', sklCallsCounter)
            print('daal_calls=', daal4pyCallsCounter)

            print("sparse_calls=", csr_counter)
            print("mse_criterion_calls=", mse_criterion_counter)
            print("mae_criterion_calls=", mae_criterion_counter)
            print("n_estimators_calls=", n_estimators_counter)
            print("max_depth_calls=", max_depth_counter)
            print("min_samples_split_calls=", min_samples_split_counter)
            print("min_samples_leaf_calls=", min_samples_leaf_counter)
            print("min_weight_fraction_leaf_calls=", min_weight_fraction_leaf_counter)
            print("auto_max_features_calls=", auto_max_features_counter)
            print("sqrt_max_features_calls=", sqrt_max_features_counter)
            print("log2_max_features_calls=", log2_max_features_counter)
            print("max_features_calls=", max_features_counter)
            print("max_leaf_nodes_calls=", max_leaf_nodes_counter)
            print("min_impurity_decrease_calls=", min_impurity_decrease_counter)
            print("min_impurity_split_calls=", min_impurity_split_counter)
            print("bootstrap_calls=", bootstrap_counter)
            print("oob_score_calls=", oob_score_counter)
            print("warm_start_calls=", warm_start_counter, "\n")

            return super().predict(X)

