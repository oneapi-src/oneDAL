import re
import sys, os

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..', 'tools'))
from conformance_functions import find_count_calls, get_testing_results, get_n_calls


log_filename = "_log.txt"

data_sparse_calls = "sparse_calls"
weight_calls = "weight_calls"

gini_criterion_calls = "gini_criterion_calls"
entropy_criterion_calls = "entropy_criterion_calls"

mse_criterion_calls = "mse_criterion_calls"
mae_criterion_calls = "mae_criterion_calls"

n_estimators_calls = "n_estimators_calls"
max_depth_calls = "max_depth_calls"
min_samples_split_calls = "min_samples_split_calls"
min_samples_leaf_calls = "min_samples_leaf_calls"
min_weight_fraction_leaf_calls = "min_weight_fraction_leaf_calls"
max_features_calls = "max_features_calls"
auto_max_features_calls = "auto_max_features_calls"
sqrt_max_features_calls = "sqrt_max_features_calls"
log2_max_features_calls = "log2_max_features_calls"
max_leaf_node_calls = "max_leaf_nodes_calls"
min_impurity_decrease_calls = "min_impurity_decrease_calls"
min_impurity_split_calls = "min_impurity_split_calls"
bootstrap_calls = "bootstrap_calls"
oob_score_calls = "oob_score_calls"
warm_start_calls = "warm_start_calls"
class_weight_calls = "class_weight_calls"
ccp_alpha_calls = "ccp_alpha_calls"
max_samples_calls = "max_samples_calls"

if __name__ == "__main__":
    n_sparse_calls = n_weight_calls = n_gini_criterion_calls = \
    n_entropy_criterion_calls = n_mse_criterion_calls = \
    n_mae_criterion_calls = n_n_estimators_calls = \
    n_max_depth_calls = n_min_samples_split_calls = \
    n_min_samples_leaf_calls = n_min_weight_fraction_leaf_calls = \
    n_max_features_calls = n_auto_max_features_calls = \
    n_sqrt_max_features_calls = n_log2_max_features_calls = \
    n_max_leaf_node_calls = n_min_impurity_decrease_calls = \
    n_min_impurity_split_calls = n_bootstrap_calls = \
    n_oob_score_calls = n_warm_start_calls = n_class_weight_calls = \
    n_ccp_alpha_calls = n_max_samples_calls = 0

    n_sparse_calls = find_count_calls(log_filename, data_sparse_calls)
    n_weight_calls = find_count_calls(log_filename, weight_calls)

    n_gini_criterion_calls = find_count_calls(log_filename, gini_criterion_calls)
    n_entropy_criterion_calls = find_count_calls(log_filename, entropy_criterion_calls)

    n_mse_criterion_calls = find_count_calls(log_filename, mse_criterion_calls)
    n_mae_criterion_calls = find_count_calls(log_filename, mae_criterion_calls)

    n_n_estimators_calls = find_count_calls(log_filename, n_estimators_calls)
    n_max_depth_calls = find_count_calls(log_filename, max_depth_calls)
    n_min_samples_split_calls = find_count_calls(log_filename, min_samples_split_calls)
    n_min_samples_leaf_calls = find_count_calls(log_filename, min_samples_leaf_calls)
    n_min_weight_fraction_leaf_calls = find_count_calls(log_filename, min_weight_fraction_leaf_calls)
    n_max_features_calls = find_count_calls(log_filename, max_features_calls)
    n_auto_max_features_calls = find_count_calls(log_filename, auto_max_features_calls)
    n_sqrt_max_features_calls = find_count_calls(log_filename, sqrt_max_features_calls)
    n_log2_max_features_calls = find_count_calls(log_filename, log2_max_features_calls)
    n_max_leaf_node_calls = find_count_calls(log_filename, max_leaf_node_calls)
    n_min_impurity_decrease_calls = find_count_calls(log_filename, min_impurity_decrease_calls)
    n_min_impurity_split_calls = find_count_calls(log_filename, min_impurity_split_calls)
    n_bootstrap_calls = find_count_calls(log_filename, bootstrap_calls)
    n_oob_score_calls = find_count_calls(log_filename, oob_score_calls)
    n_warm_start_calls = find_count_calls(log_filename, warm_start_calls)
    n_class_weight_calls = find_count_calls(log_filename, class_weight_calls)
    n_ccp_alpha_calls = find_count_calls(log_filename, ccp_alpha_calls)
    n_max_samples_calls = find_count_calls(log_filename, max_samples_calls)

     #Calculate metrics
    all_calls=get_n_calls() / 100
    sparse_using = n_sparse_calls / all_calls if all_calls else 0
    weight_using = n_weight_calls / all_calls if all_calls else 0

    gini_criterion_using = n_gini_criterion_calls / all_calls if all_calls else 0
    entropy_criterion_using = n_entropy_criterion_calls / all_calls if all_calls else 0

    mse_criterion_using = n_mse_criterion_calls / all_calls if all_calls else 0
    mae_criterion_using = n_mae_criterion_calls / all_calls if all_calls else 0

    n_estimators_using = n_n_estimators_calls / all_calls if all_calls else 0
    max_depth_using = n_max_depth_calls / all_calls if all_calls else 0
    min_samples_split_using = n_min_samples_split_calls / all_calls if all_calls else 0
    min_samples_leaf_using = n_min_samples_leaf_calls / all_calls if all_calls else 0
    min_weight_fraction_leaf_using = n_min_weight_fraction_leaf_calls / all_calls if all_calls else 0
    max_features_using = n_max_features_calls / all_calls if all_calls else 0
    auto_max_features_using = n_auto_max_features_calls / all_calls if all_calls else 0
    sqrt_max_features_using = n_sqrt_max_features_calls / all_calls if all_calls else 0
    log2_max_features_using = n_log2_max_features_calls / all_calls if all_calls else 0
    max_leaf_node_using = n_max_leaf_node_calls / all_calls if all_calls else 0
    min_impurity_decrease_using = n_min_impurity_decrease_calls / all_calls if all_calls else 0
    min_impurity_split_using = n_min_impurity_split_calls / all_calls if all_calls else 0
    bootstrap_using = n_bootstrap_calls / all_calls if all_calls else 0
    oob_score_using = n_oob_score_calls / all_calls if all_calls else 0
    warm_start_using = n_warm_start_calls / all_calls if all_calls else 0
    class_weight_using = n_class_weight_calls / all_calls if all_calls else 0
    ccp_alpha_using = n_ccp_alpha_calls / all_calls if all_calls else 0
    max_samples_using = n_max_samples_calls / all_calls if all_calls else 0

    print('\n------ Unsupported parameters/data usage % ------')
    print('Data: sparse:.......................', int(sparse_using), '%')
    print('Parameter: sample weights:..........', int(weight_using), '%')

    print("Parameter: gini_criterion:..........", int(gini_criterion_using), "%")
    print("Parameter: entropy_criterion:.......", int(entropy_criterion_using), "%")

    print("Parameter: mse_criterion:.......... ", int(mse_criterion_using), "%")
    print("Parameter: mae_criterion:...........", int(mae_criterion_using), "%")

    print("Parameter: n_estimators:............", int(n_estimators_using), "%")
    print("Parameter: max_depth:...............", int(max_depth_using), "%")
    print("Parameter: min_samples_split:.......", int(min_samples_split_using), "%")
    print("Parameter: min_samples_leaf:........", int(min_samples_leaf_using), "%")
    print("Parameter: min_weight_fraction_leaf:", int(min_weight_fraction_leaf_using), "%")
    print("Parameter: max_features:............", int(max_features_using), "%")
    print("Parameter: auto_max_features:.......", int(auto_max_features_using), "%")
    print("Parameter: sqrt_max_features:.......", int(sqrt_max_features_using), "%")
    print("Parameter: log2_max_features:.......", int(log2_max_features_using), "%")
    print("Parameter: max_leaf_node:...........", int(max_leaf_node_using), "%")
    print("Parameter: min_impurity_decrease:...", int(min_impurity_decrease_using), "%")
    print("Parameter: min_impurity_split:......", int(min_impurity_split_using), "%")
    print("Parameter: bootstrap................", int(bootstrap_using), "%")
    print("Parameter: oob_score................", int(oob_score_using), "%")
    print("Parameter: warm_start...............", int(warm_start_using), "%")
    print("Parameter: class_weight:............", int(class_weight_using), "%")
    print("Parameter: ccp_alpha:...............", int(ccp_alpha_using), "%")
    print("Parameter: max_samples:.............", int(max_samples_using), "%")

