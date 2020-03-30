import re
import sys, os

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..', 'tools'))
from conformance_functions import find_count_calls, get_testing_results, get_n_calls


log_filename = "_log.txt"

data_sparse_calls = "sparse_calls"
weight_calls = "weight_calls"
check_input_calls = "check_input_calls"
X_idx_sorted_calls = "X_idx_sorted_calls"

gini_criterion_calls = "gini_criterion_calls"
entropy_criterion_calls = "entropy_criterion_calls"

mse_criterion_calls = "mse_criterion_calls"
friedman_mse_criterion_calls = "friedman_mse_criterion_calls"
mae_criterion_calls = "mae_criterion_calls"

best_splitter_calls = "best_splitter_calls"
random_splitter_calls = "random_splitter_calls"
max_depth_calls = "max_depth_calls"
min_samples_split_calls = "min_samples_split_calls"
min_samples_leaf_calls = "min_samples_leaf_calls"
min_weight_fraction_leaf_calls = "min_weight_fraction_leaf_calls"
max_features_calls = "max_features_calls"
auto_max_features_calls = "auto_max_features_calls"
sqrt_max_features_calls = "sqrt_max_features_calls"
log2_max_features_calls = "log2_max_features_calls"
max_leaf_node_calls = "max_leaf_node_calls"
min_impurity_decrease_calls = "min_impurity_decrease_calls"
min_impurity_split_calls = "min_impurity_split_calls"
class_weight_calls = "class_weight_calls"
ccp_alpha_calls = "ccp_alpha_calls"
presort_calls = "presort_calls"

if __name__ == "__main__":
    n_sparse_calls = n_weight_calls = n_check_input_calls = \
    n_X_idx_sorted_calls = n_gini_criterion_calls = \
    n_entropy_criterion_calls = n_mse_criterion_calls = \
    n_friedman_mse_criterion_calls = n_mae_criterion_calls = \
    n_best_splitter_calls = \
    n_random_splitter_calls = n_max_depth_calls = \
    n_min_samples_split_calls = \
    n_min_samples_leaf_calls = n_min_weight_fraction_leaf_calls = \
    n_max_features_calls = n_auto_max_features_calls = \
    n_sqrt_max_features_calls = n_log2_max_features_calls = \
    n_max_leaf_node_calls = n_min_impurity_decrease_calls = \
    n_min_impurity_split_calls = n_class_weight_calls = \
    n_ccp_alpha_calls = n_presort_calls = 0

    n_sparse_calls = find_count_calls(log_filename, data_sparse_calls)
    n_weight_calls = find_count_calls(log_filename, weight_calls)
    n_check_input_calls = find_count_calls(log_filename, check_input_calls)
    n_X_idx_sorted_calls = find_count_calls(log_filename, X_idx_sorted_calls)

    n_gini_criterion_calls = find_count_calls(log_filename, gini_criterion_calls)
    n_entropy_criterion_calls = find_count_calls(log_filename, entropy_criterion_calls)

    n_mse_criterion_calls = find_count_calls(log_filename, mse_criterion_calls)
    n_friedman_mse_criterion_calls = find_count_calls(log_filename, friedman_mse_criterion_calls)
    n_mae_criterion_calls = find_count_calls(log_filename, mae_criterion_calls)

    n_best_splitter_calls = find_count_calls(log_filename, best_splitter_calls)
    n_random_splitter_calls = find_count_calls(log_filename, random_splitter_calls)
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
    n_class_weight_calls = find_count_calls(log_filename, class_weight_calls)
    n_ccp_alpha_calls = find_count_calls(log_filename, ccp_alpha_calls)
    n_presort_calls = find_count_calls(log_filename, presort_calls)

     #Calculate metrics
    all_calls=get_n_calls() / 100

    sparse_using = n_sparse_calls / all_calls if all_calls else 0
    weight_using = n_weight_calls / all_calls if all_calls else 0
    check_input_using = n_check_input_calls / all_calls if all_calls else 0
    X_idx_sorted_using = n_X_idx_sorted_calls / all_calls if all_calls else 0

    gini_criterion_using = n_gini_criterion_calls / all_calls if all_calls else 0
    entropy_criterion_using = n_entropy_criterion_calls / all_calls if all_calls else 0

    mse_criterion_using = n_mse_criterion_calls / all_calls if all_calls else 0
    friedman_mse_criterion_using = n_friedman_mse_criterion_calls / all_calls if all_calls else 0
    mae_criterion_using = n_mae_criterion_calls / all_calls if all_calls else 0

    best_splitter_using = n_best_splitter_calls / all_calls if all_calls else 0
    random_splitter_using = n_random_splitter_calls / all_calls if all_calls else 0
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
    class_weight_using = n_class_weight_calls / all_calls if all_calls else 0
    ccp_alpha_using = n_ccp_alpha_calls / all_calls if all_calls else 0
    presort_calls_using = n_presort_calls / all_calls if all_calls else 0


    print('\n------ Unsupported parameters/data usage % ------')
    print('Data: sparse:.......................', int(sparse_using), '%')
    print('Parameter: sample weights:..........', int(weight_using), '%')
    print("Parameter: check_input:.............", int(check_input_using), "%")
    print("Parameter: X_idx_sorted:............", int(X_idx_sorted_using), "%")

    print("Parameter: gini_criterion:..........", int(gini_criterion_using), "%")
    print("Parameter: entropy_criterion:.......", int(entropy_criterion_using), "%")

    print("Parameter: mse_criterion:.......... ", int(mse_criterion_using), "%")
    print("Parameter: friedman_mse_criterion...", int(friedman_mse_criterion_using), "%")
    print("Parameter: mae_criterion:...........", int(mae_criterion_using), "%")

    print("Parameter: best_splitter:...........", int(best_splitter_using), "%")
    print("Parameter: random_splitter:.........", int(random_splitter_using), "%")
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
    print("Parameter: class_weight:............", int(class_weight_using), "%")
    print("Parameter: ccp_alpha:...............", int(ccp_alpha_using), "%")
    print("Parameter: presort:.................", int(presort_calls_using), "%")
