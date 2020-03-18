import re
import sys, os

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..', 'tools'))
from conformance_functions import find_count_calls, get_testing_results, get_n_calls

log_filename = "_log.txt"

csr_data = "csr_calls"
weights_uniform_calls = "weights_uniform_calls"
weights_distance_calls = "weights_distance_calls"
algorithm_auto_calls = "algorithm_auto_calls"
algorithm_ball_tree_calls = "algorithm_ball_calls"
algorithm_kdtree_calls = "algorithm_kdtree_calls"
algorithm_brute_calls = "algorithm_brute_calls"
metric_manhattan_calls = "metric_manhattan_calls"
metric_euclidean_calls = "metric_euclidean_calls"
metric_minkowski_calls = "metric_minkowski_calls"


if __name__ == "__main__":
    n_csr_data = 0
    n_weights_uniform_calls = 0
    n_weights_distance_calls = 0
    n_algorithm_auto_calls = 0
    n_algorithm_ball_tree_calls = 0
    n_algorithm_kdtree_calls = 0
    n_algorithm_brute_calls = 0
    n_metric_manhattan_calls = 0
    n_metric_euclidean_calls = 0
    n_metric_minkowski_calls = 0

    n_csr_data = find_count_calls(log_filename, csr_data)
    n_weights_uniform_calls = find_count_calls(log_filename, weights_uniform_calls)
    n_weights_distance_calls = find_count_calls(log_filename, weights_distance_calls)
    n_algorithm_auto_calls = find_count_calls(log_filename, algorithm_auto_calls)
    n_algorithm_ball_tree_calls = find_count_calls(log_filename, algorithm_ball_tree_calls)
    n_algorithm_kdtree_calls = find_count_calls(log_filename, algorithm_kdtree_calls)
    n_algorithm_brute_calls = find_count_calls(log_filename, algorithm_brute_calls)
    n_metric_manhattan_calls = find_count_calls(log_filename, metric_manhattan_calls)
    n_metric_euclidean_calls = find_count_calls(log_filename, metric_euclidean_calls)
    n_metric_minkowski_calls = find_count_calls(log_filename, metric_minkowski_calls)

    #Calculate metrics
    all_calls=get_n_calls()/100

    sparse_using = n_csr_data / all_calls if all_calls else 0
    weights_uniform_using = n_weights_uniform_calls / all_calls if all_calls else 0
    weights_distance_using = n_weights_distance_calls / all_calls if all_calls else 0
    algorithm_auto_using = n_algorithm_auto_calls / all_calls if all_calls else 0
    algorithm_ball_tree_using = n_algorithm_ball_tree_calls / all_calls if all_calls else 0
    algorithm_kdtree_using = n_algorithm_kdtree_calls / all_calls if all_calls else 0
    algorithm_brute_using = n_algorithm_brute_calls / all_calls if all_calls else 0
    metric_manhattan_using = n_metric_manhattan_calls / all_calls if all_calls else 0
    metric_euclidean_using = n_metric_euclidean_calls / all_calls if all_calls else 0
    metric_minkowski_using = n_metric_minkowski_calls / all_calls if all_calls else 0

    print('\n------ Parameters/data usage % ------')
    print('Data: sparse:...................', int(sparse_using), '%')
    print('Parameter: weights uniform:.....', int(weights_uniform_using), '%')
    print('Parameter: weights distance:....', int(weights_distance_using), '%')
    print('Parameter: algorithm auto:......', int(algorithm_auto_using), '%')
    print('Parameter: algorithm ball_tree:.', int(algorithm_ball_tree_using), '%')
    print('Parameter: algorithm kdtree:....', int(algorithm_kdtree_using), '%')
    print('Parameter: algorithm brute:.....', int(algorithm_brute_using), '%')
    print('Parameter: metric manhattan:....', int(metric_manhattan_using), '%')
    print('Parameter: metric euclidean:....', int(metric_euclidean_using), '%')
    print('Parameter: metric minkowski:....', int(metric_minkowski_using), '%')
