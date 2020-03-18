import re
import sys, os

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..', 'tools'))
from conformance_functions import find_count_calls, get_testing_results, get_n_calls

log_filename = "_log.txt"

data_sparse_using = "data_sparse_using"
param_metric_precomputed = "param_metric_precomputed"
param_metric_minkowski = "param_metric_minkowski"
param_metric_manhattan = "param_metric_manhattan"


if __name__ == "__main__":
    n_srapse_calls = n_param_minkowski = n_param_manhat = n_param_precomp = 0

    n_srapse_calls = find_count_calls(log_filename, data_sparse_using)
    n_param_precomp = find_count_calls(log_filename, param_metric_precomputed)
    n_param_minkowski = find_count_calls(log_filename, param_metric_minkowski)
    n_param_manhat = find_count_calls(log_filename, param_metric_manhattan)

    #Calculate metrics
    all_calls=get_n_calls()/100

    sparse_using = n_srapse_calls / all_calls if all_calls else 0
    minkowski_using = n_param_minkowski / all_calls if all_calls else 0
    manhattan_using = n_param_manhat / all_calls if all_calls else 0
    precomputed_using = n_param_precomp / all_calls if all_calls else 0

    print('\n------ Unsupported parameters/data usage % ------')
    print('Data: sparse:..................', int(sparse_using), '%')
    print('Parameter: metric precomputed:.', int(precomputed_using), '%')
    print('Parameter: metric minkowski:...', int(minkowski_using), '%')
    print('Parameter: metric manhattan:...', int(manhattan_using), '%')
