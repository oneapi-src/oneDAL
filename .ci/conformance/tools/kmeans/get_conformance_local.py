import re
import sys, os

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..', 'tools'))
from conformance_functions import find_count_calls, get_testing_results, get_n_calls

log_filename = "_log.txt"

data_sparse_using = "sparse_calls"
data_not_array_using = "notArray_calls"
param_metric_precomputed = "precompute_calls"


if __name__ == "__main__":
    n_srapse_calls = n_not_array_calls = n_param_precomp = 0

    n_srapse_calls = find_count_calls(log_filename, data_sparse_using)
    n_not_array_calls = find_count_calls(log_filename, data_not_array_using)
    n_param_precomp = find_count_calls(log_filename, param_metric_precomputed)

    #Calculate metrics
    all_calls=get_n_calls()/100

    sparse_using = n_srapse_calls / all_calls if all_calls else 0
    not_array_using = n_not_array_calls / all_calls if all_calls else 0
    precomputed_using = n_param_precomp / all_calls if all_calls else 0

    print('\n------ Unsupported parameters/data usage % ------')
    print('Data: sparse:..................', int(sparse_using), '%')
    print('Data: not array:...............', int(not_array_using), '%')
    print('Parameter: metric precomputed:.', int(precomputed_using), '%')
