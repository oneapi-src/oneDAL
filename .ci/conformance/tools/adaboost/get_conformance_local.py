import re
import sys, os

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..', 'tools'))
from conformance_functions import find_count_calls, get_testing_results, get_n_calls

log_filename = "_log.txt"

data_sparse_using = "sparse_calls"
sammer_calls = "sammer_calls"
samme_calls = "samme_calls"
weight_calls = "weight_calls"


if __name__ == "__main__":
    n_srapse_calls = n_sammer_calls = n_samme_calls = n_weight_calls = 0

    n_srapse_calls = find_count_calls(log_filename, data_sparse_using)
    n_sammer_calls = find_count_calls(log_filename, sammer_calls)
    n_samme_calls = find_count_calls(log_filename, samme_calls)
    n_weight_calls = find_count_calls(log_filename, weight_calls)

    #Calculate metrics
    all_calls=get_n_calls()/100

    sparse_using = n_srapse_calls / all_calls if all_calls else 0
    sammer_using = n_sammer_calls / all_calls if all_calls else 0
    samme_using = n_samme_calls / all_calls if all_calls else 0
    weight_using = n_weight_calls / all_calls if all_calls else 0

    print('\n------ Unsupported parameters/data usage % ------')
    print('Data: sparse:..................', int(sparse_using), '%')
    print('Data: sample weights:..........', int(weight_using), '%')
    print('Parameter: algorithm SAMME.R:..', int(sammer_using), '%')
    print('Parameter: algorithm SAMME:....', int(samme_using), '%')
