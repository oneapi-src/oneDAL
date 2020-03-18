import re
import sys, os

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..', 'tools'))
from conformance_functions import find_count_calls, get_testing_results, get_n_calls

log_filename = "_log.txt"

data_not_nd_array = "nd_array_calls"
data_not_skinny = "not_skinny_calls"
param_not_daal_svd_solver = "solver_calls"
size_of_array = "size_calls"


if __name__ == "__main__":
    n_svd_solver_calls = n_not_nd_array_calls = n_not_skinny_calls = n_size_of_array= 0

    n_svd_solver_calls = find_count_calls(log_filename, param_not_daal_svd_solver)
    n_not_nd_array_calls = find_count_calls(log_filename, data_not_nd_array)
    n_not_skinny_calls = find_count_calls(log_filename, data_not_skinny)
    n_size_of_array_calls = find_count_calls(log_filename, size_of_array)

    #Calculate metrics
    all_calls=get_n_calls()/100

    param_not_daal_svd_solver_using = n_svd_solver_calls / all_calls if all_calls else 0
    data_not_nd_array_using = n_not_nd_array_calls / all_calls if all_calls else 0
    data_not_skinny_using = n_not_skinny_calls / all_calls if all_calls else 0
    size_of_array_using = n_size_of_array_calls / all_calls if all_calls else 0

    print('\n------ Unsupported parameters/data usage % ------')
    print('Data: not nd array:............', int(data_not_nd_array_using), '%')
    print('Data: not tall/skinny array:...', int(data_not_skinny_using), '%')
    print('Data: n_samples < n_features:..', int(size_of_array_using), '%')
    print('Parameter: not daal svd solver:', int(param_not_daal_svd_solver_using), '%')
