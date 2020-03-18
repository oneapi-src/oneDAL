import re
import sys, os

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..', 'tools'))
from conformance_functions import find_count_calls, get_testing_results, get_n_calls

log_filename = "_log.txt"

not_good_shape_for_daal_str = "fit_shape_not_good_for_daal"
data_not_float_str = "data_not_float"
normalize_data_str = "normalize_data"
sample_weight_str = "sample_weight"
data_sparce_using_str = "data_sparce_using"

if __name__ == "__main__":
    n_csr_calls = n_sample_weight_calls = n_normalizing_calls = \
    n_data_not_float_calls = n_not_good_for_daal_shape_calls = 0

    n_csr_calls = find_count_calls(log_filename, data_sparce_using_str)
    n_sample_weight_calls = find_count_calls(log_filename, sample_weight_str)
    n_normalizing_calls = find_count_calls(log_filename, normalize_data_str)
    n_data_not_float_calls = find_count_calls(log_filename, data_not_float_str)
    n_not_good_for_daal_shape_calls = find_count_calls(log_filename, not_good_shape_for_daal_str)

    #Calculate metrics
    all_calls = get_n_calls() / 100

    csr_using = n_csr_calls / all_calls if all_calls else 0
    sample_weight_using = n_sample_weight_calls / all_calls if all_calls else 0
    normalizing_using = n_normalizing_calls / all_calls if all_calls else 0
    data_not_float_using = n_data_not_float_calls / all_calls if all_calls else 0
    not_good_for_daal_shape_using = n_not_good_for_daal_shape_calls / all_calls if all_calls else 0


    print('\n------ Unsupported parameters/data usage % ------')
    print('Data: sparse:...................', int(csr_using), '%')
    print('Data not float:.................', int(data_not_float_using), '%')
    print('Data not a good shape for daal:.', int(not_good_for_daal_shape_using), '%')
    print('Parameter: sample_weight:.......', int(sample_weight_using), '%')
    print('Parameter normalizing:..........', int(normalizing_using), '%')
    print('\n', get_testing_results(log_filename))


