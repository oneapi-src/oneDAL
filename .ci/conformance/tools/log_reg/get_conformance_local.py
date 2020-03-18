import re
import sys, os

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..', 'tools'))
from conformance_functions import find_count_calls, get_testing_results, get_n_calls

log_filename = "_log.txt"

elasticnet_penalty_str = "elasticnet_penalty_using"
dual_str = "dual_using"
class_weight_str = "class_weight_using"
intercept_scaling_str = "intercept_scaling_using"
l1_ratio_str = "l1_ratio_using"
sag_str = "sag_using"
saga_str = "saga_using"
liblinear_str = "liblinear_using"
sample_weight_str = "sample_weight_using"
data_sparce_using_str = "data_sparce_using"

if __name__ == "__main__":
    n_csr_calls = n_sample_weight_calls = n_elasticnet_penalty_calls = \
    n_dual_calls = n_class_weight_calls = \
    n_intercept_scaling_calls = n_l1_ratio_calls = \
    n_sag_calls = n_saga_calls = n_liblinear_calls = 0

    n_csr_calls = find_count_calls(log_filename, data_sparce_using_str)
    n_sample_weight_calls = find_count_calls(log_filename, sample_weight_str)

    n_elasticnet_penalty_calls = find_count_calls(log_filename, elasticnet_penalty_str)
    n_dual_calls = find_count_calls(log_filename ,dual_str)
    n_class_weight_calls = find_count_calls(log_filename, class_weight_str)
    n_intercept_scaling_calls = find_count_calls(log_filename, intercept_scaling_str)
    n_l1_ratio_calls = find_count_calls(log_filename, l1_ratio_str)
    n_sag_calls = find_count_calls(log_filename, sag_str)
    n_saga_calls = find_count_calls(log_filename, saga_str)
    n_liblinear_calls = find_count_calls(log_filename, liblinear_str)

    #Calculate metrics
    all_calls = get_n_calls() / 100

    csr_using = n_csr_calls / all_calls if all_calls else 0
    sample_weight_using = n_sample_weight_calls / all_calls if all_calls else 0

    n_elasticnet_penalty_using = n_elasticnet_penalty_calls/ all_calls if all_calls else 0
    n_dual_using = n_dual_calls/ all_calls if all_calls else 0
    n_class_weight_using = n_class_weight_calls/ all_calls if all_calls else 0
    n_intercept_scaling_using = n_intercept_scaling_calls/ all_calls if all_calls else 0
    n_l1_ratio_using = n_l1_ratio_calls/ all_calls if all_calls else 0
    n_sag_using = n_sag_calls/ all_calls if all_calls else 0
    n_saga_using = n_saga_calls/ all_calls if all_calls else 0
    n_liblinear_using = n_liblinear_calls/ all_calls if all_calls else 0

    print('\n------ Unsupported parameters/data usage % ------')
    print('Data: sparse:...................', int(csr_using), '%')
    print('Parameter: sample_weight:.......', int(sample_weight_using), '%')
    print('Parameter: elasticnet_penalty:..', int(n_elasticnet_penalty_using), '%')
    print('Parameter: dual:................', int(n_dual_using), '%')
    print('Parameter: class_weight:........', int(n_class_weight_using), '%')
    print('Parameter: intercept_scaling:.. ', int(n_intercept_scaling_using), '%')
    print('Parameter: l1_ratio:............', int(n_l1_ratio_using), '%')
    print('Parameter: sag solver:..........', int(n_sag_using), '%')
    print('Parameter: saga solver:.........', int(n_saga_using), '%')
    print('Parameter: liblinear solver:....', int(n_liblinear_using), '%')

    print('\n', get_testing_results(log_filename))


