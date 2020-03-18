import re
import sys, os

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..', 'tools'))
from conformance_functions import find_count_calls, get_testing_results, get_n_calls

log_filename = "_log.txt"

svd_str = "param_solver_svd"
cholesky_str = "param_solver_cholesky"
lsqr_str = "param_solver_lsqr"
sparse_cg_str = "param_solver_sparse_cg"
sag_solver_str = "param_solver_sag"
saga_solver_str = "param_solver_saga"
auto_solver_str = "param_solver_auto"

not_good_shape_for_daal_str = "fit_shape_not_good_for_daal"
data_not_float_str = "data_not_float"
normalize_data_str = "normalize_data"
sample_weight_str = "sample_weight_using"
data_sparce_using_str = "data_sparse_using"


if __name__ == "__main__":
    n_param_auto = n_param_svd = n_param_cholesky = \
    n_param_lsqr = n_param_cg = n_param_sag = \
    n_param_saga = n_csr_calls = n_sample_weight_calls = \
    n_normalizing_calls = n_data_not_float_calls = \
    n_not_good_for_daal_shape_calls = 0

    n_param_auto = find_count_calls(log_filename, auto_solver_str)
    n_param_svd = find_count_calls(log_filename, svd_str)
    n_param_cholesky = find_count_calls(log_filename, cholesky_str)
    n_param_lsqr = find_count_calls(log_filename, lsqr_str)
    n_param_cg = find_count_calls(log_filename, sparse_cg_str)
    n_param_sag = find_count_calls(log_filename, sag_solver_str)
    n_param_saga = find_count_calls(log_filename, saga_solver_str)

    n_csr_calls = find_count_calls(log_filename, data_sparce_using_str)
    n_sample_weight_calls = find_count_calls(log_filename, sample_weight_str)
    n_normalizing_calls = find_count_calls(log_filename, normalize_data_str)
    n_data_not_float_calls = find_count_calls(log_filename, data_not_float_str)
    n_not_good_for_daal_shape_calls = find_count_calls(log_filename, not_good_shape_for_daal_str)

    #Calculate metrics
    all_calls = get_n_calls() / 100

    auto_using = n_param_auto / all_calls if all_calls else 0
    svd_using = n_param_svd / all_calls if all_calls else 0
    cholesky_using = n_param_cholesky / all_calls if all_calls else 0
    lsqr_using = n_param_lsqr / all_calls if all_calls else 0
    cg_using = n_param_cg / all_calls if all_calls else 0
    sag_using = n_param_sag / all_calls if all_calls else 0
    saga_using = n_param_saga / all_calls if all_calls else 0

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
    print('Parameter: normalizing:.........', int(normalizing_using), '%')
    print('Parameter: auto solver:.........', int(auto_using), '%')
    print('Parameter: svd solver:..........', int(svd_using), '%')
    print('Parameter: cholesky solver:.....', int(cholesky_using), '%')
    print('Parameter: lsqr solver:.........', int(lsqr_using), '%')
    print('Parameter: cg solver:...........', int(cg_using), '%')
    print('Parameter: sag solver:..........', int(sag_using), '%')
    print('Parameter: saga solver:.........', int(saga_using), '%')
    print('\n', get_testing_results(log_filename))


