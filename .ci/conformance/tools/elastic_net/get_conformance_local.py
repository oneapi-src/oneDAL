import re
import sys, os

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..', 'tools'))
from conformance_functions import find_count_calls, get_testing_results, get_n_calls

log_filename = "_log.txt"

data_sparce_using_str = "data_sparse_using"
intercept_estimation_str = "intercept_estimation"
normalize_data_str = "normalize_data"

precomputed_Gram_matrix_str = "precomputed_Gram_matrix"
pass_precomputed_matrix_str = "pass_precomputed_matrix"
copy_X_data_str = "copy_X_data"
reuse_previous_solution_str = "reuse_previous_solution"

positive_coefficients_str = "positive_coefficients"
random_state_instance_str = "random_state_instance"
cyclic_selection_str = "cyclic_selection"
random_selection_str = "random_selection"


if __name__ == "__main__":
    n_data_sparse_calls = n_intercept_estimation_calls = n_normalize_data_calls = \
    n_precomputed_Gram_matrix_calls = n_pass_precomputed_matrix_calls = n_copy_X_data_calls = n_reuse_previous_solution_calls = \
    n_positive_coefficients_calls = n_random_state_instance_calls = n_cyclic_selection_calls = n_random_selection_calls = 0

    n_data_sparse_calls = find_count_calls(log_filename, data_sparce_using_str)
    n_intercept_estimation_calls = find_count_calls(log_filename, intercept_estimation_str)
    n_normalize_data_calls = find_count_calls(log_filename, normalize_data_str)

    n_precomputed_Gram_matrix_calls = find_count_calls(log_filename, precomputed_Gram_matrix_str)
    n_pass_precomputed_matrix_calls = find_count_calls(log_filename, pass_precomputed_matrix_str)
    n_copy_X_data_calls = find_count_calls(log_filename, copy_X_data_str)
    n_reuse_previous_solution_calls = find_count_calls(log_filename, reuse_previous_solution_str)

    n_positive_coefficients_calls = find_count_calls(log_filename, positive_coefficients_str)
    n_random_state_instance_calls = find_count_calls(log_filename, random_state_instance_str)
    n_cyclic_selection_calls = find_count_calls(log_filename, cyclic_selection_str)
    n_random_selection_calls = find_count_calls(log_filename, random_selection_str)

    #Calculate metrics
    all_calls = get_n_calls() / 100

    data_sparse_using = n_data_sparse_calls / all_calls if all_calls else 0
    intercept_estimation = n_intercept_estimation_calls / all_calls if all_calls else 0
    normalize_data = n_normalize_data_calls / all_calls if all_calls else 0

    precomputed_Gram_matrix = n_precomputed_Gram_matrix_calls / all_calls if all_calls else 0
    pass_precomputed_matrix = n_pass_precomputed_matrix_calls / all_calls if all_calls else 0
    copy_X_data = n_copy_X_data_calls / all_calls if all_calls else 0
    reuse_previous_solution = n_reuse_previous_solution_calls / all_calls if all_calls else 0

    positive_coefficients = n_positive_coefficients_calls / all_calls if all_calls else 0
    random_state_instance = n_random_state_instance_calls / all_calls if all_calls else 0
    cyclic_selection = n_cyclic_selection_calls / all_calls if all_calls else 0
    random_selection = n_random_selection_calls / all_calls if all_calls else 0

    print('\n------ Unsupported parameters/data usage % ------')
    print('Data is sparse:.................', int(data_sparse_using), '%')
    print('Intercept is estimated:.........', int(intercept_estimation), '%')
    print('Data is normalized:.............', int(normalize_data), '%')
    print('Gram matrix is precomputed:.....', int(precomputed_Gram_matrix), '%')
    print('Gram matrix is passed:..........', int(pass_precomputed_matrix), '%')
    print('X is copied:....................', int(copy_X_data), '%')
    print('Previous Solution is reused:....', int(reuse_previous_solution), '%')
    print('Coefficients are positive:......', int(positive_coefficients), '%')
    print('Random state is inited:.........', int(random_state_instance), '%')
    print('Cyclic selection of features:...', int(cyclic_selection), '%')
    print('Random selection of features:...', int(random_selection), '%')
    print('\n', get_testing_results(log_filename))
