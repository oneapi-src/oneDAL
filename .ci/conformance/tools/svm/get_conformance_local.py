import re
import sys, os

sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'..', 'tools'))
from conformance_functions import find_count_calls, get_testing_results, get_n_calls

log_filename = "_log.txt"

csr_data_calls = "csr_data_calls"
penalty_calls = "penalty_calls"
kernel_rbf_calls = "kernel_rbf_calls"
kernel_linear_calls = "kernel_linear_calls"
kernel_poly_calls = "kernel_poly_calls"
kernel_sigmoid_calls = "kernel_sigmoid_calls"
kernel_precomputed_calls = "kernel_precomputed_calls"
shrinking_calls = "shrinking_calls"
probability_calls = "probability_calls"
class_weight_calls = "class_weight_calls"
verbose_calls = "verbose_calls"
df_shape_ovo_calls = "df_shape_ovo_calls"
df_shape_ovr_calls = "df_shape_ovr_calls"
random_state_calls = "random_state_calls"


if __name__ == "__main__":
    n_csr_data_calls = 0
    n_penalty_calls = 0
    n_kernel_rbf_calls = 0
    n_kernel_linear_calls = 0
    n_kernel_poly_calls = 0
    n_kernel_sigmoid_calls = 0
    n_kernel_precomputed_calls = 0
    n_shrinking_calls = 0
    n_probability_calls = 0
    n_class_weight_calls = 0
    n_verbose_calls = 0
    n_df_shape_ovo_calls = 0
    n_df_shape_ovr_calls = 0
    n_random_state_calls = 0

    n_csr_data_calls = find_count_calls(log_filename, csr_data_calls)
    n_penalty_calls = find_count_calls(log_filename, penalty_calls)
    n_kernel_rbf_calls = find_count_calls(log_filename, kernel_rbf_calls)
    n_kernel_linear_calls = find_count_calls(log_filename, kernel_linear_calls)
    n_kernel_poly_calls = find_count_calls(log_filename, kernel_poly_calls)
    n_kernel_sigmoid_calls = find_count_calls(log_filename, kernel_sigmoid_calls)
    n_kernel_precomputed_calls = find_count_calls(log_filename, kernel_precomputed_calls)
    n_shrinking_calls = find_count_calls(log_filename, shrinking_calls)
    n_probability_calls = find_count_calls(log_filename, probability_calls)
    n_class_weight_calls = find_count_calls(log_filename, class_weight_calls)
    n_verbose_calls = find_count_calls(log_filename, verbose_calls)
    n_df_shape_ovo_calls = find_count_calls(log_filename, df_shape_ovo_calls)
    n_df_shape_ovr_calls = find_count_calls(log_filename, df_shape_ovr_calls)
    n_random_state_calls = find_count_calls(log_filename, random_state_calls)

    #Calculate metrics
    all_calls=get_n_calls()/100

    csr_data_using = n_csr_data_calls / all_calls if all_calls else 0
    penalty_using = n_penalty_calls / all_calls if all_calls else 0
    kernel_rbf_using = n_kernel_rbf_calls / all_calls if all_calls else 0
    kernel_linear_using = n_kernel_linear_calls / all_calls if all_calls else 0
    kernel_poly_using = n_kernel_poly_calls / all_calls if all_calls else 0
    kernel_sigmoid_using = n_kernel_sigmoid_calls / all_calls if all_calls else 0
    kernel_precomputed_using = n_kernel_precomputed_calls / all_calls if all_calls else 0
    shrinking_using = n_shrinking_calls / all_calls if all_calls else 0
    probability_using = n_probability_calls / all_calls if all_calls else 0
    class_weight_using = n_class_weight_calls / all_calls if all_calls else 0
    verbose_using = n_verbose_calls / all_calls if all_calls else 0
    df_shape_ovo_using = n_df_shape_ovo_calls / all_calls if all_calls else 0
    df_shape_ovr_using = n_df_shape_ovr_calls / all_calls if all_calls else 0
    random_state_using = n_random_state_calls / all_calls if all_calls else 0

    print('\n------ Parameters/data usage % ------')
    print('Data: sparse:..........................', int(csr_data_using), '%')
    print('Parameter: penalty:....................', int(penalty_using), '%')
    print('Parameter: kernel rbf:.................', int(kernel_rbf_using), '%')
    print('Parameter: kernel linear:..............', int(kernel_linear_using), '%')
    print('Parameter: kernel poly:................', int(kernel_poly_using), '%')
    print('Parameter: kernel sigmoid:.............', int(kernel_sigmoid_using), '%')
    print('Parameter: kernel precomputed:.........', int(kernel_precomputed_using), '%')
    print('Parameter: shrinking:..................', int(shrinking_using), '%')
    print('Parameter: probability:................', int(probability_using), '%')
    print('Parameter: class weight:...............', int(class_weight_using), '%')
    print('Parameter: verbose:....................', int(verbose_using), '%')
    print('Parameter: df shape ovo:...............', int(df_shape_ovo_using), '%')
    print('Parameter: df shape ovr:...............', int(df_shape_ovr_using), '%')
    print('Parameter: not standard random state:..', int(random_state_using), '%')
