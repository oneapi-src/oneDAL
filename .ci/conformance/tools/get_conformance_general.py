import re
from conformance_functions import find_count_calls, get_testing_results

log_filename = "_log.txt"
n_calls_file = "__n_calls.tmp"

scikit_learn_str = "skl_calls"
daal4py_str = "daal_calls"

def print_calls(number):
    file=open(n_calls_file, 'w')
    file.write(str(number)+'\n')
    file.close()

if __name__ == "__main__":
    n_d4p_calls = n_skl_calls = 0

    n_skl_calls = find_count_calls(log_filename, scikit_learn_str)
    n_d4p_calls = find_count_calls(log_filename, daal4py_str)

    #Calculate metrics
    all_calls = (n_d4p_calls + n_skl_calls) / 100
    d4p_using = n_d4p_calls / all_calls if all_calls else 0
    print_calls(n_d4p_calls + n_skl_calls)

    print('Number of Scikit-learn calls: ', n_skl_calls)
    print('Number of daal4py calls:      ', n_d4p_calls)
    print('Percent of using daal4py:     ', int(d4p_using), '%')
    print('\n', get_testing_results(log_filename))
