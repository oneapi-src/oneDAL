import re

n_calls_file = "__n_calls.tmp"

def find_count_calls(file_name, var_str):
    var_calls = 0

    file_log = open(file_name, "r")
    lines = file_log.readlines()
    file_log.close()

    for line in reversed(lines):
        if var_str in line:
            var_calls = re.findall('\d+', line)
            break
    result = var_calls if type(var_calls) == int else int(var_calls[0])
    return result

def get_testing_results(file_name):
    result_str = ''

    file_log = open(file_name, "r")
    lines = file_log.readlines()
    file_log.close()

    for line in reversed(lines):
        if '====' in line:
            result_str = line
            break

    return result_str

def get_n_calls(file_name=n_calls_file):
    file_calls = open(file_name, "r")
    lines = file_calls.readlines()
    file_calls.close()

    n_calls = 0
    for line in lines:
        n_calls = re.findall('\d+', line)
        break
    return int(n_calls[0])
