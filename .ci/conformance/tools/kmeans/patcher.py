import re
import sys

fil_to_path = sys.argv[1]
dictionary = [('from sklearn.utils._testing import assert_array_equal',               'from sklearn.utils.testing import assert_array_equal'),
              ('from sklearn.utils._testing import assert_array_almost_equal',        'from sklearn.utils.testing import assert_array_almost_equal'),
              ('from sklearn.utils._testing import assert_allclose',                  'from sklearn.utils.testing import assert_allclose'),
              ('from sklearn.utils._testing import assert_almost_equal',              'from sklearn.utils.testing import assert_almost_equal'),
              ('from sklearn.utils._testing import assert_warns',                     'from sklearn.utils.testing import assert_warns'),
              ('from sklearn.utils._testing import assert_warns_message',             'from sklearn.utils.testing import assert_warns_message'),
              ('from sklearn.utils._testing import if_safe_multiprocessing_with_blas','from sklearn.utils.testing import if_safe_multiprocessing_with_blas'),
              ('from sklearn.utils._testing import assert_raise_message',             'from sklearn.utils.testing import assert_raise_message'),
              ('from sklearn.cluster._k_means import _labels_inertia',''),
              ('from sklearn.cluster._k_means import _mini_batch_step',''),
              ('from sklearn.cluster import KMeans, k_means','from sklearn.cluster import k_means \nfrom wrappers.wrapper_kmeans import KMeans')]

def path_file(file_name, dictionary):
    result_str = ''

    f = open(file_name, "r")
    lines = f.readlines()
    f.close()

    for to_path_line, path_line in dictionary:
        for i in range(len(lines)):
            if to_path_line in lines[i]:
                lines[i] = path_line + '\n'
                break

    f = open(file_name, "w")
    f.writelines(lines)
    f.close()


if __name__ == "__main__":
    path_file(fil_to_path, dictionary)


