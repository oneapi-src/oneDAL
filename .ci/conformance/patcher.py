import re
import sys

alg_name = sys.argv[1]

case = {
    'adaboost':[('from sklearn.ensemble import AdaBoostClassifier', 'from wrappers.wrapper_adaboost import AdaBoostClassifier')],

    'dbscan':[('from sklearn.cluster.dbscan_ import DBSCAN', 'from wrappers.wrapper_dbscan import DBSCAN'),
            ('from sklearn.cluster.dbscan_ import dbscan', 'from wrappers.wrapper_dbscan import dbscan')],

    'elastic_net':[('from sklearn.linear_model.coordinate_descent import Lasso','from wrappers.wrapper_elastic_net import ElasticNet'),
              ('LassoCV, ElasticNet, ElasticNetCV, MultiTaskLasso, MultiTaskElasticNet','from sklearn.linear_model.coordinate_descent import Lasso, LassoCV, ElasticNetCV, MultiTaskLasso, MultiTaskElasticNet'),
              ('MultiTaskElasticNetCV, MultiTaskLassoCV, lasso_path, enet_path','from sklearn.linear_model.coordinate_descent import MultiTaskElasticNetCV, MultiTaskLassoCV, lasso_path, enet_path')],

    'forest':[('from sklearn.ensemble import RandomForestClassifier', 'from wrappers.wrapper_forest import RandomForestClassifier'),
              ('from sklearn.ensemble import RandomForestRegressor', 'from wrappers.wrapper_forest import RandomForestRegressor')],

    'kmeans':[('from sklearn.utils._testing import assert_array_equal',               'from sklearn.utils.testing import assert_array_equal'),
              ('from sklearn.utils._testing import assert_array_almost_equal',        'from sklearn.utils.testing import assert_array_almost_equal'),
              ('from sklearn.utils._testing import assert_allclose',                  'from sklearn.utils.testing import assert_allclose'),
              ('from sklearn.utils._testing import assert_almost_equal',              'from sklearn.utils.testing import assert_almost_equal'),
              ('from sklearn.utils._testing import assert_warns',                     'from sklearn.utils.testing import assert_warns'),
              ('from sklearn.utils._testing import assert_warns_message',             'from sklearn.utils.testing import assert_warns_message'),
              ('from sklearn.utils._testing import if_safe_multiprocessing_with_blas','from sklearn.utils.testing import if_safe_multiprocessing_with_blas'),
              ('from sklearn.utils._testing import assert_raise_message',             'from sklearn.utils.testing import assert_raise_message'),
              ('from sklearn.cluster._k_means import _labels_inertia',''),
              ('from sklearn.cluster._k_means import _mini_batch_step',''),
              ('from sklearn.cluster import KMeans, k_means','from sklearn.cluster import k_means \nfrom wrappers.wrapper_kmeans import KMeans')],

    'knn':[('from sklearn import neighbors, datasets', 'from sklearn import datasets\nfrom wrappers.wrapper_knn import neighbors')],

    'lin_reg':[('from sklearn.linear_model.base import LinearRegression','from wrappers.wrapper_lin_reg import LinearRegression')],

    'log_reg':[('from sklearn.linear_model.logistic import (','from wrappers.wrapper_log_reg import (')],

    'pca':[('from sklearn.decomposition import PCA', 'from wrappers.wrapper_pca import PCA')],

    'ridge_reg':[('from sklearn.linear_model.ridge import Ridge','from wrappers.wrapper_ridge_reg import Ridge')],

    'svm':[('from sklearn import svm, linear_model, datasets, metrics, base', 'from wrappers.wrapper_svm import svm, linear_model, datasets, metrics, base')],

    'tree':[('from sklearn.tree import DecisionTreeClassifier', 'from wrappers.wrapper_tree import DecisionTreeClassifier'),
              ('from sklearn.tree import DecisionTreeRegressor', 'from wrappers.wrapper_tree import DecisionTreeRegressor')]
}

dictionary = case[alg_name]

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
    test_file = "test_" + alg_name + ".py"
    path_file(test_file, dictionary)
