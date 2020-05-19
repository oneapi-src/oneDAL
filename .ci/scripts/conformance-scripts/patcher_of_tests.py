import re
import sys

alg_name = sys.argv[1]

testToPatch = ['dbscan','elastic_net', 'lin_reg', 'ridge_reg']

case = {
    'dbscan':[('from sklearn.cluster.dbscan_ import DBSCAN', 'from sklearn.cluster import DBSCAN'),
              ('from sklearn.cluster.dbscan_ import dbscan', 'from sklearn.cluster import dbscan')
            ],

    'elastic_net':[('from sklearn.linear_model.coordinate_descent import Lasso, \\','from sklearn.linear_model import Lasso, \\')],

    'lin_reg':[('from sklearn.linear_model.base import LinearRegression','from sklearn.linear_model import LinearRegression')],

    'ridge_reg':[('from sklearn.linear_model.ridge import ridge_regression', 'from sklearn.linear_model import ridge_regression'),
                 ('from sklearn.linear_model.ridge import Ridge', 'from sklearn.linear_model import Ridge'),
                 ('from sklearn.linear_model.ridge import RidgeCV', 'from sklearn.linear_model import RidgeCV'),
                 ('from sklearn.linear_model.ridge import RidgeClassifier', 'from sklearn.linear_model import RidgeClassifier'),
                 ('from sklearn.linear_model.ridge import RidgeClassifierCV', 'from sklearn.linear_model import RidgeClassifierCV')]
}

def path_file(file_name, dictionary):
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
    if testToPatch.count(alg_name):
        dictionary = case[alg_name]
        test_file = "test_" + alg_name + ".py"
        path_file(test_file, dictionary)
