#*******************************************************************************
# Copyright 2014-2017 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#******************************************************************************/

# This files shows the use of single-process/batch-mode in daal4py.

import daal4py as d
from numpy import loadtxt
from time import gmtime, strftime

def pca():
    # let's pass in a file directly, not a table/array
    return d.pca().compute("../data/batch/pca_normalized.csv")

def svd():
    # let's pass in a table/array, not a file
    dfin = loadtxt("../data/batch/svd.csv", delimiter=',')
    return d.svd().compute(dfin)

def mvod():
    dfin = loadtxt("../data/batch/outlierdetection.csv", delimiter=',')
    return d.multivariate_outlier_detection().compute(dfin)

def uvod():
    dfin = loadtxt("../data/batch/outlierdetection.csv", delimiter=',')
    return d.univariate_outlier_detection().compute(dfin)

def kmeans():
    dfin = loadtxt("../data/batch/kmeans_dense.csv", delimiter=',')
    centroids = d.kmeans_init(10, t_method="plusPlusDense")
    return d.kmeans(10).compute(dfin, centroids.compute(dfin))

def naive_bayes():
    data = loadtxt("../data/batch/naivebayes_train_dense.csv", delimiter=',', usecols=range(20))
    gt = loadtxt("../data/batch/naivebayes_train_dense.csv", delimiter=',', usecols=range(20,21))
    gt.shape = (gt.size,1)
    res = d.multinomial_naive_bayes_training(20).compute(data, gt)
    tdata = loadtxt("../data/batch/naivebayes_test_dense.csv", delimiter=',', usecols=range(20))
    return d.multinomial_naive_bayes_prediction(20).compute(tdata, res['model'])

def svm():
    data = loadtxt("../data/batch/svm_two_class_train_dense.csv", delimiter=',', usecols=range(20))
    gt = loadtxt("../data/batch/svm_two_class_train_dense.csv", delimiter=',', usecols=range(20,21))
    gt.shape = (gt.size,1)
    res = d.svm_training(p_cacheSize=600000000).compute(data, gt)
    tdata = loadtxt("../data/batch/svm_two_class_test_dense.csv", delimiter=',', usecols=range(20))
    return d.svm_prediction().compute(tdata, res['model'])

def svm_rbf():
    data = loadtxt("../data/batch/svm_two_class_train_dense.csv", delimiter=',', usecols=range(20))
    gt = loadtxt("../data/batch/svm_two_class_train_dense.csv", delimiter=',', usecols=range(20,21))
    gt.shape = (gt.size,1)
    res = d.svm_training(p_cacheSize=600000000, p_kernel=d.rbf()).compute(data, gt)
    tdata = loadtxt("../data/batch/svm_two_class_test_dense.csv", delimiter=',', usecols=range(20))
    return d.svm_prediction().compute(tdata, res['model'])

def svm_multi():
    data = loadtxt("../data/batch/svm_multi_class_train_dense.csv", delimiter=',', usecols=range(20))
    gt = loadtxt("../data/batch/svm_multi_class_train_dense.csv", delimiter=',', usecols=range(20,21))
    gt.shape = (gt.size,1)
    predictor = d.svm_prediction()
    trainer = d.svm_training(p_cacheSize=100000000)
    res = d.multi_class_classifier_training(p_nClasses=5, p_prediction=predictor, p_training=trainer).compute(data, gt)
    tdata = loadtxt("../data/batch/svm_multi_class_test_dense.csv", delimiter=',', usecols=range(20))
    return d.multi_class_classifier_prediction(p_nClasses=5, p_prediction=predictor, p_training=trainer).compute(tdata, res['model'])

def linreg():
    data = loadtxt("../data/batch/linear_regression_train.csv", delimiter=',', usecols=range(10))
    gt = loadtxt("../data/batch/linear_regression_train.csv", delimiter=',', usecols=range(10,11))
    gt.shape = (gt.size,1)
    res = d.linear_regression_training().compute(data, gt)
    tdata = loadtxt("../data/batch/linear_regression_test.csv", delimiter=',', usecols=range(10))
    return d.linear_regression_prediction().compute(tdata, res['model'])


################################################################################################
#### run all
for func, name in [(kmeans, 'kmeans'), (pca, 'pca'), (naive_bayes, 'naive bayes'), (svd, 'svd'), 
                   (mvod, 'multivariate outlier detection'), (uvod, 'univariate outlier detection'),
                   (svm, 'svm'), (svm_multi, 'svm(multi class)'), (svm_rbf, 'svm(rbf)'), (linreg, 'linear regression')]:
#for func, name in [(pca, 'pca'),]:
    try:
        print('####################### ' + name)
        assert '__daalptr__' in func()
        print(strftime("%H:%M:%S", gmtime()) + '\tPASSED')
    except BaseException as ex:
        print(str(ex))
        print(strftime("%H:%M:%S", gmtime()) + '\tFAILED')
