# file: spark_LinearRegressionQR.py
#===============================================================================
# Copyright 2017-2019 Intel Corporation
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
#===============================================================================

#
# Content:
#     Python sample of multiple linear regression in the distributed processing
#     mode.
#
#     The program trains the multiple linear regression model on a training
#     data set with a QR decomposition-based method and computes regression
#     for the test data.
#
from __future__ import print_function
import os
import sys

from pyspark import SparkContext, SparkConf

from daal import step1Local, step2Master
from daal.algorithms.linear_regression import training, prediction

from distributed_hdfs_dataset import (
    serializeNumericTable, deserializeNumericTable, getMergedDataAndLabelsRDD, deserializePartialResult
)

utils_folder = os.path.join(os.environ['DAALROOT'], 'examples', 'python', 'source')
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable


def runLinearRegression(trainDataRDD, testDataRDD):
    result = {}
    partsRDD = trainLocal(trainDataRDD)
    model = trainMaster(partsRDD)
    result['beta'] = model.getBeta()
    result['predicted'] = testModel(testDataRDD, model)
    return result


def trainLocal(trainDataRDD):

    def mapper(tup):

        key, tables = tup
        h_table1, h_table2 = tables

        # Create an algorithm object to train the multiple linear regression model with a QR decomposition-based method
        linearRegressionTraining = training.Distributed(step1Local, method=training.qrDense)
        # Set the input data on local nodes
        deserialized_h_table1 = deserializeNumericTable(h_table1)
        deserialized_h_table2 = deserializeNumericTable(h_table2)
        linearRegressionTraining.input.set(training.data, deserialized_h_table1)
        linearRegressionTraining.input.set(training.dependentVariables, deserialized_h_table2)

        # Build a partial multiple linear regression model
        pres = linearRegressionTraining.compute()
        serialized_pres = serializeNumericTable(pres)

        return (key, serialized_pres)

    return trainDataRDD.map(mapper)


def trainMaster(partsRDD):

    # Create an algorithm object to train the multiple linear regression model with a QR decomposition-based method
    linearRegressionTraining = training.Distributed(step2Master, method=training.qrDense)

    parts_list = partsRDD.collect()

    # Add partial results computed on local nodes to the algorithm on the master node
    for key, pres in parts_list:
        deserialized_pres = deserializePartialResult(pres, training)
        linearRegressionTraining.input.add(training.partialModels, deserialized_pres)

    # Build and retrieve the final multiple linear regression model
    linearRegressionTraining.compute()

    trainingResult = linearRegressionTraining.finalizeCompute()

    return trainingResult.get(training.model)


def testModel(testData, model):

    # Create algorithm objects to predict values of multiple linear regression with the default method
    linearRegressionPredict = prediction.Batch(method=prediction.defaultDense)

    # Pass the test data to the algorithm
    parts_list = testData.collect()
    for key, (h_table1, _) in parts_list:
        deserialized_h_table1 = deserializeNumericTable(h_table1)
        linearRegressionPredict.input.setTable(prediction.data, deserialized_h_table1)

    linearRegressionPredict.input.setModel(prediction.model, model)

    # Compute and retrieve the prediction results
    predictionResult = linearRegressionPredict.compute()

    return predictionResult.get(prediction.prediction)


if __name__ == "__main__":

    # Create SparkContext that loads defaults from the system properties and the classpath and sets the name
    sc = SparkContext(conf=SparkConf().setAppName("Spark Linear Regression").setMaster('local[4]'))

    trainDataFilesPath = "/Spark/LinearRegressionQR/data/LinearRegressionQR_train_?.csv"
    trainDataLabelsFilesPath = "/Spark/LinearRegressionQR/data/LinearRegressionQR_train_labels_?.csv"
    testDataFilesPath = "/Spark/LinearRegressionQR/data/LinearRegressionQR_test_1.csv"
    testDataLabelsFilesPath = "/Spark/LinearRegressionQR/data/LinearRegressionQR_test_labels_1.csv"

    # Read the training data and labels from a specified path
    trainDataAndLabelsRDD = getMergedDataAndLabelsRDD(trainDataFilesPath, trainDataLabelsFilesPath, sc)

    # Read the test data and labels from a specified path
    testDataAndLabelsRDD = getMergedDataAndLabelsRDD(testDataFilesPath, testDataLabelsFilesPath, sc)

    # Compute linear regression for dataRDD
    res = runLinearRegression(trainDataAndLabelsRDD, testDataAndLabelsRDD)

    # Redirect stdout to a file for correctness verification
    stdout = sys.stdout
    sys.stdout = open('LinearRegressionQR.out', 'w')

    # Print the results
    parts_list = testDataAndLabelsRDD.collect()
    for key, (_, h_table2) in parts_list:
        deserialzied_expected = deserializeNumericTable(h_table2)

    printNumericTable(res['beta'], "Coefficients:")
    printNumericTable(res['predicted'], "First 10 rows of results (obtained): ", 10)
    printNumericTable(deserialzied_expected, "First 10 rows of results (expected): ", 10)

    # Restore stdout
    sys.stdout = stdout

    sc.stop()
