# file: spark_NaiveBayesCSR.py
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
#  Content:
#      Python sample of Naive Bayes classification in the distributed processing
#      mode.
#
#      The program trains the Naive Bayes model on a supplied training data set
#      and then performs classification of previously unseen data.
#
from __future__ import print_function
import os
import sys

from pyspark import SparkContext, SparkConf

from daal import step1Local, step2Master
from daal.algorithms.multinomial_naive_bayes import training, prediction
from daal.algorithms import classifier

from distributed_hdfs_dataset import (
    serializeNumericTable, deserializePartialResult, deserializeCSRNumericTable,
    deserializeNumericTable, getMergedCSRDataAndLabelsRDD
)

utils_folder = os.path.join(os.environ['DAALROOT'], 'examples', 'python', 'source')
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTables

nClasses = 20
nTestObservations = 2000


def runNaiveBayes(trainDataRDD, testDataRDD):
    partsRDD = trainLocal(trainDataRDD)
    model = trainMaster(partsRDD)
    return testModel(testDataRDD, model)


def trainLocal(trainDataRDD):

    def mapper(tup):

        key, tables = tup
        csr_table, homogen_table = tables

        # Create an algorithm to train the Naive Bayes model on local nodes
        algorithm = training.Distributed(step1Local, nClasses, method=training.fastCSR)

        # Set the input data on local nodes
        deserialized_csr_table = deserializeCSRNumericTable(csr_table)
        deserialized_homogen_table = deserializeNumericTable(homogen_table)
        algorithm.input.set(classifier.training.data, deserialized_csr_table)
        algorithm.input.set(classifier.training.labels, deserialized_homogen_table)

        # Train the Naive Bayes model on local nodes
        pres = algorithm.compute()
        serialized_pres = serializeNumericTable(pres)

        return (key, serialized_pres)
    return trainDataRDD.map(mapper)


def trainMaster(partsRDD):

    # Create an algorithm to train the Naive Bayes model on the master node
    algorithm = training.Distributed(step2Master, nClasses, gmethod=training.fastCSR)

    parts_list = partsRDD.collect()

    # Add partial results computed on local nodes to the algorithm on the master node
    for key, value in parts_list:
        deserialized_pres = deserializePartialResult(value, training)
        algorithm.input.add(training.partialModels, deserialized_pres)

    # Train the Naive Bayes model on the master node
    algorithm.compute()

    # Finalize computations and retrieve the training results
    trainingResult = algorithm.finalizeCompute()

    return trainingResult.get(classifier.training.model)


def testModel(testData, model):

    # Create algorithm objects to predict values of the Naive Bayes model with the fastCSR method
    algorithm = prediction.Batch(nClasses, method=prediction.fastCSR)

    # Pass the test data to the algorithm
    parts_list = testData.collect()
    for _, (csr, homogen) in parts_list:
        deserialized_csr = deserializeCSRNumericTable(csr)
        algorithm.input.setTable(classifier.prediction.data, deserialized_csr)

    algorithm.input.setModel(classifier.prediction.model, model)

    # Compute the prediction results
    predictionResult = algorithm.compute()

    # Retrieve the results
    return predictionResult.get(classifier.prediction.prediction)


if __name__ == "__main__":

    # Create SparkContext that loads defaults from the system properties and the classpath and sets the name
    sc = SparkContext(conf=SparkConf().setAppName("Spark Naive Bayes").setMaster('local[4]'))

    trainDataFilesPath = "/Spark/NaiveBayesCSR/data/NaiveBayesCSR_train_?.csv"
    trainDataLabelsFilesPath = "/Spark/NaiveBayesCSR/data/NaiveBayesCSR_train_labels_?.csv"
    testDataFilesPath = "/Spark/NaiveBayesCSR/data/NaiveBayesCSR_test_1.csv"
    testDataLabelsFilesPath = "/Spark/NaiveBayesCSR/data/NaiveBayesCSR_test_labels_1.csv"

    # Read the training data and labels from a specified path
    trainDataAndLabelsRDD = getMergedCSRDataAndLabelsRDD(trainDataFilesPath, trainDataLabelsFilesPath, sc)

    # Read the test data and labels from a specified path
    testDataAndLabelsRDD = getMergedCSRDataAndLabelsRDD(testDataFilesPath, testDataLabelsFilesPath, sc)

    # Compute the results of the Naive Bayes algorithm for dataRDD
    predicted = runNaiveBayes(trainDataAndLabelsRDD, testDataAndLabelsRDD)

    # Print the results
    parts_list = testDataAndLabelsRDD.collect()
    for _, (csr, homogen) in parts_list:
        expected = deserializeNumericTable(homogen)

    # Redirect stdout to a file for correctness verification
    stdout = sys.stdout
    sys.stdout = open('NaiveBayesCSR.out', 'w')

    printNumericTables(expected, predicted, "Ground truth", "Classification results",
                       "NaiveBayes classification results (first 20 observations):", 20, flt64=False)

    # Restore stdout
    sys.stdout = stdout

    sc.stop()
