# file: spark_NaiveBayesDense.py
#===============================================================================
# Copyright 2017-2019 Intel Corporation.
#
# This software and the related documents are Intel copyrighted  materials,  and
# your use of  them is  governed by the  express license  under which  they were
# provided to you (License).  Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute,  disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents  are provided as  is,  with no express
# or implied  warranties,  other  than those  that are  expressly stated  in the
# License.
#
# License:
# http://software.intel.com/en-us/articles/intel-sample-source-code-license-agr
# eement/
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
    serializeNumericTable, deserializePartialResult,
    deserializeNumericTable, getMergedDataAndLabelsRDD
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
        key, val = tup
        t1, t2 = val

        # Create an algorithm to train the Naive Bayes model on local nodes
        algorithm = training.Distributed(step1Local, nClasses)

        # Set the input data on local nodes
        deserialized_t1 = deserializeNumericTable(t1)
        deserialized_t2 = deserializeNumericTable(t2)
        algorithm.input.set(classifier.training.data, deserialized_t1)
        algorithm.input.set(classifier.training.labels, deserialized_t2)

        # Train the Naive Bayes model on local nodes
        pres = algorithm.compute()
        serialized_pres = serializeNumericTable(pres)

        return (key, serialized_pres)
    return trainDataRDD.map(mapper)


def trainMaster(partsRDD):

    # Create an algorithm to train the Naive Bayes model on the master node
    algorithm = training.Distributed(step2Master, nClasses)

    parts_List = partsRDD.collect()

    # Add partial results computed on local nodes to the algorithm on the master node
    for _, value in parts_List:
        deserialized_pres = deserializePartialResult(value, training)
        algorithm.input.add(training.partialModels, deserialized_pres)

    # Train the Naive Bayes model on the master node
    algorithm.compute()

    # Finalize computations and retrieve the training results
    trainingResult = algorithm.finalizeCompute()

    return trainingResult.get(classifier.training.model)


def testModel(testData, model):

    # Create algorithm objects to predict values of the Naive Bayes model with the defaultDense method
    algorithm = prediction.Batch(nClasses)

    # Pass the test data to the algorithm
    parts_List = testData.collect()
    for key, (t1, t2) in parts_List:
        deserialized_t1 = deserializeNumericTable(t1)
        algorithm.input.setTable(classifier.prediction.data, deserialized_t1)

    algorithm.input.setModel(classifier.prediction.model, model)

    # Compute the prediction results
    predictionResult = algorithm.compute()

    # Retrieve the results
    return predictionResult.get(classifier.prediction.prediction)


if __name__ == "__main__":

    # Create SparkContext that loads defaults from the system properties and the classpath and sets the name
    sc = SparkContext(conf=SparkConf().setAppName("Spark Naive Bayes").setMaster('local[4]'))

    trainDataFilesPath = "/Spark/NaiveBayesDense/data/NaiveBayesDense_train_?.csv"
    trainDataLabelsFilesPath = "/Spark/NaiveBayesDense/data/NaiveBayesDense_train_labels_?.csv"
    testDataFilesPath = "/Spark/NaiveBayesDense/data/NaiveBayesDense_test_1.csv"
    testDataLabelsFilesPath = "/Spark/NaiveBayesDense/data/NaiveBayesDense_test_labels_1.csv"

    # Read the training data and labels from a specified path
    trainDataAndLabelsRDD = getMergedDataAndLabelsRDD(trainDataFilesPath, trainDataLabelsFilesPath, sc)

    # Read the test data and labels from a specified path
    testDataAndLabelsRDD = getMergedDataAndLabelsRDD(testDataFilesPath, testDataLabelsFilesPath, sc)

    # Compute the results of the Naive Bayes algorithm for dataRDD
    result = runNaiveBayes(trainDataAndLabelsRDD, testDataAndLabelsRDD)

    # Print the results
    parts_List = testDataAndLabelsRDD.collect()
    for _, (t1, t2) in parts_List:
        expected = deserializeNumericTable(t2)

    # Redirect stdout to a file for correctness verification
    stdout = sys.stdout
    sys.stdout = open('NaiveBayesDense.out', 'w')

    printNumericTables(expected, result, "Ground truth", "Classification results",
                       "NaiveBayes classification results (first 20 observations):", 20, flt64=False)

    # Restore stdout
    sys.stdout = stdout

    sc.stop()
