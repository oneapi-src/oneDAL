# file: mn_naive_bayes_dense_batch.py
#===============================================================================
# Copyright 2014-2019 Intel Corporation.
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
#===============================================================================

## <a name="DAAL-EXAMPLE-PY-MULTINOMIAL_NAIVE_BAYES_DENSE_BATCH"></a>
## \example mn_naive_bayes_dense_batch.py

import os
import sys

from daal.algorithms.multinomial_naive_bayes import prediction, training
from daal.algorithms import classifier
from daal.data_management import (
    FileDataSource, HomogenNumericTable, MergedNumericTable, DataSourceIface, NumericTableIface
)

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTables

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
trainDatasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'naivebayes_train_dense.csv')
testDatasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'naivebayes_test_dense.csv')

nFeatures = 20
nClasses = 20

trainingResult = None
predictionResult = None
testGroundTruth = None


def trainModel():
    global trainingResult

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    trainDataSource = FileDataSource(
        trainDatasetFileName, DataSourceIface.notAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Create Numeric Tables for training data and labels
    trainData = HomogenNumericTable(nFeatures, 0, NumericTableIface.doNotAllocate)
    trainGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.doNotAllocate)
    mergedData = MergedNumericTable(trainData, trainGroundTruth)

    # Retrieve the data from the input file
    trainDataSource.loadDataBlock(mergedData)

    # Create an algorithm object to train the Naive Bayes model
    algorithm = training.Batch(nClasses)

    # Pass a training data set and dependent values to the algorithm
    algorithm.input.set(classifier.training.data,   trainData)
    algorithm.input.set(classifier.training.labels, trainGroundTruth)

    # Build the Naive Bayes model and retrieve the algorithm results
    trainingResult = algorithm.compute()


def testModel():
    global predictionResult, testGroundTruth

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file
    testDataSource = FileDataSource(
        testDatasetFileName, DataSourceIface.notAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Create Numeric Tables for testing data and labels
    testData = HomogenNumericTable(nFeatures, 0, NumericTableIface.doNotAllocate)
    testGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.doNotAllocate)
    mergedData = MergedNumericTable(testData, testGroundTruth)

    # Retrieve the data from input file
    testDataSource.loadDataBlock(mergedData)

    # Create an algorithm object to predict Naive Bayes values
    algorithm = prediction.Batch(nClasses)

    # Pass a testing data set and the trained model to the algorithm
    algorithm.input.setTable(classifier.prediction.data,  testData)
    algorithm.input.setModel(classifier.prediction.model, trainingResult.get(classifier.training.model))

    # Predict Naive Bayes values (Result class from classifier.prediction)
    predictionResult = algorithm.compute()  # Retrieve the algorithm results

def printResults():
    printNumericTables(
        testGroundTruth, predictionResult.get(classifier.prediction.prediction),
        "Ground truth", "Classification results",
        "NaiveBayes classification results (first 20 observations):", 20, flt64=False
    )

if __name__ == "__main__":

    trainModel()
    testModel()
    printResults()
