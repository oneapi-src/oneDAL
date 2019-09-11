# file: mn_naive_bayes_csr_batch.py
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

## <a name="DAAL-EXAMPLE-PY-MULTINOMIAL_NAIVE_BAYES_CSR_BATCH"></a>
## \example mn_naive_bayes_csr_batch.py

import os
import sys

from daal.algorithms.multinomial_naive_bayes import prediction, training
from daal.algorithms import classifier
from daal.data_management import FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTables, createSparseTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
trainDatasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'naivebayes_train_csr.csv')
trainGroundTruthFileName = os.path.join(DAAL_PREFIX, 'batch', 'naivebayes_train_labels.csv')

testDatasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'naivebayes_test_csr.csv')
testGroundTruthFileName = os.path.join(DAAL_PREFIX, 'batch', 'naivebayes_test_labels.csv')

nTrainObservations = 8000
nTestObservations = 2000
nClasses = 20

trainingResult = None
predictionResult = None


def trainModel():
    global trainingResult

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    trainGroundTruthSource = FileDataSource(
        trainGroundTruthFileName,
        DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Retrieve the data from input files
    trainData = createSparseTable(trainDatasetFileName)
    trainGroundTruthSource.loadDataBlock(nTrainObservations)

    # Create an algorithm object to train the Naive Bayes model
    algorithm = training.Batch(nClasses, method=training.fastCSR)

    # Pass a training data set and dependent values to the algorithm
    algorithm.input.set(classifier.training.data,   trainData)
    algorithm.input.set(classifier.training.labels, trainGroundTruthSource.getNumericTable())

    # Build the Naive Bayes model and retrieve the algorithm results
    trainingResult = algorithm.compute()


def testModel():
    global predictionResult

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file
    testData = createSparseTable(testDatasetFileName)

    # Create an algorithm object to predict Naive Bayes values
    algorithm = prediction.Batch(nClasses, method=prediction.fastCSR)

    # Pass a testing data set and the trained model to the algorithm
    algorithm.input.setTable(classifier.prediction.data,  testData)
    algorithm.input.setModel(classifier.prediction.model, trainingResult.get(classifier.training.model))

    # Predict Naive Bayes values and retrieve the algorithm results (Result class from classifier.prediction)
    predictionResult = algorithm.compute()


def printResults():

    testGroundTruth = FileDataSource(
        testGroundTruthFileName,
        DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    testGroundTruth.loadDataBlock(nTestObservations)

    printNumericTables(
        testGroundTruth.getNumericTable(),
        predictionResult.get(classifier.prediction.prediction),
        "Ground truth", "Classification results",
        "NaiveBayes classification results (first 20 observations):", 20, 15, flt64=False
    )

if __name__ == "__main__":

    trainModel()
    testModel()
    printResults()
