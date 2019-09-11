# file: mn_naive_bayes_csr_online.py
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

## <a name="DAAL-EXAMPLE-PY-MULTINOMIAL_NAIVE_BAYES_CSR_ONLINE"></a>
## \example mn_naive_bayes_csr_online.py

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
trainDatasetFileNames = [
    os.path.join(DAAL_PREFIX, 'batch', 'naivebayes_train_csr.csv'),
    os.path.join(DAAL_PREFIX, 'batch', 'naivebayes_train_csr.csv'),
    os.path.join(DAAL_PREFIX, 'batch', 'naivebayes_train_csr.csv'),
    os.path.join(DAAL_PREFIX, 'batch', 'naivebayes_train_csr.csv')
]

trainGroundTruthFileNames = [
    os.path.join(DAAL_PREFIX, 'batch', 'naivebayes_train_labels.csv'),
    os.path.join(DAAL_PREFIX, 'batch', 'naivebayes_train_labels.csv'),
    os.path.join(DAAL_PREFIX, 'batch', 'naivebayes_train_labels.csv'),
    os.path.join(DAAL_PREFIX, 'batch', 'naivebayes_train_labels.csv')
]

testDatasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'naivebayes_test_csr.csv')
testGroundTruthFileName = os.path.join(DAAL_PREFIX, 'batch', 'naivebayes_test_labels.csv')

nTrainVectorsInBlock = 8000
nTestObservations = 2000
nClasses = 20
nBlocks = 4

trainingResult = None
predictionResult = None
trainData = [0] * nBlocks
testData = None


def trainModel():
    global trainData, trainingResult

    # Create an algorithm object to train the Naive Bayes model
    algorithm = training.Online(nClasses, method=training.fastCSR)

    for i in range(nBlocks):
        # Read trainDatasetFileNames and create a numeric table to store the input data
        trainData[i] = createSparseTable(trainDatasetFileNames[i])
        trainLabelsSource = FileDataSource(
            trainGroundTruthFileNames[i], DataSourceIface.doAllocateNumericTable,
            DataSourceIface.doDictionaryFromContext
        )

        trainLabelsSource.loadDataBlock(nTrainVectorsInBlock)

        # Pass a training data set and dependent values to the algorithm
        algorithm.input.set(classifier.training.data,   trainData[i])
        algorithm.input.set(classifier.training.labels, trainLabelsSource.getNumericTable())

        # Build the Naive Bayes model
        algorithm.compute()

    # Finalize the Naive Bayes model and retrieve the algorithm results
    trainingResult = algorithm.finalizeCompute()


def testModel():
    global predictionResult, testData

    # Read testDatasetFileName and create a numeric table to store the input data
    testData = createSparseTable(testDatasetFileName)

    # Create an algorithm object to predict Naive Bayes values
    algorithm = prediction.Batch(nClasses, method=prediction.fastCSR)

    # Pass a testing data set and the trained model to the algorithm
    algorithm.input.setTable(classifier.prediction.data,  testData)
    algorithm.input.setModel(classifier.prediction.model, trainingResult.get(classifier.training.model))

    # Predict Naive Bayes values (Result class from classifier.prediction)
    predictionResult = algorithm.compute()  # Retrieve the algorithm results


def printResults():

    testGroundTruth = FileDataSource(
        testGroundTruthFileName, DataSourceIface.doAllocateNumericTable,
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
