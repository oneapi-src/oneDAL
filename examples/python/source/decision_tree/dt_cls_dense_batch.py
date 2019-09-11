# file: dt_cls_dense_batch.py
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

## <a name="DAAL-EXAMPLE-PY-DT_CLS_DENSE_BATCH"></a>
## \example dt_cls_dense_batch.py

import os
import sys

from daal.algorithms.decision_tree.classification import prediction, training
from daal.algorithms import classifier
from daal.data_management import (
    FileDataSource, DataSourceIface, NumericTableIface, HomogenNumericTable, MergedNumericTable
)
utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTables

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
trainDatasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'decision_tree_train.csv')
pruneDatasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'decision_tree_prune.csv')
testDatasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'decision_tree_test.csv')

nFeatures = 5
nClasses = 5

# Model object for the decision tree classification algorithm
model = None
predictionResult = None
testGroundTruth = None


def trainModel():
    global model

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    trainDataSource = FileDataSource(
        trainDatasetFileName,
        DataSourceIface.notAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Create Numeric Tables for training data and labels
    trainData = HomogenNumericTable(nFeatures, 0, NumericTableIface.notAllocate)
    trainGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.notAllocate)
    mergedData = MergedNumericTable(trainData, trainGroundTruth)

    # Retrieve the data from the input file
    trainDataSource.loadDataBlock(mergedData)

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    pruneDataSource = FileDataSource(
        pruneDatasetFileName,
        DataSourceIface.notAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Create Numeric Tables for pruning data and labels
    pruneData = HomogenNumericTable(nFeatures, 0, NumericTableIface.notAllocate)
    pruneGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.notAllocate)
    pruneMergedData = MergedNumericTable(pruneData, pruneGroundTruth)

    # Retrieve the data from the input file
    pruneDataSource.loadDataBlock(pruneMergedData)

    # Create an algorithm object to train the decision tree classification model
    algorithm = training.Batch(nClasses)

    # Pass the training data set and dependent values to the algorithm
    algorithm.input.set(classifier.training.data, trainData)
    algorithm.input.set(classifier.training.labels, trainGroundTruth)
    algorithm.input.setTable(training.dataForPruning, pruneData)
    algorithm.input.setTable(training.labelsForPruning, pruneGroundTruth)

    # Train the decision tree classification model and retrieve the results of the training algorithm
    trainingResult = algorithm.compute()
    model = trainingResult.get(classifier.training.model)

def testModel():
    global testGroundTruth, predictionResult

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file
    testDataSource = FileDataSource(
        testDatasetFileName,
        DataSourceIface.notAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Create Numeric Tables for testing data and labels
    testData = HomogenNumericTable(nFeatures, 0, NumericTableIface.notAllocate)
    testGroundTruth = HomogenNumericTable(1, 0, NumericTableIface.notAllocate)
    mergedData = MergedNumericTable(testData, testGroundTruth)

    # Retrieve the data from input file
    testDataSource.loadDataBlock(mergedData)

    # Create algorithm objects for decision tree classification prediction with the default method
    algorithm = prediction.Batch()

    # Pass the testing data set and trained model to the algorithm
    #print("Number of columns: {}".format(testData.getNumberOfColumns()))
    algorithm.input.setTable(classifier.prediction.data,  testData)
    algorithm.input.setModel(classifier.prediction.model, model)

    # Compute prediction results and retrieve algorithm results
    # (Result class from classifier.prediction)
    predictionResult = algorithm.compute()


def printResults():

    printNumericTables(
        testGroundTruth,
        predictionResult.get(classifier.prediction.prediction),
        "Ground truth", "Classification results",
        "Decision tree classification results (first 20 observations):",
        20, flt64=False
    )

if __name__ == "__main__":

    trainModel()
    testModel()
    printResults()
