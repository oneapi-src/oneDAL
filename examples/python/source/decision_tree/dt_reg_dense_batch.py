# file: dt_reg_dense_batch.py
#===============================================================================
# Copyright 2014-2019 Intel Corporation
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

## <a name="DAAL-EXAMPLE-PY-DT_REG_DENSE_BATCH"></a>
## \example dt_reg_dense_batch.py

import os
import sys

from daal.algorithms.decision_tree.regression import prediction, training
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

# Model object for the decision tree regression algorithm
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

    # Create an algorithm object to train the decision tree regression model
    algorithm = training.Batch()

    # Pass the training data set and dependent values to the algorithm
    algorithm.input.set(training.data, trainData)
    algorithm.input.set(training.dependentVariables, trainGroundTruth)
    algorithm.input.set(training.dataForPruning, pruneData)
    algorithm.input.set(training.dependentVariablesForPruning, pruneGroundTruth)

    # Train the decision tree regression model and retrieve the results of the training algorithm
    trainingResult = algorithm.compute()
    model = trainingResult.get(training.model)

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

    # Create algorithm objects for decision tree regression prediction with the default method
    algorithm = prediction.Batch()

    # Pass the testing data set and trained model to the algorithm
    #print("Number of columns: {}".format(testData.getNumberOfColumns()))
    algorithm.input.setTable(prediction.data, testData)
    algorithm.input.setModel(prediction.model, model)

    # Compute prediction results and retrieve algorithm results
    predictionResult = algorithm.compute()


def printResults():

    printNumericTables(testGroundTruth, predictionResult.get(prediction.prediction),
                       "Ground truth", "Regression results",
                       "Decision tree regression results (first 20 observations):",
                       20, flt64=False)

if __name__ == "__main__":

    trainModel()
    testModel()
    printResults()
