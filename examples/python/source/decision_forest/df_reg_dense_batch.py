# file: df_reg_dense_batch.py
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

## <a name="DAAL-EXAMPLE-PY-DF_REG_DENSE_BATCH"></a>
## \example df_reg_dense_batch.py

import os
import sys

from daal.algorithms import decision_forest
from daal.algorithms.decision_forest.regression import prediction, training
from daal.data_management import (
    FileDataSource, DataSourceIface, NumericTableIface,
    HomogenNumericTable, MergedNumericTable, features
)

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
trainDatasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'df_regression_train.csv')
testDatasetFileName = os.path.join(DAAL_PREFIX, 'batch', 'df_regression_test.csv')

nFeatures = 13

# Decision forest parameters
nTrees = 100

# Model object for the decision forest regression algorithm
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

    #  Get the dictionary and update it with additional information about data
    dict = trainData.getDictionary()

    #  Add a feature type to the dictionary
    dict[3].featureType = features.DAAL_CATEGORICAL

    # Create an algorithm object to train the decision forest regression model
    algorithm = training.Batch()
    algorithm.parameter.nTrees = nTrees
    algorithm.parameter.varImportance = decision_forest.training.MDA_Raw
    algorithm.parameter.resultsToCompute = decision_forest.training.computeOutOfBagError|decision_forest.training.computeOutOfBagErrorPerObservation;

    # Pass the training data set and dependent values to the algorithm
    algorithm.input.set(training.data, trainData)
    algorithm.input.set(training.dependentVariable, trainGroundTruth)

    # Train the decision forest regression model and retrieve the results of the training algorithm
    trainingResult = algorithm.compute()
    model = trainingResult.get(training.model)
    printNumericTable(trainingResult.getTable(training.variableImportance), "Variable importance results: ")
    printNumericTable(trainingResult.getTable(training.outOfBagError), "OOB error: ")
    printNumericTable(trainingResult.getTable(training.outOfBagError), "OOB error (first 10 rows): ", 10)

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

    #  Get the dictionary and update it with additional information about data
    dict = testData.getDictionary()

    #  Add a feature type to the dictionary
    dict[3].featureType = features.DAAL_CATEGORICAL

    # Create algorithm objects for decision forest regression prediction with the default method
    algorithm = prediction.Batch()

    # Pass the testing data set and trained model to the algorithm
    algorithm.input.setTable(prediction.data,  testData)
    algorithm.input.set(prediction.model, model)

    # Compute prediction results and retrieve algorithm results
    predictionResult = algorithm.compute()


def printResults():

    printNumericTable(
        predictionResult.get(prediction.prediction),
        "Decision forest prediction results (first 10 rows):", 10
    )
    printNumericTable(
        testGroundTruth,
        "Ground truth (first 10 rows):", 10
    )

if __name__ == "__main__":

    trainModel()
    testModel()
    printResults()
