# file: ridge_reg_norm_eq_dense_batch.py
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

#
# !  Content:
# !    Python example of ridge regression in the batch processing mode.
# !
# !    The program trains the ridge regression model on a training
# !    datasetFileName with the normal equations method and computes regression
# !    for the test data.
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-RIDGE_REGRESSION_NORM_EQ_BATCH"></a>
## \example ridge_reg_norm_eq_dense_batch.py
#

import os
import sys

from daal.algorithms.ridge_regression import training, prediction
from daal.data_management import DataSource, FileDataSource, NumericTable, HomogenNumericTable, MergedNumericTable

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

# Input data set parameters
trainDatasetFileName = os.path.join("..", "data", "batch", "linear_regression_train.csv")
testDatasetFileName = os.path.join("..", "data", "batch", "linear_regression_test.csv")

nFeatures           = 10    # Number of features in training and testing data sets
nDependentVariables = 2     # Number of dependent variables that correspond to each observation


def trainModel():
    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    trainDataSource = FileDataSource(trainDatasetFileName,
                                     DataSource.notAllocateNumericTable,
                                     DataSource.doDictionaryFromContext)

    # Create Numeric Tables for training data and dependent variables
    trainData =  HomogenNumericTable(nFeatures, 0, NumericTable.doNotAllocate)
    trainDependentVariables = HomogenNumericTable(nDependentVariables, 0, NumericTable.doNotAllocate)
    mergedData = MergedNumericTable(trainData, trainDependentVariables)

    # Retrieve the data from input file
    trainDataSource.loadDataBlock(mergedData)

    # Create an algorithm object to train the ridge regression model with the normal equations method
    algorithm = training.Batch()

    # Pass a training data set and dependent values to the algorithm
    algorithm.input.set(training.data, trainData)
    algorithm.input.set(training.dependentVariables, trainDependentVariables)

    # Build the ridge regression model and etrieve the algorithm results
    trainingResult = algorithm.compute()

    printNumericTable(trainingResult.get(training.model).getBeta(), "Ridge Regression coefficients:")
    return trainingResult


def testModel(trainingResult):
    # Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file
    testDataSource = FileDataSource(testDatasetFileName,
                                    DataSource.doAllocateNumericTable,
                                    DataSource.doDictionaryFromContext)

    # Create Numeric Tables for testing data and ground truth values
    testData = HomogenNumericTable(nFeatures, 0, NumericTable.doNotAllocate)
    testGroundTruth = HomogenNumericTable(nDependentVariables, 0, NumericTable.doNotAllocate)
    mergedData = MergedNumericTable(testData, testGroundTruth)

    # Load the data from the data file
    testDataSource.loadDataBlock(mergedData)

    # Create an algorithm object to predict values of ridge regression
    algorithm = prediction.Batch()

    # Pass a testing data set and the trained model to the algorithm
    algorithm.input.setTable(prediction.data, testData)
    algorithm.input.setModel(prediction.model, trainingResult.get(training.model))

    # Predict values of ridge regression and retrieve the algorithm results
    predictionResult = algorithm.compute()

    printNumericTable(predictionResult.get(prediction.prediction),
                      "Ridge Regression prediction results: (first 10 rows):", 10)
    printNumericTable(testGroundTruth, "Ground truth (first 10 rows):", 10)


if __name__ == "__main__":
    trainingResult = trainModel()
    testModel(trainingResult)
