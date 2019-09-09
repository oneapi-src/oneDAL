# file: lin_reg_qr_dense_online.py
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

## <a name="DAAL-EXAMPLE-PY-LINEAR_REGRESSION_QR_ONLINE"></a>
## \example lin_reg_qr_dense_online.py

import os
import sys

from daal.algorithms.linear_regression import training, prediction
from daal.data_management import (
    DataSourceIface, FileDataSource, HomogenNumericTable, MergedNumericTable, NumericTableIface
)

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
trainDatasetFileName = os.path.join(DAAL_PREFIX, 'online', 'linear_regression_train.csv')
testDatasetFileName = os.path.join(DAAL_PREFIX, 'online', 'linear_regression_test.csv')

nTrainVectorsInBlock = 250

nFeatures           = 10    # Number of features in training and testing data sets
nDependentVariables = 2     # Number of dependent variables that correspond to each observation

trainingResult = None
predictionResult = None


def trainModel():
    global trainingResult

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    trainDataSource = FileDataSource(
        trainDatasetFileName, DataSourceIface.notAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Create Numeric Tables for training data and dependent variables
    trainData = HomogenNumericTable(nFeatures, 0, NumericTableIface.doNotAllocate)
    trainDependentVariables = HomogenNumericTable(
        nDependentVariables, 0, NumericTableIface.doNotAllocate
    )
    mergedData = MergedNumericTable(trainData, trainDependentVariables)

    # Create an algorithm object to train the multiple linear regression model
    algorithm = training.Online(method=training.qrDense)

    while(trainDataSource.loadDataBlock(nTrainVectorsInBlock, mergedData) == nTrainVectorsInBlock):
        # Pass a training data set and dependent values to the algorithm
        algorithm.input.set(training.data, trainData)
        algorithm.input.set(training.dependentVariables, trainDependentVariables)

        # Update the multiple linear regression model
        algorithm.compute()

    # Finalize the multiple linear regression model and retrieve the algorithm results
    trainingResult = algorithm.finalizeCompute()
    printNumericTable(trainingResult.get(training.model).getBeta(), "Linear Regression coefficients:")


def testModel():
    global trainingResult, predictionResult

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    testDataSource = FileDataSource(
        testDatasetFileName, DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Create Numeric Tables for testing data and ground truth values
    testData = HomogenNumericTable(nFeatures, 0, NumericTableIface.doNotAllocate)
    testGroundTruth = HomogenNumericTable(nDependentVariables, 0, NumericTableIface.doNotAllocate)
    mergedData = MergedNumericTable(testData, testGroundTruth)

    # Retrieve the data from the input file
    testDataSource.loadDataBlock(mergedData)

    # Create an algorithm object to predict values of multiple linear regression
    algorithm = prediction.Batch()

    # Pass a testing data set and the trained model to the algorithm
    algorithm.input.setModel(prediction.model, trainingResult.get(training.model))
    algorithm.input.setTable(prediction.data, testData)

    # Predict values of multiple linear regression and retrieve the algorithm results
    predictionResult = algorithm.compute()
    printNumericTable(predictionResult.get(prediction.prediction), "Linear Regression prediction results: (first 10 rows):", 10)
    printNumericTable(testGroundTruth, "Ground truth (first 10 rows):", 10)

if __name__ == "__main__":

    trainModel()
    testModel()
