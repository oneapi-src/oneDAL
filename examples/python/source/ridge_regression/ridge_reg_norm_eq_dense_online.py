# file: ridge_reg_norm_eq_dense_online.py
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

#
# !  Content:
# !    Python example of ridge regression in the online processing mode.
# !
# !    The program trains the ridge regression model on a training datasetFileName
# !    with the normal equations method and computes regression for the test data.
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-RIDGE_REGRESSION_NORM_EQ_ONLINE"></a>
## \example ridge_reg_norm_eq_dense_online.py
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

nTrainVectorsInBlock = 250
nFeatures           = 10    # Number of features in training and testing data sets
nDependentVariables = 2     # Number of dependent variables that correspond to each observation


def trainModel():
    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    trainDataSource = FileDataSource(trainDatasetFileName,
                                     DataSource.notAllocateNumericTable,
                                     DataSource.doDictionaryFromContext)

    # Create Numeric Tables for training data and dependent variables
    trainData = HomogenNumericTable(nFeatures, 0, NumericTable.doNotAllocate)
    trainDependentVariables = HomogenNumericTable(nDependentVariables, 0, NumericTable.doNotAllocate)
    mergedData = MergedNumericTable(trainData, trainDependentVariables)

    # Create an algorithm object to train the ridge regression model
    algorithm = training.Online()

    while trainDataSource.loadDataBlock(nTrainVectorsInBlock, mergedData) == nTrainVectorsInBlock:
        # Pass a training data set and dependent values to the algorithm
        algorithm.input.set(training.data, trainData)
        algorithm.input.set(training.dependentVariables, trainDependentVariables)

        # Update the ridge regression model
        algorithm.compute()


    # Finalize the ridge regression model and retrieve the algorithm results
    trainingResult = algorithm.finalizeCompute()

    printNumericTable(trainingResult.get(training.model).getBeta(), "Ridge Regression coefficients:")
    return trainingResult


def testModel(trainingResult):
    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    testDataSource = FileDataSource(testDatasetFileName,
                                    DataSource.doAllocateNumericTable,
                                    DataSource.doDictionaryFromContext)

    # Create Numeric Tables for testing data and ground truth values
    testData = HomogenNumericTable(nFeatures, 0, NumericTable.doNotAllocate)
    testGroundTruth = HomogenNumericTable(nDependentVariables, 0, NumericTable.doNotAllocate)
    mergedData = MergedNumericTable(testData, testGroundTruth)

    # Retrieve the data from the input file
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
