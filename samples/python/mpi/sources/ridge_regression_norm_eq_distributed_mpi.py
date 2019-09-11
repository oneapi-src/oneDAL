# file: ridge_regression_norm_eq_distributed_mpi.py
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
# !  Content:
# !    Python sample of ridge regression in the distributed processing mode.
# !
# !    The program trains the ridge regression model on a training data set with the normal equations method and computes regression for the test data.
# !*****************************************************************************

#
## <a name = "DAAL-SAMPLE-PY-RIDGE_REGRESSION_NORM_EQ_DISTRIBUTED"></a>)
## \example ridge_regression_norm_eq_distributed_mpi.py
#

import os
import sys
from os.path import join as jp

from mpi4py import MPI

from daal import step1Local, step2Master
from daal.algorithms.ridge_regression import training, prediction
from daal.data_management import (
    DataSourceIface, FileDataSource, HomogenNumericTable, NumericTableIface,
    MergedNumericTable, InputDataArchive, OutputDataArchive
)

utils_folder = os.path.realpath(os.path.abspath(jp(os.environ['DAALROOT'], 'examples', 'python', 'source')))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

data_root = jp('data', 'distributed')

trainDatasetFileNames = [
    jp(data_root, "linear_regression_train_1.csv"),
    jp(data_root, "linear_regression_train_2.csv"),
    jp(data_root, "linear_regression_train_3.csv"),
    jp(data_root, "linear_regression_train_4.csv")

]

testDatasetFileName = jp(data_root, "linear_regression_test.csv")

NUM_BLOCKS = 4
NUM_FEATURES = 10       # Number of features in training and testing data sets
NUM_DEPENDENT_VARS = 2  # Number of dependent variables that correspond to each observation

MPI_ROOT = 0


def trainModel(comm, rankId):

    trainingResult = None

    # Initialize FileDataSource to retrieve the input data from a .csv file
    trainDataSource = FileDataSource(
        trainDatasetFileNames[rankId],
        DataSourceIface.notAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Create Numeric Tables for training data and labels
    trainData = HomogenNumericTable(NUM_FEATURES, 0, NumericTableIface.doNotAllocate)
    trainDependentVariables = HomogenNumericTable(NUM_DEPENDENT_VARS, 0, NumericTableIface.doNotAllocate)
    mergedData = MergedNumericTable(trainData, trainDependentVariables)

    # Retrieve the data from the input file
    trainDataSource.loadDataBlock(mergedData)

    # Create an algorithm object to train the ridge regression model based on the local-node data
    localAlgorithm = training.Distributed(step1Local)

    # Pass a training data set and dependent values to the algorithm
    localAlgorithm.input.set(training.data, trainData)
    localAlgorithm.input.set(training.dependentVariables, trainDependentVariables)

    # Train the ridge regression model on local nodes
    pres = localAlgorithm.compute()

    # Serialize partial results required by step 2
    dataArch = InputDataArchive()
    pres.serialize(dataArch)

    # Transfer partial results to step 2 on the root node
    nodeResults = dataArch.getArchiveAsArray()

    serializedData = comm.gather(nodeResults)

    if rankId == MPI_ROOT:

        # Create an algorithm object to build the final ridge regression model on the master node
        masterAlgorithm = training.Distributed(step2Master)

        for i in range(NUM_BLOCKS):

            # Deserialize partial results from step 1
            dataArch = OutputDataArchive(serializedData[i])
            dataForStep2FromStep1 = training.PartialResult()
            dataForStep2FromStep1.deserialize(dataArch)

            # Set the local ridge regression model as input for the master-node algorithm
            masterAlgorithm.input.add(training.partialModels, dataForStep2FromStep1)

        # Merge and finalizeCompute the ridge regression model on the master node
        masterAlgorithm.compute()
        trainingResult = masterAlgorithm.finalizeCompute()

        # Retrieve the algorithm results
        printNumericTable(trainingResult.get(training.model).getBeta(), "Ridge Regression coefficients:")

    return trainingResult


def testModel(trainingResult):

    # Initialize FileDataSource to retrieve the input data from a .csv file
    testDataSource = FileDataSource(
        testDatasetFileName,
        DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Create Numeric Tables for testing data and ground truth values
    testData = HomogenNumericTable(NUM_FEATURES, 0, NumericTableIface.doNotAllocate)
    testGroundTruth = HomogenNumericTable(NUM_DEPENDENT_VARS, 0, NumericTableIface.doNotAllocate)
    mergedData = MergedNumericTable(testData, testGroundTruth)

    # Retrieve the data from an input file
    testDataSource.loadDataBlock(mergedData)

    # Create an algorithm object to predict values of ridge regression
    algorithm = prediction.Batch()

    # Pass a testing data set and the trained model to the algorithm
    algorithm.input.setTable(prediction.data, testData)
    algorithm.input.setModel(prediction.model, trainingResult.get(training.model))

    # Predict values of ridge regression
    res = algorithm.compute()

    # Retrieve the algorithm results
    printNumericTable(res.get(prediction.prediction), "Ridge Regression prediction results: (first 10 rows):", 10)
    printNumericTable(testGroundTruth, "Ground truth (first 10 rows):", 10)


def main():
    comm = MPI.COMM_WORLD
    rankId = comm.Get_rank()

    trainingResult = trainModel(comm, rankId)

    if rankId == MPI_ROOT:
        testModel(trainingResult)

if __name__ == "__main__":
    sys.exit(main())
