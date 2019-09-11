# file: svm_multi_class_csr_batch.py
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

## <a name="DAAL-EXAMPLE-PY-SVM_MULTI_CLASS_CSR_BATCH"></a>
## \example svm_multi_class_csr_batch.py

import os
import sys

from daal.algorithms.svm import training, prediction
from daal.algorithms import classifier, kernel_function, multi_class_classifier
from daal.data_management import DataSourceIface, FileDataSource

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTables, createSparseTable

# Input data set parameters
data_dir = os.path.join('..', 'data', 'batch')
trainDatasetFileName = os.path.join(data_dir, 'svm_multi_class_train_csr.csv')
trainLabelsFileName = os.path.join(data_dir, 'svm_multi_class_train_labels.csv')
testDatasetFileName = os.path.join(data_dir, 'svm_multi_class_test_csr.csv')
testLabelsFileName = os.path.join(data_dir, 'svm_multi_class_test_labels.csv')

nClasses = 5

trainingAlg = training.Batch()
predictionAlg = prediction.Batch()

# Parameters for the SVM kernel function
kernel = kernel_function.linear.Batch(method=kernel_function.linear.fastCSR)

trainingResult = None
predictionResult = None
testGroundTruth = None


def trainModel():
    global trainingResult

    # Initialize FileDataSource to retrieve the input data from a .csv file
    trainLabelsDataSource = FileDataSource(
        trainLabelsFileName, DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Create numeric table for training data
    trainData = createSparseTable(trainDatasetFileName)

    # Retrieve the data from the input file
    trainLabelsDataSource.loadDataBlock()

    # Create an algorithm object to train the multi-class SVM model
    algorithm = multi_class_classifier.training.Batch(nClasses)

    algorithm.parameter.training = trainingAlg
    algorithm.parameter.prediction = predictionAlg

    # Pass a training data set and dependent values to the algorithm
    algorithm.input.set(classifier.training.data, trainData)
    algorithm.input.set(classifier.training.labels, trainLabelsDataSource.getNumericTable())

    # Build the multi-class SVM model and retrieve the algorithm results
    # (Result class from multi_class_classifier.training)
    trainingResult = algorithm.compute()


def testModel():
    global predictionResult

    # Create Numeric Tables for testing data
    testData = createSparseTable(testDatasetFileName)

    # Create an algorithm object to predict multi-class SVM values
    algorithm = multi_class_classifier.prediction.Batch(nClasses)

    algorithm.parameter.training = trainingAlg
    algorithm.parameter.prediction = predictionAlg

    # Pass a testing data set and the trained model to the algorithm
    algorithm.input.setTable(classifier.prediction.data, testData)
    algorithm.input.setModel(classifier.prediction.model, trainingResult.get(classifier.training.model))

    # Predict multi-class SVM values and retrieve the algorithm results
    # (Result class from classifier.prediction)
    predictionResult = algorithm.compute()


def printResults():

    # Initialize FileDataSource to retrieve the test data from a .csv file
    testLabelsDataSource = FileDataSource(
        testLabelsFileName, DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )
    # Retrieve the data from input file
    testLabelsDataSource.loadDataBlock()
    testGroundTruth = testLabelsDataSource.getNumericTable()

    printNumericTables(
        testGroundTruth, predictionResult.get(classifier.prediction.prediction),
        "Ground truth", "Classification results",
        "Multi-class SVM classification sample program results (first 20 observations):",
        20, flt64=False
    )

if __name__ == "__main__":
    trainingAlg.parameter.cacheSize = 100000000
    trainingAlg.parameter.kernel = kernel
    predictionAlg.parameter.kernel = kernel

    trainModel()
    testModel()
    printResults()
