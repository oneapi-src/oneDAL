# file: svm_two_class_csr_batch.py
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

## <a name="DAAL-EXAMPLE-PY-SVM_TWO_CLASS_CSR_BATCH"></a>
## \example svm_two_class_csr_batch.py

import os
import sys

from daal.algorithms.svm import training, prediction
from daal.algorithms import kernel_function, classifier
from daal.data_management import DataSourceIface, FileDataSource

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTables, createSparseTable

# Input data set parameters
DATA_PREFIX = os.path.join('..', 'data', 'batch')

trainDatasetFileName = os.path.join(DATA_PREFIX, 'svm_two_class_train_csr.csv')
trainLabelsFileName = os.path.join(DATA_PREFIX, 'svm_two_class_train_labels.csv')
testDatasetFileName = os.path.join(DATA_PREFIX, 'svm_two_class_test_csr.csv')
testLabelsFileName = os.path.join(DATA_PREFIX, 'svm_two_class_test_labels.csv')

# Parameters for the SVM kernel function
kernel = kernel_function.linear.Batch(method=kernel_function.linear.fastCSR)

# Model object for the SVM algorithm
trainingResult = None
predictionResult = None


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

    # Create an algorithm object to train the SVM model
    algorithm = training.Batch()

    algorithm.parameter.kernel = kernel
    algorithm.parameter.cacheSize = 40000000

    # Pass a training data set and dependent values to the algorithm
    algorithm.input.set(classifier.training.data, trainData)
    algorithm.input.set(classifier.training.labels, trainLabelsDataSource.getNumericTable())

    # Build the SVM model
    trainingResult = algorithm.compute()


def testModel():
    global predictionResult

    # Create Numeric Tables for testing data
    testData = createSparseTable(testDatasetFileName)

    # Create an algorithm object to predict SVM values
    algorithm = prediction.Batch()

    algorithm.parameter.kernel = kernel

    # Pass a testing data set and the trained model to the algorithm
    algorithm.input.setTable(classifier.prediction.data, testData)

    algorithm.input.setModel(classifier.prediction.model, trainingResult.get(classifier.training.model))

    # Predict SVM values
    algorithm.compute()

    # Retrieve the algorithm results
    predictionResult = algorithm.getResult()


def printResults():

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the test data from a .csv file
    testLabelsDataSource = FileDataSource(
        testLabelsFileName, DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )
    # Retrieve the data from input file
    testLabelsDataSource.loadDataBlock()
    testGroundTruth = testLabelsDataSource.getNumericTable()

    printNumericTables(
        testGroundTruth, predictionResult.get(classifier.prediction.prediction),
        "Ground truth\t", "Classification results",
        "SVM classification results (first 20 observations):", 20, flt64=False
    )

if __name__ == "__main__":

    trainModel()
    testModel()
    printResults()
