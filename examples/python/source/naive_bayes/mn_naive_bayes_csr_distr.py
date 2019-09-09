# file: mn_naive_bayes_csr_distr.py
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

## <a name="DAAL-EXAMPLE-PY-MULTINOMIAL_NAIVE_BAYES_CSR_DISTRIBUTED"></a>
## \example mn_naive_bayes_csr_distr.py

import os
import sys

from daal import step1Local, step2Master
from daal.algorithms import classifier
from daal.algorithms.multinomial_naive_bayes import training, prediction
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

nClasses = 20
nBlocks = 4
nTrainVectorsInBlock = 8000
nTestObservations = 2000

trainingResult = None
predictionResult = None
trainData = [0] * nBlocks
testData = None


def trainModel():
    global trainData, trainingResult

    masterAlgorithm = training.Distributed(step2Master, nClasses, method=training.fastCSR)

    for i in range(nBlocks):
        # Read trainDatasetFileNames and create a numeric table to store the input data
        trainData[i] = createSparseTable(trainDatasetFileNames[i])

        # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
        trainLabelsSource = FileDataSource(
            trainGroundTruthFileNames[i], DataSourceIface.doAllocateNumericTable,
            DataSourceIface.doDictionaryFromContext
        )

        # Retrieve the data from an input file
        trainLabelsSource.loadDataBlock(nTrainVectorsInBlock)

        # Create an algorithm object to train the Naive Bayes model on the local-node data
        localAlgorithm = training.Distributed(step1Local, nClasses, method=training.fastCSR)

        # Pass a training data set and dependent values to the algorithm
        localAlgorithm.input.set(classifier.training.data,   trainData[i])
        localAlgorithm.input.set(classifier.training.labels, trainLabelsSource.getNumericTable())

        # Build the Naive Bayes model on the local node
        # Set the local Naive Bayes model as input for the master-node algorithm
        masterAlgorithm.input.add(training.partialModels, localAlgorithm.compute())

    # Merge and finalize the Naive Bayes model on the master node
    masterAlgorithm.compute()
    trainingResult = masterAlgorithm.finalizeCompute()  # Retrieve the algorithm results


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
