# file: multinomial_naive_bayes_csr_distributed_mpi.py
#===============================================================================
# Copyright 2017-2019 Intel Corporation
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
# !    Python sample of Naive Bayes classification in the distributed processing
# !    mode.
# !
# !    The program trains the Naive Bayes model on a supplied training data set
# !    and then performs classification of previously unseen data.
# !*****************************************************************************

#
## <a name="DAAL-SAMPLE-PY-MULTINOMIAL_NAIVE_BAYES_CSR_DISTRIBUTED"></a>
## \example multinomial_naive_bayes_csr_distributed_mpi.py
#

import os
import sys
from os.path import join as jp

from mpi4py import MPI

from daal import step1Local, step2Master
from daal.algorithms import classifier
from daal.algorithms.multinomial_naive_bayes import training, prediction
from daal.data_management import DataSourceIface, FileDataSource, OutputDataArchive, InputDataArchive

utils_folder = os.path.realpath(os.path.abspath(jp(os.environ['DAALROOT'], 'examples', 'python', 'source')))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTables, createSparseTable

DATA_PREFIX = jp('data', 'distributed')

# Input data set parameters
trainDatasetFileNames = [
    jp(DATA_PREFIX, 'naivebayes_train_csr.csv'),
    jp(DATA_PREFIX, 'naivebayes_train_csr.csv'),
    jp(DATA_PREFIX, 'naivebayes_train_csr.csv'),
    jp(DATA_PREFIX, 'naivebayes_train_csr.csv')
]
trainGroundTruthFileNames = [
    jp(DATA_PREFIX, 'naivebayes_train_labels.csv'),
    jp(DATA_PREFIX, 'naivebayes_train_labels.csv'),
    jp(DATA_PREFIX, 'naivebayes_train_labels.csv'),
    jp(DATA_PREFIX, 'naivebayes_train_labels.csv')
]

testDatasetFileName = jp(DATA_PREFIX, 'naivebayes_test_csr.csv')
testGroundTruthFileName = jp(DATA_PREFIX, 'naivebayes_test_labels.csv')

nClasses = 20
nBlocks = 4

MPI_ROOT = 0

trainingResult = None
predictionResult = None


def trainModel():
    global trainingResult

    # Retrieve the input data from a .csv file
    trainDataTable = createSparseTable(trainDatasetFileNames[rankId])

    # Initialize FileDataSource to retrieve the input data from a .csv file
    trainLabelsSource = FileDataSource(trainGroundTruthFileNames[rankId],
                                       DataSourceIface.doAllocateNumericTable,
                                       DataSourceIface.doDictionaryFromContext)

    # Retrieve the data from input files
    trainLabelsSource.loadDataBlock()

    # Create an algorithm object to train the Naive Bayes model based on the local-node data
    localAlgorithm = training.Distributed(step1Local, nClasses, method=training.fastCSR)

    # Pass a training data set and dependent values to the algorithm
    localAlgorithm.input.set(classifier.training.data, trainDataTable)
    localAlgorithm.input.set(classifier.training.labels, trainLabelsSource.getNumericTable())

    # Train the Naive Bayes model on local nodes
    pres = localAlgorithm.compute()

    # Serialize partial results required by step 2
    dataArch = InputDataArchive()
    pres.serialize(dataArch)

    nodeResults = dataArch.getArchiveAsArray()

    # Transfer partial results to step 2 on the root node
    serializedData = comm.gather(nodeResults)

    if rankId == MPI_ROOT:
        # Create an algorithm object to build the final Naive Bayes model on the master node
        masterAlgorithm = training.Distributed(step2Master, nClasses, method=training.fastCSR)

        for i in range(nBlocks):
            # Deserialize partial results from step 1
            dataArch = OutputDataArchive(serializedData[i])

            dataForStep2FromStep1 = training.PartialResult()
            dataForStep2FromStep1.deserialize(dataArch)

            # Set the local Naive Bayes model as input for the master-node algorithm
            masterAlgorithm.input.add(training.partialModels, dataForStep2FromStep1)

        # Merge and finalizeCompute the Naive Bayes model on the master node
        masterAlgorithm.compute()
        trainingResult = masterAlgorithm.finalizeCompute()


def testModel():
    global predictionResult

    # Retrieve the input data from a .csv file
    testDataTable = createSparseTable(testDatasetFileName)

    # Create an algorithm object to predict values of the Naive Bayes model
    algorithm = prediction.Batch(nClasses, method=prediction.fastCSR)

    # Pass a testing data set and the trained model to the algorithm
    algorithm.input.setTable(classifier.prediction.data, testDataTable)
    algorithm.input.setModel(classifier.prediction.model, trainingResult.get(classifier.training.model))

    # Predict values of the Naive Bayes model
    # Result class from classifier.prediction
    predictionResult = algorithm.compute()


def printResults():

    testGroundTruth = FileDataSource(testGroundTruthFileName,
                                     DataSourceIface.doAllocateNumericTable,
                                     DataSourceIface.doDictionaryFromContext)
    testGroundTruth.loadDataBlock()

    printNumericTables(testGroundTruth.getNumericTable(),
                       predictionResult.get(classifier.prediction.prediction),
                       "Ground truth",
                       "Classification results",
                       "NaiveBayes classification results (first 20 observations):",
                       20,
                       interval=15,
                       flt64=False)

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rankId = comm.Get_rank()

    trainModel()

    if rankId == MPI_ROOT:
        testModel()
        printResults()
