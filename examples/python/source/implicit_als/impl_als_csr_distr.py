# file: impl_als_csr_distr.py
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

## <a name="DAAL-EXAMPLE-PY-IMPLICIT_ALS_CSR_DISTRIBUTED"></a>
## \example impl_als_csr_distr.py

import os
import sys

import numpy as np

from daal import step1Local, step2Local, step2Master, step3Local, step4Local
import daal.algorithms.implicit_als.prediction.ratings as ratings
import daal.algorithms.implicit_als.training as training
import daal.algorithms.implicit_als.training.init as init
from daal.data_management import KeyValueDataCollection, HomogenNumericTable

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import createSparseTable, printALSRatings

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
nBlocks = 4

# Number of observations in transposed training data set blocks
trainDatasetFileNames = [
    os.path.join(DAAL_PREFIX, 'distributed', 'implicit_als_trans_csr_1.csv'),
    os.path.join(DAAL_PREFIX, 'distributed', 'implicit_als_trans_csr_2.csv'),
    os.path.join(DAAL_PREFIX, 'distributed', 'implicit_als_trans_csr_3.csv'),
    os.path.join(DAAL_PREFIX, 'distributed', 'implicit_als_trans_csr_4.csv')
]

usersPartition = [0] * 1
usersPartition[0] = nBlocks

userOffsets = [0] * nBlocks
itemOffsets = [0] * nBlocks

# Algorithm parameters
nUsers = 46        # Full number of users
nFactors = 2       # Number of factors
maxIterations = 5  # Number of iterations in the implicit ALS training algorithm

dataTable = [0] * nBlocks
transposedDataTable = [0] * nBlocks

predictedRatings = [[0] * nBlocks for x in range(nBlocks)]

userStep3LocalInput = [0] * nBlocks
itemStep3LocalInput = [0] * nBlocks

itemsPartialResultLocal = [0] * nBlocks
usersPartialResultLocal = [0] * nBlocks

def readData(block):
    global dataTable

    # Read trainDatasetFileName from a file and create a numeric table to store the input data
    dataTable[block] = createSparseTable(trainDatasetFileNames[block])


def initializeStep1Local(block):
    global itemsPartialResultLocal
    global itemStep3LocalInput
    global userOffsets

    # Create an algorithm object to initialize the implicit ALS model with the fastCSR method
    initAlgorithm = init.Distributed(step=step1Local, method=init.fastCSR)
    initAlgorithm.parameter.fullNUsers = nUsers
    initAlgorithm.parameter.nFactors = nFactors
    initAlgorithm.parameter.seed += block
    usersPartitionArray = np.array(usersPartition, dtype=np.float64)
    usersPartitionArray.shape = (1, 1)

    initAlgorithm.parameter.partition = HomogenNumericTable(usersPartitionArray)

    # Pass a training data set to the algorithm
    initAlgorithm.input.set(init.data, dataTable[block])

    # Initialize the implicit ALS model
    partialResult = initAlgorithm.compute()
    itemStep3LocalInput[block] = partialResult.getCollection(init.outputOfInitForComputeStep3)
    userOffsets[block]         = partialResult.getCollection(init.offsets, block)
    partialModelLocal          = partialResult.getPartialModel(init.partialModel)

    itemsPartialResultLocal[block] = training.DistributedPartialResultStep4()
    itemsPartialResultLocal[block].set(training.outputOfStep4ForStep1, partialModelLocal)

    return partialResult.getTablesCollection(init.outputOfStep1ForStep2)

def initializeStep2Local(block, initStep2LocalInput):
    global transposedDataTable
    global userStep3LocalInput
    global itemOffsets
    # Create an algorithm object to initialize the implicit ALS model with the fastCSR method
    initAlgorithm = init.Distributed(step=step2Local, method=init.fastCSR)

    initAlgorithm.input.set(init.inputOfStep2FromStep1, initStep2LocalInput)

    # Initialize the implicit ALS model
    partialResult = initAlgorithm.compute()

    transposedDataTable[block] = partialResult.getTable(init.transposedData)
    userStep3LocalInput[block] = partialResult.getCollection(init.outputOfInitForComputeStep3)
    itemOffsets[block]         = partialResult.getCollection(init.offsets, block)

def initializeModel():
    initStep1LocalResult = [0] * nBlocks

    for i in range(nBlocks):
        initStep1LocalResult[i] = initializeStep1Local(i)

    initStep2LocalInput = [0] * nBlocks

    for i in range(nBlocks):
        initStep2LocalInput[i] = KeyValueDataCollection()
        for j in range(nBlocks):
            initStep2LocalInput[i][j] = initStep1LocalResult[j][i]

    for i in range(nBlocks):
        initializeStep2Local(i, initStep2LocalInput[i])


def computeStep1Local(partialResultLocal):

    # Create an algorithm object to perform first step of the implicit ALS training algorithm on local-node data
    algorithm = training.Distributed(step=step1Local)
    algorithm.parameter.nFactors = nFactors

    # Set input objects for the algorithm
    algorithm.input.set(training.partialModel, partialResultLocal.get(training.outputOfStep4ForStep1))

    # Compute partial results of the first step on local nodes
    # DistributedPartialResultStep1 class from training
    return algorithm.compute()


def computeStep2Master(step1LocalResult):

    # Create an algorithm object to perform second step of the implicit ALS training algorithm
    algorithm = training.Distributed(step=step2Master)
    algorithm.parameter.nFactors = nFactors

    # Set the partial results of the first local step of distributed computations
    # as input for the master-node algorithm
    for i in range(nBlocks):
        algorithm.input.add(training.inputOfStep2FromStep1, step1LocalResult[i])

    # Compute a partial result on the master node from the partial results on local nodes
    # DistributedPartialResultStep2 class from training
    res = algorithm.compute()
    return res.get(training.outputOfStep2ForStep4)


def computeStep3Local(offsets, partialResultLocal, step3LocalInput):

    # Create an algorithm object to perform third step of the implicit ALS training algorithm on local-node data
    algorithm = training.Distributed(step=step3Local)
    algorithm.parameter.nFactors = nFactors

    # Set input objects for the algorithm
    algorithm.input.setModel(training.partialModel, partialResultLocal.get(training.outputOfStep4ForStep3))
    algorithm.input.setCollection(training.inputOfStep3FromInit, step3LocalInput)
    algorithm.input.setTable(training.offset, offsets)

    # Compute partial results of the third step on local nodes
    # DistributedPartialResultStep3 class from training
    res = algorithm.compute()
    return res.get(training.outputOfStep3ForStep4)


def computeStep4Local(dataTable, step2MasterResult, step4LocalInput):

    # Create an algorithm object to perform fourth step of the implicit ALS training algorithm on local-node data
    algorithm = training.Distributed(step=step4Local)
    algorithm.parameter.nFactors = nFactors

    # Set input objects for the algorithm
    algorithm.input.setModels(training.partialModels, step4LocalInput)
    algorithm.input.setTable(training.partialData, dataTable)
    algorithm.input.setTable(training.inputOfStep4FromStep2, step2MasterResult)

    # Build the implicit ALS partial model on the local node
    # DistributedPartialResultStep4 class from training
    return algorithm.compute()


def trainModel():

    step1LocalResult = [0] * nBlocks
    step3LocalResult = [0] * nBlocks
    step4LocalInput  = [0] * nBlocks

    for i in range(nBlocks):
        step4LocalInput[i] = KeyValueDataCollection()

    for iteration in range(maxIterations):

        # Update partial users factors
        for i in range(nBlocks):
            step1LocalResult[i] = computeStep1Local(itemsPartialResultLocal[i])

        step2MasterResult = computeStep2Master(step1LocalResult)

        for i in range(nBlocks):
            step3LocalResult[i] = computeStep3Local(itemOffsets[i], itemsPartialResultLocal[i], itemStep3LocalInput[i])

        # Prepare input objects for the fourth step of the distributed algorithm
        for i in range(nBlocks):
            for j in range(nBlocks):
                step4LocalInput[i][j] = step3LocalResult[j][i]

        for i in range(nBlocks):
            usersPartialResultLocal[i] = computeStep4Local(transposedDataTable[i], step2MasterResult, step4LocalInput[i])

        # Update partial items factors
        for i in range(nBlocks):
            step1LocalResult[i] = computeStep1Local(usersPartialResultLocal[i])

        step2MasterResult = computeStep2Master(step1LocalResult)

        for i in range(nBlocks):
            step3LocalResult[i] = computeStep3Local(userOffsets[i], usersPartialResultLocal[i], userStep3LocalInput[i])

        # Prepare input objects for the fourth step of the distributed algorithm
        for i in range(nBlocks):
            for j in range(nBlocks):
                step4LocalInput[i][j] = step3LocalResult[j][i]

        for i in range(nBlocks):
            itemsPartialResultLocal[i] = computeStep4Local(dataTable[i], step2MasterResult, step4LocalInput[i])


def testModel(i, j):
    # Create an algorithm object to predict ratings based in the implicit ALS partial models
    algorithm = ratings.Distributed(step=step1Local, method=ratings.defaultDense)
    algorithm.parameter.nFactors = nFactors

    # Set input objects for the algorithm
    algorithm.input.set(ratings.usersPartialModel, usersPartialResultLocal[i].get(training.outputOfStep4))
    algorithm.input.set(ratings.itemsPartialModel, itemsPartialResultLocal[j].get(training.outputOfStep4))

    # Predict ratings and retrieve the algorithm results
    algorithm.compute()

    # Result class from ratings
    res = algorithm.finalizeCompute()
    return res.get(ratings.prediction)


def printResults():

    for i in range(nBlocks):
        for j in range(nBlocks):
            print("Ratings for users block {}, items block {} :".format(i, j))
            printALSRatings(userOffsets[i], itemOffsets[j], predictedRatings[i][j])

if __name__ == "__main__":
    for i in range(nBlocks):
        readData(i)

    initializeModel()

    trainModel()

    for i in range(nBlocks):
        for j in range(nBlocks):
            predictedRatings[i][j] = testModel(i, j)

    printResults()
