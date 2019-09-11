# file: implicit_als_csr_distributed_mpi.py
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
# !    Python example of the implicit alternating least squares (ALS) algorithm in
# !    the distributed processing mode.
# !
# !    The program trains the implicit ALS model on a training data set.
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-IMPLICIT_ALS_CSR_DISTRIBUTED"></a>
## \example implicit_als_csr_distributed_mpi.py
#

import os
import sys
from os.path import join as jp

import numpy as np
from mpi4py import MPI

from daal import step1Local, step2Local, step2Master, step3Local, step4Local

import daal.algorithms.implicit_als as implicit_als
import daal.algorithms.implicit_als.training as training
import daal.algorithms.implicit_als.training.init as init
import daal.algorithms.implicit_als.prediction.ratings as ratings

from daal.data_management import (
    KeyValueDataCollection, HomogenNumericTable, CSRNumericTable, OutputDataArchive, InputDataArchive
)

utils_folder = os.path.realpath(os.path.abspath(jp(os.environ['DAALROOT'], 'examples', 'python', 'source')))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import createSparseTable, printNumericTable


# Input data set parameters
nBlocks = 4
MPI_ROOT = 0
DATA_DIR = jp('data', 'distributed')

# Number of observations in transposed training data set blocks
transposedTrainDatasetFileNames = [
    jp(DATA_DIR, 'implicit_als_trans_csr_1.csv'),
    jp(DATA_DIR, 'implicit_als_trans_csr_2.csv'),
    jp(DATA_DIR, 'implicit_als_trans_csr_3.csv'),
    jp(DATA_DIR, 'implicit_als_trans_csr_4.csv')
]

# Algorithm parameters
nUsers = 46  # Full number of users
nFactors = 2  # Number of factors
maxIterations = 5  # Number of iterations in the implicit ALS training algorithm

itemsPartialResultLocal = None
itemsPartialResultsMaster = [0] * nBlocks

usersPartition = [[nBlocks]]

userOffset = None
itemOffset = None

userStep3LocalInput = None
itemStep3LocalInput = None

usersPartialResultLocal = None

predictedRatingsMaster = [[0] * nBlocks for i in range(nBlocks)]


def initializeStep1Local():
    global itemsPartialResultLocal, itemStep3LocalInput, userOffset, usersPartition

    # Create an algorithm object to initialize the implicit ALS model with the fastCSR method
    initAlgorithm = init.Distributed(step=step1Local)
    initAlgorithm.parameter.fullNUsers = nUsers
    initAlgorithm.parameter.nFactors = nFactors
    initAlgorithm.parameter.seed += rankId
    initAlgorithm.parameter.partition = HomogenNumericTable(np.array(usersPartition, dtype=np.float64))

    # Pass a training data set to the algorithm
    initAlgorithm.input.set(init.data, transposedDataTable)

    # Initialize the implicit ALS model
    partialResult = initAlgorithm.compute()
    itemStep3LocalInput = partialResult.getCollection(init.outputOfInitForComputeStep3)
    userOffset         = partialResult.getCollection(init.offsets, rankId)
    partialModelLocal   = partialResult.getPartialModel(init.partialModel)

    itemsPartialResultLocal = training.DistributedPartialResultStep4()
    itemsPartialResultLocal.set(training.outputOfStep4ForStep1, partialModelLocal)

    return partialResult.getTablesCollection(init.outputOfStep1ForStep2)

def initializeStep2Local(initStep2LocalInput):
    global dataTable, userStep3LocalInput, itemOffset

    # Create an algorithm object to initialize the implicit ALS model with the fastCSR method
    initAlgorithm = init.Distributed(step=step2Local)

    initAlgorithm.input.set(init.inputOfStep2FromStep1, initStep2LocalInput)

    # Initialize the implicit ALS model
    partialResult = initAlgorithm.compute()

    userStep3LocalInput = partialResult.getCollection(init.outputOfInitForComputeStep3)
    itemOffset         = partialResult.getCollection(init.offsets, rankId)

    return partialResult.getTable(init.transposedData)

def initializeModel():
    global usersPartition

    initStep1LocalResult = initializeStep1Local()

    # MPI_Alltoallv to populate initStep2LocalInput
    initStep2LocalInput = KeyValueDataCollection()
    for i in range(nBlocks):
        initStep2LocalInput[i] = CSRNumericTable()

    initStep2LocalInput = all2all(initStep1LocalResult, initStep2LocalInput)

    return initializeStep2Local(initStep2LocalInput)

def computeStep1Local(partialResultLocal):

    # Create algorithm objects to compute implicit ALS algorithm in the distributed processing mode on the local node using the default method
    algorithm = training.Distributed(step1Local)
    algorithm.parameter.nFactors = nFactors

    # Set input objects for the algorithm
    algorithm.input.set(training.partialModel, partialResultLocal.get(training.outputOfStep4ForStep1))

    # Compute partial estimates on local nodes
    return algorithm.compute()


def computeStep2Master(step1LocalResult):

    # Create algorithm objects to compute implicit ALS algorithm in the distributed processing mode on the master node using the default method
    algorithm = training.Distributed(step2Master)
    algorithm.parameter.nFactors = nFactors

    # Set input objects for the algorithm
    for i in range(nBlocks):
        algorithm.input.add(training.inputOfStep2FromStep1, step1LocalResult[i])

    # Compute a partial estimate on the master node from the partial estimates on local nodes
    pres = algorithm.compute()

    return pres.get(training.outputOfStep2ForStep4)


def computeStep3Local(offset, partialResultLocal, step3LocalInput):
    global step4LocalInput

    algorithm = training.Distributed(step3Local)
    algorithm.parameter.nFactors = nFactors

    algorithm.input.setModel(training.partialModel, partialResultLocal.get(training.outputOfStep4ForStep3))
    algorithm.input.setCollection(training.inputOfStep3FromInit, step3LocalInput)
    algorithm.input.setTable(training.offset, offset)

    pres = algorithm.compute()

    return pres.get(training.outputOfStep3ForStep4)


def computeStep4Local(dataTable, step2MasterResult, step4LocalInput):

    algorithm = training.Distributed(step4Local)
    algorithm.parameter.nFactors = nFactors
    algorithm.input.setModels(training.partialModels, step4LocalInput)
    algorithm.input.setTable(training.partialData, dataTable)
    algorithm.input.setTable(training.inputOfStep4FromStep2, step2MasterResult)

    return algorithm.compute()


def trainModel(dataTable, transposedDataTable):
    global usersPartialResultLocal, itemsPartialResultLocal

    step4LocalInput = KeyValueDataCollection()

    for iteration in range(maxIterations):
        step1LocalResult = computeStep1Local(itemsPartialResultLocal)

        # Gathering step1LocalResult on the master
        step1LocalResultsMaster = [None] * nBlocks
        if rankId == MPI_ROOT:
            for i in range(nBlocks):
                step1LocalResultsMaster[i] = training.DistributedPartialResultStep1()

        step1LocalResultsMaster = gather(step1LocalResult, step1LocalResultsMaster)

        crossProductBuf = None

        if rankId == MPI_ROOT:
            step2MasterResultRoot = computeStep2Master(step1LocalResultsMaster)
            crossProductBuf = serializeDAALObject(step2MasterResultRoot)

        crossProductBuf = comm.bcast(crossProductBuf)

        step2MasterResult = deserializeDAALObject(crossProductBuf, HomogenNumericTable())

        step3LocalResult = computeStep3Local(itemOffset, itemsPartialResultLocal, itemStep3LocalInput)

        # MPI_Alltoallv to populate step4LocalInput
        for i in range(nBlocks):
            step4LocalInput[i] = implicit_als.PartialModel()

        step4LocalInput = all2all(step3LocalResult, step4LocalInput)

        usersPartialResultLocal = computeStep4Local(dataTable, step2MasterResult, step4LocalInput)

        step1LocalResult = computeStep1Local(usersPartialResultLocal)

        # Gathering step1LocalResult on the master
        step1LocalResultsMaster = [None] * nBlocks
        if rankId == MPI_ROOT:
            for i in range(nBlocks):
                step1LocalResultsMaster[i] = training.DistributedPartialResultStep1()

        step1LocalResultsMaster = gather(step1LocalResult, step1LocalResultsMaster)

        if rankId == MPI_ROOT:
            step2MasterResultRoot = computeStep2Master(step1LocalResultsMaster)
            crossProductBuf = serializeDAALObject(step2MasterResultRoot)

        crossProductBuf = comm.bcast(crossProductBuf)

        step2MasterResult = deserializeDAALObject(crossProductBuf, HomogenNumericTable())

        step3LocalResult = computeStep3Local(userOffset, usersPartialResultLocal, userStep3LocalInput)

        # MPI_Alltoallv to populate step4LocalInput
        for i in range(nBlocks):
            step4LocalInput[i] = implicit_als.PartialModel()

        step4LocalInput = all2all(step3LocalResult, step4LocalInput)

        itemsPartialResultLocal = computeStep4Local(transposedDataTable, step2MasterResult, step4LocalInput)

    # Gather all itemsPartialResultLocal to itemsPartialResultsMaster on the master and distributing the result over other ranks
    nodeResults = serializeDAALObject(itemsPartialResultLocal)
    gatherItems(nodeResults)


def testModel():

    # Create an algorithm object to predict recommendations of the implicit ALS model
    for i in range(nBlocks):
        algorithm = ratings.Distributed(step1Local, method=ratings.allUsersAllItems)
        algorithm.parameter.nFactors = nFactors

        algorithm.input.set(ratings.usersPartialModel, usersPartialResultLocal.get(training.outputOfStep4ForStep1))
        algorithm.input.set(ratings.itemsPartialModel, itemsPartialResultsMaster[i].get(training.outputOfStep4ForStep1))

        res = algorithm.compute()

        predictedRatingsLocal = res.get(ratings.prediction)

        for j in range(nBlocks):
            predictedRatingsMaster[i][j] = ratings.Result()
        predictedRatingsMaster[i] = gather(predictedRatingsLocal, predictedRatingsMaster[i])


def gather(input, result):

    nodeResults = serializeDAALObject(input)

    serializedData = None
    perNodeArchLengthMaster = comm.gather(len(nodeResults))

    displs = [0] * nBlocks
    if rankId == MPI_ROOT:
        memoryBuf = sum(perNodeArchLengthMaster)
        serializedData = np.zeros(memoryBuf, dtype=np.uint8)
        shift = 0
        for i in range(nBlocks):
            displs[i] = shift
            shift += perNodeArchLengthMaster[i]

    # Transfer partial results to step 2 on the root node
    comm.Gatherv([nodeResults, len(nodeResults), MPI.CHAR],
                 [serializedData, perNodeArchLengthMaster, displs, MPI.CHAR])

    if rankId == MPI_ROOT:

        for i in range(nBlocks):
            # Deserialize partial results from step 1
            start = displs[i]
            end = start + perNodeArchLengthMaster[i]
            piece = serializedData[start:end]
            result[i] = deserializeDAALObject(piece, result[i])

    return result


def gatherItems(nodeResults):
    global itemsPartialResultsMaster

    perNodeArchLengthMaster = comm.allgather(len(nodeResults))

    displs = [0] * nBlocks
    memoryBuf = sum(perNodeArchLengthMaster)
    serializedData = np.zeros(memoryBuf, dtype=np.uint8)

    shift = 0
    for i in range(nBlocks):
        displs[i] = shift
        shift += perNodeArchLengthMaster[i]

    # Transfer partial results to step 2 on the root node
    comm.Allgatherv([nodeResults, len(nodeResults), MPI.CHAR],
                    [serializedData, perNodeArchLengthMaster, displs, MPI.CHAR])

    for i in range(nBlocks):
        # Deserialize partial results from step 4
        start = displs[i]
        end = start + perNodeArchLengthMaster[i]
        piece = serializedData[start:end]
        itemsPartialResultsMaster[i] = deserializeDAALObject(piece, training.DistributedPartialResultStep4())


def all2all(input, result):

    nodeResults = [0] * nBlocks
    perNodeArchLengths = [0] * nBlocks
    for i in range(nBlocks):
        nodeResults[i] = serializeDAALObject(input[i])
        perNodeArchLengths[i] = len(nodeResults[i])

    shift = 0
    sdispls = [0] * nBlocks
    for i in range(nBlocks):
        sdispls[i] = shift
        shift += perNodeArchLengths[i]

    # memcpy to avoid double compute
    serializedSendData = np.concatenate([x for x in nodeResults])

    perNodeArchLengthsRecv = comm.alltoall(perNodeArchLengths)

    memoryBuf = sum(perNodeArchLengthsRecv)
    shift = 0
    rdispls = [0] * nBlocks
    for i in range(nBlocks):
        rdispls[i] = shift
        shift += perNodeArchLengthsRecv[i]

    serializedRecvData = np.zeros(memoryBuf, dtype=np.uint8)

    # Transfer partial results to step 2 on the root node
    comm.Alltoallv([serializedSendData, perNodeArchLengths, sdispls, MPI.CHAR],
                   [serializedRecvData, perNodeArchLengthsRecv, rdispls, MPI.CHAR])

    for i in range(nBlocks):
        start = rdispls[i]
        end = start + perNodeArchLengthsRecv[i]
        piece = serializedRecvData[start:end]
        result[i] = deserializeDAALObject(piece, result[i])

    return result

def serializeDAALObject(data):
    # Create a data archive to serialize the numeric table
    dataArch = InputDataArchive()

    # Serialize the numeric table into the data archive
    data.serialize(dataArch)

    length = dataArch.getSizeOfArchive()
    buffer = np.zeros(length, dtype=np.uint8)
    dataArch.copyArchiveToArray(buffer)
    return buffer


def deserializeDAALObject(buffer, object):
    # Create a data archive to deserialize the numeric table
    dataArch = OutputDataArchive(buffer)

    # Deserialize the numeric table from the data archive
    object.deserialize(dataArch)

    return object


if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rankId = comm.Get_rank()

    transposedDataTable = createSparseTable(transposedTrainDatasetFileNames[rankId])

    step4LocalInput = KeyValueDataCollection()
    itemsPartialResultPrediction = KeyValueDataCollection()

    dataTable = initializeModel()
    trainModel(dataTable, transposedDataTable)
    testModel()

    if rankId == MPI_ROOT:
        for i in range(nBlocks):
            for j in range(nBlocks):
                print("prediction {}, {}".format(i, j))
                printNumericTable(predictedRatingsMaster[i][j].get(ratings.prediction))
