# file: qr_fast_distributed_mpi.py
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
# !    Python sample of computing QR decomposition in the distributed processing
# !    mode
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-QR_FAST_DISTRIBUTED_MPI"></a>
## \example qr_fast_distributed_mpi.py
#

import os
import sys
from os.path import join as jp

from mpi4py import MPI
import numpy as np

from daal import step1Local, step2Master, step3Local
from daal.algorithms import qr
from daal.data_management import (
    DataSourceIface, FileDataSource, OutputDataArchive, InputDataArchive, DataCollection
)

utils_folder = os.path.realpath(os.path.abspath(jp(os.environ['DAALROOT'], 'examples', 'python', 'source')))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

DATA_PREFIX = jp('data', 'distributed')

# Input data set parameters
nBlocks = 4
MPI_ROOT = 0

datasetFileNames = [
    jp(DATA_PREFIX, 'qr_1.csv'),
    jp(DATA_PREFIX, 'qr_2.csv'),
    jp(DATA_PREFIX, 'qr_3.csv'),
    jp(DATA_PREFIX, 'qr_4.csv')
]

dataFromStep1ForStep3 = None
R = None
Qi = None
serializedData = None


def computestep1Local():
    global serializedData, dataFromStep1ForStep3

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    dataSource = FileDataSource(datasetFileNames[rankId],
                                DataSourceIface.doAllocateNumericTable,
                                DataSourceIface.doDictionaryFromContext)

    # Retrieve the input data
    dataSource.loadDataBlock()

    # Create an algorithm to compute QR decomposition on local nodes
    alg = qr.Distributed(step1Local)

    alg.input.set(qr.data, dataSource.getNumericTable())

    # Compute QR decomposition
    pres = alg.compute()

    dataFromStep1ForStep2 = pres.get(qr.outputOfStep1ForStep2)
    dataFromStep1ForStep3 = pres.get(qr.outputOfStep1ForStep3)

    # Serialize partial results required by step 2
    dataArch = InputDataArchive()
    dataFromStep1ForStep2.serialize(dataArch)

    nodeResults = dataArch.getArchiveAsArray()

    # Transfer partial results to step 2 on the root node
    serializedData = comm.gather(nodeResults)


def computeOnMasterNode():
    global R, serializedData

    # Create an algorithm to compute QR decomposition on the master node
    algorithm = qr.Distributed(step2Master)

    for i in range(nBlocks):
        # Deserialize partial results from step 1
        dataArch = OutputDataArchive(serializedData[i])

        dataForStep2FromStep1 = DataCollection()
        dataForStep2FromStep1.deserialize(dataArch)

        algorithm.input.add(qr.inputOfStep2FromStep1, i, dataForStep2FromStep1)

    # Compute QR decomposition
    pres = algorithm.compute()

    inputForStep3FromStep2 = pres.getCollection(qr.outputOfStep2ForStep3)

    for i in range(nBlocks):
        # Serialize partial results to transfer to local nodes for step 3
        dataArch = InputDataArchive()
        inputForStep3FromStep2[i].serialize(dataArch)
        length = dataArch.getSizeOfArchive()

        serializedData[i] = np.empty(length, dtype=np.uint8)
        dataArch.copyArchiveToArray(serializedData[i])

    # Result class from qr
    res = algorithm.getResult()

    R = res.get(qr.matrixR)


def finalizeComputestep1Local():
    global Qi, serializedData

    # Transfer partial results from the root node
    nodeResults = comm.scatter(serializedData)

    # Deserialize partial results from step 2
    dataArch = OutputDataArchive(nodeResults)

    dataFromStep2ForStep3 = DataCollection()
    dataFromStep2ForStep3.deserialize(dataArch)

    # Create an algorithm to compute QR decomposition on the master node
    algorithm = qr.Distributed(step3Local)

    algorithm.input.set(qr.inputOfStep3FromStep1, dataFromStep1ForStep3)
    algorithm.input.set(qr.inputOfStep3FromStep2, dataFromStep2ForStep3)

    # Compute QR decomposition
    algorithm.compute()
    res = algorithm.finalizeCompute()

    Qi = res.get(qr.matrixQ)

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    rankId = comm.Get_rank()

    if nBlocks != comm_size:
        if rankId == MPI_ROOT:
            print("{} MPI ranks != {} datasets available, so please start exactly {} ranks.", comm_size, nBlocks, nBlocks)
        sys.exit(0)

    computestep1Local()

    if rankId == MPI_ROOT:
        computeOnMasterNode()

    finalizeComputestep1Local()

    # Print the results
    if rankId == MPI_ROOT:
        printNumericTable(Qi, "Part of orthogonal matrix Q from 1st node:", 10)
        printNumericTable(R, "Triangular matrix R:")
