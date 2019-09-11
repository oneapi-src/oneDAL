# file: svd_fast_distributed_mpi.py
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
# !    Python sample of computing singular value decomposition (SVD) in the
# !    distributed processing mode
# !*****************************************************************************

#
## <a name="DAAL-EXAMPLE-PY-SVD_FAST_DISTRIBUTED_MPI"></a>
## \example svd_fast_distributed_mpi.py
#

import os
import sys
from os.path import join as jp

from mpi4py import MPI
import numpy as np

from daal import step1Local, step2Master, step3Local
from daal.algorithms import svd
from daal.data_management import (
    FileDataSource, InputDataArchive, OutputDataArchive, DataSourceIface, DataCollection
)

utils_folder = os.path.realpath(os.path.abspath(jp(os.environ['DAALROOT'], 'examples', 'python', 'source')))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

# Input data set parameters
nBlocks = 4
MPI_ROOT = 0
DAAL_PREFIX = jp('data', 'distributed')

datasetFileNames = [
    jp(DAAL_PREFIX, 'svd_1.csv'),
    jp(DAAL_PREFIX, 'svd_2.csv'),
    jp(DAAL_PREFIX, 'svd_3.csv'),
    jp(DAAL_PREFIX, 'svd_4.csv')
]

dataFromStep1ForStep3 = None
Sigma = None
V = None
Ui = None

serializedData = None


def computestep1Local():
    global serializedData, dataFromStep1ForStep3

    # Initialize FileDataSource to retrieve the input data from a .csv file
    dataSource = FileDataSource(datasetFileNames[rankId],
                                DataSourceIface.doAllocateNumericTable,
                                DataSourceIface.doDictionaryFromContext)

    # Retrieve the input data
    dataSource.loadDataBlock()

    # Create an algorithm to compute SVD on local nodes
    algorithm = svd.Distributed(step1Local)

    algorithm.input.set(svd.data, dataSource.getNumericTable())

    # Compute SVD
    # OnlinePartialResult class from svd
    pres = algorithm.compute()

    dataFromStep1ForStep2 = pres.get(svd.outputOfStep1ForStep2)
    dataFromStep1ForStep3 = pres.get(svd.outputOfStep1ForStep3)

    # Serialize partial results required by step 2
    dataArch = InputDataArchive()
    dataFromStep1ForStep2.serialize(dataArch)

    nodeResults = dataArch.getArchiveAsArray()

    # Transfer partial results to step 2 on the root node
    serializedData = comm.gather(nodeResults)


def computeOnMasterNode():
    global serializedData, Sigma, V

    # Create an algorithm to compute SVD on the master node
    algorithm = svd.Distributed(step2Master)

    for i in range(nBlocks):
        # Deserialize partial results from step 1
        dataArch = OutputDataArchive(serializedData[i])

        dataForStep2FromStep1 = DataCollection()
        dataForStep2FromStep1.deserialize(dataArch)

        algorithm.input.add(svd.inputOfStep2FromStep1, i, dataForStep2FromStep1)

    # Compute SVD
    # DistributedPartialResult class from svd
    pres = algorithm.compute()

    inputForStep3FromStep2 = pres.getCollection(svd.outputOfStep2ForStep3)

    for i in range(nBlocks):
        # Serialize partial results to transfer to local nodes for step 3
        dataArch = InputDataArchive()
        inputForStep3FromStep2[i].serialize(dataArch)
        length = dataArch.getSizeOfArchive()

        serializedData[i] = np.empty(length, dtype=np.uint8)
        dataArch.copyArchiveToArray(serializedData[i])

    # DistributedPartialResult class from svd
    res = algorithm.getResult()

    Sigma = res.get(svd.singularValues)
    V = res.get(svd.rightSingularMatrix)


def finalizeComputestep1Local():
    global Ui, serializedData

    # Transfer partial results from the root node
    nodeResults = comm.scatter(serializedData)

    # Deserialize partial results from step 2
    dataArch = OutputDataArchive(nodeResults)

    dataFromStep2ForStep3 = DataCollection()
    dataFromStep2ForStep3.deserialize(dataArch)

    # Create an algorithm to compute SVD on the master node
    algorithm = svd.Distributed(step3Local)
    algorithm.input.set(svd.inputOfStep3FromStep1, dataFromStep1ForStep3)
    algorithm.input.set(svd.inputOfStep3FromStep2, dataFromStep2ForStep3)

    # Compute SVD
    algorithm.compute()
    res = algorithm.finalizeCompute()

    Ui = res.get(svd.leftSingularMatrix)

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    rankId = comm.Get_rank()

    if nBlocks != comm_size:
        if rankId == MPI_ROOT:
            frmt = "{} MPI ranks != {} datasets available, so please start exactly {} ranks."
            print(frmt.format(comm_size, nBlocks, nBlocks))
        sys.exit(0)

    computestep1Local()

    if rankId == MPI_ROOT:
        computeOnMasterNode()

    finalizeComputestep1Local()

    # Print the results
    if rankId == MPI_ROOT:
        printNumericTable(Sigma, "Singular values:")
        printNumericTable(V, "Right orthogonal matrix V:")
        printNumericTable(Ui, "Part of left orthogonal matrix U from root node:", 10)
