# file: svd_dense_distr.py
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

## <a name="DAAL-EXAMPLE-PY-SVD_DISTRIBUTED"></a>
## \example svd_dense_distr.py

import os
import sys
import numpy as np

from daal import step1Local, step2Master, step3Local
from daal.algorithms import svd
from daal.data_management import FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
nBlocks = 4

datasetFileNames = [
    os.path.join(DAAL_PREFIX, 'distributed', 'svd_1.csv'),
    os.path.join(DAAL_PREFIX, 'distributed', 'svd_2.csv'),
    os.path.join(DAAL_PREFIX, 'distributed', 'svd_3.csv'),
    os.path.join(DAAL_PREFIX, 'distributed', 'svd_4.csv')
]

dataFromStep1ForStep2 = [0] * nBlocks
dataFromStep1ForStep3 = [0] * nBlocks
dataFromStep2ForStep3 = [0] * nBlocks
Sigma = None
V = None
Ui = [0] * nBlocks


def computestep1Local(block):
    global dataFromStep1ForStep2, dataFromStep1ForStep3

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    dataSource = FileDataSource(
        datasetFileNames[block],
        DataSourceIface.doAllocateNumericTable,
        DataSourceIface.doDictionaryFromContext
    )

    # Retrieve the input data
    dataSource.loadDataBlock()

    # Create an algorithm to compute SVD on the local node
    algorithm = svd.Distributed(step1Local,fptype=np.float64)

    algorithm.input.set(svd.data, dataSource.getNumericTable())

    # Compute SVD and get OnlinePartialResult class from daal.algorithms.svd
    pres = algorithm.compute()

    dataFromStep1ForStep2[block] = pres.get(svd.outputOfStep1ForStep2)
    dataFromStep1ForStep3[block] = pres.get(svd.outputOfStep1ForStep3)


def computeOnMasterNode():
    global Sigma, V, dataFromStep2ForStep3

    # Create an algorithm to compute SVD on the master node
    algorithm = svd.Distributed(step2Master,fptype=np.float64)

    for i in range(nBlocks):
        algorithm.input.add(svd.inputOfStep2FromStep1, i, dataFromStep1ForStep2[i])

    # Compute SVD and get DistributedPartialResult class from daal.algorithms.svd
    pres = algorithm.compute()

    for i in range(nBlocks):
        dataFromStep2ForStep3[i] = pres.getCollection(svd.outputOfStep2ForStep3, i)

    res = algorithm.finalizeCompute()

    Sigma = res.get(svd.singularValues)
    V = res.get(svd.rightSingularMatrix)


def finalizeComputestep1Local(block):
    global Ui

    # Create an algorithm to compute SVD on the master node
    algorithm = svd.Distributed(step3Local,fptype=np.float64)

    algorithm.input.set(svd.inputOfStep3FromStep1, dataFromStep1ForStep3[block])
    algorithm.input.set(svd.inputOfStep3FromStep2, dataFromStep2ForStep3[block])

    # Compute SVD
    algorithm.compute()
    res = algorithm.finalizeCompute()

    Ui[block] = res.get(svd.leftSingularMatrix)

if __name__ == "__main__":

    for i in range(nBlocks):
        computestep1Local(i)

    computeOnMasterNode()

    for i in range(nBlocks):
        finalizeComputestep1Local(i)

    # Print the results
    printNumericTable(Sigma, "Singular values:")
    printNumericTable(V,     "Right orthogonal matrix V:")
    printNumericTable(Ui[0], "Part of left orthogonal matrix U from 1st node:", 10)
