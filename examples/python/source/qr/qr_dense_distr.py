# file: qr_dense_distr.py
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

## <a name="DAAL-EXAMPLE-PY-QR_DISTRIBUTED"></a>
## \example qr_dense_distr.py

import os
import sys

from daal import step1Local, step2Master, step3Local
from daal.algorithms import qr
from daal.data_management import FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

DAAL_PREFIX = os.path.join('..', 'data')

# Input data set parameters
nBlocks = 4

datasetFileNames = [
    os.path.join(DAAL_PREFIX, 'distributed', 'qr_1.csv'),
    os.path.join(DAAL_PREFIX, 'distributed', 'qr_2.csv'),
    os.path.join(DAAL_PREFIX, 'distributed', 'qr_3.csv'),
    os.path.join(DAAL_PREFIX, 'distributed', 'qr_4.csv')
]

dataFromStep1ForStep2 = [0] * nBlocks
dataFromStep1ForStep3 = [0] * nBlocks
dataFromStep2ForStep3 = [0] * nBlocks
R = None
Qi = [0] * nBlocks


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

    # Create an algorithm to compute QR decomposition on the local node
    algorithm = qr.Distributed(step1Local)

    algorithm.input.set(qr.data, dataSource.getNumericTable())

    # Compute QR decomposition and get OnlinePartialResult class from daal.algorithms.qr
    pres = algorithm.compute()

    dataFromStep1ForStep2[block] = pres.get(qr.outputOfStep1ForStep2)
    dataFromStep1ForStep3[block] = pres.get(qr.outputOfStep1ForStep3)


def computeOnMasterNode():
    global R, dataFromStep2ForStep3

    # Create an algorithm to compute QR decomposition on the master node
    algorithm = qr.Distributed(step2Master)

    for i in range(nBlocks):
        algorithm.input.add(qr.inputOfStep2FromStep1, i, dataFromStep1ForStep2[i])

    # Compute QR decomposition and get DistributedPartialResult class from daal.algorithms.qr
    pres = algorithm.compute()

    for i in range(nBlocks):
        dataFromStep2ForStep3[i] = pres.getCollection(qr.outputOfStep2ForStep3, i)

    res = algorithm.finalizeCompute()
    R = res.get(qr.matrixR)


def finalizeComputestep1Local(block):
    global Qi

    # Create an algorithm to compute QR decomposition on the master node
    algorithm = qr.Distributed(step3Local)

    algorithm.input.set(qr.inputOfStep3FromStep1, dataFromStep1ForStep3[block])
    algorithm.input.set(qr.inputOfStep3FromStep2, dataFromStep2ForStep3[block])

    # Compute QR decomposition
    algorithm.compute()

    res = algorithm.finalizeCompute()

    Qi[block] = res.get(qr.matrixQ)

if __name__ == "__main__":

    for i in range(nBlocks):
        computestep1Local(i)

    computeOnMasterNode()

    for i in range(nBlocks):
        finalizeComputestep1Local(i)

    # Print the results
    printNumericTable(Qi[0], "Part of orthogonal matrix Q from 1st node:", 10)
    printNumericTable(R, "Triangular matrix R:")
