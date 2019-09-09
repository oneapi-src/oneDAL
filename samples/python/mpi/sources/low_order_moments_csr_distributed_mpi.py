# file: low_order_moments_csr_distributed_mpi.py
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
# !    Python sample of computing low order moments in the distributed processing
# !    mode.
# !
# !    Input matrix is stored in the compressed sparse row (CSR) format.
# !*****************************************************************************

#
## <a name="DAAL-SAMPLE-CPP-LOW_ORDER_MOMENTS_CSR_DISTRIBUTED"></a>
## \example low_order_moments_csr_distributed_mpi.py
#

import os
import sys
from os.path import join as jp

from mpi4py import MPI

from daal import step1Local, step2Master
from daal.algorithms import low_order_moments
from daal.data_management import OutputDataArchive, InputDataArchive

utils_folder = os.path.realpath(os.path.abspath(jp(os.environ['DAALROOT'], 'examples', 'python', 'source')))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable, createSparseTable

DATA_PREFIX = jp('data', 'distributed')

# Input data set parameters
nBlocks = 4
MPI_ROOT = 0

datasetFileNames = [
    jp(DATA_PREFIX, 'covcormoments_csr_1.csv'),
    jp(DATA_PREFIX, 'covcormoments_csr_2.csv'),
    jp(DATA_PREFIX, 'covcormoments_csr_3.csv'),
    jp(DATA_PREFIX, 'covcormoments_csr_4.csv')
]

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rankId = comm.Get_rank()

    # Retrieve the input data from a file
    dataTable = createSparseTable(datasetFileNames[rankId])

    # Create an algorithm to compute low order moments on local nodes
    localAlgorithm = low_order_moments.Distributed(step1Local, method=low_order_moments.fastCSR)

    # Set the input data set to the algorithm
    localAlgorithm.input.set(low_order_moments.data, dataTable)

    # Compute low order moments
    pres = localAlgorithm.compute()

    # Serialize partial results required by step 2
    dataArch = InputDataArchive()
    pres.serialize(dataArch)

    nodeResults = dataArch.getArchiveAsArray()

    # Transfer partial results to step 2 on the root node
    serializedData = comm.gather(nodeResults)

    if rankId == MPI_ROOT:
        # Create an algorithm to compute low order moments on the master node
        masterAlgorithm = low_order_moments.Distributed(step2Master, method=low_order_moments.fastCSR)

        for i in range(nBlocks):
            # Deserialize partial results from step 1
            dataArch = OutputDataArchive(serializedData[i])

            dataForStep2FromStep1 = low_order_moments.PartialResult()

            dataForStep2FromStep1.deserialize(dataArch)

            # Set local partial results as input for the master-node algorithm
            masterAlgorithm.input.add(low_order_moments.partialResults, dataForStep2FromStep1)

        # Merge and finalizeCompute low order moments on the master node
        masterAlgorithm.compute()
        res = masterAlgorithm.finalizeCompute()

        # Print the results
        printNumericTable(res.get(low_order_moments.minimum),              "Minimum:")
        printNumericTable(res.get(low_order_moments.maximum),              "Maximum:")
        printNumericTable(res.get(low_order_moments.sum),                  "Sum:")
        printNumericTable(res.get(low_order_moments.sumSquares),           "Sum of squares:")
        printNumericTable(res.get(low_order_moments.sumSquaresCentered),   "Sum of squared difference from the means:")
        printNumericTable(res.get(low_order_moments.mean),                 "Mean:")
        printNumericTable(res.get(low_order_moments.secondOrderRawMoment), "Second order raw moment:")
        printNumericTable(res.get(low_order_moments.variance),             "Variance:")
        printNumericTable(res.get(low_order_moments.standardDeviation),    "Standard deviation:")
        printNumericTable(res.get(low_order_moments.variation),            "Variation:")
