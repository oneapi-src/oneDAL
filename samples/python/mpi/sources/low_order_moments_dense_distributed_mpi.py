# file: low_order_moments_dense_distributed_mpi.py
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
# !    Python sample of computing low order moments in the distributed processing
# !    mode
# !
# !*****************************************************************************

#
## <a name="DAAL-SAMPLE-PY-LOW_ORDER_MOMENTS_DENSE_DISTRIBUTED"></a>
## \example low_order_moments_dense_distributed_mpi.py
#

import os
import sys
from os.path import join as jp

from mpi4py import MPI

from daal import step1Local, step2Master
from daal.algorithms import low_order_moments
from daal.data_management import OutputDataArchive, InputDataArchive, FileDataSource, DataSourceIface

utils_folder = os.path.realpath(os.path.abspath(jp(os.environ['DAALROOT'], 'examples', 'python', 'source')))
if utils_folder not in sys.path:
    sys.path.insert(0, utils_folder)
from utils import printNumericTable

DATA_PREFIX = jp('data', 'distributed')

# Input data set parameters
nBlocks = 4

MPI_ROOT = 0

datasetFileNames = [
    jp(DATA_PREFIX, 'covcormoments_dense_1.csv'),
    jp(DATA_PREFIX, 'covcormoments_dense_2.csv'),
    jp(DATA_PREFIX, 'covcormoments_dense_3.csv'),
    jp(DATA_PREFIX, 'covcormoments_dense_4.csv')
]

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rankId = comm.Get_rank()

    # Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file
    dataSource = FileDataSource(datasetFileNames[rankId],
                                DataSourceIface.doAllocateNumericTable,
                                DataSourceIface.doDictionaryFromContext)

    # Retrieve the input data
    dataSource.loadDataBlock()

    # Create an algorithm to compute low order moments on local nodes
    localAlgorithm = low_order_moments.Distributed(step=step1Local)

    # Set the input data set to the algorithm
    localAlgorithm.input.set(low_order_moments.data, dataSource.getNumericTable())

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
        masterAlgorithm = low_order_moments.Distributed(step=step2Master)

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
